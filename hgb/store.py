"""
hgb/store.py
============

Persistence + resume.  Append-only JSONL logs make the benchmark restartable:

  runs_embed.jsonl        one record per (dataset, embedder, seed)
        {embed_time_s, ari, nmi, emb_path, emb_dim, n_nodes}
  runs_clf.jsonl          one record per (dataset, embedder, seed, classifier)
        {accuracy, macro_f1, split_accuracy[], split_macro_f1[], split}
  runs_clf_splits.jsonl   one record per finished label split of a classifier
        (so a classifier interrupted at split 13/20 resumes at split 14)

Every record is flushed + fsync'ed the moment it is produced, and the summary
CSVs are rewritten after every classifier.  A half-written last line (power
cut / Ctrl+C mid-write) is ignored on reload instead of breaking the resume.

Embeddings are found by file name ``<dataset>__<embedder>__seed<k>.npy``:
first at the path in runs_embed.jsonl, then inside the embeddings dir.  A file
with no log record is still used (``adopt_embedding``) -- it is never
regenerated.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile

import numpy as np

from . import config as C
from .metrics import aggregate


def _isnan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


class ResultStore:
    def __init__(self, results_dir: str = C.RESULTS_ROOT,
                 emb_dir: str = C.EMB_CACHE_ROOT, logger=None):
        self.results_dir = os.path.abspath(results_dir)
        self.emb_dir = os.path.abspath(emb_dir)
        self.logger = logger
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.emb_dir, exist_ok=True)
        self.embed_log = os.path.join(self.results_dir, "runs_embed.jsonl")
        self.clf_log = os.path.join(self.results_dir, "runs_clf.jsonl")
        self.split_log = os.path.join(self.results_dir, "runs_clf_splits.jsonl")
        self._embed = self._read_jsonl(self.embed_log)   # key -> record
        self._clf = self._read_jsonl(self.clf_log)
        self._splits = self._read_jsonl(self.split_log)

    # ----------------------------- keys ----------------------------------- #
    @staticmethod
    def ekey(dataset, embedder, seed) -> str:
        return f"{dataset}|{embedder}|{seed}"

    @staticmethod
    def ckey(dataset, embedder, seed, clf) -> str:
        # split setup is part of the key: results from another split protocol
        # are never treated as "already done"
        return f"{dataset}|{embedder}|{seed}|{clf}|{C.SPLIT_TAG}"

    @classmethod
    def skey(cls, dataset, embedder, seed, clf, split_id) -> str:
        return f"{cls.ckey(dataset, embedder, seed, clf)}|split{split_id}"

    def emb_path(self, dataset, embedder, seed) -> str:
        return os.path.join(self.emb_dir, f"{dataset}__{embedder}__seed{seed}.npy")

    # ----------------------------- io ------------------------------------- #
    def _warn(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print("WARNING:", msg)

    def _read_jsonl(self, path) -> dict:
        out = {}
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    out[rec["key"]] = rec           # later lines win
                except (json.JSONDecodeError, KeyError, TypeError):
                    self._warn(f"{os.path.basename(path)} line {n} is incomplete "
                               f"(interrupted write?) -- ignored")
        return out

    @staticmethod
    def _append(path, rec) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    # --------------------------- embeddings ------------------------------- #
    def find_embedding_file(self, dataset, embedder, seed) -> str | None:
        rec = self._embed.get(self.ekey(dataset, embedder, seed)) or {}
        candidates = []
        if rec.get("emb_path"):
            p = rec["emb_path"]
            candidates.append(p)
            # recorded on another machine / folder (e.g. a Windows path)
            candidates.append(os.path.join(self.emb_dir, p.replace("\\", "/").split("/")[-1]))
        candidates.append(self.emb_path(dataset, embedder, seed))
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return None

    def has_embedding(self, dataset, embedder, seed) -> bool:
        return self.find_embedding_file(dataset, embedder, seed) is not None

    def get_embed_record(self, dataset, embedder, seed) -> dict | None:
        return self._embed.get(self.ekey(dataset, embedder, seed))

    def load_embedding(self, dataset, embedder, seed) -> np.ndarray:
        path = self.find_embedding_file(dataset, embedder, seed)
        if path is None:
            raise FileNotFoundError(f"no embedding file for {dataset}/{embedder}/seed{seed}")
        return np.load(path)

    def save_embedding(self, dataset, embedder, seed, emb, embed_time_s,
                       ari, nmi, n_nodes) -> None:
        path = self.emb_path(dataset, embedder, seed)
        tmp = path + ".tmp.npy"
        np.save(tmp, np.asarray(emb, dtype=np.float32))
        os.replace(tmp, path)            # never leave a half-written .npy
        self._write_embed_record(dataset, embedder, seed, path, embed_time_s,
                                 ari, nmi, np.asarray(emb).shape[1], n_nodes)

    def adopt_embedding(self, dataset, embedder, seed, emb, ari, nmi, n_nodes,
                        embed_time_s=float("nan")) -> None:
        """Register an existing embedding file without regenerating it."""
        path = self.find_embedding_file(dataset, embedder, seed)
        self._write_embed_record(dataset, embedder, seed, path, embed_time_s,
                                 ari, nmi, np.asarray(emb).shape[1], n_nodes)

    def _write_embed_record(self, dataset, embedder, seed, path, embed_time_s,
                            ari, nmi, emb_dim, n_nodes):
        rec = dict(key=self.ekey(dataset, embedder, seed), dataset=dataset,
                   embedder=embedder, seed=seed, embed_time_s=float(embed_time_s),
                   ari=float(ari), nmi=float(nmi), emb_path=os.path.abspath(path),
                   emb_dim=int(emb_dim), n_nodes=int(n_nodes))
        self._embed[rec["key"]] = rec
        self._append(self.embed_log, rec)

    # --------------------------- classifiers ------------------------------ #
    def has_clf(self, dataset, embedder, seed, clf) -> bool:
        return self.ckey(dataset, embedder, seed, clf) in self._clf

    def done_splits(self, dataset, embedder, seed, clf) -> dict:
        """{split_id: (accuracy, macro_f1)} already finished for this classifier."""
        prefix = self.ckey(dataset, embedder, seed, clf) + "|split"
        out = {}
        for k, r in self._splits.items():
            if k.startswith(prefix):
                out[int(r["split_id"])] = (float(r["accuracy"]), float(r["macro_f1"]))
        return out

    def save_split(self, dataset, embedder, seed, clf, split_id, accuracy, macro_f1):
        rec = dict(key=self.skey(dataset, embedder, seed, clf, split_id),
                   dataset=dataset, embedder=embedder, seed=seed, classifier=clf,
                   split=C.SPLIT_TAG, split_id=int(split_id),
                   accuracy=float(accuracy), macro_f1=float(macro_f1))
        self._splits[rec["key"]] = rec
        self._append(self.split_log, rec)

    def save_clf(self, dataset, embedder, seed, clf, accuracy, macro_f1,
                 extra=None) -> None:
        rec = dict(key=self.ckey(dataset, embedder, seed, clf), dataset=dataset,
                   embedder=embedder, seed=seed, classifier=clf,
                   split=C.SPLIT_TAG,
                   accuracy=float(accuracy), macro_f1=float(macro_f1))
        if extra:
            rec.update(extra)
        self._clf[rec["key"]] = rec
        self._append(self.clf_log, rec)

    def _clf_current(self):
        return [r for r in self._clf.values() if r.get("split") == C.SPLIT_TAG]

    # --------------------------- summaries -------------------------------- #
    def _write_csv(self, path, header, rows) -> bool:
        """Write via temp file + atomic replace.  If the CSV is locked (e.g.
        open in Excel on Windows) warn and keep going -- the JSONL logs hold
        the results and the CSV is rewritten on the next flush."""
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self.results_dir, suffix=".csv.tmp")
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
            os.replace(tmp, path)
            return True
        except OSError as e:
            self._warn(f"could not write {os.path.basename(path)} ({e}); "
                       "is it open in another program? Results are safe in the "
                       ".jsonl logs and the CSV will be rewritten next time.")
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            return False

    def write_summaries(self, seeds=None) -> list[str]:
        """Rewrite all CSV tables.  ``seeds``: only include these seeds
        (the benchmark passes the seeds of the current run)."""
        written = []
        keep = (lambda r: True) if seeds is None else (lambda r: r.get("seed") in set(seeds))
        clf_recs = sorted((r for r in self._clf_current() if keep(r)), key=lambda r: r["key"])

        # ---- detailed: every (dataset, embedder, seed, classifier) row ----
        rows = []
        for cr in clf_recs:
            er = self._embed.get(self.ekey(cr["dataset"], cr["embedder"], cr["seed"]), {})
            rows.append([cr["dataset"], cr["embedder"], cr["seed"], cr["classifier"],
                         f'{cr["accuracy"]:.4f}', f'{cr["macro_f1"]:.4f}',
                         f'{er.get("ari", float("nan")):.4f}',
                         f'{er.get("nmi", float("nan")):.4f}',
                         f'{er.get("embed_time_s", float("nan")):.3f}',
                         cr.get("split", ""), len(cr.get("split_accuracy", []))])
        p = os.path.join(self.results_dir, "detailed.csv")
        if self._write_csv(p, ["dataset", "embedder", "seed", "classifier",
                               "accuracy", "macro_f1", "ari", "nmi", "embed_time_s",
                               "split", "n_splits"], rows):
            written.append(p)

        # ---- averaged classification tables (mean±std over seeds) ----------
        # extra columns at the END (existing readers keep working):
        #   *_std_all_splits = std over every (seed x split) score, the way
        #   TriCL reports its ±
        agg = {}
        for cr in clf_recs:
            key = (cr["dataset"], cr["embedder"], cr["classifier"])
            d = agg.setdefault(key, {"accuracy": [], "macro_f1": [],
                                     "all_accuracy": [], "all_macro_f1": []})
            d["accuracy"].append(cr["accuracy"])
            d["macro_f1"].append(cr["macro_f1"])
            d["all_accuracy"] += cr.get("split_accuracy", [cr["accuracy"]])
            d["all_macro_f1"] += cr.get("split_macro_f1", [cr["macro_f1"]])
        for metric in ("accuracy", "macro_f1"):
            rows = []
            for (ds, emb, clf), d in sorted(agg.items()):
                m, s = aggregate(d[metric])
                _, s_all = aggregate(d["all_" + metric])
                rows.append([ds, emb, clf, f"{m:.4f}", f"{s:.4f}", len(d[metric]),
                             f"{s_all:.4f}", len(d["all_" + metric])])
            p = os.path.join(self.results_dir, f"avg_{metric}.csv")
            if self._write_csv(p, ["dataset", "embedder", "classifier",
                                   f"{metric}_mean", f"{metric}_std", "n_seeds",
                                   f"{metric}_std_all_splits", "n_scores"], rows):
                written.append(p)

        # ---- embedding-level tables (ari, nmi, time): per (ds, emb) --------
        eagg = {}
        for er in self._embed.values():
            if not keep(er):
                continue
            key = (er["dataset"], er["embedder"])
            d = eagg.setdefault(key, {"ari": [], "nmi": [], "embed_time_s": []})
            for m in d:
                if m in er and not _isnan(er[m]):
                    d[m].append(er[m])
        for metric in ("ari", "nmi", "embed_time_s"):
            rows = []
            for (ds, emb), d in sorted(eagg.items()):
                if not d[metric]:
                    continue          # e.g. time unknown for adopted embeddings
                mval, sval = aggregate(d[metric])
                rows.append([ds, emb, f"{mval:.4f}", f"{sval:.4f}", len(d[metric])])
            p = os.path.join(self.results_dir, f"avg_{metric}.csv")
            if self._write_csv(p, ["dataset", "embedder", f"{metric}_mean",
                                   f"{metric}_std", "n_seeds"], rows):
                written.append(p)
        return written
