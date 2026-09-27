#!/usr/bin/env python3
"""
benchmark.py -- hypergraph node-representation benchmark (single entry point)
============================================================================

For every (dataset x embedder x seed):

  1. get the node embedding       -- reused if the .npy already exists,
                                     generated only if it does not
  2. cluster it with KMeans       (->  ARI, NMI)
  3. train each downstream classifier on it (->  accuracy, macro-F1)
       logreg, mlp                       structure-free (sklearn)
       allset, hnhn, hgnn, hypergcn, unigcn   official AllSet models
     on TriCL's 20 fixed 10/10/80 splits.

Reusing embeddings
------------------
An embedding is looked up as  <embeddings-dir>/<dataset>__<embedder>__seed<k>.npy
(and at the path recorded in results/runs_embed.jsonl).  If the file exists it
is used -- with or without a log record -- and never regenerated.
  --use-saved-embeddings   never generate; cells without a file are skipped
  --regenerate             ignore existing files and recompute everything

Resume
------
Every finished split, every finished classifier and every embedding is written
to results/*.jsonl immediately, and the summary CSVs are rewritten after every
classifier.  Re-run the same command after a crash / Ctrl+C and it continues
from the next unfinished split.

Examples
--------
    python benchmark.py --use-saved-embeddings
    python benchmark.py --use-saved-embeddings \\
        --datasets cora_a cora_c citeseer pubmed modelnet40 zoo \\
        --classifiers logreg mlp allset hnhn hgnn unigcn
    python benchmark.py --datasets zoo --embedders HyperFuse --classifiers logreg --seeds 0
    python benchmark.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from hgb import config as C
from hgb import data as D
from hgb import metrics as MET
from hgb.seeding import set_global_seed
from hgb.logging_utils import get_logger, tqdm_iter
from hgb.store import ResultStore
from hgb.embedders import get_embedder
from hgb.classifiers import get_classifier_method, run_downstream


def parse_args():
    p = argparse.ArgumentParser(description="Hypergraph representation benchmark")
    p.add_argument("--datasets", nargs="+", default=None,
                   help=f"subset of {C.AVAILABLE_DATASETS} (default: all available)")
    p.add_argument("--embedders", nargs="+", default=None,
                   help=f"subset of {list(C.EMBEDDERS)} (default: {C.DEFAULT_EMBEDDERS})")
    p.add_argument("--classifiers", nargs="+", default=None,
                   help=f"subset of {list(C.CLASSIFIERS)} (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"seeds (default: {C.SEEDS})")
    p.add_argument("--device", default=C.DEVICE, help="cpu / cuda")
    p.add_argument("--no-classifiers", action="store_true",
                   help="only get + cluster embeddings, skip classifiers")
    p.add_argument("--include-chgnn", action="store_true",
                   help="include the semi-supervised CHGNN embedder (caveated)")
    p.add_argument("--results-dir", default=C.RESULTS_ROOT)
    p.add_argument("--embeddings-dir", default=C.EMB_CACHE_ROOT,
                   help="folder with <ds>__<emb>__seed<k>.npy files "
                        f"(default: {C.EMB_CACHE_ROOT})")
    p.add_argument("--use-saved-embeddings", action="store_true",
                   help="never generate embeddings; skip cells with no saved file")
    p.add_argument("--regenerate", action="store_true",
                   help="ignore saved embeddings and always recompute")
    p.add_argument("--list", action="store_true", help="print registry + exit")
    return p.parse_args()


def resolve_dir(path: str) -> str:
    """Relative paths: current dir first, then the benchmark folder."""
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), ROOT):
        cand = os.path.join(base, path)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(ROOT, path))


def do_list(logger):
    logger.info("Datasets (available shown with data on disk):")
    for name in C.ALL_DATASETS:
        ok, info = D.dataset_available(name)
        mark = "OK " if ok else "-- "
        logger.info(f"  [{mark}] {name:<11} {C.DATASET_DISPLAY.get(name,name)}"
                    + ("" if ok else f"   ({info.split(':',1)[-1].strip()})"))
    logger.info("Embedders:")
    for k, v in C.EMBEDDERS.items():
        logger.info(f"  {k:<14} {v['display']}"
                    + ("" if v["unsupervised"] else "   [semi-supervised]"))
    logger.info("Classifiers:")
    for k, v in C.CLASSIFIERS.items():
        logger.info(f"  {k:<10} {v['display']}")


def validate(names, known, what):
    bad = [n for n in names if n not in known]
    if bad:
        raise SystemExit(f"unknown {what}: {bad}. Known: {list(known)}")


def main():
    args = parse_args()
    os.makedirs(C.LOGS_ROOT, exist_ok=True)
    logger = get_logger("hgb", os.path.join(C.LOGS_ROOT, "benchmark.log"))

    if args.list:
        do_list(logger)
        return
    if args.use_saved_embeddings and args.regenerate:
        raise SystemExit("--use-saved-embeddings and --regenerate cannot be combined")

    datasets = args.datasets or C.AVAILABLE_DATASETS
    embedders = args.embedders or list(C.DEFAULT_EMBEDDERS)
    if args.include_chgnn and "chgnn" not in embedders:
        embedders.append("chgnn")
    classifiers = [] if args.no_classifiers else (args.classifiers or C.DEFAULT_CLASSIFIERS)
    seeds = args.seeds if args.seeds is not None else list(C.SEEDS)
    validate(datasets, C.DATASETS, "dataset(s)")
    validate(embedders, C.EMBEDDERS, "embedder(s)")
    validate(classifiers, C.CLASSIFIERS, "classifier(s)")

    results_dir = resolve_dir(args.results_dir)
    emb_dir = resolve_dir(args.embeddings_dir)
    store = ResultStore(results_dir=results_dir, emb_dir=emb_dir, logger=logger)
    fail_log = os.path.join(results_dir, "failures.log")

    logger.info("=" * 78)
    logger.info(f"datasets={datasets}")
    logger.info(f"embedders={embedders}")
    logger.info(f"classifiers={classifiers}  seeds={seeds}  device={args.device}")
    logger.info(f"results_dir={results_dir}")
    logger.info(f"embeddings_dir={emb_dir}  "
                f"({len([f for f in os.listdir(emb_dir) if f.endswith('.npy')])} .npy files)")
    logger.info(f"embeddings: " + ("REUSE ONLY (never generate)" if args.use_saved_embeddings
                                   else "REGENERATE ALL" if args.regenerate
                                   else "reuse if present, else generate"))
    logger.info(f"splits: TriCL {C.TRAIN_PROP:.0%}/{C.VALID_PROP:.0%}/"
                f"{1 - C.TRAIN_PROP - C.VALID_PROP:.0%} x {C.N_CLF_SPLITS}  ({C.SPLIT_TAG})")
    logger.info("=" * 78)

    failures = []

    def fail(where, e):
        tb = traceback.format_exc()
        logger.error(f"[FAIL ] {where}: {type(e).__name__}: {e}")
        logger.error(tb.rstrip())
        failures.append((where, f"{type(e).__name__}: {e}"))
        try:
            with open(fail_log, "a", encoding="utf-8") as f:
                f.write(f"==== {where}\n{tb}\n")
        except OSError:
            pass

    samples = {}
    cells = [(ds, emb, sd) for ds in datasets for emb in embedders for sd in seeds]
    try:
        for (ds, emb_name, seed) in tqdm_iter(cells, desc="cells", total=len(cells)):
            cell = f"{ds}/{emb_name}/seed{seed}"

            # skip the whole cell quickly if every classifier is already done
            if classifiers and not args.regenerate and all(
                    store.has_clf(ds, emb_name, seed, c) for c in classifiers):
                logger.info(f"[cache] {cell}: all {len(classifiers)} classifiers done")
                continue

            ok, info = D.dataset_available(ds)
            if not ok:
                logger.warning(f"[skip ] {info}")
                continue
            try:
                if ds not in samples:
                    samples[ds] = D.load(ds)
                sample = samples[ds]
            except Exception as e:
                fail(f"load {ds}", e)
                continue

            # ---------- Stage 1: embedding (+ clustering) ----------
            try:
                have_file = (not args.regenerate) and store.has_embedding(ds, emb_name, seed)
                if have_file:
                    path = store.find_embedding_file(ds, emb_name, seed)
                    emb = store.load_embedding(ds, emb_name, seed)
                    if emb.ndim != 2 or emb.shape[0] != sample.num_nodes:
                        raise ValueError(f"{path} has shape {emb.shape}, expected "
                                         f"({sample.num_nodes}, d)")
                    if not np.isfinite(emb).all():
                        raise ValueError(f"{path} contains NaN/inf")
                    rec = store.get_embed_record(ds, emb_name, seed)
                    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
                    if rec is not None and same(rec.get("emb_path", ""), path):
                        logger.info(f"[cache] emb {cell} dim={emb.shape[1]} "
                                    f"ARI={rec['ari']:.3f} NMI={rec['nmi']:.3f}")
                    elif rec is not None:
                        # record exists but file moved: keep its metrics, fix the path
                        store.adopt_embedding(ds, emb_name, seed, emb, rec["ari"], rec["nmi"],
                                              sample.num_nodes, rec.get("embed_time_s", float("nan")))
                        logger.info(f"[cache] emb {cell} <- {path} (moved) dim={emb.shape[1]}")
                    else:
                        cl = MET.clustering_metrics(emb, sample.y, base_seed=seed)
                        store.adopt_embedding(ds, emb_name, seed, emb, cl["ari"], cl["nmi"],
                                              sample.num_nodes)
                        logger.info(f"[load ] emb {cell} <- {path} dim={emb.shape[1]} "
                                    f"ARI={cl['ari']:.3f} NMI={cl['nmi']:.3f} "
                                    f"(no log record; embed time unknown)")
                elif args.use_saved_embeddings:
                    logger.warning(f"[skip ] {cell}: no saved embedding "
                                   f"({store.emb_path(ds, emb_name, seed)})")
                    continue
                else:
                    spec = C.EMBEDDERS[emb_name]
                    if spec["feats"] and not sample.has_features:
                        logger.warning(f"[skip ] {emb_name} needs features but {ds} has none")
                        continue
                    logger.info(f"[embed] {cell}")
                    mod, _ = get_embedder(emb_name)
                    set_global_seed(seed)
                    with MET.timer() as t:
                        emb = mod.embed(sample, seed=seed, device=args.device, logger=logger)
                    embed_time = t()
                    emb = np.asarray(emb, dtype=np.float32)
                    cl = MET.clustering_metrics(emb, sample.y, base_seed=seed)
                    store.save_embedding(ds, emb_name, seed, emb, embed_time,
                                         cl["ari"], cl["nmi"], sample.num_nodes)
                    logger.info(f"[done ] emb {cell} dim={emb.shape[1]} "
                                f"time={embed_time:.2f}s ARI={cl['ari']:.3f} NMI={cl['nmi']:.3f}")
            except Exception as e:
                fail(f"embed {cell}", e)
                continue

            if not classifiers:
                store.write_summaries(seeds)
                continue

            # ---------- Stage 2: downstream classifiers ----------
            splits = D.get_label_splits(sample)
            for clf in classifiers:
                if store.has_clf(ds, emb_name, seed, clf):
                    logger.info(f"[cache] clf {cell}/{clf}")
                    continue
                try:
                    done = store.done_splits(ds, emb_name, seed, clf)
                    done = {k: v for k, v in done.items() if k < len(splits)}
                    resume = f", resuming at {len(done)}/{len(splits)} splits" if done else ""
                    logger.info(f"[clf  ] {cell}/{clf} ({get_classifier_method(clf)}{resume})")

                    def on_split(si, acc, f1, _clf=clf):
                        store.save_split(ds, emb_name, seed, _clf, si, acc, f1)

                    res = run_downstream(clf, emb, sample, splits, seed, args.device,
                                         logger, done=done, on_split=on_split)
                    store.save_clf(ds, emb_name, seed, clf, res["accuracy"], res["macro_f1"],
                                   extra={"split_accuracy": res["split_accuracy"],
                                          "split_macro_f1": res["split_macro_f1"]})
                    logger.info(f"[done ] clf {cell}/{clf} "
                                f"acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f}")
                except Exception as e:
                    fail(f"clf {cell}/{clf}", e)
                finally:
                    store.write_summaries(seeds)      # after EVERY classifier
    except KeyboardInterrupt:
        logger.warning("Interrupted -- everything finished so far is saved. "
                       "Re-run the same command to resume.")
    finally:
        written = store.write_summaries(seeds)
        logger.info("=" * 78)
        logger.info("Results written:")
        for w in written:
            logger.info(f"  {w}")
        if failures:
            logger.warning(f"{len(failures)} failure(s); full tracebacks in {fail_log}")
            for where, msg in failures[:15]:
                logger.warning(f"  {where} -> {msg[:160]}")
        logger.info("Done.")


if __name__ == "__main__":
    main()
