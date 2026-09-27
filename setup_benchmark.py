#!/usr/bin/env python3
"""
setup_benchmark.py -- stage the 10 available datasets into ./data
=================================================================

The benchmark reads every dataset from a ``pickle_dir`` (features.pickle,
hypergraph.pickle, labels.pickle).  That exact format is shipped by the TriCL
repo's ``dataset.zip`` and by the CHGNN repo's ``data/`` folder.  This script
copies the ten datasets we have a faithful source for into ``./data`` under the
paths the registry expects.  It does NOT invent the five datasets with no
supplied source (IMDB, AMiner, DBLP-A, DBLP-P, House).

Usage
-----
    # from TriCL's dataset.zip (recommended -- has all ten in one place)
    python setup_benchmark.py --from-tricl-zip /path/to/TriCL-main/dataset.zip

    # or from an already-extracted CHGNN data directory
    python setup_benchmark.py --from-chgnn-dir /path/to/CHGNN-master/data
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile

from hgb import config as C

# our dataset name -> (source-subpath-in-TriCL, source-subpath-in-CHGNN)
SRC_MAP = {
    "cora_c":     ("cocitation/cora",      "cocitation/cora"),
    "citeseer":   ("cocitation/citeseer",  "cocitation/citeseer"),
    "pubmed":     ("cocitation/pubmed",    "cocitation/pubmed"),
    "cora_a":     ("coauthorship/cora",    "coauthorship/cora"),
    "dblp":       ("coauthorship/dblp",    "coauthorship/dblp"),
    "modelnet40": ("ModelNet40",           "hypergraph/ModelNet40"),
    "zoo":        ("zoo",                  "hypergraph/zoo"),
    "20news":     ("20newsW100",           "hypergraph/20newsW100"),
    "mushroom":   ("Mushroom",             "hypergraph/Mushroom"),
    "ntu2012":    ("NTU2012",              "hypergraph/NTU2012"),
}
NEEDED = ("features.pickle", "hypergraph.pickle", "labels.pickle")


def _copy_one(src_dir, dst_dir):
    if not all(os.path.exists(os.path.join(src_dir, f)) for f in NEEDED):
        return False
    os.makedirs(dst_dir, exist_ok=True)
    for f in NEEDED:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    # TriCL's 20 fixed 10/10/80 node-classification splits (used + verified
    # by hgb.data.make_tricl_splits when present)
    split_src = os.path.join(src_dir, "splits")
    if os.path.isdir(split_src):
        shutil.copytree(split_src, os.path.join(dst_dir, "splits"), dirs_exist_ok=True)
    return True


def stage(src_root, which_idx):
    staged, missing = [], []
    for name, subs in SRC_MAP.items():
        rel = C.DATASETS[name][1]
        dst = os.path.join(C.DATA_ROOT, rel)
        src = os.path.join(src_root, subs[which_idx])
        if _copy_one(src, dst):
            staged.append(name)
        else:
            missing.append((name, src))
    return staged, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-tricl-zip", default=None,
                    help="path to TriCL-main/dataset.zip")
    ap.add_argument("--from-chgnn-dir", default=None,
                    help="path to an extracted CHGNN-master/data directory")
    args = ap.parse_args()

    os.makedirs(C.DATA_ROOT, exist_ok=True)
    if args.from_tricl_zip:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(args.from_tricl_zip) as z:
                z.extractall(tmp)
            root = os.path.join(tmp, "dataset")
            if not os.path.isdir(root):
                root = tmp
            staged, missing = stage(root, which_idx=0)
    elif args.from_chgnn_dir:
        staged, missing = stage(args.from_chgnn_dir, which_idx=1)
    else:
        ap.error("provide --from-tricl-zip or --from-chgnn-dir")

    print(f"Staged {len(staged)} datasets into {C.DATA_ROOT}: {staged}")
    if missing:
        print("Not found in this source (fine if you used the other one):")
        for name, src in missing:
            print(f"  {name}: expected {src}")
    print("\nRun `python benchmark.py --list` to confirm availability.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
