#!/usr/bin/env python3
"""
benchmark_flat.py -- structure-free baselines (logreg, mlp) on saved embeddings
===============================================================================

The two structure-free classifiers now live inside the main benchmark
(hgb/classifiers/flat_clf.py) so they use the same TriCL splits, the same
result files (results/avg_accuracy.csv etc., classifier = logreg / mlp) and
the same per-split resume as the AllSet classifiers.

This script is kept as a shortcut and simply runs:

    python benchmark.py --use-saved-embeddings --classifiers logreg mlp [your args]

Any other benchmark.py option can be added, e.g.
    python benchmark_flat.py --datasets cora_a zoo --seeds 0
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark  # noqa: E402

if __name__ == "__main__":
    extra = sys.argv[1:]
    if "--classifiers" not in extra:
        extra = ["--classifiers", "logreg", "mlp"] + extra
    if "--use-saved-embeddings" not in extra and "--regenerate" not in extra:
        extra = ["--use-saved-embeddings"] + extra
    sys.argv = [sys.argv[0]] + extra
    benchmark.main()
