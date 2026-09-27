# Hypergraph Representation Learning Benchmark

This repository contains a unified benchmark for evaluating **hypergraph node-representation learning methods** under a common downstream evaluation protocol.

The benchmark is designed to compare unsupervised hypergraph representation methods using the same datasets, data splits, random seeds, downstream classifiers, and evaluation metrics.

## Methods

The default unsupervised representation-learning methods are:

- **HyperFuse** — our method
- **TriCL**
- **SE-HSSL**
- **VilLain**
- **HypeBoy**

The benchmark also evaluates the learned representations with:

### Structure-free classifiers

- Logistic Regression
- MLP

### Hypergraph-aware classifiers

- AllSet / AllSetTransformer
- HNHN
- HGNN
- HyperGCN
- UniGCN (AllSet UniGCNII)

## Datasets

The benchmark currently has faithful data sources for the following datasets:

| Dataset | Benchmark name |
|---|---|
| Cora-C | `cora_c` |
| Citeseer | `citeseer` |
| Pubmed | `pubmed` |
| Cora-A | `cora_a` |
| DBLP | `dblp` |
| ModelNet40 | `modelnet40` |
| Zoo | `zoo` |
| 20News | `20news` |
| Mushroom | `mushroom` |
| NTU2012 | `ntu2012` |

Additional datasets are registered for transparency (`IMDB`, `AMiner`, `DBLP-A`, `DBLP-P`, and `House`), but are skipped unless corresponding data are supplied.

The benchmark expects each dataset in the standard pickle format:

```text
features.pickle
hypergraph.pickle
labels.pickle
```

The data themselves are **not included in this repository**.

## Evaluation protocol

For each:

```text
dataset × representation method × seed
```

the benchmark:

1. Generates or loads the node embeddings.
2. Performs KMeans clustering and reports:
   - ARI
   - NMI
3. Evaluates the embeddings with the downstream classifiers and reports:
   - Accuracy
   - Macro-F1
4. Records embedding runtime.

### Downstream splits

The benchmark follows the TriCL-style node-classification protocol:

- 10% training
- 10% validation
- 80% test
- fixed splits shared across methods

The configuration currently defaults to the first **3 splits** for quick/reproducible runs:

```text
HGB_N_SPLITS=3
```

For the full 20-split evaluation, use:

```powershell
$env:HGB_N_SPLITS=20
```

before running the benchmark.

### Random seeds

The default representation-learning seeds are:

```text
0, 1, 2
```

### Clustering

KMeans is repeated with 5 clustering seeds and the mean ARI/NMI is reported.

## Repository structure

```text
.
├── benchmark.py              # Main benchmark entry point
├── benchmark_flat.py         # Shortcut for Logistic Regression + MLP
├── setup_benchmark.py        # Dataset staging utility
│
├── hgb/
│   ├── config.py             # Datasets, methods, classifiers and protocol
│   ├── data.py               # Dataset loading and splits
│   ├── metrics.py            # Evaluation metrics
│   ├── seeding.py            # Reproducibility utilities
│   ├── store.py              # Result storage / resume support
│   ├── logging_utils.py
│   │
│   ├── embedders/            # Representation-learning adapters
│   └── classifiers/          # Downstream classifiers
│
└── vendor/                   # Vendored implementations used by the benchmark
    ├── allset/
    ├── chgnn/
    ├── fuse/
    ├── hypeboy/
    ├── sehssl/
    ├── tricl/
    └── villain/
```

## Installation

Create a Python environment and install the dependencies required by the methods and classifiers.

For example:

```bash
conda create -n hgbenchmark python=3.10
conda activate hgbenchmark
```

The exact PyTorch/CUDA installation should be selected according to the GPU and CUDA environment being used.

## Preparing the datasets

### From the TriCL dataset archive

If you have the TriCL `dataset.zip`:

```bash
python setup_benchmark.py --from-tricl-zip /path/to/TriCL-main/dataset.zip
```

### From an extracted CHGNN data directory

Alternatively:

```bash
python setup_benchmark.py --from-chgnn-dir /path/to/CHGNN-master/data
```

The script stages the available datasets into the directory expected by the benchmark.

After staging the data, check availability with:

```bash
python benchmark.py --list
```

## Running the benchmark

### List available datasets, methods and classifiers

```bash
python benchmark.py --list
```

### Run the default unsupervised benchmark

```bash
python benchmark.py
```

### Run selected datasets

```bash
python benchmark.py --datasets cora_c citeseer pubmed zoo
```

### Run selected representation methods

```bash
python benchmark.py --embedders HyperFuse tricl sehssl villain hypeboy
```

### Run selected classifiers

```bash
python benchmark.py --classifiers logreg mlp allset hnhn hgnn hypergcn unigcn
```

### Run a small test

For example:

```bash
python benchmark.py \
    --datasets zoo \
    --embedders HyperFuse \
    --classifiers logreg \
    --seeds 0
```

On Windows PowerShell, the same command can be written on one line:

```powershell
python benchmark.py --datasets zoo --embedders HyperFuse --classifiers logreg --seeds 0
```

## Using saved embeddings

The benchmark supports cached node embeddings.

To use only embeddings that already exist:

```bash
python benchmark.py --use-saved-embeddings
```

An embedding is expected at:

```text
embeddings/<dataset>__<embedder>__seed<seed>.npy
```

For example:

```text
embeddings/zoo__HyperFuse__seed0.npy
```

Cells for which no saved embedding exists are skipped.

## Recomputing embeddings

To ignore cached embeddings and regenerate them:

```bash
python benchmark.py --regenerate
```

`--use-saved-embeddings` and `--regenerate` cannot be used together.

## Structure-free benchmark shortcut

`benchmark_flat.py` is a convenience wrapper for running only the two structure-free classifiers:

```bash
python benchmark_flat.py
```

For example:

```bash
python benchmark_flat.py --datasets cora_c zoo --seeds 0
```

This is equivalent to using:

```bash
python benchmark.py --use-saved-embeddings --classifiers logreg mlp
```

## Results and reproducibility

The benchmark is designed to support long-running experiments and interruption/resumption.

Completed embeddings, classifier evaluations, and split-level results are written incrementally to the results directory. If an experiment is interrupted, rerunning the same command can continue from unfinished work.

The main output directories are:

```text
data/         # staged datasets
embeddings/   # cached node embeddings
results/      # benchmark results
logs/         # benchmark logs
```

These generated directories are intentionally separate from the source code.

## Fair comparison

The benchmark keeps the following components common across methods wherever possible:

- dataset representation
- downstream train/validation/test splits
- random seeds
- downstream classifiers
- evaluation metrics
- result aggregation
- embedding caching and resume behavior

Method-specific hyperparameters remain inside the corresponding method adapters or their original configurations rather than being placed in the common benchmark configuration.

## Notes

- Dataset files are not redistributed with this repository.
- Vendored code is retained under `vendor/` for reproducibility of the benchmark.
- Please check the licenses and citation requirements of the original methods before redistribution or reuse.

## Citation

If you use this benchmark or HyperFuse in your work, please cite the corresponding HyperFuse paper and the original papers for the benchmarked methods.

## Contact

For questions regarding the benchmark or HyperFuse, please open an issue in this repository.
