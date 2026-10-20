# CS4.406: Benchmarking Approximate Nearest Neighbour Search

This repository contains the full implementation, benchmarking scripts, synthetic data generators, analysis plots, and report for **CS4.406: Information Retrieval & Extraction Assignment 3**.

---

## 1. Directory Structure

```text
Benchmarking-ANN-Search/
├── A3.md                       # Assignment specifications
├── README.md                   # Setup and reproduction instructions
├── requirements.txt            # Python dependencies
├── run_all.sh                  # One-command reproduction script (full and --quick)
├── hypothesis.md               # Pre-registered hypotheses and empirical evaluations
├── report.tex                  # Complete academic report in LaTeX (Part 5)
├── data/                       # Dataset directory (excluded from git)
│   ├── sift-128-euclidean.hdf5
│   └── glove-100-angular.hdf5
├── results/
│   ├── results.csv             # Raw benchmark measurements (253 runs)
│   ├── exact_baseline.json     # Exact search measurements
│   ├── part2_summary.json      # Structured summary for Part 2
│   ├── part2_tables.md         # Markdown summary tables for Part 2
│   ├── part3_summary.json      # Structured summary for Part 3
│   ├── part3_tables.md         # Markdown summary tables for Part 3
│   ├── part4_summary.json      # Structured summary for Part 4
│   ├── part4_tables.md         # Markdown summary tables for Part 4
│   └── figures/                # Publication-quality benchmark figures
│       ├── part2_sift_tradeoff.png
│       ├── part2_glove_tradeoff.png
│       ├── part3_compression_tradeoff.png
│       ├── part3_compression_recall_ceiling.png
│       ├── part4_geometry_failure.png
│       └── part4_dispersion_impact.png
├── src/
│   ├── data_loader.py          # HDF5 loader and query split manager
│   ├── metrics.py              # Custom tie-aware recall@k metric
│   ├── exact_search.py         # Part 1: Exact search baseline and timing
│   ├── benchmark_part2.py      # Part 2: HNSW, IVF-Flat, IVF-PQ benchmark
│   ├── plot_part2.py           # Part 2: Figure and table generator
│   ├── experiment_part3.py     # Part 3: Compression & exact re-ranking
│   ├── plot_part3.py           # Part 3: Figure and table generator
│   ├── generate_synthetic_data.py # Part 4: Parameterized GMM generator
│   ├── experiment_part4.py     # Part 4: Geometric failure mode benchmark
│   └── plot_part4.py           # Part 4: Figure and table generator
└── tests/
    └── test_metrics.py         # Unit tests for tie-aware recall metric
```

---

## 2. Installation & Setup

### Environment Setup
Create and activate a Python 3.10+ virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Download
Download the standard datasets from ann-benchmarks into the `data/` directory:
```bash
mkdir -p data
curl -o data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5
curl -o data/glove-100-angular.hdf5 http://ann-benchmarks.com/glove-100-angular.hdf5
```

---

## 3. One-Command Reproduction

### Option A: Quick Run (`< 15 minutes`)
To run a fast validation on a subset of data (under 15 minutes on a laptop):
```bash
bash run_all.sh --quick
```
This runs:
- Exact search on 100K subset of SIFT and GloVe.
- Parameter-trimmed sweeps for HNSW, IVF-Flat, and IVF-PQ.
- Part 3 compression with $m \in \{16, 32\}$.
- Part 4 geometric failure mode with $\sigma \in \{0.05, 0.70\}$ on 50K synthetic vectors.
- Regenerates all figures and summary markdown tables.

### Option B: Full Benchmark Run
To reproduce all 253 benchmark runs across all four parts from scratch:
```bash
bash run_all.sh
```

---

## 4. Running Individual Parts

Each part can also be executed independently:

### Part 1: Exact Baseline
```bash
python src/exact_search.py
pytest tests/
```

### Part 2: Benchmark Three Indexes
```bash
python src/benchmark_part2.py
python src/plot_part2.py
```

### Part 3: Compression & Exact Re-ranking
```bash
python src/experiment_part3.py
python src/plot_part3.py
```

### Part 4: Geometric Cluster Dispersion Failure Mode
```bash
python src/experiment_part4.py
python src/plot_part4.py
```

---

## 5. Compiling the LaTeX Report (Part 5)

The complete academic report is written in `report.tex`. To compile it to `report.pdf`:
```bash
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
```

---

## 6. Pre-Registered Hypotheses & Git History

In compliance with the assignment ground rules, all hypotheses for Parts 3 and 4 were committed to git **before** conducting each respective experiment:
- `36d726d`: Add hypothesis for Part 3: Compression and Exact Re-ranking
- `b5ee09e`: Complete Part 3: Compression and Exact Re-ranking experiment, figures, and tables
- `f5c67f3`: Add hypothesis for Part 4: Cluster dispersion failure mode
- `0dc4feb`: Complete Part 4: Cluster dispersion failure mode experiment, figures, and tables

See [`hypothesis.md`](hypothesis.md) for full pre-registered hypotheses and empirical evaluations.