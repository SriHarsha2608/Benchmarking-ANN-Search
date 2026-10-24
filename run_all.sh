#!/usr/bin/env bash
set -e

# CS4.406: Assignment 3 - Benchmarking Approximate Nearest Neighbour Search
# One-command reproduction script
#
# Usage:
#   bash run_all.sh          # Full benchmark run across all parts
#   bash run_all.sh --quick  # Quick run (< 15 mins) on small subset of data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLCONFIGDIR="/tmp/matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"

QUICK_FLAG=""
if [ "$1" == "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "======================================================================"
    echo "  Running Benchmarks in QUICK Mode (< 15 mins)"
    echo "======================================================================"
    # Back up full results.csv if it exists and contains full benchmark data
    if [ -f "results/results.csv" ] && [ ! -f "results/results_full_backup.csv" ]; then
        cp results/results.csv results/results_full_backup.csv
        echo "Backed up existing full results to results/results_full_backup.csv"
    fi
    # Start fresh for quick mode so runs are not blended with full benchmark runs
    rm -f results/results.csv results/exact_baseline.json
else
    echo "======================================================================"
    echo "  Running FULL Benchmarks"
    echo "======================================================================"
    # Start fresh for clean full reproduction
    if [ -f "results/results.csv" ]; then
        cp results/results.csv results/results.csv.bak
        rm -f results/results.csv results/exact_baseline.json
        echo "Backed up existing results to results/results.csv.bak and starting fresh."
    fi
fi

# Detect python executable (prefer .venv)
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Python interpreter: $($PYTHON --version)"

# -------------------------------------------------------------
# Part 1: Unit Tests & Exact Baseline
# -------------------------------------------------------------
echo ""
echo ">>> Running Unit & Integration Tests (pytest)..."
$PYTHON -m pytest tests/

echo ""
echo ">>> [1/4] Running Part 1: Exact Search Baseline..."
$PYTHON src/exact_search.py $QUICK_FLAG

# -------------------------------------------------------------
# Part 2: Benchmark Three Indexes
# -------------------------------------------------------------
echo ""
echo ">>> [2/4] Running Part 2: Benchmark HNSW, IVF-Flat, IVF-PQ..."
$PYTHON src/benchmark_part2.py $QUICK_FLAG
$PYTHON src/plot_part2.py

# -------------------------------------------------------------
# Part 3: Compression & Re-ranking Experiment
# -------------------------------------------------------------
echo ""
echo ">>> [3/4] Running Part 3: Compression & Exact Re-ranking..."
$PYTHON src/experiment_part3.py $QUICK_FLAG
$PYTHON src/plot_part3.py

# -------------------------------------------------------------
# Part 4: Custom Geometry Failure Mode Experiment
# -------------------------------------------------------------
echo ""
echo ">>> [4/4] Running Part 4: Geometric Cluster Dispersion Failure Mode..."
$PYTHON src/experiment_part4.py $QUICK_FLAG
$PYTHON src/plot_part4.py

# -------------------------------------------------------------
# Optional PDF compilation if LaTeX is installed
# -------------------------------------------------------------
if command -v pdflatex &> /dev/null; then
    echo ""
    echo ">>> Compiling report.tex into report.pdf..."
    pdflatex -interaction=nonstopmode report.tex > pdflatex.log 2>&1
    pdflatex -interaction=nonstopmode report.tex >> pdflatex.log 2>&1
    if [ -f "report.pdf" ]; then
        echo "Successfully generated report.pdf!"
        rm -f pdflatex.log
    else
        echo "ERROR: pdflatex compilation failed. See pdflatex.log"
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "  All benchmark benchmarks completed successfully!"
echo "  - Raw numbers: results/results.csv"
echo "  - Figures:     results/figures/"
echo "  - Tables:      results/part2_tables.md, results/part3_tables.md, results/part4_tables.md"
echo "  - LaTeX:       report.tex"
echo "======================================================================"

