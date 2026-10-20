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

QUICK_FLAG=""
if [ "$1" == "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "======================================================================"
    echo "  Running Benchmarks in QUICK Mode (< 15 mins)"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "  Running FULL Benchmarks"
    echo "======================================================================"
fi

# Detect python executable (prefer .venv)
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Python interpreter: $($PYTHON --version)"

# -------------------------------------------------------------
# Part 1: Exact Baseline
# -------------------------------------------------------------
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
    pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1 || true
    if [ -f "report.pdf" ]; then
        echo "Successfully generated report.pdf!"
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

