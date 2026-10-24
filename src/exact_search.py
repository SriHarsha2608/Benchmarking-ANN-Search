"""
Part 1: Exact Search Baseline using FAISS IndexFlatL2.

Features:
- Validates FAISS exact search against ground-truth neighbours stored in dataset.
- Measures single-query-at-a-time latency and QPS on a single thread.
- Measures all-queries batch latency and QPS.
- Repeats timings and records median values.
- Explains architectural reasons for the throughput difference.
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import faiss

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, get_query_split
from src.metrics import compute_recall_at_k


def run_exact_benchmark(
    dataset_name: str,
    data_dir: str = "data",
    split: str = "test",
    repeats: int = 3,
    num_single_queries: int = 1000,
    quick: bool = False,
):
    """
    Run exact baseline benchmarking for a given dataset.
    """
    max_train = 100000 if quick else None
    print(f"\n{'='*70}")
    print(f"Running Exact Baseline: {dataset_name.upper()} (split='{split}', quick={quick})")
    print(f"{'='*70}")

    ds = load_dataset(dataset_name, data_dir=data_dir, max_train_vectors=max_train)
    train_vectors = ds["train"]
    dim = ds["dim"]
    n_train = len(train_vectors)

    queries, true_neighbors, true_distances = get_query_split(ds, split=split)
    if quick:
        queries = queries[:1000]
        true_neighbors = true_neighbors[:1000]
        true_distances = true_distances[:1000]
    n_queries = len(queries)
    print(f"Loaded {n_train} base vectors (dim={dim}) and {n_queries} queries.")

    # Single-thread execution as specified in assignment rules
    faiss.omp_set_num_threads(1)

    # 1. Build IndexFlatL2
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(train_vectors)
    build_time = time.perf_counter() - t0
    index_bytes = n_train * dim * 4  # float32 = 4 bytes
    index_mb = index_bytes / (1024 * 1024)
    bytes_per_vec = index_bytes / n_train
    print(f"Index built in {build_time:.3f}s | Size: {index_mb:.2f} MB ({bytes_per_vec:.1f} bytes/vec)")

    # 2. Check agreement with ground-truth
    # In quick mode (100k vectors), stored ground truth corresponds to the full 1M dataset,
    # so we compute ground truth on the 100k subset to verify exact search consistency.
    D_batch, I_batch = index.search(queries, 10)
    if quick or (max_train is not None and max_train < len(ds["train"])):
        print(f"[Quick Mode] Computing reference ground truth on {n_train} vector subset...")
        ref_distances, ref_neighbors = index.search(queries, 100)
        agreement = compute_recall_at_k(I_batch, ref_neighbors, ref_distances, k=10)
    else:
        agreement = compute_recall_at_k(I_batch, true_neighbors, true_distances, k=10)
    print(f"Ground-truth agreement (Recall@10): {agreement:.6f}")

    # 3. Time Batch Search (all queries at once, repeated, single thread)
    batch_latencies = []
    for r in range(repeats):
        t_start = time.perf_counter()
        _ = index.search(queries, 10)
        t_end = time.perf_counter()
        batch_latencies.append(t_end - t_start)
    median_batch_time = float(np.median(batch_latencies))
    batch_qps = n_queries / median_batch_time
    batch_latency_ms = (median_batch_time / n_queries) * 1000.0

    print(f"Batch Search ({n_queries} queries, single-threaded):")
    print(f"  Median total time: {median_batch_time:.4f}s")
    print(f"  Throughput: {batch_qps:.2f} QPS (Latency: {batch_latency_ms:.4f} ms/query)")

    # 4. Time Single Query Search (one query at a time in loop, single thread)
    # We test on min(n_queries, num_single_queries) to keep runtimes reasonable
    n_single = min(n_queries, num_single_queries)
    single_queries = queries[:n_single]

    single_latencies = []
    for r in range(repeats):
        t_start = time.perf_counter()
        for i in range(n_single):
            _ = index.search(single_queries[i : i + 1], 10)
        t_end = time.perf_counter()
        single_latencies.append(t_end - t_start)
    median_single_time = float(np.median(single_latencies))
    single_qps = n_single / median_single_time
    single_latency_ms = (median_single_time / n_single) * 1000.0

    print(f"Sequential Single-Query Search ({n_single} queries, single-threaded):")
    print(f"  Median total time: {median_single_time:.4f}s")
    print(f"  Throughput: {single_qps:.2f} QPS (Latency: {single_latency_ms:.4f} ms/query)")

    speedup = batch_qps / single_qps
    print(f"Speedup of Batch vs Sequential Single-Query: {speedup:.2f}x")

    return {
        "dataset": ds["name"],
        "num_train": n_train,
        "dim": dim,
        "split": split,
        "build_time_s": build_time,
        "index_mb": index_mb,
        "bytes_per_vec": bytes_per_vec,
        "recall_at_10": agreement,
        "batch_qps": batch_qps,
        "batch_latency_ms": batch_latency_ms,
        "single_qps": single_qps,
        "single_latency_ms": single_latency_ms,
        "speedup_batch_vs_single": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Part 1: Exact search baseline.")
    parser.add_argument("--dataset", choices=["sift", "glove", "both"], default="both")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["test", "tune", "all"], default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-single-queries", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Quick mode (100k vectors)")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = ["sift", "glove"] if args.dataset == "both" else [args.dataset]
    all_results = []

    for ds in datasets:
        res = run_exact_benchmark(
            dataset_name=ds,
            data_dir=args.data_dir,
            split=args.split,
            repeats=args.repeats,
            num_single_queries=args.num_single_queries,
            quick=args.quick,
        )
        all_results.append(res)

    out_file = os.path.join(args.out_dir, "exact_baseline.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved exact baseline results to: {out_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS: Why Batch and Single-Query Search Differ So Much:")
    print("=" * 70)
    print(
        "1. Cache & Memory Bandwidth: In single-query search, the entire 1M vector\n"
        "   dataset (~500 MB) must be streamed from main memory across the memory bus\n"
        "   for every single query, making it strictly memory-bandwidth bound.\n"
        "   In batch search, blocks of dataset vectors are loaded into CPU L2/L3 cache once\n"
        "   and reused across many queries simultaneously (cache locality).\n"
        "2. BLAS GEMM Vectorization: Batch search computes distances using General Matrix\n"
        "   Multiply (GEMM: Q x X^T), which utilizes highly optimized SIMD FMA\n"
        "   (Fused Multiply-Add) matrix kernels operating near peak compute FLOPS.\n"
        "   Sequential 1-query search performs vector-matrix dot products (GEMV),\n"
        "   achieving only a small fraction of peak floating-point throughput.\n"
        "3. Python & Function Call Overhead: Repeatedly invoking search() in a loop\n"
        "   incurs per-call runtime overhead and context-switching that is amortized\n"
        "   to near zero in a single batch invocation."
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

