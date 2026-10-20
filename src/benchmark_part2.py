"""
Part 2: Benchmark Three ANN Indexes (HNSW, IVF-Flat, IVF-PQ) against Exact Search.

Methodology:
- Single thread (faiss.omp_set_num_threads(1))
- One query at a time for timing (repeated 3 times, median taken)
- Recall@10 evaluated on the last 5,000 reporting queries (split='test')
- Measures build time, on-disk/RAM index size, and bytes per vector
- Sweeps search parameters to construct Recall vs. QPS trade-off curves
- Saves detailed rows to results/results.csv and summary to results/part2_summary.json
"""

import os
import sys
import time
import argparse
import tempfile
import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import faiss

# Ensure src in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, get_query_split
from src.metrics import compute_recall_at_k


def measure_index_size(index: faiss.Index) -> int:
    """Measure exact serialized size of a FAISS index in bytes."""
    fd, tmp_path = tempfile.mkstemp(suffix=".index")
    os.close(fd)
    try:
        faiss.write_index(index, tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return size_bytes


def time_single_queries(
    index: faiss.Index,
    queries: np.ndarray,
    repeats: int = 3,
    num_queries_to_time: int = 1000,
) -> float:
    """
    Measure single-threaded, sequential 1-query search latency.
    Returns median QPS across repeats.
    """
    n_queries = min(len(queries), num_queries_to_time)
    test_q = queries[:n_queries]

    latencies = []
    # Single-thread enforcement
    faiss.omp_set_num_threads(1)

    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(n_queries):
            _ = index.search(test_q[i : i + 1], 10)
        t_elapsed = time.perf_counter() - t0
        latencies.append(t_elapsed)

    median_time = float(np.median(latencies))
    qps = n_queries / median_time if median_time > 0 else 0.0
    return qps


def benchmark_hnsw(
    dataset_name: str,
    train: np.ndarray,
    queries: np.ndarray,
    true_neighbors: np.ndarray,
    true_distances: np.ndarray,
    m_values: List[int],
    ef_search_values: List[int],
    ef_construction: int = 64,
    repeats: int = 3,
    num_queries_to_time: int = 1000,
) -> List[Dict[str, Any]]:
    """Benchmark HNSW across M build settings and efSearch search settings."""
    dim = train.shape[1]
    n_train = len(train)
    results = []

    print(f"\n--- Benchmarking HNSW on {dataset_name.upper()} ---")

    for M in m_values:
        print(f"Building HNSW (M={M}, efConstruction={ef_construction})...")
        t0 = time.perf_counter()
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = ef_construction
        index.add(train)
        build_time = time.perf_counter() - t0

        size_bytes = measure_index_size(index)
        size_mb = size_bytes / (1024 * 1024)
        bytes_per_vec = size_bytes / n_train
        print(f"  Build time: {build_time:.2f}s | Size: {size_mb:.2f} MB ({bytes_per_vec:.1f} bytes/vec)")

        for ef in ef_search_values:
            index.hnsw.efSearch = ef

            # Evaluate recall on all test queries
            faiss.omp_set_num_threads(1)
            _, I = index.search(queries, 10)
            recall = compute_recall_at_k(I, true_neighbors, true_distances, k=10)

            # Measure sequential single-query QPS
            qps = time_single_queries(index, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            latency_ms = (1000.0 / qps) if qps > 0 else 0.0

            print(f"  [HNSW M={M}] efSearch={ef:3d} -> Recall@10: {recall:.4f} | QPS: {qps:8.1f} ({latency_ms:.3f} ms)")

            results.append({
                "dataset": dataset_name,
                "index_type": "HNSW",
                "build_params": f"M={M},efC={ef_construction}",
                "search_params": f"efSearch={ef}",
                "build_time_s": build_time,
                "index_size_bytes": size_bytes,
                "index_size_mb": size_mb,
                "bytes_per_vec": bytes_per_vec,
                "recall_at_10": recall,
                "qps": qps,
                "latency_ms": latency_ms,
                "num_train": n_train,
                "num_queries": len(queries),
                "split": "test",
            })

    return results


def benchmark_ivf_flat(
    dataset_name: str,
    train: np.ndarray,
    queries: np.ndarray,
    true_neighbors: np.ndarray,
    true_distances: np.ndarray,
    nlist_values: List[int],
    nprobe_values: List[int],
    repeats: int = 3,
    num_queries_to_time: int = 1000,
) -> List[Dict[str, Any]]:
    """Benchmark IVF-Flat across nlist build settings and nprobe search settings."""
    dim = train.shape[1]
    n_train = len(train)
    results = []

    print(f"\n--- Benchmarking IVF-Flat on {dataset_name.upper()} ---")

    for nlist in nlist_values:
        print(f"Building IVF-Flat (nlist={nlist})...")
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(train)
        index.add(train)
        build_time = time.perf_counter() - t0

        size_bytes = measure_index_size(index)
        size_mb = size_bytes / (1024 * 1024)
        bytes_per_vec = size_bytes / n_train
        print(f"  Build time: {build_time:.2f}s | Size: {size_mb:.2f} MB ({bytes_per_vec:.1f} bytes/vec)")

        for nprobe in nprobe_values:
            if nprobe > nlist:
                continue
            index.nprobe = nprobe

            faiss.omp_set_num_threads(1)
            _, I = index.search(queries, 10)
            recall = compute_recall_at_k(I, true_neighbors, true_distances, k=10)

            qps = time_single_queries(index, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            latency_ms = (1000.0 / qps) if qps > 0 else 0.0

            print(f"  [IVF-Flat nlist={nlist}] nprobe={nprobe:3d} -> Recall@10: {recall:.4f} | QPS: {qps:8.1f} ({latency_ms:.3f} ms)")

            results.append({
                "dataset": dataset_name,
                "index_type": "IVF-Flat",
                "build_params": f"nlist={nlist}",
                "search_params": f"nprobe={nprobe}",
                "build_time_s": build_time,
                "index_size_bytes": size_bytes,
                "index_size_mb": size_mb,
                "bytes_per_vec": bytes_per_vec,
                "recall_at_10": recall,
                "qps": qps,
                "latency_ms": latency_ms,
                "num_train": n_train,
                "num_queries": len(queries),
                "split": "test",
            })

    return results


def benchmark_ivf_pq(
    dataset_name: str,
    train: np.ndarray,
    queries: np.ndarray,
    true_neighbors: np.ndarray,
    true_distances: np.ndarray,
    nlist: int,
    m_values: List[int],
    nprobe_values: List[int],
    nbits: int = 8,
    repeats: int = 3,
    num_queries_to_time: int = 1000,
) -> List[Dict[str, Any]]:
    """Benchmark IVF-PQ across sub-quantizer m values and nprobe search settings."""
    dim = train.shape[1]
    n_train = len(train)
    results = []

    print(f"\n--- Benchmarking IVF-PQ on {dataset_name.upper()} ---")

    for m in m_values:
        if dim % m != 0:
            print(f"Skipping m={m} (dim {dim} not divisible by {m})")
            continue

        print(f"Building IVF-PQ (nlist={nlist}, m={m}, nbits={nbits})...")
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.train(train)
        index.add(train)
        build_time = time.perf_counter() - t0

        size_bytes = measure_index_size(index)
        size_mb = size_bytes / (1024 * 1024)
        bytes_per_vec = size_bytes / n_train
        print(f"  Build time: {build_time:.2f}s | Size: {size_mb:.2f} MB ({bytes_per_vec:.1f} bytes/vec)")

        for nprobe in nprobe_values:
            if nprobe > nlist:
                continue
            index.nprobe = nprobe

            faiss.omp_set_num_threads(1)
            _, I = index.search(queries, 10)
            recall = compute_recall_at_k(I, true_neighbors, true_distances, k=10)

            qps = time_single_queries(index, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            latency_ms = (1000.0 / qps) if qps > 0 else 0.0

            print(f"  [IVF-PQ m={m}] nprobe={nprobe:3d} -> Recall@10: {recall:.4f} | QPS: {qps:8.1f} ({latency_ms:.3f} ms)")

            results.append({
                "dataset": dataset_name,
                "index_type": "IVF-PQ",
                "build_params": f"nlist={nlist},m={m},nbits={nbits}",
                "search_params": f"nprobe={nprobe}",
                "build_time_s": build_time,
                "index_size_bytes": size_bytes,
                "index_size_mb": size_mb,
                "bytes_per_vec": bytes_per_vec,
                "recall_at_10": recall,
                "qps": qps,
                "latency_ms": latency_ms,
                "num_train": n_train,
                "num_queries": len(queries),
                "split": "test",
            })

    return results


def run_benchmark_for_dataset(
    dataset_name: str,
    data_dir: str = "data",
    split: str = "test",
    repeats: int = 3,
    num_queries_to_time: int = 1000,
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """Run full Part 2 benchmarks on a single dataset."""
    max_train = 100000 if quick else None
    ds = load_dataset(dataset_name, data_dir=data_dir, max_train_vectors=max_train)
    train = ds["train"]
    dim = ds["dim"]

    # In quick mode, compute ground truth on the 100k subset
    if quick:
        print(f"[Quick Mode] Computing ground truth on 100k {dataset_name.upper()} vectors...")
        raw_queries, _, _ = get_query_split(ds, split=split)
        queries = raw_queries[:1000]  # evaluate on 1000 queries in quick mode
        exact_idx = faiss.IndexFlatL2(dim)
        exact_idx.add(train)
        true_distances, true_neighbors = exact_idx.search(queries, 100)
    else:
        queries, true_neighbors, true_distances = get_query_split(ds, split=split)

    # Load exact baseline numbers if available
    exact_results = []
    baseline_file = os.path.join("results", "exact_baseline.json")
    if os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            base_data = json.load(f)
            for item in base_data:
                if item["dataset"] == ds["name"]:
                    exact_results.append({
                        "dataset": ds["name"],
                        "index_type": "Exact",
                        "build_params": "IndexFlatL2",
                        "search_params": "exact",
                        "build_time_s": item["build_time_s"],
                        "index_size_bytes": int(item["index_mb"] * 1024 * 1024),
                        "index_size_mb": item["index_mb"],
                        "bytes_per_vec": item["bytes_per_vec"],
                        "recall_at_10": 1.0,
                        "qps": item["single_qps"],
                        "latency_ms": item["single_latency_ms"],
                        "num_train": item["num_train"],
                        "num_queries": len(queries),
                        "split": split,
                    })

    # Parameter grids
    if quick:
        hnsw_m = [16]
        hnsw_ef_search = [10, 30, 80, 160]
        ivf_nlist = [256, 512]
        ivf_nprobe = [1, 4, 16, 64]
        pq_nlist = 512
        pq_m = [16, 32] if ds["name"] == "sift" else [20, 25]
        pq_nprobe = [1, 4, 16, 64]
    else:
        hnsw_m = [16, 32]
        hnsw_ef_search = [10, 20, 35, 60, 100, 160, 256, 400]
        ivf_nlist = [1024, 2048]
        ivf_nprobe = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        pq_nlist = 2048
        pq_m = [16, 32, 64] if ds["name"] == "sift" else [20, 25, 50]
        pq_nprobe = [1, 2, 4, 8, 16, 32, 64, 128]

    all_dataset_results = []
    all_dataset_results.extend(exact_results)

    # 1. Benchmark HNSW
    res_hnsw = benchmark_hnsw(
        ds["name"],
        train,
        queries,
        true_neighbors,
        true_distances,
        m_values=hnsw_m,
        ef_search_values=hnsw_ef_search,
        repeats=repeats,
        num_queries_to_time=num_queries_to_time,
    )
    all_dataset_results.extend(res_hnsw)

    # 2. Benchmark IVF-Flat
    res_ivf = benchmark_ivf_flat(
        ds["name"],
        train,
        queries,
        true_neighbors,
        true_distances,
        nlist_values=ivf_nlist,
        nprobe_values=ivf_nprobe,
        repeats=repeats,
        num_queries_to_time=num_queries_to_time,
    )
    all_dataset_results.extend(res_ivf)

    # 3. Benchmark IVF-PQ
    res_pq = benchmark_ivf_pq(
        ds["name"],
        train,
        queries,
        true_neighbors,
        true_distances,
        nlist=pq_nlist,
        m_values=pq_m,
        nprobe_values=pq_nprobe,
        repeats=repeats,
        num_queries_to_time=num_queries_to_time,
    )
    all_dataset_results.extend(res_pq)

    return all_dataset_results


def append_to_results_csv(new_rows: List[Dict[str, Any]], csv_path: str = "results/results.csv"):
    """Append new benchmark rows to results.csv without duplicates."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame(new_rows)

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        combined = pd.concat([df_old, df_new], ignore_index=True)
        # Deduplicate based on dataset, index_type, build_params, search_params, split
        combined = combined.drop_duplicates(
            subset=["dataset", "index_type", "build_params", "search_params", "split"],
            keep="last",
        )
        combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)

    print(f"Saved/Updated benchmark rows in: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Part 2: Benchmark HNSW, IVF-Flat, and IVF-PQ.")
    parser.add_argument("--dataset", choices=["sift", "glove", "both"], default="both")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["test", "tune", "all"], default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-single-queries", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Quick mode (100k vectors, fewer sweeps)")
    parser.add_argument("--out-csv", default="results/results.csv")
    args = parser.parse_args()

    datasets = ["sift", "glove"] if args.dataset == "both" else [args.dataset]
    all_runs = []

    for ds in datasets:
        runs = run_benchmark_for_dataset(
            dataset_name=ds,
            data_dir=args.data_dir,
            split=args.split,
            repeats=args.repeats,
            num_queries_to_time=args.num_single_queries,
            quick=args.quick,
        )
        all_runs.extend(runs)

    append_to_results_csv(all_runs, csv_path=args.out_csv)
    print("\nPart 2 benchmark execution finished successfully!")


if __name__ == "__main__":
    main()

