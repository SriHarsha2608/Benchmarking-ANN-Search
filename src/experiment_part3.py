"""
Part 3: Compression Experiment on SIFT-128.
Varies PQ sub-vectors (m in {8, 16, 32, 64}) and compares:
1. Plain IVF-PQ
2. IVF-PQ + Exact Re-ranking of top 100 candidates (k_factor = 10.0)

Measures:
- Recall@10 (using custom tie-aware metric on 5,000 test queries)
- Single-query throughput (QPS, median of 3 repeats, 1 thread)
- Index build time and memory footprint (bytes per vector, index size)
- Appends to results/results.csv and outputs results/part3_summary.json
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


def run_compression_experiment(
    data_dir: str = "data",
    results_csv: str = "results/results.csv",
    summary_json: str = "results/part3_summary.json",
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """Run Part 3 Compression & Exact Re-ranking benchmark."""
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)

    print("=" * 70)
    print("Part 3: Compression & Re-ranking Experiment on SIFT-128")
    print("=" * 70)

    # 1. Load SIFT dataset
    print("\n[1/4] Loading SIFT-128 dataset...")
    sift_data = load_dataset("sift", data_dir=data_dir)
    train = sift_data["train"]
    queries, true_neighbors, true_distances = get_query_split(sift_data, split="test")

    dim = train.shape[1]
    n_train = len(train)
    n_queries = len(queries)
    print(f"  Base vectors: {n_train:,} (dim={dim})")
    print(f"  Evaluation queries: {n_queries:,} (split='test')")

    # 2. Build reference Flat index for exact re-ranking
    print("\n[2/4] Setting up flat reference storage for exact re-ranking...")
    t0 = time.perf_counter()
    refine_flat = faiss.IndexFlatL2(dim)
    refine_flat.add(train)
    t_flat = time.perf_counter() - t0
    flat_size_bytes = n_train * dim * 4  # 512 MB
    print(f"  Reference Flat built in {t_flat:.2f}s ({flat_size_bytes / (1024*1024):.1f} MB)")

    # Settings
    if quick:
        m_values = [16, 32]
        nlist = 1024
        nprobe_values = [1, 4, 16, 64]
        num_queries_to_time = 200
        repeats = 2
    else:
        m_values = [8, 16, 32, 64]
        nlist = 4096
        nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        num_queries_to_time = 1000
        repeats = 3

    print(f"\n[3/4] Running sweep over PQ sub-vectors: m in {m_values}")
    print(f"      nlist={nlist}, nprobe in {nprobe_values}")
    print(f"      Timing: {num_queries_to_time} queries, {repeats} repeats, single thread")

    all_records = []
    summary_data = {}

    for m in m_values:
        print(f"\n" + "-" * 60)
        print(f"--> Training IVF-PQ (nlist={nlist}, m={m}, nbits=8)...")
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(dim)
        index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
        index_ivfpq.train(train)
        index_ivfpq.add(train)
        build_time = time.perf_counter() - t0

        plain_size_bytes = measure_index_size(index_ivfpq)
        plain_size_mb = plain_size_bytes / (1024 * 1024)
        plain_bytes_per_vec = plain_size_bytes / n_train

        # Setup IndexRefine wrapping index_ivfpq
        # k_factor = 10.0 => retrieves 10 * 10 = 100 candidates from IVF-PQ,
        # then computes exact L2 distance against full vectors and returns top 10
        refine_index = faiss.IndexRefine(index_ivfpq, refine_flat)
        refine_index.k_factor = 10.0
        refine_index.own_fields = False
        refine_index.own_refine_index = False

        refine_size_bytes = plain_size_bytes + flat_size_bytes
        refine_size_mb = refine_size_bytes / (1024 * 1024)
        refine_bytes_per_vec = refine_size_bytes / n_train

        print(f"  IVF-PQ (m={m}) built in {build_time:.2f}s")
        print(f"    Plain index size: {plain_size_mb:.2f} MB ({plain_bytes_per_vec:.1f} B/vec)")
        print(f"    With raw vectors (Refine): {refine_size_mb:.2f} MB ({refine_bytes_per_vec:.1f} B/vec)")

        m_summary = {
            "m": m,
            "d_prime": dim // m,
            "plain_size_bytes": plain_size_bytes,
            "plain_bytes_per_vec": plain_bytes_per_vec,
            "refine_size_bytes": refine_size_bytes,
            "refine_bytes_per_vec": refine_bytes_per_vec,
            "build_time_s": build_time,
            "nprobe_results": [],
        }

        for nprobe in nprobe_values:
            index_ivfpq.nprobe = nprobe
            faiss.omp_set_num_threads(1)

            # --- A. Plain IVF-PQ ---
            _, I_plain = index_ivfpq.search(queries, 10)
            recall_plain = compute_recall_at_k(I_plain, true_neighbors, true_distances, k=10)
            qps_plain = time_single_queries(index_ivfpq, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            lat_plain = (1000.0 / qps_plain) if qps_plain > 0 else 0.0

            # --- B. Refined IVF-PQ (re-rank top 100) ---
            _, I_refine = refine_index.search(queries, 10)
            recall_refine = compute_recall_at_k(I_refine, true_neighbors, true_distances, k=10)
            qps_refine = time_single_queries(refine_index, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            lat_refine = (1000.0 / qps_refine) if qps_refine > 0 else 0.0

            recall_delta = recall_refine - recall_plain
            qps_ratio = (qps_refine / qps_plain) if qps_plain > 0 else 0.0

            print(
                f"  nprobe={nprobe:3d} | Plain: Rec={recall_plain:.4f}, QPS={qps_plain:7.1f} | "
                f"Refined: Rec={recall_refine:.4f}, QPS={qps_refine:7.1f} | "
                f"Gain: +{recall_delta:.4f} ({qps_ratio*100:.1f}% QPS)"
            )

            # Record Plain IVF-PQ
            row_plain = {
                "dataset": "sift",
                "index_type": "IVF-PQ",
                "build_params": f"nlist={nlist},m={m},nbits=8,rerank=False",
                "search_params": f"nprobe={nprobe}",
                "build_time_s": build_time,
                "index_size_bytes": plain_size_bytes,
                "index_size_mb": plain_size_mb,
                "bytes_per_vec": plain_bytes_per_vec,
                "recall_at_10": recall_plain,
                "qps": qps_plain,
                "latency_ms": lat_plain,
                "num_train": n_train,
                "num_queries": n_queries,
                "split": "test",
            }
            all_records.append(row_plain)

            # Record Refined IVF-PQ
            row_refine = {
                "dataset": "sift",
                "index_type": "IVF-PQ-Refine",
                "build_params": f"nlist={nlist},m={m},nbits=8,rerank=True,k_factor=10",
                "search_params": f"nprobe={nprobe}",
                "build_time_s": build_time,
                "index_size_bytes": refine_size_bytes,
                "index_size_mb": refine_size_mb,
                "bytes_per_vec": refine_bytes_per_vec,
                "recall_at_10": recall_refine,
                "qps": qps_refine,
                "latency_ms": lat_refine,
                "num_train": n_train,
                "num_queries": n_queries,
                "split": "test",
            }
            all_records.append(row_refine)

            m_summary["nprobe_results"].append({
                "nprobe": nprobe,
                "recall_plain": recall_plain,
                "qps_plain": qps_plain,
                "latency_plain_ms": lat_plain,
                "recall_refine": recall_refine,
                "qps_refine": qps_refine,
                "latency_refine_ms": lat_refine,
                "recall_gain": recall_delta,
                "qps_retention": qps_ratio,
            })

        summary_data[f"m_{m}"] = m_summary

    # 4. Save results to results.csv and summary json
    print(f"\n[4/4] Saving results...")
    new_df = pd.DataFrame(all_records)

    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
        # Combine and drop duplicates based on exact configuration key, keeping newest
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=["dataset", "index_type", "build_params", "search_params"],
            keep="last",
        )
    else:
        combined_df = new_df

    combined_df.to_csv(results_csv, index=False)
    print(f"  Appended {len(all_records)} runs to {results_csv} (Total: {len(combined_df)} rows)")

    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved summary to {summary_json}")

    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 3: Compression and Exact Re-ranking Benchmark")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset HDF5 files")
    parser.add_argument("--results-csv", type=str, default="results/results.csv", help="Path to output results CSV")
    parser.add_argument("--summary-json", type=str, default="results/part3_summary.json", help="Path to summary JSON")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark on subset of params")

    args = parser.parse_args()
    run_compression_experiment(
        data_dir=args.data_dir,
        results_csv=args.results_csv,
        summary_json=args.summary_json,
        quick=args.quick,
    )
