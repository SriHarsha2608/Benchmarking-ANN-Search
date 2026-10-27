"""
Part 4: Failure Mode Experiment on Synthetic Data with Controlled Geometry.

Studies the impact of cluster dispersion (sigma in {0.05, 0.15, 0.35, 0.70})
on space-partitioning (IVF-Flat) and proximity graph (HNSW) indexes.
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

from src.generate_synthetic_data import generate_synthetic_dataset
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
    num_queries_to_time: int = 500,
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


def run_geometry_experiment(
    results_csv: str = "results/results.csv",
    summary_json: str = "results/part4_summary.json",
    quick: bool = False,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run Part 4 Cluster Dispersion Failure Mode Benchmark."""
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)

    print("=" * 70)
    print("Part 4: Failure Mode Experiment on Generated Cluster Geometry")
    print("=" * 70)

    if quick:
        sigmas = [0.05, 0.70]
        n_samples = 50000
        n_queries = 1000
        nlist = 512
        nprobe_values = [1, 4, 16, 64]
        ef_search_values = [10, 40, 160]
        num_queries_to_time = 200
        repeats = 2
    else:
        sigmas = [0.05, 0.15, 0.35, 0.70]
        n_samples = 200000
        n_queries = 2000
        nlist = 1024
        nprobe_values = [1, 2, 4, 8, 16, 32, 64, 128]
        ef_search_values = [10, 20, 40, 80, 160, 320]
        num_queries_to_time = 500
        repeats = 3

    dim = 64
    n_clusters = 50
    m_hnsw = 16
    ef_construction = 64

    print(f"Configurations:")
    print(f"  N={n_samples:,}, D={dim}, K={n_clusters} clusters, Seed={seed}")
    print(f"  Sigmas: {sigmas} (Control: sigma=0.05)")
    print(f"  IVF-Flat: nlist={nlist}, nprobe in {nprobe_values}")
    print(f"  HNSW: M={m_hnsw}, efC={ef_construction}, efSearch in {ef_search_values}")
    print(f"  Timing: {num_queries_to_time} queries, {repeats} repeats, single thread")

    all_records = []
    summary_data = {}

    for sigma in sigmas:
        dataset_name = f"synthetic_sigma_{sigma}"
        print(f"\n" + "=" * 60)
        print(f"--> Generating synthetic data for sigma = {sigma} ...")
        t0 = time.perf_counter()
        data = generate_synthetic_dataset(
            n_samples=n_samples,
            dim=dim,
            n_clusters=n_clusters,
            sigma=sigma,
            n_queries=n_queries,
            seed=seed,
        )
        gen_time = time.perf_counter() - t0
        train = data["train"]
        queries = data["queries"]
        true_neighbors = data["true_neighbors"]
        true_distances = data["true_distances"]
        print(f"    Generated {n_samples:,} vectors and {n_queries:,} queries in {gen_time:.2f}s")

        sigma_summary: Dict[str, Any] = {
            "sigma": sigma,
            "n_samples": n_samples,
            "dim": dim,
            "n_clusters": n_clusters,
            "exact": {},
            "ivf_flat": {},
            "hnsw": {},
        }

        # ---------------------------------------------------------
        # 1. Exact Search Baseline
        # ---------------------------------------------------------
        print(f"  [1/3] Benchmarking Exact Search (IndexFlatL2)...")
        index_exact = faiss.IndexFlatL2(dim)
        t0 = time.perf_counter()
        index_exact.add(train)
        exact_build_time = time.perf_counter() - t0
        exact_size_bytes = n_samples * dim * 4
        exact_qps = time_single_queries(index_exact, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
        exact_lat = (1000.0 / exact_qps) if exact_qps > 0 else 0.0

        all_records.append({
            "dataset": dataset_name,
            "index_type": "Exact",
            "build_params": "IndexFlatL2",
            "search_params": "exact",
            "build_time_s": exact_build_time,
            "index_size_bytes": exact_size_bytes,
            "index_size_mb": exact_size_bytes / (1024 * 1024),
            "bytes_per_vec": exact_size_bytes / n_samples,
            "recall_at_10": 1.0,
            "qps": exact_qps,
            "latency_ms": exact_lat,
            "num_train": n_samples,
            "num_queries": n_queries,
            "split": "test",
        })
        sigma_summary["exact"] = {
            "build_time_s": exact_build_time,
            "qps": exact_qps,
            "latency_ms": exact_lat,
        }
        print(f"    Exact: {exact_qps:.1f} QPS ({exact_lat:.2f} ms)")

        # ---------------------------------------------------------
        # 2. IVF-Flat Benchmark
        # ---------------------------------------------------------
        print(f"  [2/3] Benchmarking IVF-Flat (nlist={nlist})...")
        quantizer = faiss.IndexFlatL2(dim)
        index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        t0 = time.perf_counter()
        index_ivf.train(train)
        index_ivf.add(train)
        ivf_build_time = time.perf_counter() - t0

        ivf_size_bytes = measure_index_size(index_ivf)
        ivf_size_mb = ivf_size_bytes / (1024 * 1024)
        ivf_bpv = ivf_size_bytes / n_samples
        print(f"    IVF-Flat built in {ivf_build_time:.2f}s ({ivf_size_mb:.2f} MB, {ivf_bpv:.1f} B/vec)")

        ivf_results = []
        for np_val in nprobe_values:
            index_ivf.nprobe = np_val
            faiss.omp_set_num_threads(1)
            _, I_ivf = index_ivf.search(queries, 10)
            rec = compute_recall_at_k(I_ivf, true_neighbors, true_distances, k=10)
            qps = time_single_queries(index_ivf, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            lat = (1000.0 / qps) if qps > 0 else 0.0

            all_records.append({
                "dataset": dataset_name,
                "index_type": "IVF-Flat",
                "build_params": f"nlist={nlist}",
                "search_params": f"nprobe={np_val}",
                "build_time_s": ivf_build_time,
                "index_size_bytes": ivf_size_bytes,
                "index_size_mb": ivf_size_mb,
                "bytes_per_vec": ivf_bpv,
                "recall_at_10": rec,
                "qps": qps,
                "latency_ms": lat,
                "num_train": n_samples,
                "num_queries": n_queries,
                "split": "test",
            })
            ivf_results.append({"nprobe": np_val, "recall": rec, "qps": qps, "latency_ms": lat})
            print(f"      nprobe={np_val:3d} -> Recall@10: {rec:.4f} | QPS: {qps:8.1f} ({lat:.3f} ms)")

        sigma_summary["ivf_flat"] = {
            "build_time_s": ivf_build_time,
            "index_size_mb": ivf_size_mb,
            "bytes_per_vec": ivf_bpv,
            "results": ivf_results,
        }

        # ---------------------------------------------------------
        # 3. HNSW Benchmark
        # ---------------------------------------------------------
        print(f"  [3/3] Benchmarking HNSW (M={m_hnsw}, efC={ef_construction})...")
        index_hnsw = faiss.IndexHNSWFlat(dim, m_hnsw)
        index_hnsw.hnsw.efConstruction = ef_construction
        t0 = time.perf_counter()
        index_hnsw.add(train)
        hnsw_build_time = time.perf_counter() - t0

        hnsw_size_bytes = measure_index_size(index_hnsw)
        hnsw_size_mb = hnsw_size_bytes / (1024 * 1024)
        hnsw_bpv = hnsw_size_bytes / n_samples
        print(f"    HNSW built in {hnsw_build_time:.2f}s ({hnsw_size_mb:.2f} MB, {hnsw_bpv:.1f} B/vec)")

        hnsw_results = []
        for ef in ef_search_values:
            index_hnsw.hnsw.efSearch = ef
            faiss.omp_set_num_threads(1)
            _, I_hnsw = index_hnsw.search(queries, 10)
            rec = compute_recall_at_k(I_hnsw, true_neighbors, true_distances, k=10)
            qps = time_single_queries(index_hnsw, queries, repeats=repeats, num_queries_to_time=num_queries_to_time)
            lat = (1000.0 / qps) if qps > 0 else 0.0

            all_records.append({
                "dataset": dataset_name,
                "index_type": "HNSW",
                "build_params": f"M={m_hnsw},efC={ef_construction}",
                "search_params": f"efSearch={ef}",
                "build_time_s": hnsw_build_time,
                "index_size_bytes": hnsw_size_bytes,
                "index_size_mb": hnsw_size_mb,
                "bytes_per_vec": hnsw_bpv,
                "recall_at_10": rec,
                "qps": qps,
                "latency_ms": lat,
                "num_train": n_samples,
                "num_queries": n_queries,
                "split": "test",
            })
            hnsw_results.append({"efSearch": ef, "recall": rec, "qps": qps, "latency_ms": lat})
            print(f"      efSearch={ef:3d} -> Recall@10: {rec:.4f} | QPS: {qps:8.1f} ({lat:.3f} ms)")

        sigma_summary["hnsw"] = {
            "build_time_s": hnsw_build_time,
            "index_size_mb": hnsw_size_mb,
            "bytes_per_vec": hnsw_bpv,
            "results": hnsw_results,
        }

        summary_data[f"sigma_{sigma}"] = sigma_summary

    # Save to results.csv
    print(f"\nSaving results...")
    new_df = pd.DataFrame(all_records)
    if os.path.exists(results_csv):
        existing_df = pd.read_csv(results_csv)
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
    parser = argparse.ArgumentParser(description="Part 4: Failure Mode Experiment on Synthetic Geometry")
    parser.add_argument("--results-csv", type=str, default="results/results.csv")
    parser.add_argument("--summary-json", type=str, default="results/part4_summary.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    # Single thread for builds as well as searches, so build times are comparable
    faiss.omp_set_num_threads(1)
    run_geometry_experiment(
        results_csv=args.results_csv,
        summary_json=args.summary_json,
        quick=args.quick,
        seed=args.seed,
    )

