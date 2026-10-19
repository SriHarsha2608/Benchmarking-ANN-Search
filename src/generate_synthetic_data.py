"""
Part 4: Synthetic Data Generator with Controlled Cluster Geometry.

Generates reproducible high-dimensional datasets with parameterized
cluster dispersion (intra-cluster variance sigma).
Used to evaluate space-partitioning (IVF-Flat) and graph traversal (HNSW)
failure modes under varying data geometry.
"""

import os
import argparse
from typing import Tuple, Dict, Any
import numpy as np
import faiss


def generate_synthetic_dataset(
    n_samples: int = 200000,
    dim: int = 64,
    n_clusters: int = 50,
    sigma: float = 0.05,
    n_queries: int = 2000,
    seed: int = 42,
    cluster_scale: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data from a Gaussian Mixture Model with controlled dispersion.

    Parameters
    ----------
    n_samples : int, default=200,000
        Number of base vectors to index.
    dim : int, default=64
        Vector dimensionality.
    n_clusters : int, default=50
        Number of natural cluster centroids.
    sigma : float, default=0.05
        Intra-cluster standard deviation (cluster dispersion).
    n_queries : int, default=2,000
        Number of evaluation queries sampled from the same mixture.
    seed : int, default=42
        Random seed for full reproducibility.
    cluster_scale : float, default=3.0
        Radius of cluster centroid sphere.

    Returns
    -------
    dict with keys:
        'train': base vectors (n_samples, dim), float32
        'queries': test queries (n_queries, dim), float32
        'true_neighbors': exact 10-NN indices (n_queries, 10), int64
        'true_distances': exact 10-NN squared L2 distances (n_queries, 10), float32
        'centroids': cluster centers (n_clusters, dim), float32
        'sigma': float
    """
    rng = np.random.default_rng(seed)

    # 1. Generate cluster centroids on a sphere
    raw_centroids = rng.normal(0, 1.0, size=(n_clusters, dim)).astype(np.float32)
    norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
    centroids = (raw_centroids / norms) * float(cluster_scale)

    # 2. Sample base vectors from Gaussian clusters
    labels = rng.integers(0, n_clusters, size=n_samples)
    noise = rng.normal(0, sigma, size=(n_samples, dim)).astype(np.float32)
    train = (centroids[labels] + noise).astype(np.float32)

    # 3. Sample test queries from the same mixture
    q_labels = rng.integers(0, n_clusters, size=n_queries)
    q_noise = rng.normal(0, sigma, size=(n_queries, dim)).astype(np.float32)
    queries = (centroids[q_labels] + q_noise).astype(np.float32)

    # 4. Compute exact ground-truth nearest neighbours
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(train)
    true_distances, true_neighbors = index_flat.search(queries, 10)

    return {
        "train": train,
        "queries": queries,
        "true_neighbors": true_neighbors,
        "true_distances": true_distances,
        "centroids": centroids,
        "sigma": sigma,
        "dim": dim,
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Data Generator for Part 4")
    parser.add_argument("--n-samples", type=int, default=200000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--n-queries", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    data = generate_synthetic_dataset(
        n_samples=args.n_samples,
        dim=args.dim,
        n_clusters=args.n_clusters,
        sigma=args.sigma,
        n_queries=args.n_queries,
        seed=args.seed,
    )
    print(f"Generated synthetic dataset (sigma={data['sigma']}):")
    print(f"  Base vectors: {data['train'].shape}")
    print(f"  Queries:      {data['queries'].shape}")
    print(f"  Ground-truth: {data['true_neighbors'].shape}")

