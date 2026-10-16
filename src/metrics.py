"""
Metrics for Approximate Nearest Neighbour Search evaluation.

This module provides functions for computing Recall@k, explicitly handling
distance ties at the k-th neighbour rank so that ties are not penalized as mistakes.
"""

from typing import Optional, Union
import numpy as np


def recall_at_k_single(
    retrieved_ids: np.ndarray,
    true_ids: np.ndarray,
    true_distances: Optional[np.ndarray] = None,
    k: int = 10,
    tol: float = 1e-4,
) -> float:
    """
    Compute Recall@k for a single query, handling ties at the k-th neighbour.

    Parameters
    ----------
    retrieved_ids : np.ndarray
        1D array of retrieved neighbour IDs (length >= k). Only the first k are evaluated.
    true_ids : np.ndarray
        1D array of ground truth neighbour IDs sorted by distance ascending.
    true_distances : np.ndarray, optional
        1D array of true distances corresponding to true_ids.
        If None, standard set intersection with true_ids[:k] is used.
    k : int, default=10
        Number of top neighbours to evaluate.
    tol : float, default=1e-5
        Absolute tolerance for considering two distances tied.

    Returns
    -------
    float
        Recall@k value in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    retrieved_topk = np.asarray(retrieved_ids[:k])
    retrieved_set = set(retrieved_topk.tolist())

    if len(true_ids) == 0:
        return 0.0

    actual_k = min(k, len(true_ids))

    # Fallback to standard set intersection if distances are not provided
    if true_distances is None:
        target_set = set(true_ids[:actual_k].tolist())
        return len(retrieved_set.intersection(target_set)) / float(actual_k)

    true_ids = np.asarray(true_ids)
    true_distances = np.asarray(true_distances)

    if len(true_ids) != len(true_distances):
        raise ValueError("true_ids and true_distances must have identical length.")

    # Distance of the k-th true neighbour (0-indexed rank actual_k - 1)
    kth_distance = true_distances[actual_k - 1]

    # Partition ground-truth items into:
    # 1. Strictly closer than the k-th neighbour: dist < kth_distance - tol
    # 2. Tied with the k-th neighbour: |dist - kth_distance| <= tol
    strict_mask = true_distances < (kth_distance - tol)
    tied_mask = np.abs(true_distances - kth_distance) <= tol

    strict_ids = set(true_ids[strict_mask].tolist())
    tied_ids = set(true_ids[tied_mask].tolist())

    # Items in retrieved set that match strictly closer neighbours
    matched_strict = len(retrieved_set.intersection(strict_ids))

    # Items in retrieved set that match tied neighbours
    matched_tied = len(retrieved_set.intersection(tied_ids))

    # In any valid top-k permutation, at most (actual_k - len(strict_ids))
    # slots can be filled by tied neighbours.
    allowed_tied_slots = max(0, actual_k - len(strict_ids))
    accepted_tied = min(matched_tied, allowed_tied_slots)

    total_correct = matched_strict + accepted_tied
    return float(total_correct) / float(actual_k)


def compute_recall_at_k(
    retrieved_ids: np.ndarray,
    true_ids: np.ndarray,
    true_distances: Optional[np.ndarray] = None,
    k: int = 10,
    tol: float = 1e-4,
    return_per_query: bool = False,
) -> Union[float, np.ndarray]:
    """
    Compute average Recall@k across one or more queries with tie handling.

    Parameters
    ----------
    retrieved_ids : np.ndarray
        Array of retrieved IDs. Shape (num_queries, retrieved_k) or (retrieved_k,).
    true_ids : np.ndarray
        Array of ground truth IDs. Shape (num_queries, num_true) or (num_true,).
    true_distances : np.ndarray, optional
        Array of ground truth distances. Shape matching true_ids.
    k : int, default=10
        Cutoff rank for recall.
    tol : float, default=1e-5
        Distance tolerance for tie detection.
    return_per_query : bool, default=False
        If True, return an array of recalls per query instead of the mean.

    Returns
    -------
    Union[float, np.ndarray]
        Mean recall@k across all queries, or 1D array of per-query recalls.
    """
    retrieved_ids = np.asarray(retrieved_ids)
    true_ids = np.asarray(true_ids)
    if true_distances is not None:
        true_distances = np.asarray(true_distances)

    if retrieved_ids.ndim == 1:
        # Single query
        rec = recall_at_k_single(
            retrieved_ids,
            true_ids,
            true_distances=true_distances,
            k=k,
            tol=tol,
        )
        return np.array([rec]) if return_per_query else rec

    # Batched queries: shape (N, ...)
    num_queries = retrieved_ids.shape[0]
    recalls = np.empty(num_queries, dtype=np.float64)

    for i in range(num_queries):
        dists = true_distances[i] if true_distances is not None else None
        recalls[i] = recall_at_k_single(
            retrieved_ids[i],
            true_ids[i],
            true_distances=dists,
            k=k,
            tol=tol,
        )

    if return_per_query:
        return recalls
    return float(np.mean(recalls))
