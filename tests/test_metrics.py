"""
Unit tests for the Recall@k metric and tie-breaking handling.
"""

import pytest
import numpy as np
from src.metrics import recall_at_k_single, compute_recall_at_k


def test_perfect_match():
    retrieved = np.arange(10)
    true_ids = np.arange(20)
    true_dists = np.arange(20, dtype=float)

    rec = recall_at_k_single(retrieved, true_ids, true_dists, k=10)
    assert rec == pytest.approx(1.0)


def test_zero_overlap():
    retrieved = np.arange(100, 110)
    true_ids = np.arange(20)
    true_dists = np.arange(20, dtype=float)

    rec = recall_at_k_single(retrieved, true_ids, true_dists, k=10)
    assert rec == pytest.approx(0.0)


def test_partial_overlap():
    # 7 out of 10 matches
    retrieved = np.array([0, 1, 2, 3, 4, 5, 6, 991, 992, 993])
    true_ids = np.arange(20)
    true_dists = np.arange(20, dtype=float)

    rec = recall_at_k_single(retrieved, true_ids, true_dists, k=10)
    assert rec == pytest.approx(0.7)


def test_unordered_retrieval():
    # Top 10 returned in reversed order
    retrieved = np.arange(9, -1, -1)
    true_ids = np.arange(20)
    true_dists = np.arange(20, dtype=float)

    rec = recall_at_k_single(retrieved, true_ids, true_dists, k=10)
    assert rec == pytest.approx(1.0)


def test_tie_at_tenth_neighbour():
    """
    Crucial test: Ranks 1 to 9 have strictly smaller distances.
    Rank 10 (id 9) and Rank 11 (id 10) have the EXACT same distance (10.0).
    The index retrieved id 10 instead of id 9.
    Since id 10 is tied at the 10th distance, it must NOT be counted as a mistake.
    Recall should be 1.0 (10/10), NOT 0.9.
    """
    # true_ids: [0, 1, ..., 8, 9, 10, 11]
    true_ids = np.arange(15)
    # dists: 0.0 to 8.0 for first 9, then 10.0 for rank 10 (idx 9) and rank 11 (idx 10)
    true_dists = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 12.0, 13.0, 14.0, 15.0])

    # Retrieved has [0..8] plus id 10 (the tied rank-11 item)
    retrieved = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10])

    # Standard recall (without distances) would fail this and return 0.9
    naive_rec = recall_at_k_single(retrieved, true_ids, true_distances=None, k=10)
    assert naive_rec == pytest.approx(0.9)

    # Tie-aware recall must return 1.0
    tie_rec = recall_at_k_single(retrieved, true_ids, true_distances=true_dists, k=10)
    assert tie_rec == pytest.approx(1.0)


def test_multiple_ties_at_boundary():
    """
    Ranks 8, 9, 10, 11, 12 all have distance 5.0.
    Ranks 0..6 (7 items) have strictly smaller distance (< 5.0).
    k = 10, so there are 3 slots available for items with distance 5.0.
    If retrieved contains the 7 strict items and any 3 of the tied items, recall is 1.0.
    """
    true_ids = np.array([0, 1, 2, 3, 4, 5, 6,  7, 8, 9, 10, 11, 12,  13, 14])
    true_dists = np.array([1, 1, 2, 2, 3, 3, 4,  5, 5, 5,  5,  5,  5,   8,  9], dtype=float)

    # Pick 7 strict items + items 10, 11, 12 (which were ranks 11, 12, 13 in the ground truth file)
    retrieved = np.array([0, 1, 2, 3, 4, 5, 6, 10, 11, 12])

    rec = recall_at_k_single(retrieved, true_ids, true_distances=true_dists, k=10)
    assert rec == pytest.approx(1.0)


def test_tied_items_cannot_overfill_slots():
    """
    If a query misses strict closer neighbours, retrieving extra tied items
    must NOT count toward the missed strict neighbours.
    """
    true_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    true_dists = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0, 10.0])
    # Strict items: 0..8 (9 items).
    # Tied at 10.0: items 9, 10, 11.
    # Allowed tied slot in top-10: 10 - 9 = 1.

    # Suppose retrieved has only 5 strict items (0..4) and all 3 tied items (9, 10, 11) + 2 garbage items
    retrieved = np.array([0, 1, 2, 3, 4, 9, 10, 11, 998, 999])

    # Correct items: 5 strict + min(3, 1) tied = 6.
    # Recall should be 6/10 = 0.6.
    rec = recall_at_k_single(retrieved, true_ids, true_distances=true_dists, k=10)
    assert rec == pytest.approx(0.6)


def test_all_identical_distances():
    """
    Edge case: All true items have identical distance 1.0.
    Any subset of k true items retrieved must achieve recall 1.0.
    """
    true_ids = np.arange(50)
    true_dists = np.ones(50, dtype=float)

    retrieved = np.array([45, 46, 47, 48, 49, 30, 31, 32, 33, 34])
    rec = recall_at_k_single(retrieved, true_ids, true_distances=true_dists, k=10)
    assert rec == pytest.approx(1.0)


def test_batch_recall_computation():
    true_ids = np.tile(np.arange(20), (5, 1))
    true_dists = np.tile(np.arange(20, dtype=float), (5, 1))

    # Construct retrieved with varying overlaps: 1.0, 0.8, 0.6, 0.4, 0.2
    retrieved = []
    for overlap in [10, 8, 6, 4, 2]:
        row = list(range(overlap)) + list(range(900, 900 + (10 - overlap)))
        retrieved.append(row)
    retrieved = np.array(retrieved)

    # Per-query recalls
    per_query = compute_recall_at_k(retrieved, true_ids, true_dists, k=10, return_per_query=True)
    assert np.allclose(per_query, [1.0, 0.8, 0.6, 0.4, 0.2])

    # Mean recall
    mean_rec = compute_recall_at_k(retrieved, true_ids, true_dists, k=10, return_per_query=False)
    assert mean_rec == pytest.approx(0.6)


def test_input_validation():
    with pytest.raises(ValueError):
        recall_at_k_single(np.arange(10), np.arange(10), k=0)

    with pytest.raises(ValueError):
        recall_at_k_single(np.arange(10), np.arange(10), true_distances=np.arange(5), k=10)


def test_real_data_exact_search_agreement():
    """
    Integration test confirming that FAISS exact search achieves Recall@10 == 1.0
    against the ground-truth stored in real datasets (as required by Part 1 of A3.md).
    """
    import os
    import faiss
    from src.data_loader import load_dataset, get_query_split

    for ds_name in ["sift", "glove"]:
        filename = "sift-128-euclidean.hdf5" if ds_name == "sift" else "glove-100-angular.hdf5"
        filepath = os.path.join("data", filename)
        if not os.path.exists(filepath):
            pytest.skip(f"Dataset file {filepath} not found.")

        ds = load_dataset(ds_name, data_dir="data")
        queries, true_neighbors, true_distances = get_query_split(ds, split="test")

        sub_queries = queries[:100]
        sub_neighbors = true_neighbors[:100]
        sub_distances = true_distances[:100]

        index = faiss.IndexFlatL2(ds["dim"])
        index.add(ds["train"])
        D, I = index.search(sub_queries, 10)

        agreement = compute_recall_at_k(I, sub_neighbors, sub_distances, k=10)
        assert agreement == pytest.approx(1.0, abs=1e-5)


