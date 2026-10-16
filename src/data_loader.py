"""
Data loader and preprocessing for SIFT-128 and GloVe-100 datasets.
"""

import os
from typing import Dict, Tuple, Optional
import h5py
import numpy as np
import faiss


def load_dataset(
    dataset_name: str,
    data_dir: str = "data",
    max_train_vectors: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load SIFT or GloVe dataset from HDF5 file.

    For GloVe vectors, vectors are normalized to unit L2 norm as required by A3.md,
    enabling standard Euclidean distance (IndexFlatL2) to match angular nearest neighbours.

    Parameters
    ----------
    dataset_name : str
        'sift' or 'glove'
    data_dir : str, default='data'
        Path to data folder containing the HDF5 files.
    max_train_vectors : int, optional
        Limit number of train vectors loaded (useful for scaling experiments and quick runs).

    Returns
    -------
    dict with keys:
        'train': np.ndarray of shape (N, D), float32
        'test': np.ndarray of shape (10000, D), float32
        'neighbors': np.ndarray of shape (10000, 100), int32
        'distances': np.ndarray of shape (10000, 100), float32
        'name': str
        'dim': int
    """
    name_lower = dataset_name.lower()
    if "sift" in name_lower:
        filename = "sift-128-euclidean.hdf5"
        is_glove = False
    elif "glove" in name_lower:
        filename = "glove-100-angular.hdf5"
        is_glove = True
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Must contain 'sift' or 'glove'.")

    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    with h5py.File(filepath, "r") as f:
        if max_train_vectors is not None and max_train_vectors > 0:
            train = np.array(f["train"][:max_train_vectors], dtype=np.float32)
        else:
            train = np.array(f["train"], dtype=np.float32)

        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
        distances = np.array(f["distances"], dtype=np.float32)

    # GloVe requires L2 normalization
    if is_glove:
        faiss.normalize_L2(train)
        faiss.normalize_L2(test)
        # Note: In glove-100-angular, distance is 1 - cos(theta).
        # For unit vectors: ||u - v||^2 = 2 * (1 - cos(theta)), so ||u - v|| = sqrt(2 * (1 - cos(theta))).
        # The ordering of neighbors is identical.

    dim = train.shape[1]
    return {
        "name": "sift" if not is_glove else "glove",
        "train": train,
        "test": test,
        "neighbors": neighbors,
        "distances": distances,
        "dim": dim,
    }


def get_query_split(
    dataset: Dict[str, np.ndarray],
    split: str = "test",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the 10,000 queries into tuning (first 5,000) or reporting (last 5,000).

    Parameters
    ----------
    dataset : dict
        Dataset dict from load_dataset.
    split : str, default='test'
        'tune' / 'val' for first 5,000 queries.
        'test' for last 5,000 queries.
        'all' for all 10,000 queries.

    Returns
    -------
    Tuple[queries, true_neighbors, true_distances]
    """
    split_lower = split.lower()
    if split_lower in ("tune", "val", "validation"):
        slice_obj = slice(0, 5000)
    elif split_lower in ("test", "report"):
        slice_obj = slice(5000, 10000)
    elif split_lower == "all":
        slice_obj = slice(0, 10000)
    else:
        raise ValueError(f"Unknown split '{split}'. Choose 'tune', 'test', or 'all'.")

    queries = dataset["test"][slice_obj]
    neighbors = dataset["neighbors"][slice_obj]
    distances = dataset["distances"][slice_obj]
    return queries, neighbors, distances

