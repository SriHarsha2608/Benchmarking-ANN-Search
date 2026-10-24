### SIFT-128 (1M vectors)

#### Table 1: QPS at Recall 0.90 and 0.95
| Index | QPS @ Recall 0.90 | QPS @ Recall 0.95 |
| --- | --- | --- |
| HNSW | 13528.9 | 8278.4 |
| IVF-Flat | 3616.5 | 2030.0 |
| IVF-PQ | N/A (max 0.885) | N/A (max 0.885) |
| Exact (IndexFlatL2) | 42.2 | 42.2 |

#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)
| Index | Build Time (s) | Index Size (MB) | Bytes / Vector | Note |
| --- | --- | --- | --- | --- |
| HNSW | 130.63s | 747.8 MB | 784.1 | Achieved recall: 0.9566 (M=32,efC=64) |
| IVF-Flat | 37.60s | 496.9 MB | 521.1 | Achieved recall: 0.9609 (nlist=2048) |
| IVF-PQ | 143.15s | 70.8 MB | 74.3 | Max recall reached: 0.885 (< 0.95) |
| Exact (IndexFlatL2) | 0.22s | 488.3 MB | 512.0 | Exact search (Recall 1.0) |

----------------------------------------------------------------------

### GloVe-100 (1.18M vectors)

#### Table 1: QPS at Recall 0.90 and 0.95
| Index | QPS @ Recall 0.90 | QPS @ Recall 0.95 |
| --- | --- | --- |
| HNSW | 1419.9 | 446.0 |
| IVF-Flat | 940.2 | 359.2 |
| IVF-PQ | N/A (max 0.829) | N/A (max 0.829) |
| Exact (IndexFlatL2) | 45.7 | 45.7 |

#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)
| Index | Build Time (s) | Index Size (MB) | Bytes / Vector | Note |
| --- | --- | --- | --- | --- |
| HNSW | 208.27s | 758.6 MB | 672.2 | Achieved recall: 0.9597 (M=32,efC=64) |
| IVF-Flat | 9.90s | 460.9 MB | 408.4 | Achieved recall: 0.9502 (nlist=1024) |
| IVF-PQ | 42.31s | 66.4 MB | 58.8 | Max recall reached: 0.829 (< 0.95) |
| Exact (IndexFlatL2) | 0.20s | 451.5 MB | 400.0 | Exact search (Recall 1.0) |

----------------------------------------------------------------------
