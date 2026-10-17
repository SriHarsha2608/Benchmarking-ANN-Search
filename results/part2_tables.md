### SIFT-128 (1M vectors)

#### Table 1: QPS at Recall 0.90 and 0.95
| Index | QPS @ Recall 0.90 | QPS @ Recall 0.95 |
| --- | --- | --- |
| HNSW | 12489.9 | 7188.5 |
| IVF-Flat | 3035.8 | 1947.9 |
| IVF-PQ | N/A (max 0.883) | N/A (max 0.883) |
| Exact (IndexFlatL2) | 36.1 | 36.1 |

#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)
| Index | Build Time (s) | Index Size (MB) | Bytes / Vector | Note |
| --- | --- | --- | --- | --- |
| HNSW | 144.82s | 747.8 MB | 784.1 | Achieved recall: 0.9566 (M=32,efC=64) |
| IVF-Flat | 38.10s | 496.9 MB | 521.1 | Achieved recall: 0.9609 (nlist=2048) |
| IVF-PQ | 49.38s | 69.8 MB | 73.2 | Max recall reached: 0.883 (< 0.95) |
| Exact (IndexFlatL2) | 0.48s | 488.3 MB | 512.0 | Exact search (Recall 1.0) |

----------------------------------------------------------------------

### GloVe-100 (1.18M vectors)

#### Table 1: QPS at Recall 0.90 and 0.95
| Index | QPS @ Recall 0.90 | QPS @ Recall 0.95 |
| --- | --- | --- |
| HNSW | 1192.1 | 446.0 |
| IVF-Flat | 874.5 | 387.0 |
| IVF-PQ | N/A (max 0.830) | N/A (max 0.830) |
| Exact (IndexFlatL2) | 38.8 | 38.8 |

#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)
| Index | Build Time (s) | Index Size (MB) | Bytes / Vector | Note |
| --- | --- | --- | --- | --- |
| HNSW | 208.27s | 758.6 MB | 672.2 | Achieved recall: 0.9597 (M=32,efC=64) |
| IVF-Flat | 33.97s | 461.3 MB | 408.7 | Achieved recall: 0.9585 (nlist=2048) |
| IVF-PQ | 43.53s | 66.4 MB | 58.8 | Max recall reached: 0.830 (< 0.95) |
| Exact (IndexFlatL2) | 0.21s | 451.5 MB | 400.0 | Exact search (Recall 1.0) |

----------------------------------------------------------------------
