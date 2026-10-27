### SIFT-128 (1M vectors)

#### Table 1: QPS at Recall 0.90 and 0.95
Configuration chosen on the tuning split (first 5,000 queries); QPS and recall R reported on the reporting split (last 5,000).

| Index | QPS @ Recall 0.90 | QPS @ Recall 0.95 |
| --- | --- | --- |
| HNSW | 14035.5 (R=0.913) | 8638.1 (R=0.957) |
| IVF-Flat | 2256.7 (R=0.937) | 2175.7 (R=0.961) |
| IVF-PQ | N/A (max 0.883) | N/A (max 0.883) |
| Exact (IndexFlatL2) | 45.0 | 45.0 |

#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)
| Index | Build Time (s) | Index Size (MB) | Bytes / Vector | Note |
| --- | --- | --- | --- | --- |
| HNSW | 114.60s | 747.8 MB | 784.1 | Achieved recall: 0.9566 (M=32,efC=64, efSearch=60) |
| IVF-Flat | 34.14s | 496.9 MB | 521.1 | Achieved recall: 0.9609 (nlist=2048, nprobe=32) |
| IVF-PQ | 44.56s | 69.8 MB | 73.2 | Max recall reached: 0.883 (< 0.95) |
| Exact (IndexFlatL2) | 0.20s | 488.3 MB | 512.0 | Exact search (Recall 1.0) |

----------------------------------------------------------------------

### GloVe-100 (1.18M vectors)

#### Table 1: QPS at Recall 0.90 and 0.95
Configuration chosen on the tuning split (first 5,000 queries); QPS and recall R reported on the reporting split (last 5,000).

| Index | QPS @ Recall 0.90 | QPS @ Recall 0.95 |
| --- | --- | --- |
| HNSW | 1328.5 (R=0.911) | 410.3 (R=0.960) |
| IVF-Flat | 720.3 (R=0.908) | 357.7 (R=0.958) |
| IVF-PQ | N/A (max 0.829) | N/A (max 0.829) |
| Exact (IndexFlatL2) | 49.1 | 49.1 |

#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)
| Index | Build Time (s) | Index Size (MB) | Bytes / Vector | Note |
| --- | --- | --- | --- | --- |
| HNSW | 170.51s | 758.6 MB | 672.2 | Achieved recall: 0.9596 (M=32,efC=64, efSearch=1200) |
| IVF-Flat | 29.09s | 461.3 MB | 408.7 | Achieved recall: 0.9584 (nlist=2048, nprobe=256) |
| IVF-PQ | 38.27s | 66.4 MB | 58.8 | Max recall reached: 0.829 (< 0.95) |
| Exact (IndexFlatL2) | 0.18s | 451.5 MB | 400.0 | Exact search (Recall 1.0) |

----------------------------------------------------------------------
