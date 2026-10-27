# Part 3: Compression and Exact Re-ranking Results (SIFT-128)

## 1. Index Size and Memory Footprint Comparison

| Sub-vectors ($m$) | Sub-dim ($d'$) | Plain Index Size | Plain Bytes/Vec | Refined Index Size | Refined Bytes/Vec | Raw Size Ratio |
|:-----------------:|:--------------:|:----------------:|:---------------:|:------------------:|:-----------------:|:--------------:|
| 8 | 16 | 17.42 MB | 18.3 B | 505.70 MB | 530.3 B | 28.0x |
| 16 | 8 | 25.04 MB | 26.3 B | 513.33 MB | 538.3 B | 19.5x |
| 32 | 4 | 40.30 MB | 42.3 B | 528.58 MB | 554.3 B | 12.1x |
| 64 | 2 | 70.82 MB | 74.3 B | 559.10 MB | 586.3 B | 6.9x |

*Note: Plain IVF-PQ stores only quantized sub-vector centroids and inverted lists. Exact re-ranking requires retaining the uncompressed 128-d float32 vectors (512 MB, 512 B/vec).*

## 2. Performance Comparison: Plain IVF-PQ vs. Re-ranked IVF-PQ (Top 100 Candidates)

| Setting ($m$) | Plain Max Recall | Plain QPS @ Max | Refined Recall @ nprobe=64 | Refined QPS | Refined Recall @ nprobe=128 | Refined QPS | Reaches Recall 0.95? |
|:-------------:|:----------------:|:---------------:|:---------------------------:|:-----------:|:----------------------------:|:-----------:|:--------------------:|
| m=8 | 0.4174 | 2088.1 | 0.8818 | 5204.5 | 0.8879 | 3327.1 | No |
| m=16 | 0.5958 | 1424.0 | 0.9661 | 3973.4 | 0.9806 | 2155.5 | Yes |
| m=32 | 0.7449 | 852.1 | 0.9765 | 2744.8 | 0.9940 | 1587.8 | Yes |
| m=64 | 0.8854 | 498.1 | 0.9765 | 1694.3 | 0.9941 | 845.3 | Yes |

## 3. QPS and Recall at Standard Targets (Recall = 0.90 and 0.95)

nprobe chosen on the tuning split (first 5,000 queries); QPS and recall R reported on the reporting split.

| Index Variant | QPS @ Recall 0.90 | nprobe @ 0.90 | QPS @ Recall 0.95 | nprobe @ 0.95 |
|:--------------|:-----------------:|:-------------:|:-----------------:|:-------------:|
| Plain IVF-PQ (m=8) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=8) | N/A | - | N/A | - |
| Plain IVF-PQ (m=16) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=16) | 6175.2 (R=0.925) | nprobe=32 | 3973.4 (R=0.966) | nprobe=64 |
| Plain IVF-PQ (m=32) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=32) | 4534.3 (R=0.932) | nprobe=32 | 2744.8 (R=0.976) | nprobe=64 |
| Plain IVF-PQ (m=64) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=64) | 3016.4 (R=0.932) | nprobe=32 | 1694.3 (R=0.977) | nprobe=64 |
