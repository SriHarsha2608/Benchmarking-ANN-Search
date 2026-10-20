# Part 3: Compression and Exact Re-ranking Results (SIFT-128)

## 1. Index Size and Memory Footprint Comparison

| Sub-vectors ($m$) | Sub-dim ($d'$) | Plain Index Size | Plain Bytes/Vec | Refined Index Size | Refined Bytes/Vec | Raw Size Ratio |
|:-----------------:|:--------------:|:----------------:|:---------------:|:------------------:|:-----------------:|:--------------:|
| 8 | 16 | 17.42 MB | 18.3 B | 505.70 MB | 530.3 B | 28.0x |
| 16 | 8 | 24.03 MB | 25.2 B | 513.33 MB | 538.3 B | 20.3x |
| 32 | 4 | 39.29 MB | 41.2 B | 528.58 MB | 554.3 B | 12.4x |
| 64 | 2 | 69.81 MB | 73.2 B | 559.10 MB | 586.3 B | 7.0x |

*Note: Plain IVF-PQ stores only quantized sub-vector centroids and inverted lists. Exact re-ranking requires retaining the uncompressed 128-d float32 vectors (512 MB, 512 B/vec).*

## 2. Performance Comparison: Plain IVF-PQ vs. Re-ranked IVF-PQ (Top 100 Candidates)

| Setting ($m$) | Plain Max Recall | Plain QPS @ Max | Refined Recall @ nprobe=64 | Refined QPS | Refined Recall @ nprobe=128 | Refined QPS | Reaches Recall 0.95? |
|:-------------:|:----------------:|:---------------:|:---------------------------:|:-----------:|:----------------------------:|:-----------:|:--------------------:|
| m=8 | 0.4138 | 1941.5 | 0.8821 | 4723.3 | 0.8883 | 3100.1 | No |
| m=16 | 0.5958 | 1295.0 | 0.9661 | 3664.4 | 0.9806 | 2221.5 | Yes |
| m=32 | 0.7449 | 819.4 | 0.9765 | 2556.3 | 0.9940 | 1470.1 | Yes |
| m=64 | 0.8854 | 452.6 | 0.9765 | 1543.8 | 0.9941 | 839.1 | Yes |

## 3. QPS and Recall at Standard Targets (Recall = 0.90 and 0.95)

| Index Variant | QPS @ Recall 0.90 | nprobe @ 0.90 | QPS @ Recall 0.95 | nprobe @ 0.95 |
|:--------------|:-----------------:|:-------------:|:-----------------:|:-------------:|
| Plain IVF-PQ (m=8) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=8) | N/A | - | N/A | - |
| Plain IVF-PQ (m=16) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=16) | 5613.7 | nprobe=32 | 3664.4 | nprobe=64 |
| Plain IVF-PQ (m=32) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=32) | 4213.9 | nprobe=32 | 2556.3 | nprobe=64 |
| Plain IVF-PQ (m=64) | N/A | - | N/A (Ceiling) | - |
| Re-ranked IVF-PQ (m=64) | 2707.9 | nprobe=32 | 1543.8 | nprobe=64 |
