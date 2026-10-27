# Part 4: Failure Mode Experiment Results (Cluster Dispersion)

## 1. Index Build Time and Resource Footprint Across Dispersion ($\sigma$)

| Setting | Dispersion ($\sigma$) | IVF-Flat Build (s) | IVF Size (MB) | HNSW Build (s) | HNSW Size (MB) | Exact Brute QPS |
|:-------:|:--------------------:|:------------------:|:-------------:|:--------------:|:--------------:|:---------------:|
| Control (sigma=0.05, Tight) | 0.05 | 4.24s | 50.6 MB | 13.77s | 76.4 MB | 427.2 QPS |
| sigma=0.15 (Mild) | 0.15 | 3.96s | 50.6 MB | 14.20s | 76.4 MB | 426.1 QPS |
| sigma=0.35 (Diffuse) | 0.35 | 3.95s | 50.6 MB | 14.69s | 76.4 MB | 389.0 QPS |
| sigma=0.70 (Ambient Noise) | 0.70 | 3.84s | 50.6 MB | 15.74s | 76.4 MB | 401.0 QPS |

## 2. Recall and Speed Under Low vs. High Search Budget

| Dispersion ($\sigma$) | IVF Rec (np=4) | IVF QPS | IVF Rec (np=64) | IVF QPS | HNSW Rec (ef=20) | HNSW QPS | HNSW Rec (ef=160) | HNSW QPS |
|:--------------------:|:--------------:|:-------:|:---------------:|:-------:|:----------------:|:--------:|:-----------------:|:--------:|
| sigma=0.05 | 0.4622 | 43629.9 | 0.9999 | 5087.1 | 0.7140 | 31591.8 | 0.9670 | 5532.3 |
| sigma=0.15 | 0.4636 | 42488.3 | 1.0000 | 4926.6 | 0.7168 | 27264.3 | 0.9825 | 5262.2 |
| sigma=0.35 | 0.4597 | 48287.8 | 1.0000 | 5224.1 | 0.6835 | 20762.3 | 0.9609 | 4968.9 |
| sigma=0.70 | 0.3460 | 47107.4 | 0.8766 | 5732.9 | 0.5239 | 28331.7 | 0.8757 | 4920.6 |

## 3. High-Recall Target Performance (Recall = 0.95)

| Dispersion ($\sigma$) | IVF-Flat QPS @ 0.95 | Required nprobe | HNSW QPS @ 0.95 | Required efSearch | HNSW / IVF Throughput Ratio |
|:--------------------:|:-------------------:|:---------------:|:----------------:|:-----------------:|:---------------------------:|
| sigma=0.05 | 9526.9 | nprobe=32 | 5532.3 | efSearch=160 | 0.58x |
| sigma=0.15 | 9931.3 | nprobe=32 | 5262.2 | efSearch=160 | 0.53x |
| sigma=0.35 | 8737.2 | nprobe=32 | 4968.9 | efSearch=160 | 0.57x |
| sigma=0.70 | Failed | >128 | Failed | >320 | N/A |
