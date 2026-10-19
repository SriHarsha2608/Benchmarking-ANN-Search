# Part 4: Failure Mode Experiment Results (Cluster Dispersion)

## 1. Index Build Time and Resource Footprint Across Dispersion ($\sigma$)

| Setting | Dispersion ($\sigma$) | IVF-Flat Build (s) | IVF Size (MB) | HNSW Build (s) | HNSW Size (MB) | Exact Brute QPS |
|:-------:|:--------------------:|:------------------:|:-------------:|:--------------:|:--------------:|:---------------:|
| Control (sigma=0.05, Tight) | 0.05 | 7.09s | 50.6 MB | 23.95s | 76.4 MB | 288.1 QPS |
| sigma=0.15 (Mild) | 0.15 | 7.20s | 50.6 MB | 23.99s | 76.4 MB | 252.7 QPS |
| sigma=0.35 (Diffuse) | 0.35 | 7.02s | 50.6 MB | 25.36s | 76.4 MB | 293.9 QPS |
| sigma=0.70 (Ambient Noise) | 0.70 | 7.02s | 50.6 MB | 29.05s | 76.4 MB | 285.2 QPS |

## 2. Recall and Speed Under Low vs. High Search Budget

| Dispersion ($\sigma$) | IVF Rec (np=4) | IVF QPS | IVF Rec (np=64) | IVF QPS | HNSW Rec (ef=20) | HNSW QPS | HNSW Rec (ef=160) | HNSW QPS |
|:--------------------:|:--------------:|:-------:|:---------------:|:-------:|:----------------:|:--------:|:-----------------:|:--------:|
| sigma=0.05 | 0.4622 | 25185.0 | 0.9999 | 3108.1 | 0.7140 | 17750.8 | 0.9670 | 3080.0 |
| sigma=0.15 | 0.4636 | 28002.2 | 1.0000 | 3316.1 | 0.7168 | 19860.0 | 0.9825 | 3726.7 |
| sigma=0.35 | 0.4597 | 28356.9 | 1.0000 | 3472.1 | 0.6835 | 18873.9 | 0.9609 | 3463.5 |
| sigma=0.70 | 0.3460 | 22863.8 | 0.8766 | 3473.6 | 0.5239 | 16824.8 | 0.8757 | 3228.4 |

## 3. High-Recall Target Performance (Recall = 0.95)

| Dispersion ($\sigma$) | IVF-Flat QPS @ 0.95 | Required nprobe | HNSW QPS @ 0.95 | Required efSearch | HNSW / IVF Throughput Ratio |
|:--------------------:|:-------------------:|:---------------:|:----------------:|:-----------------:|:---------------------------:|
| sigma=0.05 | 3987.2 | nprobe=32 | 3080.0 | efSearch=160 | 0.77x |
| sigma=0.15 | 6493.6 | nprobe=32 | 3726.7 | efSearch=160 | 0.57x |
| sigma=0.35 | 6484.0 | nprobe=32 | 3463.5 | efSearch=160 | 0.53x |
| sigma=0.70 | Failed | >128 | Failed | >320 | N/A |
