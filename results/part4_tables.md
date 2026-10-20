# Part 4: Failure Mode Experiment Results (Cluster Dispersion)

## 1. Index Build Time and Resource Footprint Across Dispersion ($\sigma$)

| Setting | Dispersion ($\sigma$) | IVF-Flat Build (s) | IVF Size (MB) | HNSW Build (s) | HNSW Size (MB) | Exact Brute QPS |
|:-------:|:--------------------:|:------------------:|:-------------:|:--------------:|:--------------:|:---------------:|
| Control (sigma=0.05, Tight) | 0.05 | 4.19s | 50.6 MB | 14.61s | 76.4 MB | 413.5 QPS |
| sigma=0.15 (Mild) | 0.15 | 4.17s | 50.6 MB | 14.79s | 76.4 MB | 410.9 QPS |
| sigma=0.35 (Diffuse) | 0.35 | 4.14s | 50.6 MB | 15.21s | 76.4 MB | 412.3 QPS |
| sigma=0.70 (Ambient Noise) | 0.70 | 4.14s | 50.6 MB | 17.41s | 76.4 MB | 412.2 QPS |
\n## 2. Recall and Speed Under Low vs. High Search Budget

| Dispersion ($\sigma$) | IVF Rec (np=4) | IVF QPS | IVF Rec (np=64) | IVF QPS | HNSW Rec (ef=20) | HNSW QPS | HNSW Rec (ef=160) | HNSW QPS |
|:--------------------:|:--------------:|:-------:|:---------------:|:-------:|:----------------:|:--------:|:-----------------:|:--------:|
| sigma=0.05 | 0.4622 | 43733.8 | 0.9999 | 4490.0 | 0.7140 | 28250.7 | 0.9670 | 5275.5 |
| sigma=0.15 | 0.4636 | 45287.0 | 1.0000 | 4993.4 | 0.7168 | 28101.0 | 0.9825 | 5518.3 |
| sigma=0.35 | 0.4597 | 44132.3 | 1.0000 | 5008.8 | 0.6835 | 27713.7 | 0.9609 | 5113.7 |
| sigma=0.70 | 0.3460 | 43084.0 | 0.8766 | 5366.6 | 0.5239 | 26495.1 | 0.8757 | 4716.6 |
\n## 3. High-Recall Target Performance (Recall = 0.95)

| Dispersion ($\sigma$) | IVF-Flat QPS @ 0.95 | Required nprobe | HNSW QPS @ 0.95 | Required efSearch | HNSW / IVF Throughput Ratio |
|:--------------------:|:-------------------:|:---------------:|:----------------:|:-----------------:|:---------------------------:|
| sigma=0.05 | 9453.2 | nprobe=32 | 5275.5 | efSearch=160 | 0.56x |
| sigma=0.15 | 9421.6 | nprobe=32 | 5518.3 | efSearch=160 | 0.59x |
| sigma=0.35 | 9401.6 | nprobe=32 | 5113.7 | efSearch=160 | 0.54x |
| sigma=0.70 | Failed | >128 | Failed | >320 | N/A |
