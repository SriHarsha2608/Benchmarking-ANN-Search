# Hypotheses: Benchmarking Approximate Nearest Neighbour Search

This document records the formal hypotheses for Part 3 and Part 4 before conducting each respective experiment, as required by the assignment specification.

---

## Part 3: Compression and Exact Re-ranking (SIFT-128)

### Selected Experiment
**Compression:** On SIFT-128 (1,000,000 base vectors, $D = 128$), vary the number of Product Quantization (PQ) sub-vectors $m \in \{8, 16, 32, 64\}$ (corresponding to sub-vector dimensions $d' \in \{16, 8, 4, 2\}$, with 8 bits per sub-vector code) under an inverted file index ($nlist = 4096$). We compare **Plain IVF-PQ** against **IVF-PQ with Exact Re-ranking** of the top $k_{refine} = 100$ candidate vectors using full-precision $L_2$ distance across varying search budgets ($nprobe \in \{1, 2, 4, 8, 16, 32, 64, 128\}$).

### Background & Motivation
In Part 2, plain IVF-PQ ($m=16$) exhibited an asymptotic recall ceiling ($\approx 0.883$ on SIFT-128), failing to reach the target 0.95 recall regardless of the `nprobe` parameter. 
Product Quantization maps continuous sub-vectors to discrete centroid codes ($k^* = 256$), introducing quantization distortion $\epsilon = \|x - q(x)\|$. During asymmetric distance computation (ADC), this distortion causes ranking errors: true nearest neighbours that are successfully retrieved from inverted lists are misranked due to quantized distance noise and drop out of the top 10.

### Formal Hypotheses

1. **Recall Ceiling & Quantization Resolution**:
   - Plain IVF-PQ will suffer from an intrinsic recall ceiling that scales monotonically with $m$:
     - For coarse quantization ($m=8$), the maximum achievable recall will plateau well below $0.75$.
     - For $m=16$ and $m=32$, the ceiling will plateau around $0.88$ and $0.93$ respectively.
     - Only at fine quantization ($m=64$) might plain IVF-PQ approach or marginally reach $0.95$ recall at large $nprobe$.

2. **Recall Restoration via Exact Re-ranking**:
   - Exact re-ranking of the top $k_{refine} = 100$ candidates using original uncompressed vectors will eliminate rank distortion among candidate vectors.
   - For $m \ge 16$, re-ranking will successfully breach the recall ceiling and achieve Recall@10 $> 0.95$ at sufficiently high `nprobe` ($nprobe \ge 32$).
   - For $m = 8$, because severe quantization distortion displaces true neighbours beyond even the top 100 retrieved candidates, re-ranking the top 100 will improve recall significantly but may still struggle to reach $0.95$.

3. **Throughput (QPS) Overhead**:
   - Computing exact $L_2$ distances for 100 candidates in 128 dimensions requires only $100 \times 128 \approx 1.28 \times 10^4$ floating-point operations per query, which is minor compared to scanning thousands of inverted list entries.
   - Therefore, the throughput penalty of re-ranking will be less than $20\%$ compared to plain IVF-PQ at equal `nprobe`. Re-ranked IVF-PQ will remain significantly faster than exact brute-force search ($>30\times$ speedup).

4. **Memory Footprint Trade-off**:
   - Plain IVF-PQ achieves massive memory compression, storing only $m$ bytes per vector ($8$ to $64$ bytes/vector, an $8\times$ to $64\times$ reduction from the raw 512 bytes/vector).
   - Exact re-ranking necessitates storing or accessing the full-precision 512 bytes/vector base dataset in addition to the quantized index. Thus, re-ranking restores recall at the fundamental cost of sacrificing pure in-memory compression.

---

## Part 4: Experiment on Your Own Data (Failure Mode)

*(To be formulated and committed prior to running Part 4)*
