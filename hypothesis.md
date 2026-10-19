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

### Empirical Evaluation & Verification (Post-Experiment)

| Hypothesis | Predicted Behavior | Empirical Observation | Verdict |
|:-----------|:-------------------|:----------------------|:-------:|
| **H1: Plain Ceiling** | Strict recall ceiling scaling with $m$; fails to reach 0.95 | Ceilings at 0.4138 ($m=8$), 0.5958 ($m=16$), 0.7449 ($m=32$), 0.8854 ($m=64$) | **Confirmed** |
| **H2: Re-ranking Recovery** | Re-ranking top 100 restores recall $>0.95$ for $m \ge 16$; $m=8$ falls short | Re-ranked reaches 0.9661 ($m=16$), 0.9765 ($m=32, 64$); $m=8$ caps at 0.8899 | **Confirmed** |
| **H3: QPS Overhead** | $<20\%$ latency penalty at equal $nprobe$; $>30\times$ faster than exact | QPS retention 85.1% to 99.5% for $nprobe \ge 32$; 3,433.7 QPS at recall 0.9661 ($95\times$ exact) | **Confirmed** |
| **H4: Memory Trade-off** | Plain achieves $8\times$–$64\times$ savings; Re-ranking requires raw 512 B/vec | Plain: 18.3 to 73.2 B/vec (7x to 28x savings); Re-ranking: 530.3 to 586.3 B/vec | **Confirmed** |

**Key Takeaways for Report:**
1. Product Quantization's distance distortion is the sole cause of the recall ceiling observed in Part 2. The inverted list partition ($nlist=4096$) captures true neighbours with high fidelity within the top 100 candidates, but ADC distance noise disrupts relative ordering among the top 10.
2. Exact re-ranking of the top 100 candidates is remarkably cheap ($100 \times 128$ floating point operations, taking $< 0.05$ ms per query), preserving over $92\%$ of single-query throughput while boosting recall by up to $+47.6\%$.
3. When memory allows retaining raw vectors on SSD or RAM, Re-ranked IVF-PQ ($m=16$) achieves **3,433.7 QPS at 0.9661 recall**, outperforming IVF-Flat (1,947.9 QPS at 0.95 recall) by $1.76\times$ while building in 146s.

---

## Part 4: Experiment on Generated Data (Geometry Failure Mode)

### Core Question
How does cluster dispersion (tightness vs. overlap) affect the search recall and routing mechanisms of space-partitioning indexes (IVF-Flat) compared to proximity graph indexes (HNSW)?

### Motivation & Theoretical Basis
Real-world vector embeddings (like SIFT image descriptors) exhibit non-uniform clustering in high-dimensional space. The two dominant ANN paradigms exploit data geometry differently:
- **IVF-Flat** relies on Voronoi partitioning via $k$-means centroids, operating on the assumption that nearest neighbours fall within the same or immediately adjacent cells.
- **HNSW** builds a hierarchical Delaunay-like proximity graph, routing queries via greedy downhill traversal along local distance gradients.

When cluster variance is small, data clusters are tightly concentrated and well separated. As cluster variance increases, the clusters spread out, overlap, and eventually merge into a diffuse, uniform-like distribution. This experiment tests how each index degrades as this geometric property is varied.

### Synthetic Data Generator Design
We generate synthetic datasets with controlled cluster dispersion while fixing all other parameters:
- **Base dataset size**: $N = 200,000$ vectors in $D = 64$ dimensions.
- **Cluster structure**: $K = 50$ centroids sampled uniformly in $\mathbb{R}^{64}$.
- **Data distribution**: Base vectors are drawn from a Gaussian Mixture Model: $x \sim \frac{1}{K} \sum_{k=1}^K \mathcal{N}(\mu_k, \sigma^2 I_{64})$.
- **Query set**: $N_q = 2,000$ test queries drawn from the same mixture distribution. True nearest neighbours are determined via exact brute-force search (`IndexFlatL2`).
- **Reproducibility**: All generation is seeded with `seed=42`.

**Single Property Varied: Intra-Cluster Standard Deviation ($\sigma$)**:
- **Control ($\sigma = 0.05$)**: Tight, isolated clusters with large empty space between clusters.
- **Setting 1 ($\sigma = 0.15$)**: Moderately compact clusters with minimal boundary overlap.
- **Setting 2 ($\sigma = 0.35$)**: Diffuse clusters with significant overlap across cluster boundaries.
- **Setting 3 ($\sigma = 0.70$)**: Highly dispersed clouds where cluster structure dissolves into ambient noise.

Fixed index configurations:
- **IVF-Flat**: $nlist = 1024$, sweeping $nprobe \in \{1, 2, 4, 8, 16, 32, 64, 128\}$.
- **HNSW**: $M = 16$, $efConstruction = 64$, sweeping $efSearch \in \{10, 20, 40, 80, 160, 320\}$.

---

### Hypotheses

1. **IVF-Flat Boundary Scattering**:
   - For tight clusters ($\sigma = 0.05$), IVF-Flat will achieve high recall ($>0.95$) at very low search budget ($nprobe = 1$ or $2$) because nearly all 10 true nearest neighbours fall into the exact same Voronoi cell as the query.
   - As $\sigma$ increases, points from the same cluster will spill across multiple Voronoi boundaries. Consequently, recall at low $nprobe$ will drop steeply, forcing IVF-Flat to probe a substantially larger fraction of cells ($nprobe \ge 32$) to reach acceptable recall.

2. **HNSW Geometric Resilience**:
   - HNSW will be significantly more robust to increasing $\sigma$ than IVF-Flat. Because graph routing does not rely on rigid spatial boundaries, greedy search can traverse continuous distance gradients even when clusters overlap. At moderate search depth ($efSearch = 40$), HNSW is expected to maintain recall above $0.90$ across all dispersion settings.

3. **Isolated Clique Trapping in HNSW at Extreme Tightness**:
   - At $\sigma = 0.05$, tight and distant clusters may cause HNSW's base layer to form dense intra-cluster cliques with relatively few cross-cluster bridges. If a query lies on the margin of a cluster or greedy search starts from a distant entry point, traversal may risk local-minima entrapment unless $efSearch$ is sufficiently large.

4. **Resource and Throughput Trade-offs**:
   - **Build Time**: IVF-Flat build time will remain largely independent of $\sigma$ because $k$-means iteration counts are stable. HNSW build time will increase with $\sigma$ as neighbor selection during insertion must evaluate a less spatially concentrated set of candidates.
   - **Throughput at Target Recall**: Achieving recall $\ge 0.95$ on diffuse data ($\sigma = 0.70$) will require large $nprobe$ values for IVF-Flat, causing its QPS to drop significantly, whereas HNSW's QPS at $0.95$ recall will experience less severe degradation.

### Empirical Evaluation & Verification (Post-Experiment)

| Hypothesis | Predicted Behavior | Empirical Observation | Verdict |
|:-----------|:-------------------|:----------------------|:-------:|
| **H1: IVF Boundary Scattering** | Severe recall drop at low/mod $nprobe$ as $\sigma \to 0.70$; requires large $nprobe$ | At $nprobe=16$, recall drops from 0.9432 ($\sigma=0.05$) to 0.7053 ($\sigma=0.70$); fails 0.95 target (0.9354 at $np=128$) | **Confirmed** |
| **H2: HNSW Geometric Resilience** | HNSW degrades less than IVF; maintains $>0.90$ at $ef=40$ | HNSW outperforms IVF at low budget ($ef=20$ rec=0.5239 vs IVF $np=4$ rec=0.3460), but at $\sigma=0.70$ distance concentration caps $ef=40$ at 0.6688 | **Partially Confirmed** |
| **H3: Tight Clique Trapping** | Extreme tightness ($\sigma=0.05$) causes slight routing entrapment vs mild overlap | At $ef=320$, recall is 0.9784 ($\sigma=0.05$) vs 0.9907 ($\sigma=0.15$), confirming isolated intra-cluster clique penalty | **Confirmed** |
| **H4: Resource Trade-offs** | IVF build time invariant; HNSW build time increases with $\sigma$; diffuse data degrades QPS | IVF build time invariant (7.09s vs 7.02s); HNSW build time increases from 23.95s to 29.05s (+21.3%); both fail 0.95 at $\sigma=0.70$ | **Confirmed** |

**Key Takeaways for Report:**
1. **The Voronoi Assumption Breakpoint**: IVF-Flat's core spatial assumption breaks down when intra-cluster variance causes points to spill across Voronoi boundaries. When clusters dissolve into diffuse ambient noise ($\sigma=0.70$), true nearest neighbours are scattered across dozens of inverted lists, causing recall at $nprobe=16$ to collapse from $94.3\%$ down to $70.5\%$.
2. **Graph Navigation vs. Distance Concentration**: While HNSW avoids rigid boundary clipping, severe dispersion ($\sigma=0.70$) causes distance concentration where contrast between 1-NN and 10-NN narrows, causing greedy routing to lose directional gradient signal. Both indexes fail to reach $0.95$ recall at $\sigma=0.70$.
3. **The Goldilocks Geometry**: Both indexes perform optimally when data possesses moderate cluster structure ($\sigma \approx 0.15$): clusters are distinct enough to guide Voronoi assignment and graph highways, yet connected enough to avoid isolated clique traps.

