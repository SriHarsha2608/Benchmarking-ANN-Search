"""
Part 3: Visualization and Summary Table Generator for Compression Experiment.
Generates:
1. results/figures/part3_compression_tradeoff.png: Recall@10 vs. QPS for Plain vs. Re-ranked IVF-PQ
2. results/figures/part3_compression_recall_ceiling.png: Recall@10 vs. nprobe demonstrating recall ceiling and restoration
3. results/part3_tables.md: Detailed markdown table comparing memory, recall, and throughput
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling for scientific publication
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
})


def generate_part3_figures_and_tables(
    results_csv: str = "results/results.csv",
    summary_json: str = "results/part3_summary.json",
    figures_dir: str = "results/figures",
    tables_md: str = "results/part3_tables.md",
):
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} does not exist.")
        return

    df = pd.read_csv(results_csv)
    sift_df = df[df["dataset"] == "sift"].copy()

    # Extract m from build_params for IVF-PQ
    def extract_m(val):
        if not isinstance(val, str) or "m=" not in val:
            return None
        for part in val.split(","):
            if part.startswith("m="):
                return int(part.split("=")[1])
        return None

    def extract_nprobe(val):
        if not isinstance(val, str) or "nprobe=" not in val:
            return None
        return int(val.split("=")[1])

    sift_df["m"] = sift_df["build_params"].apply(extract_m)
    sift_df["nprobe"] = sift_df["search_params"].apply(extract_nprobe)

    # Only Part 3 rows (tagged "rerank="); Part 2 IVF-PQ rows use a different nlist
    pq_df = sift_df[
        sift_df["index_type"].isin(["IVF-PQ", "IVF-PQ-Refine"])
        & sift_df["build_params"].str.contains("rerank=")
    ].copy()

    # Colors for m values
    colors = {8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c", 64: "#9467bd"}
    markers = {8: "o", 16: "s", 32: "^", 64: "D"}

    # -------------------------------------------------------------
    # Figure 1: Recall vs. QPS Trade-off Curve (Plain vs. Re-ranked)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Reference exact baseline
    exact_sift = sift_df[sift_df["index_type"] == "Exact"]
    if not exact_sift.empty:
        exact_qps = exact_sift["qps"].values[0]
        ax.axvline(
            x=exact_qps,
            color="black",
            linestyle=":",
            alpha=0.7,
            label=f"Exact (Brute Force: {exact_qps:.1f} QPS)",
        )

    # Reference IVF-Flat if available
    ivf_flat = sift_df[sift_df["index_type"] == "IVF-Flat"]
    if not ivf_flat.empty:
        flat_sorted = ivf_flat.sort_values(by="recall_at_10")
        ax.plot(
            flat_sorted["qps"],
            flat_sorted["recall_at_10"],
            color="gray",
            linestyle="-.",
            alpha=0.6,
            linewidth=1.8,
            label="IVF-Flat Reference (Uncompressed)",
        )

    m_values = sorted(pq_df["m"].unique())

    for m in m_values:
        m_int = int(m)
        c = colors.get(m_int, "blue")
        m_marker = markers.get(m_int, "o")

        # Plain IVF-PQ
        plain = pq_df[(pq_df["index_type"] == "IVF-PQ") & (pq_df["m"] == m_int)].sort_values(by="nprobe")
        if not plain.empty:
            ax.plot(
                plain["qps"],
                plain["recall_at_10"],
                label=f"Plain IVF-PQ (m={m_int}, {m_int} B/vec)",
                color=c,
                linestyle="--",
                marker=m_marker,
                markersize=6,
                alpha=0.85,
            )

        # Refined IVF-PQ (Re-rank top 100)
        refine = pq_df[(pq_df["index_type"] == "IVF-PQ-Refine") & (pq_df["m"] == m_int)].sort_values(by="nprobe")
        if not refine.empty:
            ax.plot(
                refine["qps"],
                refine["recall_at_10"],
                label=f"Re-ranked IVF-PQ (m={m_int} + top-100 L2)",
                color=c,
                linestyle="-",
                marker=m_marker,
                markersize=7,
                linewidth=2.0,
            )

    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Target Recall = 0.95")
    ax.set_xscale("log")
    ax.set_xlabel("Queries Per Second (QPS) [Log Scale, higher is better]")
    ax.set_ylabel("Recall@10 (higher is better)")
    ax.set_title("Part 3: Plain IVF-PQ vs. Exact Re-ranking (Top 100 Candidates) on SIFT-128")
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=9)
    plt.tight_layout()

    fig1_path = os.path.join(figures_dir, "part3_compression_tradeoff.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Recall@10 vs. Search Budget (nprobe) - Recall Ceiling
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    for m in m_values:
        m_int = int(m)
        c = colors.get(m_int, "blue")
        m_marker = markers.get(m_int, "o")

        plain = pq_df[(pq_df["index_type"] == "IVF-PQ") & (pq_df["m"] == m_int)].sort_values(by="nprobe")
        refine = pq_df[(pq_df["index_type"] == "IVF-PQ-Refine") & (pq_df["m"] == m_int)].sort_values(by="nprobe")

        if not plain.empty:
            ax.plot(
                plain["nprobe"],
                plain["recall_at_10"],
                color=c,
                linestyle="--",
                marker=m_marker,
                markersize=6,
                label=f"Plain m={m_int} (Ceiling at ~{plain['recall_at_10'].max():.3f})",
            )
        if not refine.empty:
            ax.plot(
                refine["nprobe"],
                refine["recall_at_10"],
                color=c,
                linestyle="-",
                marker=m_marker,
                markersize=7,
                linewidth=2.0,
                label=f"Re-ranked m={m_int} (Max ~{refine['recall_at_10'].max():.3f})",
            )

    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.6, label="Recall 0.95 Target")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Search Budget: nprobe (log2 scale)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Quantization Recall Ceiling vs. Exact Re-ranking Across nprobe on SIFT-128")
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)
    plt.tight_layout()

    fig2_path = os.path.join(figures_dir, "part3_compression_recall_ceiling.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Generate Markdown Summary Tables
    # -------------------------------------------------------------
    lines = []
    lines.append("# Part 3: Compression and Exact Re-ranking Results (SIFT-128)\n")
    lines.append("## 1. Index Size and Memory Footprint Comparison\n")
    lines.append("| Sub-vectors ($m$) | Sub-dim ($d'$) | Plain Index Size | Plain Bytes/Vec | Refined Index Size | Refined Bytes/Vec | Raw Size Ratio |")
    lines.append("|:-----------------:|:--------------:|:----------------:|:---------------:|:------------------:|:-----------------:|:--------------:|")

    for m in m_values:
        m_int = int(m)
        p_row = pq_df[(pq_df["index_type"] == "IVF-PQ") & (pq_df["m"] == m_int)].iloc[0]
        r_row = pq_df[(pq_df["index_type"] == "IVF-PQ-Refine") & (pq_df["m"] == m_int)].iloc[0]

        d_prime = 128 // m_int
        p_mb = p_row["index_size_mb"]
        p_bpv = p_row["bytes_per_vec"]
        r_mb = r_row["index_size_mb"]
        r_bpv = r_row["bytes_per_vec"]
        ratio = f"{512.0 / p_bpv:.1f}x"

        lines.append(f"| {m_int} | {d_prime} | {p_mb:.2f} MB | {p_bpv:.1f} B | {r_mb:.2f} MB | {r_bpv:.1f} B | {ratio} |")

    lines.append("\n*Note: Plain IVF-PQ stores only quantized sub-vector centroids and inverted lists. Exact re-ranking requires retaining the uncompressed 128-d float32 vectors (512 MB, 512 B/vec).*\n")

    lines.append("## 2. Performance Comparison: Plain IVF-PQ vs. Re-ranked IVF-PQ (Top 100 Candidates)\n")
    lines.append("| Setting ($m$) | Plain Max Recall | Plain QPS @ Max | Refined Recall @ nprobe=64 | Refined QPS | Refined Recall @ nprobe=128 | Refined QPS | Reaches Recall 0.95? |")
    lines.append("|:-------------:|:----------------:|:---------------:|:---------------------------:|:-----------:|:----------------------------:|:-----------:|:--------------------:|")

    for m in m_values:
        m_int = int(m)
        p_sub = pq_df[(pq_df["index_type"] == "IVF-PQ") & (pq_df["m"] == m_int)]
        r_sub = pq_df[(pq_df["index_type"] == "IVF-PQ-Refine") & (pq_df["m"] == m_int)]

        p_max_idx = p_sub["recall_at_10"].idxmax()
        p_max_rec = p_sub.loc[p_max_idx, "recall_at_10"]
        p_max_qps = p_sub.loc[p_max_idx, "qps"]

        r_64 = r_sub[r_sub["nprobe"] == 64]
        r_64_rec = r_64["recall_at_10"].values[0] if not r_64.empty else np.nan
        r_64_qps = r_64["qps"].values[0] if not r_64.empty else np.nan

        r_128 = r_sub[r_sub["nprobe"] == 128]
        r_128_rec = r_128["recall_at_10"].values[0] if not r_128.empty else np.nan
        r_128_qps = r_128["qps"].values[0] if not r_128.empty else np.nan

        reaches_95 = "Yes" if (r_sub["recall_at_10"] >= 0.95).any() else "No"

        lines.append(
            f"| m={m_int} | {p_max_rec:.4f} | {p_max_qps:.1f} | "
            f"{r_64_rec:.4f} | {r_64_qps:.1f} | "
            f"{r_128_rec:.4f} | {r_128_qps:.1f} | {reaches_95} |"
        )

    lines.append("\n## 3. QPS and Recall at Standard Targets (Recall = 0.90 and 0.95)\n")
    lines.append("nprobe chosen on the tuning split (first 5,000 queries); QPS and recall R reported on the reporting split.\n")
    lines.append("| Index Variant | QPS @ Recall 0.90 | nprobe @ 0.90 | QPS @ Recall 0.95 | nprobe @ 0.95 |")
    lines.append("|:--------------|:-----------------:|:-------------:|:-----------------:|:-------------:|")

    for m in m_values:
        m_int = int(m)
        p_sub = pq_df[(pq_df["index_type"] == "IVF-PQ") & (pq_df["m"] == m_int)].sort_values(by="nprobe")
        r_sub = pq_df[(pq_df["index_type"] == "IVF-PQ-Refine") & (pq_df["m"] == m_int)].sort_values(by="nprobe")

        # Check 0.90 and 0.95 for plain
        p_90 = p_sub[p_sub["recall_at_10_tune"] >= 0.90]
        p_90_str = f"{p_90.iloc[0]['qps']:.1f} (np={int(p_90.iloc[0]['nprobe'])})" if not p_90.empty else "N/A"
        p_95 = p_sub[p_sub["recall_at_10_tune"] >= 0.95]
        p_95_str = f"{p_95.iloc[0]['qps']:.1f} (np={int(p_95.iloc[0]['nprobe'])})" if not p_95.empty else "N/A (Ceiling)"

        lines.append(f"| Plain IVF-PQ (m={m_int}) | {p_90_str} | - | {p_95_str} | - |")

        # Check 0.90 and 0.95 for refine
        r_90 = r_sub[r_sub["recall_at_10_tune"] >= 0.90]
        r_90_str = f"{r_90.iloc[0]['qps']:.1f} (R={r_90.iloc[0]['recall_at_10']:.3f})" if not r_90.empty else "N/A"
        r_90_np = f"nprobe={int(r_90.iloc[0]['nprobe'])}" if not r_90.empty else "-"

        r_95 = r_sub[r_sub["recall_at_10_tune"] >= 0.95]
        r_95_str = f"{r_95.iloc[0]['qps']:.1f} (R={r_95.iloc[0]['recall_at_10']:.3f})" if not r_95.empty else "N/A"
        r_95_np = f"nprobe={int(r_95.iloc[0]['nprobe'])}" if not r_95.empty else "-"

        lines.append(f"| Re-ranked IVF-PQ (m={m_int}) | {r_90_str} | {r_90_np} | {r_95_str} | {r_95_np} |")

    table_content = "\n".join(lines) + "\n"
    with open(tables_md, "w") as f:
        f.write(table_content)
    print(f"Saved: {tables_md}")


if __name__ == "__main__":
    generate_part3_figures_and_tables()

