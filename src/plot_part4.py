"""
Part 4: Visualization and Summary Table Generator for Failure Mode Experiment.

Generates:
1. results/figures/part4_geometry_failure.png: 2-panel trade-off & search budget curve across sigma
2. results/figures/part4_dispersion_impact.png: Direct impact of sigma on recall and QPS
3. results/part4_tables.md: Detailed markdown table
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def generate_part4_figures_and_tables(
    results_csv: str = "results/results.csv",
    summary_json: str = "results/part4_summary.json",
    figures_dir: str = "results/figures",
    tables_md: str = "results/part4_tables.md",
):
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(results_csv):
        print(f"Error: {results_csv} does not exist.")
        return

    df = pd.read_csv(results_csv)
    synth_df = df[df["dataset"].str.startswith("synthetic_sigma_")].copy()

    def extract_sigma(val):
        try:
            return float(val.replace("synthetic_sigma_", ""))
        except Exception:
            return None

    def extract_param_val(val):
        if not isinstance(val, str) or "=" not in val:
            return None
        return int(val.split("=")[1])

    synth_df["sigma"] = synth_df["dataset"].apply(extract_sigma)
    synth_df["param_val"] = synth_df["search_params"].apply(extract_param_val)

    sigmas = sorted(synth_df["sigma"].unique())
    sigma_colors = {0.05: "#1f77b4", 0.15: "#2ca02c", 0.35: "#ff7f0e", 0.70: "#d62728"}
    sigma_labels = {
        0.05: "Control (sigma=0.05, Tight)",
        0.15: "sigma=0.15 (Mild)",
        0.35: "sigma=0.35 (Diffuse)",
        0.70: "sigma=0.70 (Ambient Noise)",
    }

    # -------------------------------------------------------------
    # Figure 1: 2-Panel Main Figure for Report (part4_geometry_failure.png)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Panel 1: Recall vs QPS Trade-off Curves
    for s in sigmas:
        c = sigma_colors.get(s, "blue")
        s_lbl = sigma_labels.get(s, f"sigma={s}")

        # HNSW curves (solid)
        hnsw_s = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "HNSW")].sort_values(by="qps")
        if not hnsw_s.empty:
            ax1.plot(
                hnsw_s["qps"],
                hnsw_s["recall_at_10"],
                label=f"HNSW [{s_lbl}]",
                color=c,
                linestyle="-",
                marker="o",
                linewidth=1.8,
                markersize=5,
            )

        # IVF-Flat curves (dashed)
        ivf_s = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "IVF-Flat")].sort_values(by="qps")
        if not ivf_s.empty:
            ax1.plot(
                ivf_s["qps"],
                ivf_s["recall_at_10"],
                label=f"IVF-Flat [{s_lbl}]",
                color=c,
                linestyle="--",
                marker="s",
                linewidth=1.5,
                markersize=5,
                alpha=0.85,
            )

    ax1.axhline(y=0.95, color="gray", linestyle=":", alpha=0.7, label="Recall 0.95 Target")
    ax1.set_xscale("log")
    ax1.set_xlabel("Queries Per Second (QPS) [Log scale, higher is better]")
    ax1.set_ylabel("Recall@10 (higher is better)")
    ax1.set_title("(a) Recall vs. Throughput Across Cluster Dispersion")
    ax1.set_ylim([0.0, 1.02])
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # Panel 2: Recall@10 vs. Search Budget (nprobe / efSearch)
    for s in sigmas:
        c = sigma_colors.get(s, "blue")
        s_lbl = sigma_labels.get(s, f"sigma={s}")

        # HNSW efSearch
        hnsw_s = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "HNSW")].sort_values(by="param_val")
        if not hnsw_s.empty:
            ax2.plot(
                hnsw_s["param_val"],
                hnsw_s["recall_at_10"],
                color=c,
                linestyle="-",
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=f"HNSW (efSearch) [{s_lbl}]",
            )

        # IVF-Flat nprobe
        ivf_s = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "IVF-Flat")].sort_values(by="param_val")
        if not ivf_s.empty:
            ax2.plot(
                ivf_s["param_val"],
                ivf_s["recall_at_10"],
                color=c,
                linestyle="--",
                marker="s",
                linewidth=1.5,
                markersize=5,
                alpha=0.85,
                label=f"IVF-Flat (nprobe) [{s_lbl}]",
            )

    ax2.axhline(y=0.95, color="gray", linestyle=":", alpha=0.7, label="Recall 0.95 Target")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Search Parameter Budget (log2 scale: nprobe for IVF / efSearch for HNSW)")
    ax2.set_ylabel("Recall@10")
    ax2.set_title("(b) Search Budget Sensitivity Across Cluster Dispersion")
    ax2.set_ylim([0.0, 1.02])
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Part 4: Failure Mode Analysis under Geometric Cluster Dispersion (N=200K, D=64, K=50)", y=0.98)
    plt.tight_layout()

    fig1_path = os.path.join(figures_dir, "part4_geometry_failure.png")
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Direct Impact of Dispersion on Recall at Fixed Budget
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Low budget comparison: IVF nprobe=4 vs HNSW ef=20
    # Moderate budget comparison: IVF nprobe=16 vs HNSW ef=40
    ivf_low = []
    ivf_mod = []
    hnsw_low = []
    hnsw_mod = []

    for s in sigmas:
        i_l = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "IVF-Flat") & (synth_df["param_val"] == 4)]
        i_m = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "IVF-Flat") & (synth_df["param_val"] == 16)]
        h_l = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "HNSW") & (synth_df["param_val"] == 20)]
        h_m = synth_df[(synth_df["sigma"] == s) & (synth_df["index_type"] == "HNSW") & (synth_df["param_val"] == 40)]

        ivf_low.append(i_l["recall_at_10"].values[0] if not i_l.empty else np.nan)
        ivf_mod.append(i_m["recall_at_10"].values[0] if not i_m.empty else np.nan)
        hnsw_low.append(h_l["recall_at_10"].values[0] if not h_l.empty else np.nan)
        hnsw_mod.append(h_m["recall_at_10"].values[0] if not h_m.empty else np.nan)

    ax.plot(sigmas, hnsw_mod, "o-", color="#1f77b4", linewidth=2.2, label="HNSW (efSearch=40, Moderate Budget)")
    ax.plot(sigmas, ivf_mod, "s--", color="#ff7f0e", linewidth=2.0, label="IVF-Flat (nprobe=16, Moderate Budget)")
    ax.plot(sigmas, hnsw_low, "^-.", color="#2ca02c", linewidth=1.8, label="HNSW (efSearch=20, Low Budget)")
    ax.plot(sigmas, ivf_low, "d:", color="#d62728", linewidth=1.8, label="IVF-Flat (nprobe=4, Low Budget)")

    ax.set_xlabel("Cluster Dispersion sigma (Control=0.05 -> Diffuse=0.70)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Degradation of Recall@10 as Cluster Dispersion Increases")
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9.5)
    plt.tight_layout()

    fig2_path = os.path.join(figures_dir, "part4_dispersion_impact.png")
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Generate Markdown Summary Tables
    # -------------------------------------------------------------
    lines = []
    lines.append("# Part 4: Failure Mode Experiment Results (Cluster Dispersion)\n")
    lines.append(r"## 1. Index Build Time and Resource Footprint Across Dispersion ($\sigma$)" + "\n")
    lines.append(r"| Setting | Dispersion ($\sigma$) | IVF-Flat Build (s) | IVF Size (MB) | HNSW Build (s) | HNSW Size (MB) | Exact Brute QPS |")
    lines.append("|:-------:|:--------------------:|:------------------:|:-------------:|:--------------:|:--------------:|:---------------:|")

    for s in sigmas:
        sub = synth_df[synth_df["sigma"] == s]
        ivf_row = sub[sub["index_type"] == "IVF-Flat"].iloc[0]
        hnsw_row = sub[sub["index_type"] == "HNSW"].iloc[0]
        exact_row = sub[sub["index_type"] == "Exact"].iloc[0]

        lbl = sigma_labels.get(s, f"sigma={s}")
        lines.append(
            f"| {lbl} | {s:.2f} | {ivf_row['build_time_s']:.2f}s | {ivf_row['index_size_mb']:.1f} MB | "
            f"{hnsw_row['build_time_s']:.2f}s | {hnsw_row['index_size_mb']:.1f} MB | {exact_row['qps']:.1f} QPS |"
        )

    lines.append("\n" + r"## 2. Recall and Speed Under Low vs. High Search Budget" + "\n")
    lines.append(r"| Dispersion ($\sigma$) | IVF Rec (np=4) | IVF QPS | IVF Rec (np=64) | IVF QPS | HNSW Rec (ef=20) | HNSW QPS | HNSW Rec (ef=160) | HNSW QPS |")
    lines.append("|:--------------------:|:--------------:|:-------:|:---------------:|:-------:|:----------------:|:--------:|:-----------------:|:--------:|")

    for s in sigmas:
        sub = synth_df[synth_df["sigma"] == s]
        i4 = sub[(sub["index_type"] == "IVF-Flat") & (sub["param_val"] == 4)]
        i64 = sub[(sub["index_type"] == "IVF-Flat") & (sub["param_val"] == 64)]
        h20 = sub[(sub["index_type"] == "HNSW") & (sub["param_val"] == 20)]
        h160 = sub[(sub["index_type"] == "HNSW") & (sub["param_val"] == 160)]

        i4_r = i4["recall_at_10"].values[0] if not i4.empty else np.nan
        i4_q = i4["qps"].values[0] if not i4.empty else np.nan
        i64_r = i64["recall_at_10"].values[0] if not i64.empty else np.nan
        i64_q = i64["qps"].values[0] if not i64.empty else np.nan

        h20_r = h20["recall_at_10"].values[0] if not h20.empty else np.nan
        h20_q = h20["qps"].values[0] if not h20.empty else np.nan
        h160_r = h160["recall_at_10"].values[0] if not h160.empty else np.nan
        h160_q = h160["qps"].values[0] if not h160.empty else np.nan

        lines.append(
            f"| sigma={s:.2f} | {i4_r:.4f} | {i4_q:.1f} | {i64_r:.4f} | {i64_q:.1f} | "
            f"{h20_r:.4f} | {h20_q:.1f} | {h160_r:.4f} | {h160_q:.1f} |"
        )

    lines.append("\n" + r"## 3. High-Recall Target Performance (Recall = 0.95)" + "\n")
    lines.append(r"| Dispersion ($\sigma$) | IVF-Flat QPS @ 0.95 | Required nprobe | HNSW QPS @ 0.95 | Required efSearch | HNSW / IVF Throughput Ratio |")
    lines.append("|:--------------------:|:-------------------:|:---------------:|:----------------:|:-----------------:|:---------------------------:|")

    for s in sigmas:
        sub = synth_df[synth_df["sigma"] == s]
        ivf_sub = sub[sub["index_type"] == "IVF-Flat"].sort_values(by="param_val")
        hnsw_sub = sub[sub["index_type"] == "HNSW"].sort_values(by="param_val")

        i95 = ivf_sub[ivf_sub["recall_at_10"] >= 0.95]
        h95 = hnsw_sub[hnsw_sub["recall_at_10"] >= 0.95]

        if not i95.empty:
            i_row = i95.iloc[0]
            i_qps_str = f"{i_row['qps']:.1f}"
            i_np_str = f"nprobe={int(i_row['param_val'])}"
            i_qps_val = i_row["qps"]
        else:
            i_qps_str = "Failed"
            i_np_str = ">128"
            i_qps_val = np.nan

        if not h95.empty:
            h_row = h95.iloc[0]
            h_qps_str = f"{h_row['qps']:.1f}"
            h_ef_str = f"efSearch={int(h_row['param_val'])}"
            h_qps_val = h_row["qps"]
        else:
            h_qps_str = "Failed"
            h_ef_str = ">320"
            h_qps_val = np.nan

        if not np.isnan(i_qps_val) and not np.isnan(h_qps_val) and i_qps_val > 0:
            ratio_str = f"{h_qps_val / i_qps_val:.2f}x"
        else:
            ratio_str = "N/A"

        lines.append(f"| sigma={s:.2f} | {i_qps_str} | {i_np_str} | {h_qps_str} | {h_ef_str} | {ratio_str} |")

    table_content = "\n".join(lines) + "\n"
    with open(tables_md, "w") as f:
        f.write(table_content)
    print(f"Saved: {tables_md}")


if __name__ == "__main__":
    generate_part4_figures_and_tables()

