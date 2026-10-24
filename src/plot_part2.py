"""
Generate Part 2 Plots and Summary Tables from results/results.csv.

Outputs:
- results/figures/part2_sift_tradeoff.png
- results/figures/part2_glove_tradeoff.png
- results/part2_tables.md (Markdown tables of QPS at 0.90/0.95 and resources at 0.95)
- results/part2_summary.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Avoid cache directory issues
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import matplotlib.pyplot as plt

# Ensure matplotlib runs in headless mode
plt.switch_backend("Agg")


def records_to_markdown(records: list) -> str:
    """Format list of dicts as a Markdown table without external tabulate dependency."""
    if not records:
        return ""
    headers = list(records[0].keys())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in records:
        row = [str(r.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def get_pareto_frontier(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Pareto optimal frontier for (QPS, Recall@10).
    A point is Pareto optimal if no other point has both higher QPS and higher Recall.
    Sorted by recall ascending.
    """
    sorted_df = df_subset.sort_values(by=["recall_at_10", "qps"], ascending=[True, False])
    frontier = []
    max_qps_so_far = -1.0

    # Walk from highest recall down to lowest to filter dominated points
    reversed_records = sorted_df.to_dict("records")[::-1]
    for r in reversed_records:
        if r["qps"] > max_qps_so_far:
            frontier.append(r)
            max_qps_so_far = r["qps"]

    frontier_df = pd.DataFrame(frontier[::-1])
    return frontier_df


def interpolate_qps_at_recall(frontier_df: pd.DataFrame, target_recall: float) -> str:
    """Find or linearly interpolate QPS at a target recall value."""
    if frontier_df.empty:
        return "N/A"

    max_rec = frontier_df["recall_at_10"].max()
    if max_rec < target_recall:
        return f"N/A (max {max_rec:.3f})"

    # If an exact match exists or points straddle target_recall
    under = frontier_df[frontier_df["recall_at_10"] <= target_recall]
    over = frontier_df[frontier_df["recall_at_10"] >= target_recall]

    if under.empty:
        # Lowest recall point is already >= target
        best_point = over.iloc[0]
        return f"{best_point['qps']:.1f}"

    p_low = under.iloc[-1]
    p_high = over.iloc[0]

    if p_low["recall_at_10"] == p_high["recall_at_10"]:
        return f"{p_high['qps']:.1f}"

    # Linear interpolation on log(QPS)
    r1, r2 = p_low["recall_at_10"], p_high["recall_at_10"]
    log_q1, log_q2 = np.log10(p_low["qps"]), np.log10(p_high["qps"])
    interp_log_q = log_q1 + (target_recall - r1) * (log_q2 - log_q1) / (r2 - r1)
    interp_qps = 10 ** interp_log_q
    return f"{interp_qps:.1f}"


def get_resource_row_at_recall(df_index: pd.DataFrame, target_recall: float = 0.95) -> dict:
    """Get build time, index size, and bytes/vec for configuration reaching target recall."""
    qualifying = df_index[df_index["recall_at_10"] >= target_recall]
    if qualifying.empty:
        # Fall back to best recall point
        best_pt = df_index.sort_values(by="recall_at_10", ascending=False).iloc[0]
        return {
            "build_time_s": f"{best_pt['build_time_s']:.2f}s",
            "index_mb": f"{best_pt['index_size_mb']:.1f} MB",
            "bytes_per_vec": f"{best_pt['bytes_per_vec']:.1f}",
            "note": f"Max recall reached: {best_pt['recall_at_10']:.3f} (< 0.95)",
        }
    else:
        # Pick configuration with highest QPS at >= target_recall
        best_pt = qualifying.sort_values(by="qps", ascending=False).iloc[0]
        return {
            "build_time_s": f"{best_pt['build_time_s']:.2f}s",
            "index_mb": f"{best_pt['index_size_mb']:.1f} MB",
            "bytes_per_vec": f"{best_pt['bytes_per_vec']:.1f}",
            "note": f"Achieved recall: {best_pt['recall_at_10']:.4f} ({best_pt['build_params']})",
        }


def generate_plots_and_tables(csv_path: str = "results/results.csv", out_dir: str = "results"):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Part 2 benchmark is exclusively on standard datasets (SIFT and GloVe)
    df = df[df["dataset"].isin(["sift", "glove"])].copy()

    # Styles
    colors = {
        "HNSW": "#1f77b4",     # Blue
        "IVF-Flat": "#2ca02c", # Green
        "IVF-PQ": "#ff7f0e",   # Orange
        "Exact": "#000000",    # Black
    }
    markers = {
        "HNSW": "o",
        "IVF-Flat": "s",
        "IVF-PQ": "^",
        "Exact": "*",
    }

    summary_tables = {}
    markdown_report = []

    # Sort so SIFT appears first, then GloVe
    part2_datasets = [d for d in ["sift", "glove"] if d in df["dataset"].unique()]
    for dataset in part2_datasets:
        df_ds = df[df["dataset"] == dataset]
        plt.figure(figsize=(9, 6), dpi=300)

        dataset_display = "SIFT-128 (1M vectors)" if dataset == "sift" else "GloVe-100 (1.18M vectors)"

        # 1. Plot Exact Baseline
        df_exact = df_ds[df_ds["index_type"] == "Exact"]
        if not df_exact.empty:
            exact_pt = df_exact.iloc[0]
            plt.scatter(
                [exact_pt["recall_at_10"]],
                [exact_pt["qps"]],
                color=colors["Exact"],
                marker=markers["Exact"],
                s=220,
                zorder=6,
                label=f"Exact (IndexFlatL2: {exact_pt['qps']:.1f} QPS)",
            )

        # 2. Plot Pareto frontiers for HNSW, IVF-Flat, IVF-PQ
        for itype in ["HNSW", "IVF-Flat", "IVF-PQ"]:
            df_type = df_ds[df_ds["index_type"] == itype]
            if df_type.empty:
                continue

            # Plot all measured raw points softly
            plt.scatter(
                df_type["recall_at_10"],
                df_type["qps"],
                color=colors[itype],
                alpha=0.25,
                s=35,
                marker=markers[itype],
            )

            # Extract and plot Pareto frontier
            frontier = get_pareto_frontier(df_type)
            if not frontier.empty:
                plt.plot(
                    frontier["recall_at_10"],
                    frontier["qps"],
                    color=colors[itype],
                    marker=markers[itype],
                    linewidth=2.5,
                    markersize=7,
                    label=f"{itype} (Pareto frontier)",
                    zorder=5,
                )

        plt.title(f"Recall@10 vs. Queries Per Second (QPS) on {dataset_display}", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Recall@10 (Tolerant to boundary ties)", fontsize=11, fontweight="semibold")
        plt.ylabel("Queries Per Second (QPS, single thread, log scale)", fontsize=11, fontweight="semibold")
        plt.yscale("log")
        plt.xlim(0.2, 1.02)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=10, loc="lower left", framealpha=0.9)
        plt.tight_layout()

        fig_path = os.path.join(out_dir, "figures", f"part2_{dataset}_tradeoff.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Generated plot: {fig_path}")

        # Construct Tables
        qps_table = []
        resource_table = []

        for itype in ["HNSW", "IVF-Flat", "IVF-PQ", "Exact"]:
            df_type = df_ds[df_ds["index_type"] == itype]
            if df_type.empty:
                continue

            if itype == "Exact":
                qps_table.append({
                    "Index": "Exact (IndexFlatL2)",
                    "QPS @ Recall 0.90": f"{exact_pt['qps']:.1f}",
                    "QPS @ Recall 0.95": f"{exact_pt['qps']:.1f}",
                })
                resource_table.append({
                    "Index": "Exact (IndexFlatL2)",
                    "Build Time (s)": f"{exact_pt['build_time_s']:.2f}s",
                    "Index Size (MB)": f"{exact_pt['index_size_mb']:.1f} MB",
                    "Bytes / Vector": f"{exact_pt['bytes_per_vec']:.1f}",
                    "Note": "Exact search (Recall 1.0)",
                })
            else:
                frontier = get_pareto_frontier(df_type)
                q90 = interpolate_qps_at_recall(frontier, 0.90)
                q95 = interpolate_qps_at_recall(frontier, 0.95)
                qps_table.append({
                    "Index": itype,
                    "QPS @ Recall 0.90": q90,
                    "QPS @ Recall 0.95": q95,
                })
                res_info = get_resource_row_at_recall(df_type, target_recall=0.95)
                resource_table.append({
                    "Index": itype,
                    "Build Time (s)": res_info["build_time_s"],
                    "Index Size (MB)": res_info["index_mb"],
                    "Bytes / Vector": res_info["bytes_per_vec"],
                    "Note": res_info["note"],
                })

        summary_tables[dataset] = {
            "qps_at_recall": qps_table,
            "resources_at_095": resource_table,
        }

        # Format into Markdown
        markdown_report.append(f"### {dataset_display}\n")
        markdown_report.append("#### Table 1: QPS at Recall 0.90 and 0.95")
        markdown_report.append(records_to_markdown(qps_table))
        markdown_report.append("\n#### Table 2: Resource Footprint at Recall 0.95 (or Peak Recall)")
        markdown_report.append(records_to_markdown(resource_table))
        markdown_report.append("\n" + "-" * 70 + "\n")

    # Save Markdown and JSON summaries
    report_text = "\n".join(markdown_report)
    md_path = os.path.join(out_dir, "part2_tables.md")
    with open(md_path, "w") as f:
        f.write(report_text)
    print(f"Generated tables: {md_path}")

    json_path = os.path.join(out_dir, "part2_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_tables, f, indent=2)
    print(f"Saved summary JSON: {json_path}")

    print("\n" + "=" * 70)
    print("PART 2 SUMMARY TABLES")
    print("=" * 70)
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Part 2: Plotting and table extraction.")
    parser.add_argument("--csv", default="results/results.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    generate_plots_and_tables(csv_path=args.csv, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
