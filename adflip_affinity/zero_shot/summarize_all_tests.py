import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY_DIR = HERE / "outputs" / "summary"


def _safe_num(df: pd.DataFrame, col: str):
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _build_figure(df: pd.DataFrame, fig_path: str):
    plot_df = df.copy()
    plot_df = plot_df.sort_values("spearman", ascending=False, na_position="last").reset_index(drop=True)
    names = plot_df["dataset_name"].astype(str).to_list()
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(names) * 0.35)), constrained_layout=True)

    axes[0].barh(y, plot_df["spearman"], alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Spearman (score vs true label)")
    axes[0].set_xlabel("Spearman")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, plot_df["top_hit_overall"], alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_title("Top-Hit@k (overall)")
    axes[1].set_xlabel("Top-Hit")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].barh(y, plot_df["avg_nll"], alpha=0.85)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].set_title("Average NLL (lower is better)")
    axes[2].set_xlabel("Avg NLL")
    axes[2].grid(axis="x", alpha=0.25)

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def _write_markdown(df: pd.DataFrame, md_path: str):
    cols = [
        "dataset_name",
        "status",
        "n_rows_scored",
        "avg_nll",
        "avg_ppl",
        "spearman",
        "pearson",
        "top_hit_overall",
        "output_csv",
        "plot_png",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    show = df[cols].copy()
    show = show.sort_values(["status", "spearman"], ascending=[True, False], na_position="last")

    lines = []
    lines.append("# Zero-shot All-Test Summary")
    lines.append("")
    lines.append(f"- Datasets: {len(show)}")
    lines.append(f"- Success: {(show['status'] == 'ok').sum()}")
    lines.append(f"- Failed: {(show['status'] != 'ok').sum()}")
    lines.append("")
    lines.append(show.to_markdown(index=False))
    lines.append("")

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=str(DEFAULT_SUMMARY_DIR / "all_tests_summary.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_SUMMARY_DIR),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    ok_df = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()

    _safe_num(ok_df, "avg_nll")
    _safe_num(ok_df, "avg_ppl")
    _safe_num(ok_df, "spearman")
    _safe_num(ok_df, "pearson")
    _safe_num(ok_df, "top_hit_overall")

    ranked_csv = os.path.join(args.out_dir, "all_tests_ranked.csv")
    fig_png = os.path.join(args.out_dir, "all_tests_dashboard.png")
    report_md = os.path.join(args.out_dir, "all_tests_report.md")

    ok_df.sort_values("spearman", ascending=False, na_position="last").to_csv(ranked_csv, index=False)
    if len(ok_df) > 0:
        _build_figure(ok_df, fig_png)
    _write_markdown(df, report_md)

    print(f"Saved ranked csv: {ranked_csv}")
    if len(ok_df) > 0:
        print(f"Saved dashboard png: {fig_png}")
    print(f"Saved markdown report: {report_md}")


if __name__ == "__main__":
    main()
