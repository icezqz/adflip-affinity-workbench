#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SWK_DATASETS = [
    "Shanehsazzadeh2023_trastuzumab_zero_kd",
    "Warszawski2019_d44_Kd",
    "Koenig2017_g6_Kd",
]

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
ZERO_SHOT_DIR = PROJECT_DIR.parent / "zero_shot"


def _to_float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_affinity_metrics(affinity_run_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted(affinity_run_dir.glob("*.json")):
        if path.name == "run_meta.json":
            continue
        payload = _load_json(path)
        metrics = payload.get("metrics", {})
        dataset = str(payload.get("eval_dataset", "")).strip()
        if not dataset or not isinstance(metrics, dict):
            continue
        out[dataset] = {
            "source_json": str(path.resolve()),
            "eval_csv": str(payload.get("eval_csv", "")),
            "num_rows": int(payload.get("num_rows", 0) or 0),
            "spearman": _to_float_or_nan(metrics.get("test_spearman")),
            "pearson": _to_float_or_nan(metrics.get("test_pearson")),
            "rmse": _to_float_or_nan(metrics.get("test_rmse")),
            "top_hit": _to_float_or_nan(metrics.get("test_top_hit")),
            "mse": _to_float_or_nan(metrics.get("test_mse")),
            "loss": _to_float_or_nan(metrics.get("test_loss")),
        }
    return out


def _pick_zero_shot_metrics_file(
    zero_shot_scores_dir: Path,
    dataset_name: str,
    preferred_split: str,
) -> tuple[Path | None, str]:
    preferred = zero_shot_scores_dir / f"{dataset_name}_{preferred_split}.metrics.json"
    if preferred.is_file():
        return preferred, preferred_split

    # 回退：若期望文件不存在，尝试另一个split，尽量保证可对比。
    fallback_split = "all" if preferred_split == "test" else "test"
    fallback = zero_shot_scores_dir / f"{dataset_name}_{fallback_split}.metrics.json"
    if fallback.is_file():
        return fallback, fallback_split

    return None, ""


def _load_zero_shot_metrics(
    zero_shot_scores_dir: Path,
    datasets: list[str],
    swk_datasets: set[str],
) -> dict[str, dict]:
    out = {}
    for ds in datasets:
        preferred_split = "test" if ds in swk_datasets else "all"
        metrics_path, used_split = _pick_zero_shot_metrics_file(
            zero_shot_scores_dir=zero_shot_scores_dir,
            dataset_name=ds,
            preferred_split=preferred_split,
        )
        if metrics_path is None:
            out[ds] = {
                "source_json": "",
                "used_split": preferred_split,
                "status": "missing",
                "n_rows_scored": float("nan"),
                "avg_nll": float("nan"),
                "avg_ppl": float("nan"),
                "spearman": float("nan"),
                "pearson": float("nan"),
                "top_hit": float("nan"),
            }
            continue

        payload = _load_json(metrics_path)
        metrics = payload.get("metrics", {})
        out[ds] = {
            "source_json": str(metrics_path.resolve()),
            "used_split": used_split,
            "status": "ok",
            "n_rows_scored": _to_float_or_nan(metrics.get("n_rows_scored")),
            "avg_nll": _to_float_or_nan(metrics.get("avg_nll")),
            "avg_ppl": _to_float_or_nan(metrics.get("avg_ppl")),
            "spearman": _to_float_or_nan(metrics.get("spearman")),
            "pearson": _to_float_or_nan(metrics.get("pearson")),
            "top_hit": _to_float_or_nan(metrics.get("top_hit_overall")),
        }
    return out


def _build_summary_df(
    affinity_metrics: dict[str, dict],
    zero_metrics: dict[str, dict],
    swk_datasets: set[str],
) -> pd.DataFrame:
    rows = []
    for ds in sorted(affinity_metrics.keys()):
        a = affinity_metrics[ds]
        z = zero_metrics.get(ds, {})

        affinity_spearman = _to_float_or_nan(a.get("spearman"))
        zero_spearman = _to_float_or_nan(z.get("spearman"))
        affinity_pearson = _to_float_or_nan(a.get("pearson"))
        zero_pearson = _to_float_or_nan(z.get("pearson"))
        affinity_top_hit = _to_float_or_nan(a.get("top_hit"))
        zero_top_hit = _to_float_or_nan(z.get("top_hit"))

        rows.append(
            {
                "dataset_name": ds,
                "is_swk": ds in swk_datasets,
                "zero_shot_split_rule": "test" if ds in swk_datasets else "all",
                "zero_shot_used_split": str(z.get("used_split", "")),
                "zero_shot_status": str(z.get("status", "missing")),
                "n_rows_affinity": int(a.get("num_rows", 0) or 0),
                "n_rows_zeroshot": _to_float_or_nan(z.get("n_rows_scored")),
                "affinity_spearman": affinity_spearman,
                "zeroshot_spearman": zero_spearman,
                "delta_spearman_affinity_minus_zeroshot": affinity_spearman - zero_spearman,
                "affinity_pearson": affinity_pearson,
                "zeroshot_pearson": zero_pearson,
                "delta_pearson_affinity_minus_zeroshot": affinity_pearson - zero_pearson,
                "affinity_top_hit": affinity_top_hit,
                "zeroshot_top_hit": zero_top_hit,
                "delta_top_hit_affinity_minus_zeroshot": affinity_top_hit - zero_top_hit,
                "affinity_rmse": _to_float_or_nan(a.get("rmse")),
                "affinity_loss": _to_float_or_nan(a.get("loss")),
                "zeroshot_avg_nll": _to_float_or_nan(z.get("avg_nll")),
                "zeroshot_avg_ppl": _to_float_or_nan(z.get("avg_ppl")),
                "affinity_json": str(a.get("source_json", "")),
                "zeroshot_json": str(z.get("source_json", "")),
            }
        )

    return pd.DataFrame(rows)


def _save_boxplot_spearman(summary_df: pd.DataFrame, out_path: Path):
    model_a = summary_df["affinity_spearman"].to_numpy(dtype=np.float64)
    model_b = summary_df["zeroshot_spearman"].to_numpy(dtype=np.float64)

    valid_a = model_a[np.isfinite(model_a)]
    valid_b = model_b[np.isfinite(model_b)]

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot(
        [valid_a, valid_b],
        labels=["Affinity-Head Model", "Zero-Shot Backbone"],
        patch_artist=True,
        boxprops={"facecolor": "#9ecae1"},
        medianprops={"color": "#d62728", "linewidth": 2},
    )

    # 叠加散点，展示每个数据集具体点位
    rng = np.random.default_rng(42)
    x1 = 1.0 + rng.uniform(-0.07, 0.07, size=valid_a.shape[0])
    x2 = 2.0 + rng.uniform(-0.07, 0.07, size=valid_b.shape[0])
    plt.scatter(x1, valid_a, alpha=0.7, s=24, color="#1f77b4")
    plt.scatter(x2, valid_b, alpha=0.7, s=24, color="#ff7f0e")

    plt.ylabel("Spearman")
    plt.title("Spearman Distribution Across Datasets")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_dataset_bar(summary_df: pd.DataFrame, out_path: Path):
    plot_df = summary_df.sort_values("affinity_spearman", ascending=False).reset_index(drop=True)
    names = plot_df["dataset_name"].tolist()
    x = np.arange(len(names))
    width = 0.38

    a = plot_df["affinity_spearman"].to_numpy(dtype=np.float64)
    z = plot_df["zeroshot_spearman"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(max(12, len(names) * 0.6), 6))
    plt.bar(x - width / 2, a, width=width, label="Affinity-Head")
    plt.bar(x + width / 2, z, width=width, label="Zero-Shot")
    plt.xticks(x, names, rotation=55, ha="right")
    plt.ylabel("Spearman")
    plt.title("Per-Dataset Spearman Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_delta_bar(summary_df: pd.DataFrame, out_path: Path):
    plot_df = summary_df.sort_values(
        "delta_spearman_affinity_minus_zeroshot",
        ascending=False,
    ).reset_index(drop=True)

    names = plot_df["dataset_name"].tolist()
    vals = plot_df["delta_spearman_affinity_minus_zeroshot"].to_numpy(dtype=np.float64)
    x = np.arange(len(names))
    colors = ["#2ca02c" if (np.isfinite(v) and v >= 0) else "#d62728" for v in vals]

    plt.figure(figsize=(max(12, len(names) * 0.6), 6))
    plt.bar(x, vals, color=colors, alpha=0.9)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(x, names, rotation=55, ha="right")
    plt.ylabel("Affinity - Zero-Shot (Spearman)")
    plt.title("Spearman Gain/Loss by Dataset")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_scatter(summary_df: pd.DataFrame, out_path: Path):
    plot_df = summary_df.copy()
    x = plot_df["zeroshot_spearman"].to_numpy(dtype=np.float64)
    y = plot_df["affinity_spearman"].to_numpy(dtype=np.float64)
    names = plot_df["dataset_name"].tolist()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    names = [n for n, m in zip(names, mask) if m]

    if x.size == 0:
        return

    lo = min(float(np.min(x)), float(np.min(y))) - 0.03
    hi = max(float(np.max(x)), float(np.max(y))) + 0.03

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=35, alpha=0.8)
    for xi, yi, name in zip(x, y, names):
        plt.text(xi, yi, name, fontsize=8, alpha=0.85)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="gray")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Zero-Shot Spearman")
    plt.ylabel("Affinity-Head Spearman")
    plt.title("Spearman Scatter: Zero-Shot vs Affinity-Head")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_boxplot_top_hit(summary_df: pd.DataFrame, out_path: Path):
    model_a = summary_df["affinity_top_hit"].to_numpy(dtype=np.float64)
    model_b = summary_df["zeroshot_top_hit"].to_numpy(dtype=np.float64)

    valid_a = model_a[np.isfinite(model_a)]
    valid_b = model_b[np.isfinite(model_b)]

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot(
        [valid_a, valid_b],
        labels=["Affinity-Head Model", "Zero-Shot Backbone"],
        patch_artist=True,
        boxprops={"facecolor": "#c7e9c0"},
        medianprops={"color": "#d62728", "linewidth": 2},
    )

    rng = np.random.default_rng(43)
    x1 = 1.0 + rng.uniform(-0.07, 0.07, size=valid_a.shape[0])
    x2 = 2.0 + rng.uniform(-0.07, 0.07, size=valid_b.shape[0])
    plt.scatter(x1, valid_a, alpha=0.7, s=24, color="#2ca02c")
    plt.scatter(x2, valid_b, alpha=0.7, s=24, color="#ff7f0e")

    plt.ylabel("Top-Hit")
    plt.title("Top-Hit Distribution Across Datasets")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_top_hit_bar(summary_df: pd.DataFrame, out_path: Path):
    plot_df = summary_df.sort_values("affinity_top_hit", ascending=False).reset_index(drop=True)
    names = plot_df["dataset_name"].tolist()
    x = np.arange(len(names))
    width = 0.38

    a = plot_df["affinity_top_hit"].to_numpy(dtype=np.float64)
    z = plot_df["zeroshot_top_hit"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(max(12, len(names) * 0.6), 6))
    plt.bar(x - width / 2, a, width=width, label="Affinity-Head")
    plt.bar(x + width / 2, z, width=width, label="Zero-Shot")
    plt.xticks(x, names, rotation=55, ha="right")
    plt.ylabel("Top-Hit")
    plt.title("Per-Dataset Top-Hit Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _save_top_hit_delta_bar(summary_df: pd.DataFrame, out_path: Path):
    plot_df = summary_df.sort_values(
        "delta_top_hit_affinity_minus_zeroshot",
        ascending=False,
    ).reset_index(drop=True)

    names = plot_df["dataset_name"].tolist()
    vals = plot_df["delta_top_hit_affinity_minus_zeroshot"].to_numpy(dtype=np.float64)
    x = np.arange(len(names))
    colors = ["#2ca02c" if (np.isfinite(v) and v >= 0) else "#d62728" for v in vals]

    plt.figure(figsize=(max(12, len(names) * 0.6), 6))
    plt.bar(x, vals, color=colors, alpha=0.9)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(x, names, rotation=55, ha="right")
    plt.ylabel("Affinity - Zero-Shot (Top-Hit)")
    plt.title("Top-Hit Gain/Loss by Dataset")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--affinity_run_dir",
        type=str,
        default=str(PROJECT_DIR / "outputs" / "runs" / "SWK_joint_20260323_134723"),
        help="Directory containing affinity_all eval json files.",
    )
    parser.add_argument(
        "--zero_shot_scores_dir",
        type=str,
        default=str(ZERO_SHOT_DIR / "outputs" / "scores"),
        help="Directory containing zero-shot *.metrics.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for summary csv and figures. Default: <affinity_run_dir>/comparison",
    )
    parser.add_argument(
        "--swk_datasets",
        type=str,
        nargs="*",
        default=DEFAULT_SWK_DATASETS,
        help="Datasets treated as SWK; zero-shot uses test split for them.",
    )
    args = parser.parse_args()

    affinity_run_dir = Path(args.affinity_run_dir).resolve()
    zero_scores_dir = Path(args.zero_shot_scores_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (affinity_run_dir / "comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    swk_datasets = set(args.swk_datasets)

    affinity_metrics = _load_affinity_metrics(affinity_run_dir)
    if not affinity_metrics:
        raise ValueError(f"No affinity eval json found in: {affinity_run_dir}")

    datasets = sorted(affinity_metrics.keys())
    zero_metrics = _load_zero_shot_metrics(
        zero_shot_scores_dir=zero_scores_dir,
        datasets=datasets,
        swk_datasets=swk_datasets,
    )
    summary_df = _build_summary_df(affinity_metrics, zero_metrics, swk_datasets)

    # 主汇总
    summary_csv = output_dir / "affinity_vs_zeroshot_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 长表：便于后续可视化扩展
    long_rows = []
    for _, r in summary_df.iterrows():
        long_rows.append(
            {
                "dataset_name": r["dataset_name"],
                "is_swk": r["is_swk"],
                "model_type": "affinity_head",
                "spearman": r["affinity_spearman"],
                "pearson": r["affinity_pearson"],
                "top_hit": r["affinity_top_hit"],
            }
        )
        long_rows.append(
            {
                "dataset_name": r["dataset_name"],
                "is_swk": r["is_swk"],
                "model_type": "zero_shot",
                "spearman": r["zeroshot_spearman"],
                "pearson": r["zeroshot_pearson"],
                "top_hit": r["zeroshot_top_hit"],
            }
        )
    long_df = pd.DataFrame(long_rows)
    long_csv = output_dir / "affinity_vs_zeroshot_long.csv"
    long_df.to_csv(long_csv, index=False)

    # 图表
    _save_boxplot_spearman(summary_df, output_dir / "boxplot_spearman_models.png")
    _save_dataset_bar(summary_df, output_dir / "bar_spearman_per_dataset.png")
    _save_delta_bar(summary_df, output_dir / "bar_spearman_delta_affinity_minus_zeroshot.png")
    _save_scatter(summary_df, output_dir / "scatter_spearman_zeroshot_vs_affinity.png")
    _save_boxplot_top_hit(summary_df, output_dir / "boxplot_top_hit_models.png")
    _save_top_hit_bar(summary_df, output_dir / "bar_top_hit_per_dataset.png")
    _save_top_hit_delta_bar(summary_df, output_dir / "bar_top_hit_delta_affinity_minus_zeroshot.png")

    n_ok_zero = int((summary_df["zero_shot_status"] == "ok").sum())
    print(f"Saved summary csv: {summary_csv}")
    print(f"Saved long csv: {long_csv}")
    print(f"Saved figures to: {output_dir}")
    print(f"Datasets compared: {len(summary_df)} (zero-shot found: {n_ok_zero})")
    print("Zero-shot split rule: SWK datasets -> test, others -> all")


if __name__ == "__main__":
    main()
