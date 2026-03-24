import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "outputs"
DEFAULT_SCORE_SCRIPT = HERE / "score_zero_shot_nll.py"


def _print_progress(prefix: str, current: int, total: int, width: int = 30):
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({ratio * 100:5.1f}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="test", choices=["all", "train", "val", "test"])
    parser.add_argument("--eval_csv_col", type=str, default="")
    parser.add_argument("--pdb_col", type=str, default="")
    parser.add_argument("--seq_col", type=str, default="")
    parser.add_argument("--y_col", type=str, default="")
    parser.add_argument("--label_mode", type=str, default="")
    parser.add_argument("--output_root", type=str, default="")
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    manifest_path = args.manifest or str(cfg.get("data", {}).get("manifest", "")).strip()
    if not manifest_path:
        raise ValueError("Manifest path is required. Provide --manifest or data.manifest in config.")
    manifest_df = pd.read_csv(manifest_path)
    if "dataset_name" not in manifest_df.columns:
        raise ValueError(f"manifest missing dataset_name column: {manifest_path}")

    output_root = args.output_root or str(
        cfg.get("evaluation", {}).get("output_dir", str(DEFAULT_OUTPUT_ROOT))
    )
    scores_dir = os.path.join(output_root, "scores")
    summary_dir = os.path.join(output_root, "summary")
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    script_path = str(DEFAULT_SCORE_SCRIPT)
    rows = []

    total_datasets = len(manifest_df)
    if total_datasets == 0:
        print("No datasets found in manifest.")
    else:
        _print_progress("Datasets progress", 0, total_datasets)

    for ds_idx, (_, row) in enumerate(manifest_df.iterrows(), start=1):
        dataset_name = str(row["dataset_name"]).strip()
        eval_csv_col = args.eval_csv_col.strip()
        if not eval_csv_col:
            eval_csv_col = "all_csv" if args.eval_split == "all" else f"{args.eval_split}_csv"
        if eval_csv_col not in manifest_df.columns or not str(row.get(eval_csv_col, "")).strip():
            raise ValueError(f"manifest missing valid {eval_csv_col} for dataset: {dataset_name}")

        eval_csv = str(row[eval_csv_col]).strip()
        pdb_dir = str(row["structure_dir"]).strip()
        out_csv = os.path.join(scores_dir, f"{dataset_name}_{args.eval_split}.csv")
        out_metrics = os.path.join(scores_dir, f"{dataset_name}_{args.eval_split}.metrics.json")

        cmd = [
            sys.executable,
            script_path,
            "--config",
            args.config,
            "--eval_split",
            args.eval_split,
            "--eval_csv",
            eval_csv,
            "--pdb_dir",
            pdb_dir,
            "--output_csv",
            out_csv,
            "--metrics_json",
            out_metrics,
            "--dataset_name_override",
            dataset_name,
        ]
        if args.pdb_col:
            cmd.extend(["--pdb_col", args.pdb_col])
        if args.seq_col:
            cmd.extend(["--seq_col", args.seq_col])
        if args.y_col:
            cmd.extend(["--y_col", args.y_col])
        if args.label_mode:
            cmd.extend(["--label_mode", args.label_mode])
        if args.cuda_visible_devices:
            cmd.extend(["--cuda_visible_devices", args.cuda_visible_devices])

        print(f"\n[RUN {ds_idx}/{total_datasets}] {dataset_name}")
        completed = subprocess.run(cmd)
        status = "ok" if completed.returncode == 0 else "failed"
        err_msg = "" if completed.returncode == 0 else "Failed; see terminal output."

        metric_payload = {}
        if status == "ok" and os.path.isfile(out_metrics):
            with open(out_metrics, "r", encoding="utf-8") as f:
                metric_payload = json.load(f).get("metrics", {})

        rows.append(
            {
                "dataset_name": dataset_name,
                "status": status,
                "eval_csv": eval_csv,
                "pdb_dir": pdb_dir,
                "output_csv": out_csv,
                "metrics_json": out_metrics,
                "plot_png": out_csv.replace(".csv", ".neglogkd_vs_nll.png"),
                "n_rows_scored": metric_payload.get("n_rows_scored", None),
                "avg_nll": metric_payload.get("avg_nll", None),
                "avg_ppl": metric_payload.get("avg_ppl", None),
                "spearman": metric_payload.get("spearman", None),
                "pearson": metric_payload.get("pearson", None),
                "top_hit_overall": metric_payload.get("top_hit_overall", None),
                "error": err_msg,
            }
        )

        _print_progress("Datasets progress", ds_idx, total_datasets)

    summary_csv = os.path.join(summary_dir, "all_tests_summary.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"[DONE] summary saved: {summary_csv}")


if __name__ == "__main__":
    main()
