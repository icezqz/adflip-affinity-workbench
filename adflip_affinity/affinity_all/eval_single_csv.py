#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from train_joint_affinity import (
    JointAffinityDataset,
    _build_zoidberg_from_adflip_ckpt,
    _compute_loss_and_metrics,
    _forward_affinity,
    _seed_everything,
    _to_device_batch,
)


class Config:
    """
    Compatibility shim for loading legacy checkpoints that pickle Config under __main__.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value


def _register_legacy_config_for_torch_load():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "Config"):
        setattr(main_mod, "Config", Config)


def _top_hit_ratio(y_true: torch.Tensor, y_pred: torch.Tensor, topk: int) -> torch.Tensor:
    n = int(y_true.numel())
    if n <= 0:
        return torch.tensor(0.0, device=y_true.device)
    k = max(1, min(int(topk), n))
    gt_idx = torch.topk(y_true.reshape(-1), k=k).indices
    pr_idx = torch.topk(y_pred.reshape(-1), k=k).indices
    gt_set = set(gt_idx.detach().cpu().tolist())
    pr_set = set(pr_idx.detach().cpu().tolist())
    hit = len(gt_set.intersection(pr_set))
    return torch.tensor(float(hit) / float(k), device=y_true.device)


def _prepare_eval_csv_with_optional_pdb_dir(
    eval_csv: str,
    pdb_col: str,
    pdb_dir: str,
) -> tuple[str, str]:
    csv_abs = str(Path(eval_csv).resolve())
    if not pdb_dir:
        return csv_abs, ""

    pdb_dir_abs = str(Path(pdb_dir).resolve())
    with open(csv_abs, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_abs}")
    if pdb_col not in fieldnames:
        raise ValueError(f"CSV missing pdb column '{pdb_col}': {csv_abs}")

    changed = False
    for row in rows:
        raw = str(row.get(pdb_col, "")).strip()
        if not raw:
            continue
        if not os.path.isabs(raw):
            row[pdb_col] = str(Path(pdb_dir_abs, raw).resolve())
            changed = True

    if not changed:
        return csv_abs, ""

    tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False)
    try:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        tmp.close()
    return tmp.name, tmp.name


def _safe_name(raw: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip()).strip("._")
    return value or fallback


def _infer_dataset_tag(eval_csv: str) -> str:
    try:
        with open(eval_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        if first and str(first.get("dataset_name", "")).strip():
            return str(first.get("dataset_name", "")).strip()
    except Exception:
        pass
    parent_name = Path(eval_csv).resolve().parent.name
    return parent_name or Path(eval_csv).resolve().stem


def _write_pred_csv(path: str, records: list[dict[str, Any]]) -> None:
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "dataset_name",
        "name",
        "pdb_path",
        "y_true",
        "y_pred",
        "abs_error",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained affinity_all checkpoint on one specific CSV."
    )
    parser.add_argument("--config", type=str, required=True, help="config yaml used for joint training")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path from affinity_all training")
    parser.add_argument("--eval_csv", type=str, required=True, help="path to the exact CSV to evaluate")
    parser.add_argument("--pdb_dir", type=str, default="", help="optional base dir when pdb column is relative")
    parser.add_argument("--pdb_col", type=str, default="", help="override pdb column name")
    parser.add_argument("--y_col", type=str, default="", help="override label column name")
    parser.add_argument("--seq_col", type=str, default="", help="override sequence column name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="directory to save outputs; always writes both json and csv",
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    _seed_everything(int(train_cfg.get("seed", 42)))

    pdb_col = str(args.pdb_col or data_cfg.get("pdb_col", "pdb_abs"))
    y_col = str(args.y_col or data_cfg.get("y_col", "label_neglog_m"))
    seq_col = str(args.seq_col or data_cfg.get("seq_col", "sequence"))
    chain_order = tuple(data_cfg.get("chains", ["H", "L"]))
    parser_chain_id = tuple(data_cfg.get("parser_chain_id", chain_order))

    eval_csv, tmp_csv_to_clean = _prepare_eval_csv_with_optional_pdb_dir(
        eval_csv=args.eval_csv,
        pdb_col=pdb_col,
        pdb_dir=args.pdb_dir,
    )

    dataset_tag = _safe_name(_infer_dataset_tag(eval_csv), "custom_csv")
    model_tag = _safe_name(Path(args.ckpt).resolve().stem, "model")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.ckpt).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{model_tag}+{dataset_tag}"
    output_json_path = output_dir / f"{output_stem}.json"
    output_pred_csv_path = output_dir / f"{output_stem}.csv"

    rank_weight = float(train_cfg.get("rank_weight", 0.0))
    timestep = float(model_cfg.get("timestep", 0.0))
    topk = int(train_cfg.get("topk", 10))

    try:
        _register_legacy_config_for_torch_load()
        ds = JointAffinityDataset(
            csv_path=eval_csv,
            adflip_root=str(cfg["adflip_root"]),
            pdb_col=pdb_col,
            y_col=y_col,
            seq_col=seq_col,
            chain_order=chain_order,
            parser_chain_id=parser_chain_id,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = _build_zoidberg_from_adflip_ckpt(
            adflip_ckpt=str(cfg["adflip_ckpt"]),
            model_cfg=model_cfg,
            adflip_root=str(cfg["adflip_root"]),
            device=device,
        )
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        y_true_list = []
        y_pred_list = []
        pred_records = []
        with torch.no_grad():
            for idx in range(len(ds)):
                sample = ds[idx]
                batch_dict, y_true = _to_device_batch(sample, device)
                y_pred = _forward_affinity(model=model, batch_dict=batch_dict, timestep_value=timestep, device=device)

                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

                y_t = float(y_true.reshape(-1)[0].detach().cpu().item())
                y_p = float(y_pred.reshape(-1)[0].detach().cpu().item())
                meta = sample.get("meta", {})
                pred_records.append(
                    {
                        "row_index": idx,
                        "dataset_name": str(meta.get("dataset_name", "")).strip(),
                        "name": str(meta.get("name", "")).strip(),
                        "pdb_path": str(meta.get("pdb_path", "")).strip(),
                        "y_true": y_t,
                        "y_pred": y_p,
                        "abs_error": abs(y_t - y_p),
                    }
                )

        y_true_vec = torch.cat(y_true_list, dim=0)
        y_pred_vec = torch.cat(y_pred_list, dim=0)
        metric_t = _compute_loss_and_metrics(
            y_true=y_true_vec,
            y_pred=y_pred_vec,
            train_cfg=train_cfg,
            rank_weight=rank_weight,
        )
        top_hit = _top_hit_ratio(y_true_vec, y_pred_vec, topk=topk)

        metrics = {
            "test_loss": float(metric_t["loss"].detach().cpu().item()),
            "test_kl": float(metric_t["kl"].detach().cpu().item()),
            "test_mse": float(metric_t["mse"].detach().cpu().item()),
            "test_rank_loss": float(metric_t["rank_loss"].detach().cpu().item()),
            "test_rank_weight": rank_weight,
            "test_rmse": float(metric_t["rmse"].detach().cpu().item()),
            "test_spearman": float(metric_t["spearman"].detach().cpu().item()),
            "test_pearson": float(metric_t["pearson"].detach().cpu().item()),
            "test_top_hit": float(top_hit.detach().cpu().item()),
        }

        payload = {
            "config": str(Path(args.config).resolve()),
            "ckpt": str(Path(args.ckpt).resolve()),
            "eval_dataset": dataset_tag,
            "eval_csv": str(Path(eval_csv).resolve()),
            "output_csv": str(output_pred_csv_path.resolve()),
            "pdb_col": pdb_col,
            "y_col": y_col,
            "seq_col": seq_col,
            "list_size": None,
            "num_units": 1,
            "num_rows": len(ds),
            "metrics": metrics,
        }
        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        _write_pred_csv(str(output_pred_csv_path), pred_records)

        print(f"[Progress] eval_rows={len(ds)}")
        print(f"Saved eval metrics to: {output_json_path}")
        print(f"Saved per-row predictions to: {output_pred_csv_path}")
    finally:
        if tmp_csv_to_clean and Path(tmp_csv_to_clean).is_file():
            Path(tmp_csv_to_clean).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
