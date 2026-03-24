import argparse
import json
import os
import re
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from adflip_affinity_dataset import ADFLIPAffinityDataset, collate_affinity_adflip
from train_affinity_adflip_fewshot import (
    ADFLIPAffinityPL,
    _pearson_corr,
    _pairwise_logistic_rank_loss,
    _spearman_corr,
    _top_hit_ratio,
)


class Config:
    """
    Compatibility shim for legacy checkpoints that pickle Config under __main__.
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


def _manifest_row_by_dataset(manifest_path: str, dataset_name: str) -> pd.Series:
    manifest_df = pd.read_csv(manifest_path)
    matched = manifest_df[manifest_df["dataset_name"].astype(str) == dataset_name].reset_index(drop=True)
    if len(matched) != 1:
        raise ValueError(f"Dataset '{dataset_name}' not found uniquely in manifest: {manifest_path}")
    return matched.iloc[0]


def _resolve_eval_sources(
    cfg: dict,
    eval_split: str,
    eval_csv_override: str,
    pdb_dir_override: str,
) -> tuple[str, str, str]:
    data_cfg = cfg.get("data", {})
    manifest_path = str(data_cfg.get("manifest", "")).strip()

    def _default_pdb_dir_for_eval_csv(eval_csv_path: str, fallback_pdb_dir: str) -> tuple[str, str]:
        inferred_dataset = os.path.basename(os.path.dirname(os.path.abspath(eval_csv_path))).strip()
        if manifest_path and inferred_dataset:
            try:
                row = _manifest_row_by_dataset(manifest_path, inferred_dataset)
                return str(row["structure_dir"]), inferred_dataset
            except Exception:
                pass
        return fallback_pdb_dir, ""

    if eval_csv_override:
        eval_csv = eval_csv_override
        eval_dataset = (
            str(data_cfg.get("eval_data", "")).strip()
            or str(data_cfg.get("dataset_name", "")).strip()
            or "custom_csv"
        )
        if pdb_dir_override:
            pdb_dir = pdb_dir_override
        else:
            fallback_pdb_dir = str(data_cfg.get("pdb_dir", "")).strip()
            pdb_dir, inferred_dataset = _default_pdb_dir_for_eval_csv(eval_csv, fallback_pdb_dir)
            if inferred_dataset:
                eval_dataset = inferred_dataset
        return eval_csv, pdb_dir, eval_dataset

    cfg_eval_csv = str(data_cfg.get("eval_csv", "")).strip()
    if cfg_eval_csv:
        eval_dataset = (
            str(data_cfg.get("eval_data", "")).strip()
            or str(data_cfg.get("dataset_name", "")).strip()
            or "custom_csv"
        )
        if pdb_dir_override:
            pdb_dir = pdb_dir_override
        else:
            fallback_pdb_dir = str(data_cfg.get("pdb_dir", "")).strip()
            pdb_dir, inferred_dataset = _default_pdb_dir_for_eval_csv(cfg_eval_csv, fallback_pdb_dir)
            if inferred_dataset:
                eval_dataset = inferred_dataset
        return cfg_eval_csv, pdb_dir, eval_dataset

    eval_data = str(data_cfg.get("eval_data", "")).strip()
    if eval_data:
        manifest_path = str(data_cfg.get("manifest", "")).strip()
        if not manifest_path:
            raise ValueError("data.eval_data is set but data.manifest is empty.")
        row = _manifest_row_by_dataset(manifest_path, eval_data)
        eval_csv_key = f"{eval_split}_csv"
        eval_csv = str(row[eval_csv_key])
        default_pdb_dir = str(row["structure_dir"])
        pdb_dir = pdb_dir_override if pdb_dir_override else default_pdb_dir
        return eval_csv, pdb_dir, eval_data

    if eval_split == "train":
        eval_csv = cfg["data"]["train_csv"]
    elif eval_split == "val":
        eval_csv = cfg["data"]["val_csv"]
    else:
        eval_csv = cfg["data"]["test_csv"]
    eval_dataset = str(data_cfg.get("dataset_name", "")).strip()
    pdb_dir = pdb_dir_override if pdb_dir_override else cfg["data"]["pdb_dir"]
    return eval_csv, pdb_dir, eval_dataset


def _hydrate_data_paths_from_manifest(cfg: dict):
    data_cfg = cfg.get("data", {})
    has_explicit_paths = all(
        str(data_cfg.get(k, "")).strip() for k in ["train_csv", "val_csv", "test_csv", "pdb_dir"]
    )
    if has_explicit_paths:
        return

    manifest_path = str(data_cfg.get("manifest", "")).strip()
    dataset_name = str(data_cfg.get("dataset_name", "")).strip()
    if not manifest_path or not dataset_name:
        raise ValueError(
            "Need either explicit data.train_csv/val_csv/test_csv/pdb_dir, "
            "or data.manifest + data.dataset_name."
        )

    row = _manifest_row_by_dataset(manifest_path, dataset_name)
    cfg["data"]["train_csv"] = str(row["train_csv"])
    cfg["data"]["val_csv"] = str(row["val_csv"])
    cfg["data"]["test_csv"] = str(row["test_csv"])
    cfg["data"]["pdb_dir"] = str(row["structure_dir"])


def _load_model_from_ckpt(cfg: dict, ckpt_path: str) -> ADFLIPAffinityPL:
    _register_legacy_config_for_torch_load()
    model = ADFLIPAffinityPL(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    return model


def _compute_global_metrics(model: ADFLIPAffinityPL, loader: DataLoader, cfg: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_all = []
    y_hat_all = []
    with torch.no_grad():
        for sample in loader:
            batch_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in sample["batch_dict"].items()
            }
            y = sample["y"].to(device).reshape(-1)
            y_hat = model.forward(batch_dict).reshape(-1)
            y_all.append(y)
            y_hat_all.append(y_hat)

    if not y_all:
        return {
            "test_loss": 0.0,
            "test_kl": 0.0,
            "test_mse": 0.0,
            "test_rank_loss": 0.0,
            "test_rank_weight": 0.0,
            "test_rmse": 0.0,
            "test_spearman": 0.0,
            "test_pearson": 0.0,
            "test_top_hit": 0.0,
        }

    y_vec = torch.cat(y_all, dim=0)
    y_hat_vec = torch.cat(y_hat_all, dim=0)
    train_cfg = cfg.get("training", {})
    tau_true = float(train_cfg.get("tau_true", 1.0))
    tau_pred = float(train_cfg.get("tau_pred", 1.0))
    kl_weight = float(train_cfg.get("kl_weight", 1.0))
    mse_weight = float(train_cfg.get("mse_weight", 0.1))
    rank_weight = float(train_cfg.get("rank_weight", 0.0))
    rank_margin = float(train_cfg.get("rank_margin", 0.0))
    rank_tie_eps = float(train_cfg.get("rank_tie_eps", 1e-8))
    topk = int(train_cfg.get("topk", 10))

    p = torch.softmax(y_vec / tau_true, dim=0)
    log_q = torch.log_softmax(y_hat_vec / tau_pred, dim=0)
    kl = F.kl_div(log_q, p, reduction="batchmean")
    mse = F.mse_loss(y_hat_vec, y_vec)
    rank_loss = _pairwise_logistic_rank_loss(
        y_true=y_vec,
        y_pred=y_hat_vec,
        margin=rank_margin,
        tie_eps=rank_tie_eps,
    )
    rmse = torch.sqrt(mse + 1e-12)
    spearman = _spearman_corr(y_hat_vec, y_vec)
    pearson = _pearson_corr(y_hat_vec, y_vec)
    top_hit = _top_hit_ratio(y_vec, y_hat_vec, topk)
    loss = kl_weight * kl + mse_weight * mse + rank_weight * rank_loss

    return {
        "test_loss": float(loss.detach().cpu().item()),
        "test_kl": float(kl.detach().cpu().item()),
        "test_mse": float(mse.detach().cpu().item()),
        "test_rank_loss": float(rank_loss.detach().cpu().item()),
        "test_rank_weight": rank_weight,
        "test_rmse": float(rmse.detach().cpu().item()),
        "test_spearman": float(spearman.detach().cpu().item()),
        "test_pearson": float(pearson.detach().cpu().item()),
        "test_top_hit": float(top_hit.detach().cpu().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="run config produced from base config + manifest")
    parser.add_argument("--ckpt", type=str, required=True, help="lightning checkpoint path")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval_csv", type=str, default="", help="optional override csv path (highest priority)")
    parser.add_argument("--pdb_dir", type=str, default="", help="optional override pdb dir")
    parser.add_argument("--list_size", type=int, default=0, help="deprecated; ignored in global evaluation")
    parser.add_argument("--output_json", type=str, default="", help="optional metrics output json path")
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    _hydrate_data_paths_from_manifest(cfg)
    pl.seed_everything(int(cfg.get("training", {}).get("seed", 42)), workers=True)

    eval_csv, pdb_dir, eval_dataset = _resolve_eval_sources(
        cfg=cfg,
        eval_split=args.eval_split,
        eval_csv_override=args.eval_csv,
        pdb_dir_override=args.pdb_dir,
    )
    chain_order = list(cfg["data"].get("chains", ["A", "H", "L"]))

    ds = ADFLIPAffinityDataset(
        csv_path=eval_csv,
        pdb_dir=pdb_dir,
        adflip_root=cfg["adflip_root"],
        pdb_col=cfg["data"].get("pdb_col", "pdb"),
        y_col=cfg["data"].get("y_col", "KD"),
        seq_col=cfg["data"].get("seq_col", "sequence"),
        chain_order=chain_order,
        label_mode=cfg["data"].get("label_mode", "log10"),
        parser_chain_id=chain_order,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=collate_affinity_adflip,
    )
    print(f"[Progress] eval_rows={len(ds)}")

    model = _load_model_from_ckpt(cfg, args.ckpt)
    metrics_dict = _compute_global_metrics(model, loader, cfg)

    out_path = args.output_json
    eval_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(eval_dataset).strip()) or args.eval_split
    if not out_path:
        out_path = os.path.splitext(args.ckpt)[0] + f".eval_{eval_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "ckpt": os.path.abspath(args.ckpt),
                "eval_split": args.eval_split,
                "eval_dataset": eval_dataset,
                "eval_csv": os.path.abspath(eval_csv),
                "pdb_dir": os.path.abspath(pdb_dir),
                "list_size": None,
                "num_units": 1,
                "num_rows": len(ds),
                "metrics": metrics_dict,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved eval metrics to: {out_path}")


if __name__ == "__main__":
    main()
