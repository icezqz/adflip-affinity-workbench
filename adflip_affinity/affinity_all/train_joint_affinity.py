#!/usr/bin/env python3
from __future__ import annotations
import argparse
import contextlib
import csv
import datetime
import json
import math
import os
import random
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import prody
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "outputs"


try:
    prody.confProDy(verbosity="none")
except Exception:
    pass


class Config:
    """
    Compatibility shim for loading legacy checkpoints that pickle Config under __main__.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value


class _SimpleProgressBar:
    def __init__(self, total: int, desc: str, width: int = 28):
        self.total = max(1, int(total))
        self.desc = desc
        self.width = max(10, int(width))
        self.current = 0
        self.postfix = {}
        self._render()

    def set_postfix(self, values: dict[str, Any]):
        self.postfix = dict(values or {})

    def update(self, n: int = 1):
        self.current = min(self.total, self.current + int(n))
        self._render()

    def close(self):
        self.current = self.total
        self._render(final=True)

    def _render(self, final: bool = False):
        frac = float(self.current) / float(self.total)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = int(round(frac * 100))
        postfix = " ".join([f"{k}={v}" for k, v in self.postfix.items()]) if self.postfix else ""
        line = f"\r{self.desc} [{bar}] {self.current}/{self.total} {pct:3d}%"
        if postfix:
            line += f" {postfix}"
        end = "\n" if final else ""
        print(line, end=end, flush=True)


def _create_progress_bar(total: int, desc: str):
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, dynamic_ncols=True, leave=False)
    except Exception:
        return _SimpleProgressBar(total=total, desc=desc)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_adflip_importable(adflip_root: str):
    if adflip_root not in sys.path:
        sys.path.insert(0, adflip_root)


@contextlib.contextmanager
def _temporary_cwd(target_dir: str):
    prev = os.getcwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(prev)


@lru_cache(maxsize=8192)
def _chain_order_in_pdb(pdb_path: str) -> tuple[str, ...]:
    ag = prody.parsePDB(pdb_path)
    atoms = ag.select("not water and not hydrogen")
    if atoms is None:
        return tuple()

    order = []
    for atom in atoms:
        ch = atom.getChid() or "?"
        if len(ch) > 1:
            ch = ch[0]
        if ch not in order:
            order.append(ch)
    return tuple(order)


class JointAffinityDataset:
    def __init__(
        self,
        csv_path: str,
        adflip_root: str,
        pdb_col: str = "pdb_abs",
        y_col: str = "label_neglog_m",
        seq_col: str = "sequence",
        chain_order: tuple[str, ...] = ("H", "L"),
        parser_chain_id: Optional[Tuple[str, ...]] = None,
    ):
        _ensure_adflip_importable(adflip_root)
        with _temporary_cwd(adflip_root):
            from data import all_atom_parse as aap  # noqa: WPS433

        self.aap = aap
        self.rows = self._read_rows(csv_path, pdb_col=pdb_col, y_col=y_col)
        self.pdb_col = pdb_col
        self.y_col = y_col
        self.seq_col = seq_col
        self.chain_order = list(chain_order)
        self.parser_chain_id = list(parser_chain_id) if parser_chain_id is not None else list(chain_order)

    @staticmethod
    def _read_rows(csv_path: str, pdb_col: str, y_col: str) -> list[dict[str, str]]:
        rows = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if pdb_col not in row or y_col not in row:
                    raise ValueError(f"CSV missing required columns: {pdb_col}, {y_col}")
                y_raw = str(row[y_col]).strip()
                if not y_raw:
                    continue
                try:
                    float(y_raw)
                except ValueError:
                    continue
                rows.append(row)

        if not rows:
            raise ValueError(f"No valid rows found in {csv_path}")
        return rows

    def __len__(self):
        return len(self.rows)

    def _infer_chain_letter_to_internal_id(self, pdb_path: str) -> dict[str, int]:
        encountered = list(_chain_order_in_pdb(pdb_path))
        encountered = [c for c in encountered if c in self.parser_chain_id]
        if not encountered:
            encountered = list(self.parser_chain_id)
        return {c: i for i, c in enumerate(encountered)}

    def _override_chain_sequence(self, struct_data, chain_internal_id: int, seq: str):
        seq = (seq or "").strip()
        if not seq:
            return struct_data

        is_chain = struct_data.chain_id == chain_internal_id
        center_mask = is_chain & struct_data.is_center & struct_data.is_protein
        if center_mask.sum() == 0:
            return struct_data

        res_ids = np.unique(struct_data.residue_index[center_mask])
        max_len = min(len(res_ids), len(seq))
        for i in range(max_len):
            aa1 = seq[i].upper()
            aa3 = self.aap.restype_1to3.get(aa1, "<UNK>")
            token_idx = self.aap.get_token_index(aa3)
            rid = res_ids[i]
            struct_data.residue_token[struct_data.residue_index == rid] = token_idx
        return struct_data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        pdb_path = str(row[self.pdb_col]).strip()
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        y = torch.tensor([[float(row[self.y_col])]], dtype=torch.float32)
        struct_data = self.aap.parse_mmcif_to_structure_data(
            pdb_path,
            parser_chain_id=self.parser_chain_id,
        )

        raw_seq = str(row.get(self.seq_col, "")).strip()
        if raw_seq and raw_seq.lower() != "nan":
            sep = "\\" if "\\" in raw_seq else "/"
            seqs = raw_seq.split(sep)
            chain_map = self._infer_chain_letter_to_internal_id(pdb_path)
            target_chains = [c for c in self.chain_order if c != "A"] or list(self.chain_order)
            for i, seq in enumerate(seqs):
                if i >= len(target_chains):
                    break
                chain_letter = target_chains[i]
                if chain_letter not in chain_map:
                    continue
                struct_data = self._override_chain_sequence(struct_data, chain_map[chain_letter], seq)

        batch_dict = {}
        for k, v in struct_data.__dict__.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v).unsqueeze(0)
            else:
                t = torch.tensor([v]).unsqueeze(0)
            if t.dtype == torch.float64:
                t = t.float()
            batch_dict[k] = t

        batch_dict["batch_index"] = torch.zeros_like(batch_dict["residue_index"])

        return {
            "batch_dict": batch_dict,
            "y": y,
            "meta": {
                "dataset_name": str(row.get("dataset_name", "")).strip(),
                "pdb_path": pdb_path,
                "name": str(row.get("name", "")).strip(),
            },
        }


class GroupCycler:
    def __init__(self, groups: list[list[int]], seed: int):
        if not groups:
            raise ValueError("GroupCycler requires non-empty groups.")
        self.groups = [list(g) for g in groups]
        self.rng = random.Random(seed)
        self.order = list(range(len(self.groups)))
        self.rng.shuffle(self.order)
        self.pos = 0

    def next_group(self) -> list[int]:
        if self.pos >= len(self.order):
            self.rng.shuffle(self.order)
            self.pos = 0
        gid = self.order[self.pos]
        self.pos += 1
        return self.groups[gid]


def _build_groups(n_items: int, list_size: int, shuffle: bool, seed: int) -> list[list[int]]:
    if n_items < 2:
        raise ValueError(f"Need at least 2 rows to form listwise groups, got {n_items}")
    if list_size < 2:
        raise ValueError("training.list_size must be >= 2")

    indices = list(range(n_items))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    groups = [indices[i : i + list_size] for i in range(0, len(indices), list_size)]
    if len(groups) >= 2 and len(groups[-1]) == 1:
        groups[-2].extend(groups[-1])
        groups = groups[:-1]

    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        raise ValueError("No valid groups were created.")
    return groups


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = torch.sqrt((xc.square().sum() + 1e-12) * (yc.square().sum() + 1e-12))
    return (xc * yc).sum() / denom


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.zeros_like(x, dtype=torch.float32)
    ranks[order] = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float32)
    return ranks


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    return _pearson_corr(_rankdata(x.reshape(-1)), _rankdata(y.reshape(-1)))


def _pairwise_logistic_rank_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    margin: float,
    tie_eps: float,
) -> torch.Tensor:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n = int(y_true.numel())
    if n < 2:
        return torch.tensor(0.0, device=y_true.device)

    diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)

    pos_mask = diff_true > float(tie_eps)
    if not pos_mask.any():
        return torch.tensor(0.0, device=y_true.device)

    logits = diff_pred[pos_mask] - float(margin)
    return F.softplus(-logits).mean()


def _build_zoidberg_from_adflip_ckpt(adflip_ckpt: str, model_cfg: dict, adflip_root: str, device: torch.device):
    _ensure_adflip_importable(adflip_root)
    from model.zoidberg.zoidberg_GNN_affinity import Zoidberg_GNN_Affinity  # noqa: WPS433

    ckpt = torch.load(adflip_ckpt, map_location="cpu")
    cfg = ckpt["config"]
    zcfg = cfg.zoidberg_denoiser

    model = Zoidberg_GNN_Affinity(
        hidden_dim=zcfg.hidden_dim,
        encoder_hidden_dim=zcfg.hidden_dim,
        num_blocks=zcfg.num_layers,
        num_heads=zcfg.num_heads,
        k=zcfg.k_neighbors,
        num_positional_embeddings=zcfg.num_positional_embeddings,
        num_rbf=zcfg.num_rbf,
        augment_eps=zcfg.augment_eps,
        backbone_diheral=zcfg.backbone_diheral,
        dropout=zcfg.dropout,
        denoiser=True,
        update_atom=zcfg.update_atom,
        num_decoder_blocks=zcfg.num_decoder_blocks,
        num_tfmr_heads=zcfg.num_tfmr_heads,
        num_tfmr_layers=zcfg.num_tfmr_layers,
        number_ligand_atom=zcfg.number_ligand_atom,
        mpnn_cutoff=zcfg.mpnn_cutoff,
        affinity_head_use_lightattn=bool(model_cfg.get("affinity_head_use_lightattn", True)),
        affinity_head_lightattn_dropout=float(model_cfg.get("affinity_head_lightattn_dropout", 0.25)),
    ).to(device)

    sd = ckpt["model"]
    z_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.backbone.load_state_dict(z_sd, strict=False)
    return model, {"missing": missing, "unexpected": unexpected}


def _forward_affinity(model, batch_dict: dict[str, torch.Tensor], timestep_value: float, device: torch.device) -> torch.Tensor:
    timestep = torch.full((1, 1), float(timestep_value), device=device)
    _, _, affinity = model(batch_dict, timestep, return_affinity=True)
    return affinity.reshape(-1)


def _compute_loss_and_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, train_cfg: dict, rank_weight: float) -> dict[str, torch.Tensor]:
    tau_true = float(train_cfg.get("tau_true", 1.0))
    tau_pred = float(train_cfg.get("tau_pred", 1.0))
    kl_weight = float(train_cfg.get("kl_weight", 1.0))
    mse_weight = float(train_cfg.get("mse_weight", 0.0))
    rank_margin = float(train_cfg.get("rank_margin", 0.0))
    rank_tie_eps = float(train_cfg.get("rank_tie_eps", 1e-8))

    p = torch.softmax(y_true / tau_true, dim=0)
    log_q = torch.log_softmax(y_pred / tau_pred, dim=0)
    kl = F.kl_div(log_q, p, reduction="batchmean")
    mse = F.mse_loss(y_pred, y_true)
    rank = _pairwise_logistic_rank_loss(y_true=y_true, y_pred=y_pred, margin=rank_margin, tie_eps=rank_tie_eps)

    loss = kl_weight * kl + mse_weight * mse + float(rank_weight) * rank
    rmse = torch.sqrt(mse + 1e-12)
    spearman = _spearman_corr(y_pred, y_true)
    pearson = _pearson_corr(y_pred, y_true)

    return {
        "loss": loss,
        "kl": kl,
        "mse": mse,
        "rank_loss": rank,
        "rmse": rmse,
        "spearman": spearman,
        "pearson": pearson,
    }


def _current_rank_weight(epoch_idx_0based: int, train_cfg: dict) -> float:
    base = float(train_cfg.get("rank_weight", 0.0))
    if base <= 0.0:
        return 0.0
    warmup = int(train_cfg.get("rank_warmup_epochs", 0))
    if warmup <= 0:
        return base
    factor = min(1.0, float(epoch_idx_0based + 1) / float(warmup))
    return base * factor


def _float(v: torch.Tensor) -> float:
    return float(v.detach().cpu().item())


def _to_device_batch(sample: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in sample["batch_dict"].items()
    }
    y = sample["y"].to(device).reshape(-1)
    return batch_dict, y


def _run_group(
    model,
    dataset: JointAffinityDataset,
    group_indices: list[int],
    device: torch.device,
    timestep: float,
    train_cfg: dict,
    rank_weight: float,
) -> dict[str, Any]:
    y_true_list = []
    y_pred_list = []

    for idx in group_indices:
        sample = dataset[idx]
        batch_dict, y = _to_device_batch(sample, device)
        y_hat = _forward_affinity(model=model, batch_dict=batch_dict, timestep_value=timestep, device=device)
        y_true_list.append(y)
        y_pred_list.append(y_hat)

    y_true = torch.cat(y_true_list, dim=0)
    y_pred = torch.cat(y_pred_list, dim=0)
    metric_t = _compute_loss_and_metrics(y_true=y_true, y_pred=y_pred, train_cfg=train_cfg, rank_weight=rank_weight)

    return {
        "loss": metric_t["loss"],
        "metrics": {
            "loss": _float(metric_t["loss"]),
            "kl": _float(metric_t["kl"]),
            "mse": _float(metric_t["mse"]),
            "rank_loss": _float(metric_t["rank_loss"]),
            "rmse": _float(metric_t["rmse"]),
            "spearman": _float(metric_t["spearman"]),
            "pearson": _float(metric_t["pearson"]),
            "rank_weight": float(rank_weight),
        },
        "y_true": y_true.detach(),
        "y_pred": y_pred.detach(),
    }


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_dataset_paths(data_cfg: dict) -> dict[str, dict[str, str]]:
    manifest_path = str(data_cfg.get("manifest", "")).strip()
    dataset_names = list(data_cfg.get("datasets", []))
    if not manifest_path:
        raise ValueError("data.manifest is required")
    if not dataset_names:
        raise ValueError("data.datasets must include at least one dataset")

    manifest_rows = _read_csv_rows(manifest_path)
    row_map = {
        str(r.get("dataset_name", "")).strip(): r
        for r in manifest_rows
    }

    resolved = {}
    for name in dataset_names:
        if name not in row_map:
            raise ValueError(f"Dataset '{name}' not found in manifest: {manifest_path}")
        row = row_map[name]
        resolved[name] = {
            "train_csv": str(row["train_csv"]).strip(),
            "val_csv": str(row["val_csv"]).strip(),
            "structure_dir": str(row.get("structure_dir", "")).strip(),
        }
    return resolved


def _sampling_probabilities(dataset_names: list[str], group_counts: dict[str, int], train_cfg: dict) -> dict[str, float]:
    raw_custom = train_cfg.get("dataset_probs", {})
    if raw_custom:
        probs = {name: float(raw_custom[name]) for name in dataset_names}
        s = sum(probs.values())
        if s <= 0:
            raise ValueError("Sum of training.dataset_probs must be > 0")
        return {k: v / s for k, v in probs.items()}

    mode = str(train_cfg.get("sampling_mode", "uniform")).strip().lower()
    weights = {}
    if mode == "uniform":
        for name in dataset_names:
            weights[name] = 1.0
    elif mode == "inverse":
        for name in dataset_names:
            weights[name] = 1.0 / float(group_counts[name])
    elif mode in {"sqrt_inverse", "inv_sqrt", "sqrt-inverse"}:
        for name in dataset_names:
            weights[name] = 1.0 / math.sqrt(float(group_counts[name]))
    else:
        raise ValueError(f"Unknown training.sampling_mode: {mode}")

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _append_metrics_csv(path: Path, row: dict[str, Any], fieldnames: list[str]):
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _save_training_curves(metrics_csv: Path, out_png: Path) -> str:
    if not metrics_csv.is_file():
        return "metrics_csv_missing"

    rows = []
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return "metrics_csv_empty"

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return "matplotlib_not_available"

    def _to_float(v: str) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    epochs = [int(_to_float(r.get("epoch", "nan"))) for r in rows]
    keys = list(rows[0].keys())

    train_loss_cols = [k for k in keys if k.startswith("train_") and k.endswith("_loss")]
    val_loss_cols = [k for k in keys if k.startswith("val_") and k.endswith("_loss")]
    val_spear_cols = [k for k in keys if k.startswith("val_") and k.endswith("_spearman")]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    if "macro_val_loss" in keys:
        ax.plot(epochs, [_to_float(r.get("macro_val_loss", "nan")) for r in rows], label="macro_val_loss", linewidth=2)
    for col in train_loss_cols:
        ax.plot(epochs, [_to_float(r.get(col, "nan")) for r in rows], label=col, alpha=0.8)
    for col in val_loss_cols:
        ax.plot(epochs, [_to_float(r.get(col, "nan")) for r in rows], label=col, alpha=0.8, linestyle="--")
    ax.set_ylabel("Loss")
    ax.set_title("Training/Validation Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    if "macro_val_spearman" in keys:
        ax.plot(
            epochs,
            [_to_float(r.get("macro_val_spearman", "nan")) for r in rows],
            label="macro_val_spearman",
            linewidth=2,
        )
    for col in val_spear_cols:
        ax.plot(epochs, [_to_float(r.get(col, "nan")) for r in rows], label=col, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman")
    ax.set_title("Validation Spearman Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    return "ok"


def _save_checkpoint(path: Path, model, optimizer, epoch: int, best_macro_spearman: float, cfg: dict, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "best_macro_spearman": float(best_macro_spearman),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "extra": extra,
    }
    torch.save(payload, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    seed = int(cfg.get("training", {}).get("seed", 42))
    _seed_everything(seed)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    dataset_paths = _resolve_dataset_paths(data_cfg)
    dataset_names = list(dataset_paths.keys())

    chain_order = tuple(data_cfg.get("chains", ["H", "L"]))
    parser_chain_id = tuple(data_cfg.get("parser_chain_id", chain_order))

    train_ds = {}
    val_ds = {}
    train_groups = {}
    val_groups = {}

    list_size = int(train_cfg.get("list_size", 12))
    for i, ds_name in enumerate(dataset_names):
        paths = dataset_paths[ds_name]
        train_ds[ds_name] = JointAffinityDataset(
            csv_path=paths["train_csv"],
            adflip_root=cfg["adflip_root"],
            pdb_col=str(data_cfg.get("pdb_col", "pdb_abs")),
            y_col=str(data_cfg.get("y_col", "label_neglog_m")),
            seq_col=str(data_cfg.get("seq_col", "sequence")),
            chain_order=chain_order,
            parser_chain_id=parser_chain_id,
        )
        val_ds[ds_name] = JointAffinityDataset(
            csv_path=paths["val_csv"],
            adflip_root=cfg["adflip_root"],
            pdb_col=str(data_cfg.get("pdb_col", "pdb_abs")),
            y_col=str(data_cfg.get("y_col", "label_neglog_m")),
            seq_col=str(data_cfg.get("seq_col", "sequence")),
            chain_order=chain_order,
            parser_chain_id=parser_chain_id,
        )

        train_groups[ds_name] = _build_groups(
            n_items=len(train_ds[ds_name]),
            list_size=list_size,
            shuffle=True,
            seed=seed + 100 * (i + 1),
        )
        val_groups[ds_name] = _build_groups(
            n_items=len(val_ds[ds_name]),
            list_size=list_size,
            shuffle=False,
            seed=seed + 200 * (i + 1),
        )

    group_counts = {k: len(v) for k, v in train_groups.items()}
    probs = _sampling_probabilities(dataset_names=dataset_names, group_counts=group_counts, train_cfg=train_cfg)

    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 0))
    if steps_per_epoch <= 0:
        steps_per_epoch = int(sum(group_counts.values()))

    output_root = Path(str(train_cfg.get("output_root", str(DEFAULT_OUTPUT_ROOT)))).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"SWK_joint_{timestamp}"
    run_dir = output_root / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, load_info = _build_zoidberg_from_adflip_ckpt(
        adflip_ckpt=cfg["adflip_ckpt"],
        model_cfg=model_cfg,
        adflip_root=cfg["adflip_root"],
        device=device,
    )

    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    if freeze_backbone:
        model.backbone.eval()
        for p in model.backbone.parameters():
            p.requires_grad = False

    lr = float(train_cfg.get("learn_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    params = [{"params": model.affinity_head.parameters(), "lr": lr}]
    if not freeze_backbone:
        bb_lr = float(train_cfg.get("backbone_learn_rate", lr * 0.1))
        params.append({"params": model.backbone.parameters(), "lr": bb_lr})
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
    timestep = float(model_cfg.get("timestep", 0.0))

    cyclers = {
        ds_name: GroupCycler(groups=train_groups[ds_name], seed=seed + 1234 + i)
        for i, ds_name in enumerate(dataset_names)
    }

    sample_rng = random.Random(seed + 999)
    sampling_names = list(dataset_names)
    sampling_weights = [probs[n] for n in sampling_names]

    max_epochs = int(train_cfg.get("epochs", 20))
    early_stopping_enabled = bool(train_cfg.get("early_stopping_enabled", False))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))

    best_macro_spearman = -float("inf")
    best_epoch = -1
    epochs_without_improve = 0

    print(f"[Info] datasets={dataset_names}")
    print(f"[Info] train_group_counts={group_counts}")
    print(f"[Info] sampling_probs={probs}")
    print(f"[Info] steps_per_epoch={steps_per_epoch}")

    metric_fields = [
        "epoch",
        "rank_weight",
        "steps_per_epoch",
        "macro_val_spearman",
        "macro_val_pearson",
        "macro_val_loss",
    ]
    for ds_name in dataset_names:
        metric_fields.extend(
            [
                f"train_{ds_name}_loss",
                f"train_{ds_name}_spearman",
                f"val_{ds_name}_loss",
                f"val_{ds_name}_spearman",
                f"val_{ds_name}_pearson",
                f"val_{ds_name}_n_rows",
            ]
        )

    for epoch_idx in range(max_epochs):
        rank_weight = _current_rank_weight(epoch_idx_0based=epoch_idx, train_cfg=train_cfg)

        model.train()
        if freeze_backbone:
            model.backbone.eval()

        train_stats = {
            ds_name: {
                "loss": [],
                "spearman": [],
            }
            for ds_name in dataset_names
        }

        pbar = _create_progress_bar(total=steps_per_epoch, desc=f"Epoch {epoch_idx + 1}/{max_epochs}")
        for step_idx in range(steps_per_epoch):
            ds_name = sample_rng.choices(sampling_names, weights=sampling_weights, k=1)[0]
            group_indices = cyclers[ds_name].next_group()

            optimizer.zero_grad(set_to_none=True)
            out = _run_group(
                model=model,
                dataset=train_ds[ds_name],
                group_indices=group_indices,
                device=device,
                timestep=timestep,
                train_cfg=train_cfg,
                rank_weight=rank_weight,
            )
            out["loss"].backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_stats[ds_name]["loss"].append(float(out["metrics"]["loss"]))
            train_stats[ds_name]["spearman"].append(float(out["metrics"]["spearman"]))
            pbar.set_postfix(
                {
                    "step": f"{step_idx + 1}",
                    "ds": ds_name,
                    "loss": f"{float(out['metrics']['loss']):.4f}",
                }
            )
            pbar.update(1)
        pbar.close()

        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for ds_name in dataset_names:
                y_all = []
                yhat_all = []
                for group_indices in val_groups[ds_name]:
                    out = _run_group(
                        model=model,
                        dataset=val_ds[ds_name],
                        group_indices=group_indices,
                        device=device,
                        timestep=timestep,
                        train_cfg=train_cfg,
                        rank_weight=rank_weight,
                    )
                    y_all.append(out["y_true"])
                    yhat_all.append(out["y_pred"])

                y_vec = torch.cat(y_all, dim=0).to(device)
                y_hat_vec = torch.cat(yhat_all, dim=0).to(device)
                m = _compute_loss_and_metrics(
                    y_true=y_vec,
                    y_pred=y_hat_vec,
                    train_cfg=train_cfg,
                    rank_weight=rank_weight,
                )
                val_metrics[ds_name] = {
                    "loss": _float(m["loss"]),
                    "spearman": _float(m["spearman"]),
                    "pearson": _float(m["pearson"]),
                    "n_rows": int(y_vec.numel()),
                }

        macro_val_spearman = sum(val_metrics[n]["spearman"] for n in dataset_names) / float(len(dataset_names))
        macro_val_pearson = sum(val_metrics[n]["pearson"] for n in dataset_names) / float(len(dataset_names))
        macro_val_loss = sum(val_metrics[n]["loss"] for n in dataset_names) / float(len(dataset_names))

        metric_row = {
            "epoch": epoch_idx + 1,
            "rank_weight": rank_weight,
            "steps_per_epoch": steps_per_epoch,
            "macro_val_spearman": macro_val_spearman,
            "macro_val_pearson": macro_val_pearson,
            "macro_val_loss": macro_val_loss,
        }
        for ds_name in dataset_names:
            metric_row[f"train_{ds_name}_loss"] = _mean_or_nan(train_stats[ds_name]["loss"])
            metric_row[f"train_{ds_name}_spearman"] = _mean_or_nan(train_stats[ds_name]["spearman"])
            metric_row[f"val_{ds_name}_loss"] = val_metrics[ds_name]["loss"]
            metric_row[f"val_{ds_name}_spearman"] = val_metrics[ds_name]["spearman"]
            metric_row[f"val_{ds_name}_pearson"] = val_metrics[ds_name]["pearson"]
            metric_row[f"val_{ds_name}_n_rows"] = val_metrics[ds_name]["n_rows"]

        _append_metrics_csv(run_dir / "metrics.csv", metric_row, fieldnames=metric_fields)

        _save_checkpoint(
            ckpt_dir / "last.ckpt",
            model=model,
            optimizer=optimizer,
            epoch=epoch_idx + 1,
            best_macro_spearman=best_macro_spearman,
            cfg=cfg,
            extra={"metric_row": metric_row, "sampling_probs": probs, "load_info": load_info},
        )

        improved = macro_val_spearman > (best_macro_spearman + early_stopping_min_delta)
        if improved:
            best_macro_spearman = macro_val_spearman
            best_epoch = epoch_idx + 1
            epochs_without_improve = 0
            _save_checkpoint(
                ckpt_dir / "best_macro_spearman.ckpt",
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx + 1,
                best_macro_spearman=best_macro_spearman,
                cfg=cfg,
                extra={"metric_row": metric_row, "sampling_probs": probs, "load_info": load_info},
            )
        else:
            epochs_without_improve += 1

        print(
            "[Progress] "
            f"epoch={epoch_idx + 1}/{max_epochs} "
            f"macro_val_spearman={macro_val_spearman:.6f} "
            f"macro_val_loss={macro_val_loss:.6f} "
            f"best_macro_val_spearman={best_macro_spearman:.6f}"
        )

        if early_stopping_enabled and epochs_without_improve >= early_stopping_patience:
            print(
                "[EarlyStopping] "
                f"stopped at epoch={epoch_idx + 1}, "
                f"best_epoch={best_epoch}, "
                f"best_macro_spearman={best_macro_spearman:.6f}"
            )
            break

    run_meta = {
        "run_name": run_name,
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "device": str(device),
        "datasets": dataset_names,
        "train_group_counts": group_counts,
        "val_group_counts": {k: len(v) for k, v in val_groups.items()},
        "sampling_probs": probs,
        "steps_per_epoch": steps_per_epoch,
        "epochs_configured": max_epochs,
        "best_epoch": best_epoch,
        "best_macro_spearman": best_macro_spearman,
        "best_ckpt": str((ckpt_dir / "best_macro_spearman.ckpt").resolve()),
        "last_ckpt": str((ckpt_dir / "last.ckpt").resolve()),
        "load_info": load_info,
    }

    curve_status = _save_training_curves(
        metrics_csv=run_dir / "metrics.csv",
        out_png=run_dir / "training_curves.png",
    )
    run_meta["training_curves_png"] = (
        str((run_dir / "training_curves.png").resolve())
        if (run_dir / "training_curves.png").is_file()
        else ""
    )
    run_meta["curve_status"] = curve_status

    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    if curve_status == "ok":
        print(f"[Info] saved training curves: {run_dir / 'training_curves.png'}")
    else:
        print(f"[Warn] training curves not generated: {curve_status}")
    print(f"Saved run directory: {run_dir}")


if __name__ == "__main__":
    main()
