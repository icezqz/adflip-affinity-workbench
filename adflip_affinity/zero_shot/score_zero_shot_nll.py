import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "outputs"


class Config:
    """
    Compatibility shim for loading legacy checkpoints that pickle Config
    under __main__.
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


def _ensure_importable(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


@contextlib.contextmanager
def _temporary_cwd(target_dir: str):
    prev = os.getcwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(prev)


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

    manifest_df = pd.read_csv(manifest_path)
    matched = manifest_df[manifest_df["dataset_name"].astype(str) == dataset_name].reset_index(drop=True)
    if len(matched) != 1:
        raise ValueError(f"Dataset '{dataset_name}' not found uniquely in manifest: {manifest_path}")
    row = matched.iloc[0]
    cfg["data"]["train_csv"] = str(row["train_csv"])
    cfg["data"]["val_csv"] = str(row["val_csv"])
    cfg["data"]["test_csv"] = str(row["test_csv"])
    cfg["data"]["all_csv"] = str(row["all_csv"]) if "all_csv" in row else ""
    cfg["data"]["pdb_dir"] = str(row["structure_dir"])


def _resolve_eval_csv(cfg: dict, eval_split: str, eval_csv_override: str) -> str:
    if eval_csv_override:
        return eval_csv_override
    if eval_split == "all":
        all_csv = str(cfg.get("data", {}).get("all_csv", "")).strip()
        if not all_csv:
            raise ValueError("eval_split=all requires data.all_csv or --eval_csv.")
        return all_csv
    if eval_split == "train":
        return cfg["data"]["train_csv"]
    if eval_split == "val":
        return cfg["data"]["val_csv"]
    return cfg["data"]["test_csv"]


def _label_transform(y_raw: float, mode: str):
    if y_raw is None or (isinstance(y_raw, float) and not np.isfinite(y_raw)):
        return np.nan
    y = float(y_raw)
    if mode == "raw":
        return y
    if y <= 0:
        return np.nan
    if mode == "log10":
        return math.log10(y)
    if mode == "neg_log10":
        return -math.log10(y)
    raise ValueError(f"Unknown label_mode: {mode}")


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    y = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    return _pearson_corr(x, y)


def _top_hit_ratio(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]
    n = y_true.size
    if n == 0:
        return float("nan")
    kk = max(1, min(int(k), n))
    true_top = set(np.argsort(-y_true)[:kk].tolist())
    pred_top = set(np.argsort(-score)[:kk].tolist())
    return float(len(true_top & pred_top) / kk)


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


def _save_neglogkd_vs_nll_plot(df: pd.DataFrame, out_path: str):
    if "y_true_neg_log_kd" not in df.columns or "nll" not in df.columns:
        return
    plot_df = df.dropna(subset=["y_true_neg_log_kd", "nll"]).copy()
    if len(plot_df) < 2:
        return
    plot_df = plot_df[np.isfinite(plot_df["y_true_neg_log_kd"]) & np.isfinite(plot_df["nll"])]
    if len(plot_df) < 2:
        return

    import matplotlib.pyplot as plt

    x = plot_df["y_true_neg_log_kd"].to_numpy(dtype=np.float64)
    y = plot_df["nll"].to_numpy(dtype=np.float64)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.75, s=28)
    if len(plot_df) >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coeff[0] * x_line + coeff[1]
        plt.plot(x_line, y_line, linewidth=1.2)
    plt.xlabel("True -log10(KD)")
    plt.ylabel("NLL")
    plt.title("Zero-shot NLL vs True -log10(KD)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _resolve_pdb_path(pdb_key: str, pdb_dir: str) -> str:
    key = str(pdb_key).strip()
    if os.path.isabs(key) and os.path.isfile(key):
        return key
    fname = key
    if not fname.endswith((".pdb", ".cif", ".mmcif")):
        fname = fname + ".pdb"
    return os.path.join(pdb_dir, fname)


def _chain_order_in_structure(aap, pdb_path: str) -> list[str]:
    ag = aap.parse_structure(pdb_path)
    atoms = ag.select("not water and not hydrogen")
    if atoms is None:
        return []
    order = []
    for atom in atoms:
        ch = atom.getChid()
        if not ch:
            ch = "?"
        if len(ch) > 1:
            ch = ch[0]
        if ch not in order:
            order.append(ch)
    return order


def _override_sequence(aap, struct_data, internal_chain_id: int, seq: str):
    seq = (seq or "").strip()
    if not seq:
        return struct_data

    is_chain = struct_data.chain_id == internal_chain_id
    center_mask = is_chain & struct_data.is_center & struct_data.is_protein
    if center_mask.sum() == 0:
        return struct_data

    res_ids = np.unique(struct_data.residue_index[center_mask])
    max_len = min(len(res_ids), len(seq))
    for i in range(max_len):
        aa1 = seq[i].upper()
        aa3 = aap.restype_1to3.get(aa1, "<UNK>")
        tok = aap.get_token_index(aa3)
        rid = res_ids[i]
        struct_data.residue_token[struct_data.residue_index == rid] = tok
    return struct_data


def _build_batch_dict(aap, pdb_path: str, chains: list[str], sequence: str):
    struct_data = aap.parse_mmcif_to_structure_data(pdb_path, parser_chain_id=chains)

    if sequence:
        sep = "\\" if "\\" in sequence else "/"
        seqs = sequence.split(sep)
        target_chains = [c for c in chains if c != "A"] or list(chains)
        encountered = [c for c in _chain_order_in_structure(aap, pdb_path) if c in chains]
        if not encountered:
            encountered = list(chains)
        chain_map = {c: i for i, c in enumerate(encountered)}
        for i, seq in enumerate(seqs):
            if i >= len(target_chains):
                break
            chain_letter = target_chains[i]
            if chain_letter not in chain_map:
                continue
            struct_data = _override_sequence(aap, struct_data, chain_map[chain_letter], seq)

    batch_dict = {}
    for k, v in struct_data.__dict__.items():
        if isinstance(v, np.ndarray):
            batch_dict[k] = torch.from_numpy(v).unsqueeze(0)
        else:
            batch_dict[k] = torch.tensor([v]).unsqueeze(0)
    batch_dict["batch_index"] = torch.zeros_like(batch_dict["residue_index"])
    for k, v in list(batch_dict.items()):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            batch_dict[k] = v.float()
    return batch_dict


def _build_backbone_model(cfg: dict, device: torch.device):
    adflip_root = cfg["adflip_root"]
    _ensure_importable(adflip_root)
    from model.zoidberg.zoidberg_GNN import Zoidberg_GNN  # noqa: WPS433

    _register_legacy_config_for_torch_load()
    ckpt = torch.load(cfg["adflip_ckpt"], map_location="cpu")
    zcfg = ckpt["config"].zoidberg_denoiser
    model = Zoidberg_GNN(
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
    ).to(device)
    z_sd = {k[len("model.") :]: v for k, v in ckpt["model"].items() if k.startswith("model.")}
    model.load_state_dict(z_sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def _score_one(model, batch_dict: dict, timestep: float, device: torch.device):
    batch_dict = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch_dict.items()}
    t = torch.full((1, 1), float(timestep), device=device)
    logits, _ = model(batch_dict, t)
    target = batch_dict["residue_token"][batch_dict["is_center"] & batch_dict["is_protein"]].long()
    nll = F.cross_entropy(logits, target, reduction="mean")
    nll_value = float(nll.detach().cpu().item())
    return {
        "nll": nll_value,
        "ppl": float(math.exp(nll_value)),
        "n_tokens": int(target.numel()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval_split", type=str, default="", choices=["", "all", "train", "val", "test"])
    parser.add_argument("--eval_csv", type=str, default="")
    parser.add_argument("--pdb_dir", type=str, default="")
    parser.add_argument("--pdb_col", type=str, default="")
    parser.add_argument("--seq_col", type=str, default="")
    parser.add_argument("--y_col", type=str, default="")
    parser.add_argument("--label_mode", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--metrics_json", type=str, default="")
    parser.add_argument("--dataset_name_override", type=str, default="")
    parser.add_argument("--cuda_visible_devices", type=str, default="")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    _hydrate_data_paths_from_manifest(cfg)

    eval_cfg = cfg.get("evaluation", {})
    eval_split = args.eval_split if args.eval_split else str(eval_cfg.get("eval_split", "test"))
    eval_csv = _resolve_eval_csv(cfg, eval_split, args.eval_csv)
    pdb_dir = args.pdb_dir if args.pdb_dir else cfg["data"]["pdb_dir"]
    chains = list(cfg["data"].get("chains", ["H", "L"]))
    timestep = float(cfg.get("model", {}).get("timestep", 0.0))

    df = pd.read_csv(eval_csv).reset_index(drop=True)

    adflip_root = cfg["adflip_root"]
    _ensure_importable(adflip_root)
    with _temporary_cwd(adflip_root):
        from data import all_atom_parse as aap  # noqa: WPS433

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_backbone_model(cfg, device)

    pdb_col = str(args.pdb_col or cfg["data"].get("pdb_col", "pdb"))
    seq_col = str(args.seq_col or cfg["data"].get("seq_col", "sequence"))
    y_col = str(args.y_col or cfg["data"].get("y_col", "KD"))
    label_mode = str(args.label_mode or cfg["data"].get("label_mode", "neg_log10"))
    topk = int(eval_cfg.get("topk", 5))

    rows = []
    total_rows = len(df)
    if total_rows == 0:
        print("Scoring samples: no rows in evaluation CSV.")
    else:
        _print_progress("Scoring samples", 0, total_rows)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        pdb_path = _resolve_pdb_path(row[pdb_col], pdb_dir)
        if not os.path.isfile(pdb_path):
            _print_progress("Scoring samples", idx, total_rows)
            continue
        sequence = ""
        if seq_col in df.columns and not pd.isna(row[seq_col]):
            sequence = str(row[seq_col]).strip()
        batch_dict = _build_batch_dict(aap, pdb_path=pdb_path, chains=chains, sequence=sequence)
        score = _score_one(model, batch_dict=batch_dict, timestep=timestep, device=device)

        y_raw = np.nan
        y_label = np.nan
        y_true_neg_log_kd = np.nan
        if y_col in df.columns:
            y_num = pd.to_numeric(row[y_col], errors="coerce")
            if not pd.isna(y_num):
                y_raw = float(y_num)
                y_label = float(_label_transform(y_raw, label_mode))
                if label_mode == "raw" and ("neglog" in y_col.lower() or "negative_log_kd" in y_col.lower()):
                    y_true_neg_log_kd = y_raw
                elif y_raw > 0:
                    y_true_neg_log_kd = float(-math.log10(y_raw))

        sample_name = str(row["name"]) if "name" in df.columns else ""
        rows.append(
            {
                "name": sample_name,
                "pdb": str(row[pdb_col]),
                "n_tokens": score["n_tokens"],
                "nll": score["nll"],
                "ppl": score["ppl"],
                "score": -score["nll"],
                "y_true_raw": y_raw,
                "y_true_label": y_label,
                "y_true_neg_log_kd": y_true_neg_log_kd,
            }
        )
        _print_progress("Scoring samples", idx, total_rows)

    out_df = pd.DataFrame(rows)
    output_dir = str(eval_cfg.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
    dataset_name = str(cfg.get("data", {}).get("dataset_name", "dataset"))
    if args.dataset_name_override:
        dataset_name = str(args.dataset_name_override)
    elif args.eval_csv:
        inferred = os.path.basename(os.path.dirname(os.path.abspath(eval_csv))).strip()
        if inferred:
            dataset_name = inferred
    tag = os.path.splitext(os.path.basename(eval_csv))[0]
    if args.output_csv:
        out_csv = args.output_csv
    else:
        out_csv = os.path.join(output_dir, "scores", f"{dataset_name}_{tag}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    plot_png = out_csv.replace(".csv", ".neglogkd_vs_nll.png")
    _save_neglogkd_vs_nll_plot(out_df, plot_png)

    labeled = out_df.dropna(subset=["y_true_label", "score"]).copy()
    metrics = {
        "n_rows_input": int(len(df)),
        "n_rows_scored": int(len(out_df)),
        "n_rows_labeled": int(len(labeled)),
        "avg_nll": float(out_df["nll"].mean()) if len(out_df) > 0 else float("nan"),
        "avg_ppl": float(out_df["ppl"].mean()) if len(out_df) > 0 else float("nan"),
        "spearman": float("nan"),
        "pearson": float("nan"),
        "top_hit_overall": float("nan"),
    }
    if len(labeled) >= 2:
        y_true = labeled["y_true_label"].to_numpy(dtype=np.float64)
        score = labeled["score"].to_numpy(dtype=np.float64)
        metrics["spearman"] = _spearman_corr(score, y_true)
        metrics["pearson"] = _pearson_corr(score, y_true)
        metrics["top_hit_overall"] = _top_hit_ratio(y_true, score, topk)

    if args.metrics_json:
        metrics_json = args.metrics_json
    else:
        metrics_json = out_csv.replace(".csv", ".metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "adflip_ckpt": os.path.abspath(cfg["adflip_ckpt"]),
                "eval_split": eval_split,
                "eval_csv": os.path.abspath(eval_csv),
                "pdb_dir": os.path.abspath(pdb_dir),
                "chains": chains,
                "label_mode": label_mode,
                "topk": topk,
                "metrics": metrics,
                "output_csv": os.path.abspath(out_csv),
                "plot_png": os.path.abspath(plot_png),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved scores to: {out_csv}")
    print(f"Saved metrics to: {metrics_json}")
    print(f"Saved plot to: {plot_png}")


if __name__ == "__main__":
    main()
