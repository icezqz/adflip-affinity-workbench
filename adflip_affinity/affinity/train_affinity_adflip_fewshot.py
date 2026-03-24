import argparse
import contextlib
import datetime
import json
import os
import random
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value


def _ensure_adflip_importable(adflip_root: str):
    if adflip_root not in sys.path:
        sys.path.insert(0, adflip_root)


class GroupedAffinityDataset(Dataset):
    def __init__(self, base_ds, grouped_indices: list[list[int]]):
        self.base_ds = base_ds
        self.grouped_indices = grouped_indices

    def __len__(self):
        return len(self.grouped_indices)

    def __getitem__(self, idx):
        sample_indices = self.grouped_indices[idx]
        return [self.base_ds[i] for i in sample_indices]


def collate_affinity_group(batch):
    assert len(batch) == 1
    return batch[0]


def _sample_fewshot_indices(indices: list[int], cfg: dict) -> list[int]:
    enabled = bool(cfg.get("few_shot_enabled", False))
    if not enabled:
        return indices

    rng = random.Random(int(cfg.get("few_shot_seed", 42)))
    sampled = list(indices)
    rng.shuffle(sampled)

    frac = float(cfg.get("few_shot_frac", 1.0))
    if frac < 1.0:
        keep = max(1, int(round(len(sampled) * frac)))
        sampled = sampled[:keep]

    n_samples = int(cfg.get("few_shot_n_samples", cfg.get("few_shot_n_groups", 0)))
    if n_samples > 0:
        sampled = sampled[: min(n_samples, len(sampled))]

    return sampled


def _build_units_from_indices(
    indices: list[int],
    unit_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[int]]:
    if unit_size < 2:
        raise ValueError("training.list_size must be >= 2 to make KL meaningful.")

    ordered = list(indices)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)

    units = []
    for i in range(0, len(ordered), unit_size):
        chunk = ordered[i : i + unit_size]
        if len(chunk) >= 2:
            units.append(chunk)
    if not units:
        raise ValueError("No valid training units built. Need at least 2 samples.")
    return units


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    x_center = x - x.mean()
    y_center = y - y.mean()
    denom = torch.sqrt((x_center.square().sum() + 1e-12) * (y_center.square().sum() + 1e-12))
    return (x_center * y_center).sum() / denom


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.zeros_like(x, dtype=torch.float32)
    ranks[order] = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float32)
    return ranks


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() < 2:
        return torch.tensor(0.0, device=x.device)
    return _pearson_corr(_rankdata(x.reshape(-1)), _rankdata(y.reshape(-1)))


def _top_hit_ratio(y_true: torch.Tensor, y_pred: torch.Tensor, k: int) -> torch.Tensor:
    n = int(y_true.numel())
    if n == 0:
        return torch.tensor(0.0, device=y_true.device)
    kk = max(1, min(int(k), n))
    true_top = torch.topk(y_true.reshape(-1), k=kk).indices
    pred_top = torch.topk(y_pred.reshape(-1), k=kk).indices
    return torch.isin(pred_top, true_top).float().mean()


def _pairwise_logistic_rank_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    margin: float = 0.0,
    tie_eps: float = 1e-8,
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


@contextlib.contextmanager
def _redirect_stdout_stderr(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield


def _save_training_curves(log_dir: str):
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        return
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(metrics_path)
    if "epoch" not in df.columns:
        return

    df = df.sort_values("step")
    # Lightning writes train/val rows separately in the same epoch; keep the last non-null
    # per metric to avoid dropping val_* fields when the final row is train-only.
    def _last_non_null(series):
        s = series.dropna()
        return s.iloc[-1] if len(s) > 0 else float("nan")

    agg_cols = {c: _last_non_null for c in df.columns if c != "epoch"}
    df = df.groupby("epoch", as_index=False).agg(agg_cols)
    specs = [
        ("train_loss", "val_loss", "Loss"),
        ("train_kl", "val_kl", "KL"),
        ("train_rmse", "val_rmse", "RMSE"),
        ("train_mse", "val_mse", "MSE"),
        ("train_rank_loss", "val_rank_loss", "RankLoss"),
        ("train_spearman", "val_spearman", "Spearman"),
        ("train_pearson", "val_pearson", "Pearson"),
        ("train_top_hit", "val_top_hit", "Top-Hit"),
        ("train_rank_weight", "val_rank_weight", "RankWeight"),
    ]
    specs = [x for x in specs if x[0] in df.columns or x[1] in df.columns]
    if not specs:
        return

    fig, axes = plt.subplots(len(specs), 1, figsize=(7, 3 * len(specs)), sharex=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, (train_col, val_col, title) in zip(axes, specs):
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], label=train_col)
        if val_col in df.columns:
            ax.plot(df["epoch"], df[val_col], label=val_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def build_zoidberg_from_adflip_ckpt(ckpt_path: str, device: torch.device, model_cfg: dict, adflip_root: str):
    _ensure_adflip_importable(adflip_root)
    from model.zoidberg.zoidberg_GNN_affinity import Zoidberg_GNN_Affinity  # noqa: WPS433

    ckpt = torch.load(ckpt_path, map_location="cpu")
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


class ADFLIPAffinityPL(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = self.cfg.get("model", {})
        data_cfg = self.cfg.get("data", {})
        train_cfg = self.cfg.get("training", {})
        self.save_hyperparameters(
            {
                "dataset_name": str(data_cfg.get("dataset_name", "")),
                "epochs": int(train_cfg.get("epochs", 20)),
                "batch_size": int(train_cfg.get("batch_size", 1)),
                "list_size": int(train_cfg.get("list_size", 8)),
                "learn_rate": float(train_cfg.get("learn_rate", 1e-3)),
                "backbone_learn_rate": float(train_cfg.get("backbone_learn_rate", 1e-5)),
                "tau_true": float(train_cfg.get("tau_true", 1.0)),
                "tau_pred": float(train_cfg.get("tau_pred", 1.0)),
                "kl_weight": float(train_cfg.get("kl_weight", 1.0)),
                "mse_weight": float(train_cfg.get("mse_weight", 0.1)),
                "rank_weight": float(train_cfg.get("rank_weight", 0.0)),
                "rank_margin": float(train_cfg.get("rank_margin", 0.0)),
                "rank_tie_eps": float(train_cfg.get("rank_tie_eps", 1e-8)),
                "rank_warmup_epochs": int(train_cfg.get("rank_warmup_epochs", 0)),
                "topk": int(train_cfg.get("topk", 10)),
                "freeze_backbone": bool(model_cfg.get("freeze_backbone", True)),
                "early_stopping_enabled": bool(train_cfg.get("early_stopping_enabled", False)),
                "early_stopping_monitor": str(train_cfg.get("early_stopping_monitor", "val_loss")),
                "early_stopping_mode": str(train_cfg.get("early_stopping_mode", "min")),
                "early_stopping_patience": int(train_cfg.get("early_stopping_patience", 5)),
                "early_stopping_min_delta": float(train_cfg.get("early_stopping_min_delta", 0.0)),
            }
        )

        self.model, self.load_info = build_zoidberg_from_adflip_ckpt(
            ckpt_path=self.cfg["adflip_ckpt"],
            device=torch.device("cpu"),
            model_cfg=model_cfg,
            adflip_root=self.cfg["adflip_root"],
        )
        self.tau_true = float(train_cfg.get("tau_true", 1.0))
        self.tau_pred = float(train_cfg.get("tau_pred", 1.0))
        self.kl_weight = float(train_cfg.get("kl_weight", 1.0))
        self.mse_weight = float(train_cfg.get("mse_weight", 0.1))
        self.rank_weight = float(train_cfg.get("rank_weight", 0.0))
        self.rank_margin = float(train_cfg.get("rank_margin", 0.0))
        self.rank_tie_eps = float(train_cfg.get("rank_tie_eps", 1e-8))
        self.rank_warmup_epochs = int(train_cfg.get("rank_warmup_epochs", 0))
        self.topk = int(train_cfg.get("topk", 10))

    def on_fit_start(self):
        self.model = self.model.to(self.device)
        if bool(self.cfg["model"].get("freeze_backbone", True)):
            self.model.backbone.eval()
            for p in self.model.backbone.parameters():
                p.requires_grad = False

    def forward(self, batch_dict):
        t = float(self.cfg["model"].get("timestep", 0.0))
        timestep = torch.full((1, 1), t, device=self.device)
        _, _, affinity = self.model(batch_dict, timestep, return_affinity=True)
        return affinity

    def _current_rank_weight(self) -> float:
        if self.rank_weight <= 0.0:
            return 0.0
        warmup = max(0, int(self.rank_warmup_epochs))
        if warmup == 0:
            return self.rank_weight
        factor = min(1.0, float(self.current_epoch + 1) / float(warmup))
        return self.rank_weight * factor

    def _step(self, group_batch, stage: str):
        y_list = []
        y_hat_list = []
        for sample in group_batch:
            batch_dict = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in sample["batch_dict"].items()
            }
            y = sample["y"].to(self.device).reshape(-1)
            y_hat = self.forward(batch_dict).reshape(-1)
            y_list.append(y)
            y_hat_list.append(y_hat)

        y_vec = torch.cat(y_list, dim=0)
        y_hat_vec = torch.cat(y_hat_list, dim=0)

        p = torch.softmax(y_vec / self.tau_true, dim=0)
        log_q = torch.log_softmax(y_hat_vec / self.tau_pred, dim=0)
        kl = F.kl_div(log_q, p, reduction="batchmean")
        mse = F.mse_loss(y_hat_vec, y_vec)
        rank_loss = _pairwise_logistic_rank_loss(
            y_true=y_vec,
            y_pred=y_hat_vec,
            margin=self.rank_margin,
            tie_eps=self.rank_tie_eps,
        )
        rank_weight = self._current_rank_weight()
        rmse = torch.sqrt(mse + 1e-12)
        spearman = _spearman_corr(y_hat_vec, y_vec)
        pearson = _pearson_corr(y_hat_vec, y_vec)
        top_hit = _top_hit_ratio(y_vec, y_hat_vec, self.topk)
        loss = self.kl_weight * kl + self.mse_weight * mse + rank_weight * rank_loss

        self.log(f"{stage}_loss", loss, prog_bar=(stage != "test"), on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_kl", kl, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_mse", mse, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_rank_loss", rank_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_rank_weight", float(rank_weight), prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_spearman", spearman, prog_bar=(stage != "test"), on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_pearson", pearson, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self.log(f"{stage}_top_hit", top_hit, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        lr = float(self.cfg["training"].get("learn_rate", 1e-3))
        params = [{"params": self.model.affinity_head.parameters(), "lr": lr}]
        if not bool(self.cfg["model"].get("freeze_backbone", True)):
            bb_lr = float(self.cfg["training"].get("backbone_learn_rate", lr * 0.1))
            params.append({"params": self.model.backbone.parameters(), "lr": bb_lr})
        return torch.optim.AdamW(params, lr=lr)


class EpochProgressCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_idx = int(trainer.current_epoch) + 1
        max_epochs = int(trainer.max_epochs)
        m = trainer.callback_metrics
        train_loss = float(m.get("train_loss", float("nan")))
        val_loss = float(m.get("val_loss", float("nan")))
        train_s = float(m.get("train_spearman", float("nan")))
        val_s = float(m.get("val_spearman", float("nan")))
        print(
            "[Progress] "
            f"epoch={epoch_idx}/{max_epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_spearman={train_s:.6f} "
            f"val_spearman={val_s:.6f}"
        )


def _build_grouped_loader(ds, grouped_units: list[list[int]], train_cfg: dict, is_train: bool):
    grouped_ds = GroupedAffinityDataset(ds, grouped_units)
    batch_size = int(train_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("Current listwise collate expects training.batch_size=1.")
    return DataLoader(
        grouped_ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_affinity_group,
    ), len(grouped_units)


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
    cfg["data"]["pdb_dir"] = str(row["structure_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    _hydrate_data_paths_from_manifest(cfg)
    seed = int(cfg.get("training", {}).get("seed", 42))
    pl.seed_everything(seed, workers=True)

    from adflip_affinity_dataset import ADFLIPAffinityDataset  # local file

    chain_order = list(cfg["data"].get("chains", ["A", "H", "L"]))
    common_kwargs = dict(
        pdb_dir=cfg["data"]["pdb_dir"],
        adflip_root=cfg["adflip_root"],
        pdb_col=cfg["data"].get("pdb_col", "pdb"),
        y_col=cfg["data"].get("y_col", "KD"),
        seq_col=cfg["data"].get("seq_col", "sequence"),
        chain_order=chain_order,
        label_mode=cfg["data"].get("label_mode", "log10"),
        parser_chain_id=chain_order,
    )
    train_ds = ADFLIPAffinityDataset(csv_path=cfg["data"]["train_csv"], **common_kwargs)
    val_ds = ADFLIPAffinityDataset(csv_path=cfg["data"]["val_csv"], **common_kwargs)

    train_cfg = cfg.get("training", {})
    list_size = int(train_cfg.get("list_size", 8))
    seed = int(train_cfg.get("seed", 42))

    train_indices = list(range(len(train_ds)))
    train_indices = _sample_fewshot_indices(train_indices, train_cfg)
    val_indices = list(range(len(val_ds)))

    train_units = _build_units_from_indices(train_indices, unit_size=list_size, shuffle=True, seed=seed)
    val_units = _build_units_from_indices(val_indices, unit_size=list_size, shuffle=False, seed=seed)

    train_loader, n_train_groups = _build_grouped_loader(
        train_ds, train_units, train_cfg, is_train=True
    )
    val_loader, n_val_groups = _build_grouped_loader(
        val_ds, val_units, train_cfg, is_train=False
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = cfg.get("data", {}).get("dataset_name", "dataset")
    run_name = f"{dataset_name}_{timestamp}"

    ckpt = ModelCheckpoint(
        dirpath=train_cfg.get("ckpt_dir", "affinity_checkpoints"),
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.4f}",
        monitor=str(train_cfg.get("checkpoint_monitor", "val_loss")),
        mode=str(train_cfg.get("checkpoint_mode", "min")),
        save_top_k=1,
    )
    logger = CSVLogger(
        save_dir=train_cfg.get("log_dir", "affinity_logs"),
        name=run_name,
    )

    callbacks = [ckpt, EpochProgressCallback()]
    if bool(train_cfg.get("early_stopping_enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(train_cfg.get("early_stopping_monitor", train_cfg.get("checkpoint_monitor", "val_loss"))),
                mode=str(train_cfg.get("early_stopping_mode", train_cfg.get("checkpoint_mode", "min"))),
                patience=int(train_cfg.get("early_stopping_patience", 5)),
                min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
                verbose=True,
            )
        )

    model = ADFLIPAffinityPL(cfg)
    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=int(train_cfg.get("epochs", 20)),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=True,
    )

    silent = bool(train_cfg.get("silent", True))
    plot_curves = bool(train_cfg.get("plot_curves", True))
    meta = {
        "dataset_name": dataset_name,
        "list_size": list_size,
        "batch_size": int(train_cfg.get("batch_size", 1)),
        "train_samples_after_fewshot": len(train_indices),
        "train_groups_after_fewshot": n_train_groups,
        "val_groups": n_val_groups,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "config_path": os.path.abspath(args.config),
    }

    if silent:
        os.makedirs(logger.log_dir, exist_ok=True)
        run_log_path = os.path.join(logger.log_dir, "run.log")
        with _redirect_stdout_stderr(run_log_path):
            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(model, train_loader, val_loader)
            best_model = ckpt.best_model_path if ckpt.best_model_path else None
            if plot_curves:
                _save_training_curves(logger.log_dir)
    else:
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, train_loader, val_loader)
        best_model = ckpt.best_model_path if ckpt.best_model_path else None
        if plot_curves:
            _save_training_curves(logger.log_dir)

    meta["best_model_path"] = best_model
    with open(os.path.join(logger.log_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
