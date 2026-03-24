#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import math
import os
import random
from pathlib import Path

DEFAULT_DATASETS = [
    "Shanehsazzadeh2023_trastuzumab_zero_kd",
    "Warszawski2019_d44_Kd",
    "Koenig2017_g6_Kd",
]

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
AFFINITY_DIR = PROJECT_DIR.parent / "affinity"
DEFAULT_SOURCE_MANIFEST = AFFINITY_DIR / "outputs" / "prepared" / "manifest.csv"
DEFAULT_OUT_DATA_ROOT = PROJECT_DIR / "data"

# FLAb binding README indicates unit mismatch for these three sets:
# S: -log10(Kd[nM]) ; W/K: -log10(Kd[M]).
# Convert all labels to -log10(Kd[M]).
UNIT_BY_DATASET = {
    "Shanehsazzadeh2023_trastuzumab_zero_kd": "nM",
    "Warszawski2019_d44_Kd": "M",
    "Koenig2017_g6_Kd": "M",
}

ALL_FIELDS = [
    "dataset_name",
    "name",
    "pdb",
    "pdb_abs",
    "sequence",
    "negative_log_kd_raw",
    "label_neglog_m",
    "KD_M",
    "unit_source",
    "label_shift",
    "source_row_index",
]


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_pdb_abs(pdb_key: str, structure_dir: str) -> str:
    key = str(pdb_key).strip()
    if not key:
        raise ValueError("Empty pdb key.")

    if os.path.isabs(key):
        return key

    filename = key
    if not (filename.endswith(".pdb") or filename.endswith(".cif") or filename.endswith(".mmcif")):
        filename += ".pdb"
    return str((Path(structure_dir) / filename).resolve())


def _to_neglog_m(neg_log_kd: float, unit: str) -> float:
    unit = unit.strip()
    if unit == "M":
        return float(neg_log_kd)
    if unit == "nM":
        return float(neg_log_kd) + 9.0
    raise ValueError(f"Unsupported unit: {unit}")


def _find_manifest_row(manifest_rows: list[dict], dataset_name: str) -> dict:
    matched = [r for r in manifest_rows if str(r.get("dataset_name", "")).strip() == dataset_name]
    if len(matched) != 1:
        raise ValueError(f"Dataset '{dataset_name}' not found uniquely in manifest.")
    return matched[0]


def _split_sizes(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 samples to split train/val/test, got {n}")

    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    n_test = n - n_train - n_val

    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1

    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    return n_train, n_val, n_test


def _split_rows(rows: list[dict], seed: int, train_ratio: float, val_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    n_train, n_val, n_test = _split_sizes(n, train_ratio=train_ratio, val_ratio=val_ratio)

    ids = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(ids)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val : n_train + n_val + n_test])

    train_rows = []
    val_rows = []
    test_rows = []
    for i, row in enumerate(rows):
        if i in train_ids:
            r = dict(row)
            r["split"] = "train"
            train_rows.append(r)
        elif i in val_ids:
            r = dict(row)
            r["split"] = "val"
            val_rows.append(r)
        elif i in test_ids:
            r = dict(row)
            r["split"] = "test"
            test_rows.append(r)

    return train_rows, val_rows, test_rows


def _discover_local_datasets(out_data_root: Path) -> list[str]:
    names = []
    for p in sorted(out_data_root.iterdir()) if out_data_root.exists() else []:
        if not p.is_dir():
            continue
        if p.name == "prepared":
            continue
        if (p / "all.csv").is_file():
            names.append(p.name)
    return names


def _from_existing_all_csv(dataset_name: str, all_csv_path: Path) -> list[dict]:
    rows = _read_csv_rows(all_csv_path)
    if not rows:
        raise ValueError(f"{all_csv_path}: empty all.csv")

    required = ["pdb_abs", "label_neglog_m"]
    for col in required:
        if col not in rows[0]:
            raise ValueError(f"{all_csv_path}: missing required column '{col}'")

    out_rows = []
    for i, r in enumerate(rows):
        label_neglog_m = float(r["label_neglog_m"])
        unit_source = str(r.get("unit_source", UNIT_BY_DATASET.get(dataset_name, "M"))).strip() or "M"
        label_shift = str(r.get("label_shift", "9.0" if unit_source == "nM" else "0.0")).strip()
        kd_m = float(r.get("KD_M", 10 ** (-label_neglog_m)))
        neg_raw = r.get("negative_log_kd_raw")
        if neg_raw is None or str(neg_raw).strip() == "":
            neg_raw = label_neglog_m - 9.0 if unit_source == "nM" else label_neglog_m

        pdb_abs = str(r.get("pdb_abs", "")).strip()
        if not pdb_abs:
            raise ValueError(f"{all_csv_path}: row {i} has empty pdb_abs")
        if not Path(pdb_abs).is_file():
            raise FileNotFoundError(f"PDB missing for {dataset_name}: {pdb_abs}")

        out_rows.append(
            {
                "dataset_name": dataset_name,
                "name": str(r.get("name", "")).strip(),
                "pdb": str(r.get("pdb", "")).strip(),
                "pdb_abs": pdb_abs,
                "sequence": str(r.get("sequence", "")).strip(),
                "negative_log_kd_raw": f"{float(neg_raw):.12g}",
                "label_neglog_m": f"{label_neglog_m:.12g}",
                "KD_M": f"{kd_m:.12g}",
                "unit_source": unit_source,
                "label_shift": label_shift,
                "source_row_index": str(r.get("source_row_index", i)),
            }
        )
    return out_rows


def _from_source_manifest(dataset_name: str, manifest_rows: list[dict], unit: str) -> tuple[list[dict], str]:
    row = _find_manifest_row(manifest_rows, dataset_name)
    structure_dir = str(row["structure_dir"])
    all_src = Path(str(row["all_csv"])).resolve()
    all_rows = _read_csv_rows(all_src)

    processed_all = []
    for r in all_rows:
        neg_log_kd = float(r["negative_log_kd"])
        neg_log_m = _to_neglog_m(neg_log_kd, unit=unit)
        kd_m = 10 ** (-neg_log_m)

        out_r = {
            "dataset_name": dataset_name,
            "name": str(r.get("name", "")).strip(),
            "pdb": str(r.get("pdb", "")).strip(),
            "pdb_abs": _ensure_pdb_abs(r.get("pdb", ""), structure_dir=structure_dir),
            "sequence": str(r.get("sequence", "")).strip(),
            "negative_log_kd_raw": f"{neg_log_kd:.12g}",
            "label_neglog_m": f"{neg_log_m:.12g}",
            "KD_M": f"{kd_m:.12g}",
            "unit_source": unit,
            "label_shift": "9.0" if unit == "nM" else "0.0",
            "source_row_index": str(r.get("source_row_index", "")),
        }
        if not Path(out_r["pdb_abs"]).is_file():
            raise FileNotFoundError(f"PDB missing for {dataset_name}: {out_r['pdb_abs']}")
        processed_all.append(out_r)

    if not processed_all:
        raise ValueError(f"No valid rows after processing dataset: {dataset_name}")
    return processed_all, structure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_manifest",
        type=str,
        default=str(DEFAULT_SOURCE_MANIFEST),
    )
    parser.add_argument(
        "--out_data_root",
        type=str,
        default=str(DEFAULT_OUT_DATA_ROOT),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Dataset names. If omitted, use local data/*/all.csv; fallback to default S/W/K.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be > 0")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    source_manifest = Path(args.source_manifest).resolve()
    out_data_root = Path(args.out_data_root).resolve()
    prepared_root = out_data_root / "prepared"
    out_data_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv_rows(source_manifest) if source_manifest.is_file() else []

    requested_datasets = args.datasets
    if requested_datasets:
        dataset_names = requested_datasets
    else:
        local_ds = _discover_local_datasets(out_data_root)
        dataset_names = local_ds if local_ds else list(DEFAULT_DATASETS)

    out_manifest = []
    train_all = []
    val_all = []
    test_all = []

    for idx, dataset_name in enumerate(dataset_names):
        raw_ds_dir = out_data_root / dataset_name
        prepared_ds_dir = prepared_root / dataset_name
        raw_ds_dir.mkdir(parents=True, exist_ok=True)
        prepared_ds_dir.mkdir(parents=True, exist_ok=True)

        all_out = raw_ds_dir / "all.csv"

        if all_out.is_file():
            processed_all = _from_existing_all_csv(dataset_name=dataset_name, all_csv_path=all_out)
            unit = str(processed_all[0]["unit_source"])
            first_manifest = next(
                (r for r in manifest_rows if str(r.get("dataset_name", "")).strip() == dataset_name),
                None,
            )
            structure_dir = str(first_manifest["structure_dir"]) if first_manifest else ""
        else:
            unit = UNIT_BY_DATASET.get(dataset_name)
            if unit is None:
                raise ValueError(
                    f"No unit mapping for dataset '{dataset_name}', and no local all.csv found at {all_out}. "
                    "Add unit to UNIT_BY_DATASET or provide local data/<dataset>/all.csv."
                )
            if not manifest_rows:
                raise ValueError(
                    f"source_manifest not found: {source_manifest}. "
                    f"Cannot build {dataset_name} all.csv from upstream manifest."
                )
            processed_all, structure_dir = _from_source_manifest(
                dataset_name=dataset_name,
                manifest_rows=manifest_rows,
                unit=unit,
            )
            _write_csv_rows(all_out, processed_all, ALL_FIELDS)

        base_fields = list(processed_all[0].keys())
        split_fields = ["split"] + base_fields

        ds_seed = int(args.seed) + (idx + 1) * 1000
        train_rows, val_rows, test_rows = _split_rows(
            processed_all,
            seed=ds_seed,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
        )

        train_out = prepared_ds_dir / "train.csv"
        val_out = prepared_ds_dir / "val.csv"
        test_out = prepared_ds_dir / "test.csv"

        _write_csv_rows(train_out, train_rows, split_fields)
        _write_csv_rows(val_out, val_rows, split_fields)
        _write_csv_rows(test_out, test_rows, split_fields)

        train_all.extend(train_rows)
        val_all.extend(val_rows)
        test_all.extend(test_rows)

        out_manifest.append(
            {
                "dataset_name": dataset_name,
                "unit_source": unit,
                "label_space": "-log10(Kd[M])",
                "label_shift": "9.0" if unit == "nM" else "0.0",
                "structure_dir": structure_dir,
                "all_csv": str(all_out),
                "train_csv": str(train_out),
                "val_csv": str(val_out),
                "test_csv": str(test_out),
                "n_total": str(len(processed_all)),
                "n_train": str(len(train_rows)),
                "n_val": str(len(val_rows)),
                "n_test": str(len(test_rows)),
                "split_seed": str(ds_seed),
            }
        )

    manifest_out = prepared_root / "manifest.csv"
    _write_csv_rows(
        manifest_out,
        out_manifest,
        [
            "dataset_name",
            "unit_source",
            "label_space",
            "label_shift",
            "structure_dir",
            "all_csv",
            "train_csv",
            "val_csv",
            "test_csv",
            "n_total",
            "n_train",
            "n_val",
            "n_test",
            "split_seed",
        ],
    )

    combined_fields = ["split"] + list(train_all[0].keys()) if train_all else []
    _write_csv_rows(prepared_root / "train_all.csv", train_all, combined_fields)
    _write_csv_rows(prepared_root / "val_all.csv", val_all, combined_fields)
    _write_csv_rows(prepared_root / "test_all.csv", test_all, combined_fields)

    print(f"Saved raw dataset root: {out_data_root}")
    print(f"Saved prepared manifest: {manifest_out}")
    for r in out_manifest:
        print(
            f"[OK] {r['dataset_name']}: total={r['n_total']} train={r['n_train']} "
            f"val={r['n_val']} test={r['n_test']} unit={r['unit_source']}"
        )


if __name__ == "__main__":
    main()
