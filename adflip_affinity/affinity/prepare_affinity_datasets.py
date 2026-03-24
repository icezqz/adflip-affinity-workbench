import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


NEG_LOG_CANDIDATES = [
    "negative log Kd",
    "negative_log_kd",
    "neg_log_kd",
    "neglogkd",
]


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise ValueError(f"None of candidate columns found: {candidates}")
    return None


def _split_sizes(n: int) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 samples for 8/1/1 split, got n={n}")

    n_train = int(math.floor(n * 0.8))
    n_val = int(math.floor(n * 0.1))
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


def _normalize_name(raw_name, fallback_idx: int, dataset_name: str) -> str:
    name = str(raw_name).strip()
    if not name or name.lower() == "nan":
        name = f"{dataset_name}_{fallback_idx}"
    if name.startswith(">"):
        name = name[1:]
    return name


def prepare_one_dataset(
    ppl_csv: Path,
    structure_root: Path,
    out_root: Path,
    seed: int,
) -> dict:
    dataset_name = ppl_csv.parent.name
    df = pd.read_csv(ppl_csv).reset_index(drop=True)

    heavy_col = _find_column(df, ["heavy"])
    light_col = _find_column(df, ["light"])
    neg_col = _find_column(df, NEG_LOG_CANDIDATES)
    name_col = _find_column(df, ["name", "unnamed: 0"], required=False)

    output_rows = []
    missing_pdb = 0
    for idx, row in df.iterrows():
        pdb_key = f"{dataset_name}_{idx}"
        pdb_path = structure_root / dataset_name / f"{pdb_key}.pdb"
        if not pdb_path.is_file():
            missing_pdb += 1
            continue

        neg_log_kd = float(row[neg_col])
        kd = float(10 ** (-neg_log_kd))
        heavy = str(row[heavy_col]).strip()
        light = str(row[light_col]).strip()
        name_value = row[name_col] if name_col else f"{dataset_name}_{idx}"
        name = _normalize_name(name_value, idx, dataset_name)

        output_rows.append(
            {
                "name": name,
                "pdb": pdb_key,
                "sequence": f"{heavy}/{light}",
                "KD": kd,
                "negative_log_kd": neg_log_kd,
                "dataset": dataset_name,
                "source_row_index": idx,
            }
        )

    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(output_rows) < 3:
        raise ValueError(f"{dataset_name}: valid rows < 3 after filtering missing PDBs.")

    clean_df = pd.DataFrame(output_rows).reset_index(drop=True)
    all_csv = out_dir / "all.csv"
    clean_df.to_csv(all_csv, index=False)

    rng = pd.Series(range(len(clean_df))).sample(frac=1.0, random_state=seed).to_list()
    n_train, n_val, n_test = _split_sizes(len(clean_df))
    train_ids = set(rng[:n_train])
    val_ids = set(rng[n_train : n_train + n_val])
    test_ids = set(rng[n_train + n_val : n_train + n_val + n_test])

    train_df = clean_df.loc[[i in train_ids for i in range(len(clean_df))]].reset_index(drop=True)
    val_df = clean_df.loc[[i in val_ids for i in range(len(clean_df))]].reset_index(drop=True)
    test_df = clean_df.loc[[i in test_ids for i in range(len(clean_df))]].reset_index(drop=True)

    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    meta = {
        "dataset_name": dataset_name,
        "source_csv": str(ppl_csv),
        "structure_dir": str(structure_root / dataset_name),
        "all_csv": str(all_csv),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "n_total": int(len(clean_df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "missing_pdb_rows": int(missing_pdb),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--structure_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    structure_root = Path(args.structure_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ppl_files = sorted(data_root.glob("*/*_ppl.csv"))
    if not ppl_files:
        raise FileNotFoundError(f"No *_ppl.csv found under: {data_root}")

    manifests = []
    for ppl_csv in ppl_files:
        meta = prepare_one_dataset(
            ppl_csv=ppl_csv,
            structure_root=structure_root,
            out_root=out_root,
            seed=args.seed,
        )
        manifests.append(meta)
        print(f"[OK] {meta['dataset_name']}: {meta['n_train']}/{meta['n_val']}/{meta['n_test']}")

    manifest_df = pd.DataFrame(manifests)
    manifest_csv = out_root / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Saved manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
