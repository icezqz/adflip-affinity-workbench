import os
import sys
import contextlib
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import prody

try:
    prody.confProDy(verbosity="none")
except Exception:
    pass


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


def kd_to_label(kd_value: float, mode: str = "log10") -> float:
    kd = float(kd_value)
    if mode == "raw":
        return kd
    import math

    if kd <= 0:
        raise ValueError(f"KD must be > 0, got {kd}")
    if mode == "log10":
        return math.log10(kd)
    if mode == "neg_log10":
        return -math.log10(kd)
    raise ValueError(f"Unknown label mode: {mode}")


@lru_cache(maxsize=4096)
def _chain_order_in_pdb(pdb_path: str) -> tuple[str, ...]:
    ag = prody.parsePDB(pdb_path)
    atoms = ag.select("not water and not hydrogen")
    if atoms is None:
        return tuple()
    order: list[str] = []
    for atom in atoms:
        ch = atom.getChid()
        if not ch:
            ch = "?"
        if len(ch) > 1:
            ch = ch[0]
        if not order or order[-1] != ch:
            if ch not in order:
                order.append(ch)
    return tuple(order)


class ADFLIPAffinityDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        pdb_dir: str,
        adflip_root: str,
        pdb_col: str = "pdb",
        y_col: str = "KD",
        seq_col: str = "sequence",
        chain_order: list[str] | tuple[str, ...] = ("A", "H", "L"),
        label_mode: str = "log10",
        parser_chain_id: list[str] | None = None,
    ):
        super().__init__()
        _ensure_adflip_importable(adflip_root)
        # all_atom_parse.py reads relative files at import time (data/misc/...),
        # so import it under the ADFLIP project root.
        with _temporary_cwd(adflip_root):
            from data import all_atom_parse as aap  # noqa: WPS433

        self.aap = aap
        df = pd.read_csv(csv_path)
        self.pdb_dir = pdb_dir
        self.pdb_col = pdb_col
        self.y_col = y_col
        self.seq_col = seq_col if seq_col in df.columns else None
        self.chain_order = list(chain_order)
        self.label_mode = label_mode
        self.parser_chain_id = parser_chain_id

        for col in [pdb_col, y_col]:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        y_numeric = pd.to_numeric(df[y_col], errors="coerce")
        keep_mask = ~y_numeric.isna()
        if int((~keep_mask).sum()) > 0:
            df = df.loc[keep_mask].copy()
            df[y_col] = y_numeric[keep_mask].astype(float)
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_pdb_path(self, pdb_key: str) -> str:
        pdb_key = str(pdb_key).strip()
        if os.path.isabs(pdb_key) and os.path.isfile(pdb_key):
            return pdb_key
        fname = pdb_key
        if not (fname.endswith(".pdb") or fname.endswith(".cif") or fname.endswith(".mmcif")):
            fname = fname + ".pdb"
        return os.path.join(self.pdb_dir, fname)

    def _infer_chain_letter_to_internal_id(self, pdb_path: str) -> dict[str, int]:
        encountered = list(_chain_order_in_pdb(pdb_path))
        allowed = self.parser_chain_id if self.parser_chain_id is not None else self.chain_order
        encountered = [c for c in encountered if c in allowed]
        if not encountered:
            encountered = list(allowed)
        return {c: i for i, c in enumerate(encountered)}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        pdb_path = self._resolve_pdb_path(row[self.pdb_col])
        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(f"PDB/mmCIF not found: {pdb_path}")

        y_raw = float(row[self.y_col])
        y = torch.tensor([[kd_to_label(y_raw, mode=self.label_mode)]], dtype=torch.float32)

        parser_chain_id = self.parser_chain_id if self.parser_chain_id is not None else self.chain_order
        struct_data = self.aap.parse_mmcif_to_structure_data(pdb_path, parser_chain_id=parser_chain_id)

        if self.seq_col is not None:
            raw_seq = str(row[self.seq_col]).strip()
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
                batch_dict[k] = torch.from_numpy(v).unsqueeze(0)
            else:
                batch_dict[k] = torch.tensor([v]).unsqueeze(0)

        batch_dict["batch_index"] = torch.zeros_like(batch_dict["residue_index"])
        for k, v in list(batch_dict.items()):
            if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
                batch_dict[k] = v.float()

        return {
            "batch_dict": batch_dict,
            "y": y,
            "meta": {"pdb_path": pdb_path, "y_raw": y_raw},
        }

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


def collate_affinity_adflip(batch: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(batch) == 1
    return batch[0]
