"""Inference helpers: load attack/normal parquets to tensors in SI units."""

import pandas as pd
from pathlib import Path
import numpy as np
import torch
from typing import Optional
from data.registry_norm import VariableRegistry

RAW_CSV_ROWS = 449_919  # SWaT_Dataset_Attack_v0 rows (no NaN, no resampling)


def _to_si(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c.startswith("LIT"):
            df[c] = df[c] / 1000.0
        elif c.startswith("FIT"):
            df[c] = df[c] / 3600.0
        elif c.startswith(("PIT", "DPIT")):
            df[c] = df[c] * 1000.0
    return df


def ensure_attack_parquet(
    xlsx_path: Path,
    sheet: str,
    out_path: Path,
    *,
    labels_out_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Attack Excel not found: {xlsx_path}")

    if labels_out_path is None:
        labels_out_path = out_path.with_name(out_path.stem + "_labels.parquet")

    # load and normalize headers
    df = pd.read_excel(xlsx_path, sheet_name=sheet, skiprows=1, engine="openpyxl")
    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    df = df.drop(columns=["Normal/Attack"], errors="ignore")  # raw column is misaligned

    # timestamp handling (parse then drop)
    if " Timestamp" in df.columns:
        df = df.rename(columns={" Timestamp": "Timestamp"})
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"].astype(str).str.strip(),
            format="%d/%m/%Y %I:%M:%S %p",
            errors="coerce",
        )
        df = df.drop(columns=["Timestamp"])

    # convert to SI, drop any incomplete rows
    df_si = _to_si(df)
    df_si = df_si.loc[df_si.notna().all(axis=1)].reset_index(drop=True)

    # Labels come from the Fung-corrected attack slices, NOT the raw Normal/Attack
    # column (Fung et al. showed the raw column is misaligned with the data).
    # Building them from the same slices the evaluator uses keeps labels, slices,
    # and data in one coordinate system. The raw attack CSV has no NaN, so no rows
    # are dropped above and Fung's raw-CSV slice indices apply 1:1.
    from scripts.evaluation import SWAT_ATTACK_SLICES, SWAT_ATTACK_SLICES_DROPPED

    n = len(df_si)
    if n != RAW_CSV_ROWS:
        raise ValueError(
            f"attack parquet has {n} rows, expected {RAW_CSV_ROWS}: rows were "
            "dropped, so Fung's raw-CSV slice indices no longer map 1:1."
        )
    lab_arr = np.zeros(n, dtype="int8")
    for info in {**SWAT_ATTACK_SLICES, **SWAT_ATTACK_SLICES_DROPPED}.values():
        lab_arr[int(info["Start_Row"]) : int(info["End_Row"])] = 1
    labels_df = pd.DataFrame({"attack": lab_arr})

    # write both Parquets
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_si.to_parquet(out_path, index=False)
    labels_out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(labels_out_path, index=False)

    return out_path, labels_out_path


def parquet_to_tensor(
    path: Path, *, registry: Optional[VariableRegistry] = None, device=None
):
    df = pd.read_parquet(path)

    if registry is not None:
        reg_order = [registry._by_index[i].name for i in range(registry.num_features)]
        # Derive UF_REJECT_MASK when a full-SWaT registry needs it but the parquet lacks it.
        if "UF_REJECT_MASK" in reg_order and "UF_REJECT_MASK" not in df.columns:
            df["UF_REJECT_MASK"] = (
                (df["MV304"] == 2) & ((df["P301"] == 2) | (df["P302"] == 2))
            ).astype(float)

    cols = list(df.columns)
    x = torch.tensor(df.values, dtype=torch.float32, device=device or "cpu")

    if registry:
        reg_order = [registry._by_index[i].name for i in range(registry.num_features)]

        if reg_order != cols:
            # If a registry column is missing from the parquet, fail loudly.
            missing = [c for c in reg_order if c not in cols]
            if missing:
                raise ValueError(
                    f"parquet {path} is missing registry column(s) required by the "
                    f"model: {missing}. Available columns: {cols}"
                )
            x = x[
                :,
                torch.tensor([cols.index(c) for c in reg_order], device=x.device),
            ]

    return x
