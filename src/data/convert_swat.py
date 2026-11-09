"""Preprocess SWaT: convert the 1 Hz xlsx logs to SI-unit parquet (run once)."""

#!/usr/bin/env python
import argparse
import pandas as pd
import pathlib as P


def to_si(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c.startswith("LIT"):
            df[c] = df[c] / 1000.0
        elif c.startswith("FIT"):
            df[c] = df[c] / 3600.0
        elif c.startswith(("PIT", "DPIT")):
            df[c] = df[c] * 1000.0

    df["UF_REJECT_MASK"] = (
        (df["MV304"] == 2) & ((df["P301"] == 2) | (df["P302"] == 2))
    ).astype(float)

    return df


def read_xlsx(xlsx_path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, skiprows=1, sheet_name=sheet)
    if " Timestamp" in df.columns:
        df = df.rename(columns={" Timestamp": "Timestamp"})
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"].astype(str).str.strip(),
            format="%d/%m/%Y %I:%M:%S %p",
            errors="coerce",
        )
        df = df.drop(columns=["Timestamp"])
    if "Normal/Attack" in df.columns:
        df = df.drop(columns=["Normal/Attack"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to SWaT_Dataset_Normal_v0.xlsx")
    ap.add_argument("--sheet", default="Normal.csv")
    ap.add_argument(
        "--out", required=True, help="Output parquet file (preprocessed, SI)."
    )
    args = ap.parse_args()

    out = P.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = read_xlsx(args.xlsx, args.sheet)
    df = to_si(df).dropna(axis=0, how="any")
    df.to_parquet(out, index=False)
    print(f"wrote {out} with shape={df.shape}")


if __name__ == "__main__":
    main()
