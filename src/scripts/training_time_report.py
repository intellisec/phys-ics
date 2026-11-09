"""Aggregate per-run training cost from ``run_config.json`` files and compare
Data-only (w_phys=0) vs PINN (w_phys>0) variants.

Each finished run writes ``run_config.json`` with a ``run`` section holding
``train_wall_time_sec``, ``total_epochs``, ``best_epoch`` (plus ``args`` with
``w_phys`` / ``model`` / ``seed`` / ...). This walks the gridsearch output tree,
builds one row per run, and reports the cost decomposition

    total wall-time  =  per-epoch time  x  epochs-to-converge

so you can say *why* a PINN costs more (per-step compute) or less (faster
convergence), not just by how much.

Usage::

    python -m src.scripts.training_time_report --root outputs/v5-gridsearch
    python -m src.scripts.training_time_report --root outputs/v5-gridsearch \
        --experiment v5-timing-s10w50 --pgfplots-csv outputs/v5-plots/pgfplots/fig_training_time.csv

CAVEAT: if runs were trained concurrently (``xargs -P N``), absolute wall-times
include GPU contention. The dedicated ``v5_timing`` config should be run with
``-P 1`` precisely so these numbers are clean. ``per_epoch_sec`` and
``total_epochs`` need ``run_config.json`` written by the updated ``save_run_json``
(older runs report wall-time only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("run_config.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        run = d.get("run", {}) or {}
        args = d.get("args", {}) or {}
        wt = run.get("train_wall_time_sec")
        if wt is None:
            continue
        wphys = args.get("w_phys")
        if wphys is None:
            wphys = (d.get("pinn", {}) or {}).get("w_phys")
        wphys = float(wphys) if wphys is not None else None
        total_ep = run.get("total_epochs")
        best_ep = run.get("best_epoch")
        wall = float(wt)
        rows.append(
            {
                "experiment": p.parent.parent.name,  # grid_name directory
                "run_dir": p.parent.name,
                "model": args.get("model"),
                "w_phys": wphys,
                "pinn_flag": 0 if (wphys is None or wphys == 0.0) else 1,
                "corrupt_param": args.get("corrupt_param"),
                "seed": args.get("seed"),
                "lr": args.get("lr"),
                "full_swat": bool(args.get("full_swat", False)),
                "wall_sec": wall,
                "total_epochs": int(total_ep) if total_ep else None,
                "best_epoch": int(best_ep) if best_ep is not None else None,
                "per_epoch_sec": (wall / total_ep) if total_ep else None,
            }
        )
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, by) -> pd.DataFrame:
    agg = {
        "wall_sec": ["count", "mean", "std"],
        "total_epochs": ["mean"],
        "per_epoch_sec": ["mean"],
    }
    g = df.groupby(by).agg(agg)
    g.columns = ["n", "wall_mean_s", "wall_std_s", "epochs_mean", "per_epoch_s"]
    g["wall_mean_min"] = g["wall_mean_s"] / 60.0
    return g.round(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/v5-gridsearch")
    ap.add_argument(
        "--experiment",
        default=None,
        help="Restrict to one grid_name dir (e.g. v5-timing-s10w50).",
    )
    ap.add_argument("--csv", default=None, help="Optional: write the per-run table.")
    ap.add_argument(
        "--pgfplots-csv",
        default=None,
        help="Write a per-architecture wide CSV (Data-only vs PINN: wall-time, "
        "epochs, per-epoch time) ready for a pgfplots grouped bar chart.",
    )
    args = ap.parse_args()

    df = collect(Path(args.root))
    if df.empty:
        print(f"No run_config.json with timing found under {args.root}")
        return
    if args.experiment:
        df = df[df["experiment"] == args.experiment]
        if df.empty:
            print(f"No runs for experiment={args.experiment}")
            return

    df["variant"] = df["pinn_flag"].map({0: "Data-only", 1: "PINN"})
    have_epochs = df["total_epochs"].notna().any()
    print(f"Collected {len(df)} finished runs from {args.root}")
    if not have_epochs:
        print(
            "(no epoch info — run_config.json predates the epoch fields; "
            "wall-time only)"
        )
    print()

    print("=== Overall: Data-only vs PINN ===")
    print(_summary(df, "variant"), "\n")

    print("=== By experiment x variant ===")
    print(_summary(df, ["experiment", "variant"]), "\n")

    print("=== By model x variant (PLC1 vs full_swat differ in size) ===")
    print(_summary(df, ["full_swat", "model", "variant"]), "\n")

    # decomposition ratios per experiment: total / per-epoch / epochs
    print("=== PINN vs Data-only ratios (per experiment) ===")
    for metric, label in (
        ("wall_sec", "wall-time"),
        ("per_epoch_sec", "per-epoch time"),
        ("total_epochs", "epochs-to-converge"),
    ):
        if df[metric].notna().any():
            m = (
                df.groupby(["experiment", "pinn_flag"])[metric]
                .mean()
                .unstack("pinn_flag")
            )
            if 0 in m.columns and 1 in m.columns:
                m["ratio"] = (m[1] / m[0]).round(2)
                print(f"-- {label} (PINN/Data-only) --")
                print(m.rename(columns={0: "dataonly", 1: "pinn"}).round(2), "\n")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"Wrote per-run table -> {args.csv}")

    if args.pgfplots_csv:
        # Per-architecture wide format: one row per model, Data-only vs PINN
        # wall-time (min), epochs, and per-epoch time (s).
        g = (
            df.groupby(["model", "pinn_flag"])
            .agg(
                wall_mean=("wall_sec", "mean"),
                wall_std=("wall_sec", "std"),
                epochs_mean=("total_epochs", "mean"),
                per_epoch_mean=("per_epoch_sec", "mean"),
            )
            .reset_index()
        )
        g["wall_mean_min"] = g["wall_mean"] / 60.0
        g["wall_std_min"] = g["wall_std"].fillna(0.0) / 60.0
        wide = None
        for pf, key in ((0, "dataonly"), (1, "pinn")):
            s = g[g["pinn_flag"] == pf][
                [
                    "model",
                    "wall_mean_min",
                    "wall_std_min",
                    "epochs_mean",
                    "per_epoch_mean",
                ]
            ].rename(
                columns={
                    "wall_mean_min": f"{key}_wall_min",
                    "wall_std_min": f"{key}_wall_std_min",
                    "epochs_mean": f"{key}_epochs",
                    "per_epoch_mean": f"{key}_per_epoch_s",
                }
            )
            wide = s if wide is None else wide.merge(s, on="model", how="outer")
        order = ["mlp", "cnn1d", "lstm", "transformer"]
        wide["model"] = pd.Categorical(wide["model"], order, ordered=True)
        wide = wide.sort_values("model").round(3).reset_index(drop=True)
        Path(args.pgfplots_csv).parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(args.pgfplots_csv, index=False)
        print(f"\nWrote pgfplots per-arch wide CSV -> {args.pgfplots_csv}")
        print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
