"""
Select the best models for a gridsearch based on test RMSE.

Selection modes:
  - "default": groups by (model, pinn_flag), optimises over lr and w_phys.
    Used for ad-hoc single-directory selection (--base/--dest).
  - "v5_*": the per-experiment selectors of the v5 retraining set — group by
    (model, exp_var, pinn_flag, seed) and optimise over lr and w_phys.

The v5 batch runner (--batch-all) iterates BATCH_DIRS, applies the matching
v5_* selector, and copies the winners to ``outputs/selected_models/v5-<name>/``.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil

import pandas as pd


# --- implementation ----------------------------------------------------------
def scan_runs(
    base_dir: Path, alert_empty: bool = True
) -> tuple[list[dict], list[Path]]:
    """
    Collect config & metrics for every run under `base_dir` and
    report any completely empty run-folders.

    Returns
    -------
    records      : list of dicts with run statistics
    empty_dirs   : list of Path objects that were empty
    """
    records: list[dict] = []
    empty_dirs: list[Path] = []

    # ---- gather runs ----------------------------------------------------
    for cfg_path in base_dir.rglob("run_config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
            args = cfg.get("args", {}) or {}

            records.append(
                {
                    "_abs_run_path": str(cfg_path.parent),
                    "run_dir": cfg_path.parent.name,
                    "model": args.get("model"),
                    "lr": float(args.get("lr")),
                    "w_phys": float(args.get("w_phys", 0.0)),
                    "seed": int(args.get("seed", 0)),
                    "rows": int(args.get("rows", 0)),
                    "corrupt_type": args.get("corrupt_type"),
                    "corrupt_param": float(args.get("corrupt_param", 0.0)),
                    "ramp_rate": float(args.get("ramp_rate", 0.0)),
                    "physics_filter": float(args.get("physics_filter", 0.0)),
                    "bias_source": str(args.get("bias_source", "train")),
                    # Provenance fields: none of these are grouping keys, but a
                    # run that differs on any of them is not comparable by
                    # test_rmse to one that does not. _assert_homogeneous below
                    # turns accidental mixing into a hard failure.
                    "wrapper": str(args.get("wrapper", "absolute")),
                    "drop_cols": str(args.get("drop_cols", "")),
                    "test_rows": int(args.get("test_rows", 0)),
                    "test_rmse": extract_test_metric(cfg_path.parent),
                }
            )
        except Exception as e:
            print(f"[WARN] bad json in {cfg_path}: {e}")

    # ---- find empty folders --------------------------------------------
    for dir_path in base_dir.rglob("*"):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            empty_dirs.append(dir_path)

    if alert_empty and empty_dirs:
        print("[ALERT] empty run folders found:")
        for p in empty_dirs:
            print("   •", p.relative_to(base_dir))

    return records, empty_dirs


def extract_test_metric(run_dir: Path):
    """
    Return the test metric from the newest *_group_summary_1step.csv under run_dir/metrics:
    the weighted mean of 'value' over rows with
    (type=='data', metric=='mean_rmse_z', group!='ALL'), using 'n' as weights,
    i.e. sum(value_i * n_i) / sum(n_i).

    Returns None if the file or required rows/columns are missing.
    """
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return None

    csvs = sorted(
        metrics_dir.rglob("*_group_summary_1step.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csvs:
        return None

    csv = csvs[0]
    try:
        df = pd.read_csv(csv)
    except Exception:
        df = pd.read_csv(csv, sep=None, engine="python")

    # Normalize column names for robust selection
    df.columns = [c.lower().strip() for c in df.columns]

    # Basic sanity checks on required columns
    required_cols = {"type", "group", "metric", "value"}
    if not required_cols.issubset(df.columns):
        return None

    # Common filter for the metric of interest
    base_mask = df["type"].astype(str).str.lower().eq("data") & df["metric"].astype(
        str
    ).str.lower().eq("mean_rmse_z")

    # Weighted across groups (exclude ALL)
    if "n" not in df.columns:
        return None

    mask = base_mask & ~df["group"].astype(str).str.upper().eq("ALL")
    rows = df.loc[mask, ["value", "n"]].copy()
    if rows.empty:
        return None

    # Coerce to numeric and drop rows with invalid numbers
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows["n"] = pd.to_numeric(rows["n"], errors="coerce")
    rows = rows.dropna(subset=["value", "n"])

    if rows.empty:
        return None

    # Keep only strictly positive weights
    rows = rows[rows["n"] > 0]
    if rows.empty:
        return None

    numerator = (rows["value"] * rows["n"]).sum()
    denominator = rows["n"].sum()
    if denominator == 0:
        return None

    return float(numerator / denominator)


def pick_best(records: list[dict]) -> list[dict]:
    """
    For each combination of (model, pinn_flag) keep the run with the lowest test RMSE.
    pinn_flag = 0 if w_phys == 0.0 else 1, so any w_phys != 0 counts as PINN.
    """
    best = {}
    for rec in records:
        if rec["test_rmse"] is None:
            print(f"Skipping {rec['run_dir']} with no test RMSE")
            continue  # skip runs without test metric
        pinn_flag = 0 if float(rec["w_phys"]) == 0.0 else 1
        key = (rec["model"], pinn_flag)

        if key not in best or rec["test_rmse"] < best[key]["test_rmse"]:
            best[key] = rec
    return list(best.values())


def pick_best_v2(records: list[dict], exp_var: str = "corrupt_param") -> list[dict]:
    """Group by (model, exp_var, pinn_flag, seed) and pick the lowest test_rmse.

    Optimises over lr and w_phys within each group.
    """
    best = {}
    for rec in records:
        if rec["test_rmse"] is None:
            print(f"Skipping {rec['run_dir']} with no test RMSE")
            continue
        pinn_flag = 0 if float(rec["w_phys"]) == 0.0 else 1
        key = (rec["model"], rec[exp_var], pinn_flag, int(rec["seed"]))
        if key not in best or rec["test_rmse"] < best[key]["test_rmse"]:
            best[key] = {**rec, "pinn_flag": pinn_flag}
    return list(best.values())


def pick_best_v3_per_wphys(
    records: list[dict], exp_var: str = "corrupt_param"
) -> list[dict]:
    """Per-w_phys selector: group by (model, exp_var, w_phys, seed), pick lowest test_rmse.

    Optimises over lr only (w_phys becomes a grouping key instead of a binary
    pinn_flag). Used to plot one line per w_phys value in dose-response plots.
    """
    best = {}
    for rec in records:
        if rec["test_rmse"] is None:
            print(f"Skipping {rec['run_dir']} with no test RMSE")
            continue
        wp = float(rec["w_phys"])
        pinn_flag = 0 if wp == 0.0 else 1
        key = (rec["model"], rec[exp_var], wp, int(rec["seed"]))
        if key not in best or rec["test_rmse"] < best[key]["test_rmse"]:
            best[key] = {**rec, "pinn_flag": pinn_flag}
    return list(best.values())


def pick_best_qsweep(records: list[dict], exp_var: str = "physics_filter") -> list[dict]:
    """Filter-quantile sweep selector: keep BOTH corrupt_param and physics_filter
    as grouping keys — group by (model, corrupt_param, physics_filter, seed) and
    pick lowest test_rmse over lr. All models are data-only (pinn_flag=0)."""
    best = {}
    for rec in records:
        if rec["test_rmse"] is None:
            print(f"Skipping {rec['run_dir']} with no test RMSE")
            continue
        key = (
            rec["model"],
            float(rec["corrupt_param"]),
            float(rec.get("physics_filter", 0.0)),
            int(rec["seed"]),
        )
        if key not in best or rec["test_rmse"] < best[key]["test_rmse"]:
            best[key] = {**rec, "pinn_flag": 0}
    return list(best.values())


SELECTION_MODES = {
    "default": pick_best,
    # --- v5 retraining set ---
    "v5_scarcity": lambda recs: pick_best_v2(recs, "rows"),
    "v5_faulty_gaussian": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_faulty_stuck": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_poison_lit": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_poison_fit": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_consistent_lit_sin": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_consistent_fit": lambda recs: pick_best_v2(recs, "corrupt_param"),
    # LIT ramp-rate sweep at fixed corrupt_param=0.3
    "v5_poison_lit_rate_sweep": lambda recs: pick_best_v2(recs, "ramp_rate"),
    # Same sweep with the physics filter enabled (data-only + hard constraint).
    "v5_poison_lit_rate_sweep_filter": lambda recs: pick_best_v2(recs, "ramp_rate"),
    # PLC1 LIT poisoning filter baseline across poison fractions.
    "v5_poison_lit_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
    # (A) invariant-filter run; (B) physics-filter quantile sweep.
    "v5_poison_lit_invariant": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_poison_lit_filter_qsweep": lambda recs: pick_best_qsweep(recs, "physics_filter"),
    # Filter baselines for the remaining PLC1 poisoning families.
    "v5_poison_fit_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_consistent_lit_sin_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_consistent_fit_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
    # Per-w_phys selection on the ablation grid (one line per w_phys in plots).
    # Reads the v5-poison-lit-wphys-ablation training runs and keeps w_phys
    # as a grouping key. Only consumer is fig3_inconsistent_lit_*_per_wphys.
    "v5_poison_lit_per_wphys": lambda recs: pick_best_v3_per_wphys(
        recs, "corrupt_param"
    ),
    # Full-SWaT multi-tank LIT poisoning (PINN/data-only) + filter baseline.
    "v5_full_poison_lit": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_full_poison_lit_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
    # Filter baselines for scarcity / faulty-Gaussian / faulty-stuck.
    "v5_scarcity_filter": lambda recs: pick_best_v2(recs, "rows"),
    "v5_faulty_gaussian_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
    "v5_faulty_stuck_filter": lambda recs: pick_best_v2(recs, "corrupt_param"),
}


def copy_selected(selections: list[dict], dest_root: Path) -> None:
    """Copy each selected run directory into dest_root, preserving folder name."""
    dest_root.mkdir(parents=True, exist_ok=True)
    for rec in selections:
        src = Path(rec["_abs_run_path"])
        dst = dest_root / src.name
        print(f"Copying {src} → {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


_V2_SCARCITY_COLS = [
    "model",
    "rows",
    "pinn_flag",
    "seed",
    "w_phys",
    "lr",
    "test_rmse",
    "run_dir",
]
_V2_CORRUPT_COLS = [
    "model",
    "corrupt_param",
    "pinn_flag",
    "seed",
    "w_phys",
    "lr",
    "test_rmse",
    "run_dir",
]
_V2_RATE_COLS = [
    "model",
    "ramp_rate",
    "pinn_flag",
    "seed",
    "w_phys",
    "lr",
    "test_rmse",
    "run_dir",
]

_SUMMARY_COLS = {
    "default": ["model", "w_phys", "lr", "test_rmse", "run_dir"],
    # v5 retraining set
    "v5_scarcity": _V2_SCARCITY_COLS,
    "v5_faulty_gaussian": _V2_CORRUPT_COLS,
    "v5_faulty_stuck": _V2_CORRUPT_COLS,
    "v5_poison_lit": _V2_CORRUPT_COLS,
    "v5_poison_fit": _V2_CORRUPT_COLS,
    "v5_consistent_lit_sin": _V2_CORRUPT_COLS,
    "v5_consistent_fit": _V2_CORRUPT_COLS,
    # v5 ramp-rate sweep (corrupt_param fixed at 0.3, sweeps ramp_rate)
    "v5_poison_lit_rate_sweep": _V2_RATE_COLS,
    # Filter variant — same columns, no need to expose tau/kept here
    # (those live in checkpoint meta["physics_filter"]).
    "v5_poison_lit_rate_sweep_filter": _V2_RATE_COLS,
    # PLC1 LIT poisoning filter baseline across poison fractions.
    "v5_poison_lit_filter": _V2_CORRUPT_COLS,
    "v5_poison_lit_invariant": _V2_CORRUPT_COLS,
    "v5_poison_lit_filter_qsweep": _V2_CORRUPT_COLS + ["physics_filter"],
    "v5_poison_fit_filter": _V2_CORRUPT_COLS,
    "v5_consistent_lit_sin_filter": _V2_CORRUPT_COLS,
    "v5_consistent_fit_filter": _V2_CORRUPT_COLS,
    # Per-w_phys selection on the ablation grid (keeps w_phys as a grouping key).
    "v5_poison_lit_per_wphys": _V2_CORRUPT_COLS,
    # Full-SWaT multi-tank LIT poisoning (PINN/data-only) + filter baseline.
    "v5_full_poison_lit": _V2_CORRUPT_COLS,
    "v5_full_poison_lit_filter": _V2_CORRUPT_COLS,
    "v5_scarcity_filter": _V2_SCARCITY_COLS,
    "v5_faulty_gaussian_filter": _V2_CORRUPT_COLS,
    "v5_faulty_stuck_filter": _V2_CORRUPT_COLS,
}

# --- v5 batch: experiment mode → training output directory under outputs/v5-gridsearch/ ---
BATCH_DIRS = {
    "v5_scarcity": "outputs/v5-gridsearch/v5-scarcity-s10w50",
    "v5_faulty_gaussian": "outputs/v5-gridsearch/v5-faulty-gauss-s10w50",
    "v5_faulty_stuck": "outputs/v5-gridsearch/v5-faulty-stuck-s10w50",
    "v5_poison_lit": "outputs/v5-gridsearch/v5-poison-lit-s10w50",
    "v5_poison_fit": "outputs/v5-gridsearch/v5-poison-fit-s10w50",
    "v5_consistent_lit_sin": "outputs/v5-gridsearch/v5-consistent-lit-sin-s10w50",
    "v5_consistent_fit": "outputs/v5-gridsearch/v5-consistent-fit-s10w50",
    "v5_poison_lit_rate_sweep": "outputs/v5-gridsearch/v5-poison-lit-rate-sweep-s10w50",
    "v5_poison_lit_rate_sweep_filter": "outputs/v5-gridsearch/v5-poison-lit-rate-sweep-filter-s10w50",
    # PLC1 LIT poisoning filter baseline across poison fractions.
    "v5_poison_lit_filter": "outputs/v5-gridsearch/v5-poison-lit-filter-s10w50",
    "v5_poison_lit_invariant": "outputs/v5-gridsearch/v5-poison-lit-invariant-s10w50",
    "v5_poison_lit_filter_qsweep": "outputs/v5-gridsearch/v5-poison-lit-filter-qsweep-s10w50",
    "v5_poison_fit_filter": "outputs/v5-gridsearch/v5-poison-fit-filter-s10w50",
    "v5_consistent_lit_sin_filter": "outputs/v5-gridsearch/v5-consistent-lit-sin-filter-s10w50",
    "v5_consistent_fit_filter": "outputs/v5-gridsearch/v5-consistent-fit-filter-s10w50",
    # Per-w_phys selection reads the ablation grid (5 w_phys values) and
    # keeps w_phys as a grouping key; lands in its own selected_dir.
    "v5_poison_lit_per_wphys": "outputs/v5-gridsearch/v5-poison-lit-wphys-ablation-s10w50",
    # Full-SWaT multi-tank LIT poisoning (PINN/data-only) + filter baseline.
    # dest_root → outputs/selected_models/v5-full_poison_lit{,_filter}
    # Full-SWaT runs use the delta head, drop AIT201/AIT203 and the fused
    # threshold slice; they live in their own grid dir (see gridsearch.py).
    "v5_full_poison_lit": "outputs/v5-gridsearch/v5-full-poison-lit-delta-s10w50",
    "v5_full_poison_lit_filter": "outputs/v5-gridsearch/v5-full-poison-lit-delta-filter-s10w50",
    "v5_scarcity_filter": "outputs/v5-gridsearch/v5-scarcity-filter-s10w50",
    "v5_faulty_gaussian_filter": "outputs/v5-gridsearch/v5-faulty-gauss-filter-s10w50",
    "v5_faulty_stuck_filter": "outputs/v5-gridsearch/v5-faulty-stuck-filter-s10w50",
}


_PROVENANCE_FIELDS = ("wrapper", "drop_cols", "test_rows")


def _assert_homogeneous(records: list[dict], base_dir: Path) -> None:
    """Fail if a grid dir mixes runs that are not comparable by test_rmse.

    Selection minimises test_rmse within (model, corrupt_param, pinn_flag, seed),
    and none of the provenance fields are part of that key. So a directory holding
    e.g. both absolute/51-channel and delta/49-channel runs would silently let them
    compete on a metric that does not mean the same thing in both -- and would keep
    selecting one population for any cell the other does not cover.
    """
    if not records:
        return
    bad = {}
    for f in _PROVENANCE_FIELDS:
        seen = sorted({str(r.get(f)) for r in records})
        if len(seen) > 1:
            bad[f] = seen
    if bad:
        detail = "; ".join(f"{k}: {v}" for k, v in bad.items())
        raise RuntimeError(
            f"{base_dir} mixes runs with different training provenance ({detail}). "
            "test_rmse is not comparable across these, so selection would pick "
            "arbitrarily. Separate them by grid_name, or move the legacy runs aside."
        )


def _run_selection(mode: str, base_dir: Path, dest_root: Path) -> list[dict]:
    """Run model selection for a single mode. Returns the selected records."""
    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode}")
    print(f"  Base: {base_dir}")
    print(f"  Dest: {dest_root}")
    print(f"{'=' * 60}")

    records, _ = scan_runs(base_dir)
    print(f"Scanned {len(records)} runs")
    _assert_homogeneous(records, base_dir)

    selector = SELECTION_MODES[mode]
    selections = selector(records)
    copy_selected(selections, dest_root)

    print(f"\nFinished. Copied {len(selections)} best models to {dest_root}")

    if selections:
        df = pd.DataFrame(selections)
        cols = _SUMMARY_COLS.get(mode, _SUMMARY_COLS["default"])
        cols = [c for c in cols if c in df.columns]
        sort_col = (
            "rows"
            if "rows" in cols
            else "corrupt_param"
            if "corrupt_param" in cols
            else "test_rmse"
        )
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(f"\nSelected models ({mode} mode):")
            print(df[cols].sort_values(sort_col).to_string(index=False))

        # Save summary CSV
        csv_path = dest_root / "summary.csv"
        df[cols].sort_values(sort_col).to_csv(csv_path, index=False)
        print(f"Saved summary: {csv_path}")

    return selections


def main():
    ap = argparse.ArgumentParser(
        description="Pick the best run per hyper-group and copy its folder."
    )
    ap.add_argument(
        "--base",
        default=None,
        help="Directory that contains all run folders. "
        "Required when --batch-all is not set.",
    )
    ap.add_argument(
        "--dest",
        default=None,
        help="Where to copy the winning run folders. "
        "Required when --batch-all is not set.",
    )
    ap.add_argument(
        "--mode",
        choices=list(SELECTION_MODES.keys()),
        default="default",
        help="Selection mode: 'default' (generic grouper) or a 'v5_*' mode for the "
        "v5 retraining set (paired with --batch-all).",
    )
    ap.add_argument(
        "--batch-all",
        action="store_true",
        help="Run selection for every v5 experiment in BATCH_DIRS "
        "(reads from outputs/v5-gridsearch/v5-<name>-s10w50/, "
        "writes to outputs/selected_models/v5-<name>/).",
    )
    args = ap.parse_args()

    if args.batch_all:
        total = 0
        for mode, base_rel in BATCH_DIRS.items():
            base_dir = Path(base_rel).resolve()
            # All BATCH_DIRS keys are v5 modes; selected models land under
            # outputs/selected_models/v5-<rest>, matching the selected_dir
            # paths in evaluation.EXPERIMENTS.
            assert mode.startswith("v5_"), f"BATCH_DIRS mode must be v5_*: {mode}"
            dest_root = Path(
                f"outputs/selected_models/v5-{mode[len('v5_') :]}"
            ).resolve()
            if not base_dir.exists():
                print(f"[SKIP] {base_dir} does not exist")
                continue
            sels = _run_selection(mode, base_dir, dest_root)
            total += len(sels)
        print(f"\n{'=' * 60}")
        print(
            f"Batch complete. {total} total models selected across {len(BATCH_DIRS)} experiments."
        )
    else:
        if args.base is None or args.dest is None:
            ap.error("--base and --dest are required when --batch-all is not set")
        base_dir = Path(args.base).resolve()
        dest_root = Path(args.dest).resolve()
        _run_selection(args.mode, base_dir, dest_root)


if __name__ == "__main__":
    main()
