"""Reshape the flat per-figure CSVs in ``outputs/v5-plots/`` into wide-format
CSVs that pgfplots can ingest directly (``table[x=..., y=pinn_mean, ...]``).

Each plot in ``plots.py`` writes a long-form ``<name>.csv`` sidecar next to its
PDF (all flat in ``outputs/v5-plots/``). For every figure this:

 * loads that long-form CSV
 * pivots the series into columns. Multi-series figures are keyed either by
   ``pinn_flag`` (1=PINN, 0=Data-only, -1=Filter → ``pinn_*`` / ``dataonly_*`` /
   ``filter_*`` columns) or by a ``line`` label (Data-only / PINN / Filter).
 * normalises string labels (spaces/dashes → underscores) so they are valid
   ``symbolic x coords`` in pgfplots
 * writes the result to ``outputs/v5-plots/pgfplots/<figure>.csv``

Run after ``evaluation.py`` (any time the underlying PDFs/CSVs are regenerated):

    python -m src.plotting.export_pgfplots_csvs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PLOTS_DIR = Path("outputs/v5-plots")  # flat: all figure CSVs live here
OUT_DIR = PLOTS_DIR / "pgfplots"

_ARCH_ORDER = ["mlp", "cnn1d", "lstm", "transformer"]

# Canonical, pgfplots-safe column keys for the named lines in the multi-line
# figures (avoids "+"/spaces ending up in column headers).
_LINE_KEYS = {
    "Data-only": "dataonly",
    "PINN": "pinn",
    "Filter": "filter",
    "Poisoned Data-Only": "dataonly",
    "Poisoned PINN": "pinn",
    "Poisoned Data-Only + filter": "filter",
    "Clean Data-Only": "clean",
}

# pinn_flag → pgfplots-safe column prefix (Filter is encoded as -1).
_PINN_FLAG_KEYS = {1: "pinn", 0: "dataonly", -1: "filter"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise(s: str) -> str:
    """Make a label safe for pgfplots' symbolic x coords."""
    return str(s).strip().replace(" ", "_").replace("-", "_").replace("/", "_").lower()


def _load(name: str) -> pd.DataFrame | None:
    p = PLOTS_DIR / f"{name}.csv"
    if not p.exists():
        print(f"  [SKIP] {p} does not exist (run evaluation.py first)")
        return None
    return pd.read_csv(p)


def _x_col(df: pd.DataFrame) -> str:
    """Pick the dose-response x column, tolerating either naming."""
    for c in ("corrupt_param_raw", "exp_var_raw", "corrupt_param"):
        if c in df.columns:
            return c
    return df.columns[0]


def _pivot_pinn(
    df: pd.DataFrame, key_cols: list[str], value_renames: dict[str, str]
) -> pd.DataFrame:
    """Pivot rows-per-``pinn_flag`` into ``pinn_*`` / ``dataonly_*`` / ``filter_*``
    columns (the Filter series, pinn_flag == -1, only appears if present)."""
    out: pd.DataFrame | None = None
    for flag, prefix in _PINN_FLAG_KEYS.items():
        sub = df[df["pinn_flag"] == flag]
        if sub.empty:
            continue
        sub = sub[key_cols + list(value_renames)].rename(
            columns={k: f"{prefix}_{v}" for k, v in value_renames.items()}
        )
        out = sub if out is None else out.merge(sub, on=key_cols, how="outer")
    return out if out is not None else pd.DataFrame()


def _line_key(line: str) -> str:
    return _LINE_KEYS.get(str(line), _normalise(line))


def _pivot_by_line(
    df: pd.DataFrame, x_col: str, mean_col: str, std_col: str, line_col: str = "line"
) -> pd.DataFrame:
    """Pivot rows-per-``line`` into ``<key>_mean`` / ``<key>_std`` columns."""
    out: pd.DataFrame | None = None
    for line in df[line_col].dropna().unique():
        key = _line_key(line)
        sub = df[df[line_col] == line][[x_col, mean_col, std_col]].rename(
            columns={mean_col: f"{key}_mean", std_col: f"{key}_std"}
        )
        out = sub if out is None else out.merge(sub, on=x_col, how="outer")
    if out is None:
        return pd.DataFrame()
    return out.sort_values(x_col).reset_index(drop=True)


def _wphys_key(wp: float) -> str:
    """w_phys → pgfplots-safe column prefix, e.g. 0.5→w0p5, 1.0→w1, 100→w100."""
    return "w" + format(float(wp), "g").replace(".", "p").replace("-", "m")


def _pivot_by_wphys(
    df: pd.DataFrame, x_col: str, mean_col: str, std_col: str
) -> pd.DataFrame:
    """Pivot rows-per-``w_phys`` into ``w<val>_mean`` / ``w<val>_std`` columns."""
    out: pd.DataFrame | None = None
    for wp in sorted(df["w_phys"].dropna().unique()):
        key = _wphys_key(wp)
        sub = df[df["w_phys"] == wp][[x_col, mean_col, std_col]].rename(
            columns={mean_col: f"{key}_mean", std_col: f"{key}_std"}
        )
        out = sub if out is None else out.merge(sub, on=x_col, how="outer")
    if out is None:
        return pd.DataFrame()
    return out.sort_values(x_col).reset_index(drop=True)


def _save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  wrote {p}  ({len(df)} rows, cols: {list(df.columns)})")


# ---------------------------------------------------------------------------
# Per-figure exporters — bar groups
# ---------------------------------------------------------------------------
def clean_data() -> None:
    """4 archs × 3 subsets {All, Phys-valid, Phys-invalid} × {PINN, Data-only,
    Filter} grouped bars. Episode-uniform recall, timestep-pooled recall, and
    FPR, one file per arch. Filter is pinn_flag == -1."""
    for suffix in ("", "_timestep_recall", "_fpr"):
        df = _load(f"clean_data{suffix}")
        if df is None:
            continue
        out = _pivot_pinn(
            df,
            key_cols=["arch", "subset"],
            value_renames={"mean": "mean", "std": "std"},
        )
        if out.empty:
            continue
        out["arch"] = pd.Categorical(out["arch"], _ARCH_ORDER, ordered=True)
        out["subset"] = pd.Categorical(
            out["subset"], ["All attacks", "Phys-valid", "Phys-invalid"], ordered=True
        )
        out = out.sort_values(["arch", "subset"]).reset_index(drop=True)
        out["arch_label"] = out["arch"].astype(str).map(_normalise)
        out["subset_label"] = out["subset"].astype(str).map(_normalise)
        out["label"] = out["arch_label"] + "__" + out["subset_label"]
        for arch in _ARCH_ORDER:
            sub = (
                out[out["arch_label"] == arch]
                .drop(columns=["arch", "arch_label"])
                .reset_index(drop=True)
            )
            if not sub.empty:
                _save(sub, f"fig_clean_data_{arch}{suffix}")


def cross_sensor() -> None:
    """2 × 2 grouped bars: (trained_on, tested_on) × pinn."""
    df = _load("cross_sensor")
    if df is None:
        return
    out = _pivot_pinn(
        df,
        key_cols=["trained_on", "tested_on"],
        value_renames={"mean_recall": "mean", "std_recall": "std"},
    )
    if out.empty:
        return
    out["trained_on"] = pd.Categorical(out["trained_on"], ["LIT", "FIT"], ordered=True)
    out["tested_on"] = pd.Categorical(out["tested_on"], ["LIT", "FIT"], ordered=True)
    out = out.sort_values(["trained_on", "tested_on"]).reset_index(drop=True)
    out["label"] = (
        out["trained_on"].astype(str).map(_normalise)
        + "_to_"
        + out["tested_on"].astype(str).map(_normalise)
    )
    _save(out, "fig_cross_sensor")


def cross_shape() -> None:
    """4 shapes grouped bars per sensor (Linear is the trained shape)."""
    df = _load("cross_shape")
    if df is None:
        return
    df["shape"] = pd.Categorical(
        df["shape"], ["Linear", "Staircase", "Sigmoid", "Noisy Ramp"], ordered=True
    )
    df["sensor"] = pd.Categorical(df["sensor"], ["lit", "fit"], ordered=True)
    out = _pivot_pinn(
        df,
        key_cols=["sensor", "shape"],
        value_renames={"mean_recall": "mean", "std_recall": "std"},
    )
    if out.empty:
        return
    out = out.sort_values(["sensor", "shape"]).reset_index(drop=True)
    out["shape_label"] = out["shape"].astype(str).map(_normalise)
    out["label"] = out["sensor"].astype(str) + "__" + out["shape_label"]
    for sensor in ("lit", "fit"):
        sub = (
            out[out["sensor"] == sensor].drop(columns=["sensor"]).reset_index(drop=True)
        )
        if not sub.empty:
            _save(sub, f"fig_cross_shape_{sensor}")


# ---------------------------------------------------------------------------
# Per-figure exporters — dose-response lines (pinn-pivot, now incl. Filter)
# ---------------------------------------------------------------------------
def _dose_pinn(name: str, out_name: str, splits: list[tuple[str, str]]) -> None:
    """Recall dose-response split by ``training_exp`` and pivoted by pinn_flag
    (incl. Filter as pinn_flag == -1)."""
    df = _load(name)
    if df is None:
        return
    x = _x_col(df)
    for tag, training_exp in splits:
        sub = (
            df[df["training_exp"] == training_exp]
            if "training_exp" in df.columns
            else df
        )
        if sub.empty:
            print(f"  [SKIP] {name} / {tag}: no rows for training_exp={training_exp}")
            continue
        out = _pivot_pinn(
            sub,
            key_cols=[x],
            value_renames={"mean_recall": "mean", "std_recall": "std"},
        )
        if out.empty:
            continue
        out = out.rename(columns={x: "corrupt_param"}).sort_values("corrupt_param")
        _save(out.reset_index(drop=True), f"{out_name}_{tag}")


def inconsistent_poisoning() -> None:
    _dose_pinn(
        "inconsistent_poisoning",
        "fig_inconsistent_poisoning",
        [("lit", "v5_poison_lit"), ("fit", "v5_poison_fit")],
    )


def inconsistent_poisoning_mse() -> None:
    _dose_pinn(
        "inconsistent_poisoning_mse",
        "fig_inconsistent_poisoning_mse",
        [("lit", "v5_poison_lit"), ("fit", "v5_poison_fit")],
    )


def consistent_poisoning() -> None:
    _dose_pinn(
        "consistent_poisoning",
        "fig_consistent_poisoning",
        [("lit", "v5_consistent_lit_sin"), ("fit", "v5_consistent_fit")],
    )


def inconsistent_poisoning_per_architecture() -> None:
    """Per-architecture PINN-vs-Data-only lines over poison fraction (+ CSV
    sidecar). Recall and FPR, with per-arch split files for one-panel pgfplots."""
    df = _load("inconsistent_poisoning_per_architecture")
    if df is None:
        return
    xcol = "corrupt_param" if "corrupt_param" in df.columns else _x_col(df)
    for metric, suffix in (("recall", ""), ("fpr", "_fpr")):
        sub = df[df["metric"] == metric] if "metric" in df.columns else df
        if sub.empty:
            continue
        out = _pivot_pinn(
            sub, key_cols=["model", xcol], value_renames={"mean": "mean", "std": "std"}
        )
        if out.empty:
            continue
        out["model"] = pd.Categorical(out["model"], _ARCH_ORDER, ordered=True)
        out = out.sort_values(["model", xcol]).rename(columns={xcol: "corrupt_param"})
        out = out.reset_index(drop=True)
        for arch in _ARCH_ORDER:
            s = out[out["model"] == arch].drop(columns=["model"]).reset_index(drop=True)
            if not s.empty:
                _save(s, f"fig_inconsistent_poisoning_per_architecture_{arch}{suffix}")


# ---------------------------------------------------------------------------
# Per-figure exporters — single-line / per-w_phys / line-pivot dose-response
# ---------------------------------------------------------------------------
def ramp_rate() -> None:
    df = _load("ramp_rate")
    if df is None:
        return
    out = _pivot_pinn(
        df,
        key_cols=["rate_mm_per_s", "is_training_rate"],
        value_renames={"mean_recall": "mean", "std_recall": "std"},
    )
    if out.empty:
        return
    out = out.sort_values("rate_mm_per_s").reset_index(drop=True)
    _save(out, "fig_ramp_rate")


_V3_TRAIN_ROWS = 345_600


def scarcity() -> None:
    df = _load("scarcity")
    if df is None:
        return
    out = _pivot_pinn(
        df,
        key_cols=["rows"],
        value_renames={"mean_recall": "mean", "std_recall": "std"},
    )
    if out.empty:
        return
    out = out.sort_values("rows").reset_index(drop=True)
    out.insert(1, "fraction", out["rows"] / _V3_TRAIN_ROWS)
    _save(out, "fig_scarcity")


def _single_line(name: str, x_col: str, out_name: str) -> None:
    df = _load(name)
    if df is None:
        return
    if x_col not in df.columns:
        print(f"  [SKIP] {name}: no column {x_col}")
        return
    out = _pivot_pinn(
        df, key_cols=[x_col], value_renames={"mean_recall": "mean", "std_recall": "std"}
    )
    if out.empty:
        return
    _save(out.sort_values(x_col).reset_index(drop=True), out_name)


def gaussian_noise() -> None:
    _single_line("gaussian_noise", "corrupt_param", "fig_gaussian_noise")


def stale_episodes() -> None:
    _single_line("stale_episodes", "corrupt_param", "fig_stale_episodes")


def _dose_by_line(name: str, mean_col: str, std_col: str, out_name: str) -> None:
    df = _load(name)
    if df is None:
        return
    if mean_col not in df.columns or "line" not in df.columns:
        print(f"  [SKIP] {name}: missing {mean_col!r}/line")
        return
    x = _x_col(df)
    out = _pivot_by_line(df, x, mean_col, std_col)
    if out.empty:
        return
    _save(out.rename(columns={x: "corrupt_param"}), out_name)


def _line_stats(name: str, mean_col: str, std_col: str, out_name: str) -> None:
    """Per-curve stats table: one row per (line, x) with mean/std."""
    df = _load(name)
    if df is None:
        return
    if mean_col not in df.columns or "line" not in df.columns:
        print(f"  [SKIP] {name}: missing {mean_col!r}/line")
        return
    x = _x_col(df)
    cols = ["line", x, mean_col, std_col, "n"]
    out = (
        df[[c for c in cols if c in df.columns]]
        .rename(columns={x: "corrupt_param", mean_col: "mean", std_col: "std"})
        .sort_values(["line", "corrupt_param"])
        .reset_index(drop=True)
    )
    _save(out, out_name)


def rate_sweep() -> None:
    """Recall → wide pgfplots CSV (Data-only / Filter / PINN over ramp rate).
    FPR → per-curve stats table only."""
    df = _load("rate_sweep")
    if df is not None and "rate_mm_per_s" in df.columns:
        df = df[df["rate_mm_per_s"].notna()]
        _save(_pivot_by_line(df, "rate_mm_per_s", "mean", "std"), "fig_rate_sweep")
    df = _load("rate_sweep_fpr")
    if df is not None and "rate_mm_per_s" in df.columns:
        df = df[df["rate_mm_per_s"].notna()]
        cols = ["line", "rate_mm_per_s", "mean", "std", "n"]
        _save(
            df[[c for c in cols if c in df.columns]]
            .sort_values(["line", "rate_mm_per_s"])
            .reset_index(drop=True),
            "fig_rate_sweep_fpr_stats",
        )


def wphys_ablation() -> None:
    """Recall → wide pgfplots CSV (one column-pair per w_phys).
    FPR → per-curve stats table only."""
    df = _load("wphys_ablation")
    if df is not None:
        x = _x_col(df)
        out = _pivot_by_wphys(df, x, "mean_recall", "std_recall")
        if not out.empty:
            _save(out.rename(columns={x: "corrupt_param"}), "fig_wphys_ablation_recall")
    df = _load("wphys_ablation_fpr")
    if df is not None:
        x = _x_col(df)
        cols = [x, "w_phys", "mean_fpr", "std_fpr", "n"]
        _save(
            df[[c for c in cols if c in df.columns]]
            .rename(columns={x: "corrupt_param"})
            .sort_values(["w_phys", "corrupt_param"])
            .reset_index(drop=True),
            "fig_wphys_ablation_fpr_stats",
        )


def full_swat_inconsistent_poisoning() -> None:
    """Recall → wide (Data-only / PINN / Filter); FPR → stats table."""
    _dose_by_line(
        "full_swat_inconsistent_poisoning",
        "mean_recall",
        "std_recall",
        "fig_full_swat_inconsistent_poisoning",
    )
    _line_stats(
        "full_swat_inconsistent_poisoning_fpr",
        "mean_fpr",
        "std_fpr",
        "fig_full_swat_inconsistent_poisoning_fpr_stats",
    )


def plc1_poison_on_original() -> None:
    """PLC1 filter-vs-PINN on the real SWaT attacks. Recall wide; FPR stats.

    Two recall tables: the episode-uniform ``mean_recall`` and its pooled
    ``mean_timestep_recall`` companion (see plots.py). Same pivot, different
    metric column.
    """
    _dose_by_line(
        "plc1_poison_on_original",
        "mean_recall",
        "std_recall",
        "fig_plc1_poison_on_original",
    )
    _dose_by_line(
        "plc1_poison_on_original_timestep_recall",
        "mean_timestep_recall",
        "std_timestep_recall",
        "fig_plc1_poison_on_original_timestep_recall",
    )
    _line_stats(
        "plc1_poison_on_original_fpr",
        "mean_fpr",
        "std_fpr",
        "fig_plc1_poison_on_original_fpr_stats",
    )


def full_swat_poison_on_original() -> None:
    """Full-SWaT filter-vs-PINN on the real SWaT attacks (Data-only / PINN /
    Filter over poison fraction), MSE detector. Recall (episode-uniform and
    timestep-pooled) + FPR, all wide (for \\bandline)."""
    _dose_by_line(
        "full_swat_poison_on_original_mse",
        "mean_recall",
        "std_recall",
        "fig_full_swat_poison_on_original_mse",
    )
    _dose_by_line(
        "full_swat_poison_on_original_timestep_recall_mse",
        "mean_timestep_recall",
        "std_timestep_recall",
        "fig_full_swat_poison_on_original_timestep_recall_mse",
    )
    _dose_by_line(
        "full_swat_poison_on_original_fpr_mse",
        "mean_fpr",
        "std_fpr",
        "fig_full_swat_poison_on_original_fpr_mse",
    )


def full_swat_clean_data() -> None:
    """Full-SWaT clean-training per-arch bars (PINN / Data-only / Filter) on all
    real SWaT attacks, MSE detector, recall + FPR."""
    for name in ("full_swat_clean_data_mse", "full_swat_clean_data_fpr_mse"):
        df = _load(name)
        if df is None:
            continue
        out = _pivot_pinn(
            df, key_cols=["arch"], value_renames={"mean": "mean", "std": "std"}
        )
        if out.empty:
            continue
        out["arch"] = pd.Categorical(out["arch"], _ARCH_ORDER, ordered=True)
        out["arch_label"] = out["arch"].astype(str).map(_normalise)
        out = out.sort_values("arch").reset_index(drop=True)
        _save(out, f"fig_{name}")


def invariant_vs_dataonly() -> None:
    """(A) Data-only vs Invariant-filter recall over poison fraction (line-pivot)."""
    _dose_by_line(
        "invariant_vs_dataonly",
        "mean_recall",
        "std_recall",
        "fig_invariant_vs_dataonly",
    )


def filter_qsweep() -> None:
    """(B) Physics-filter quantile sweep: filter recall vs q, plus constant
    Data-only / PINN baseline columns (drawn as horizontal lines in the figure)."""
    df = _load("filter_qsweep_recall_vs_q")
    if df is None:
        return
    cols = [
        "q",
        "poison_mean",
        "poison_std",
        "dataonly_mean",
        "dataonly_std",
        "pinn_mean",
        "pinn_std",
    ]
    _save(
        df[[c for c in cols if c in df.columns]]
        .sort_values("q")
        .reset_index(drop=True),
        "fig_filter_qsweep",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # evaluation.py's --plot-dir is configurable, so this must be too: reading a
    # different directory than the one the figures were written to silently
    # exports stale tables (_load just prints [SKIP] for anything missing).
    global PLOTS_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plots-dir",
        default=str(PLOTS_DIR),
        help="Directory holding the flat per-figure CSVs written by plots.py "
        "(must match evaluation.py's --plot-dir). "
        f"Default: {PLOTS_DIR}",
    )
    args = ap.parse_args()
    PLOTS_DIR = Path(args.plots_dir)
    OUT_DIR = PLOTS_DIR / "pgfplots"
    if not PLOTS_DIR.is_dir():
        raise SystemExit(f"--plots-dir not found: {PLOTS_DIR}")

    print(f"Reading from {PLOTS_DIR}/")
    print(f"Writing to   {OUT_DIR}/")
    print()
    # bar groups
    clean_data()
    cross_sensor()
    cross_shape()
    # dose-response (pinn-pivot, incl. Filter)
    inconsistent_poisoning()
    inconsistent_poisoning_mse()
    consistent_poisoning()
    inconsistent_poisoning_per_architecture()
    # single-line companions
    ramp_rate()
    scarcity()
    gaussian_noise()
    stale_episodes()
    # line-pivot / per-w_phys / stats
    rate_sweep()
    wphys_ablation()
    full_swat_inconsistent_poisoning()
    plc1_poison_on_original()
    full_swat_poison_on_original()
    full_swat_clean_data()
    invariant_vs_dataonly()
    filter_qsweep()
    print()
    print(f"Done.  Open {OUT_DIR}/ for the pgfplots-ready CSVs.")


if __name__ == "__main__":
    main()
