"""v5 evaluation plots — single module, single flat output directory.

Reads the cache parquets produced by ``evaluation.py`` and emits the final
plot set (per-sensor is the headline detector; MSE variants carry a ``_mse``
suffix). Everything lands directly in ``plot_dir`` (no per-detector subfolders);
``generate_grid_plots`` is the single entry point. Functions print ``[SKIP]``
and continue when their cache is missing.

Output set (``plot_dir/<name>.pdf`` + a ``.csv`` sidecar, which
``export_pgfplots_csvs.py`` reshapes into the paper's tables):
  clean_data{,_timestep_recall,_fpr}      Per-arch bars on clean SWaT, split by
                                          {all, phys-valid, phys-invalid} attacks
                                          (sourced from v5_poison_lit@0).
  scarcity / gaussian_noise / stale_episodes
                                          Data-condition sweeps.
  cross_sensor / cross_shape / ramp_rate  Transfer sweeps of the LIT models.
  inconsistent_poisoning{,_fpr,_mse}      Inconsistent-poisoning dose-response
                                          (PINN / Data-only / Filter).
  consistent_poisoning                    Consistent-poisoning honest failure.
  wphys_ablation{,_fpr}                   One line per physics weight.
  rate_sweep{,_fpr}                       Filter-vs-PINN over training ramp rate.
  plc1_poison_on_original{,_timestep_recall,_fpr}
  full_swat_poison_on_original{,_timestep_recall,_fpr}_mse
                                          Filter-vs-PINN on the real SWaT attacks.
  full_swat_inconsistent_poisoning{,_fpr}
  full_swat_clean_data{,_fpr}_mse         Full-SWaT clean-training bars.
  invariant_vs_dataonly / filter_qsweep   Invariant filter; filter-quantile sweep.
  inconsistent_poisoning_per_architecture{,_fpr}   Per-arch lines (+ CSV).
  appendix/appendix_timestep*             Appendix tables (timestep-wise metrics).

Cache naming convention (from evaluation): ``{exp}.parquet`` (headline) and
``{exp}__grid__{test}.parquet`` (transfer grid).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ordering & labels (physics-validity order)
# ---------------------------------------------------------------------------
# Cross-regime labels (regime key → display name).
REGIME_LABELS = {
    "normal_swat": "Normal SWaT",
    "scarce_swat": "Scarce SWaT",
    "consistent_lit": "Poison LIT (consistent)",
    "consistent_fit": "Poison FIT (consistent)",
    "poison_fit": "Poison FIT (inconsistent)",
    "poison_lit": "Poison LIT (inconsistent)",
}

ARCHS = ["mlp", "cnn1d", "lstm", "transformer"]
ARCH_LABELS = {
    "mlp": "MLP",
    "cnn1d": "CNN1d",
    "lstm": "LSTM",
    "transformer": "Transformer",
}

DETECTOR_LABELS = {
    "mse": "MSE",
    "per_sensor": "Per-Sensor",
}
HERO_DETECTOR = "per_sensor"

# Axis labels per metric family. A dict rather than a ternary on purpose: the old
# `"Recall" if metric == "recall" else "FPR"` silently mislabelled every metric
# that was not "recall" (timestep_recall plotted under an "FPR" axis).
_METRIC_LABELS = {
    "recall": "Recall",
    "timestep_recall": "Timestep recall",
    "fpr": "FPR",
}


def _metric_label(metric: str) -> str:
    try:
        return _METRIC_LABELS[metric]
    except KeyError:
        raise ValueError(
            f"Unknown metric {metric!r}; expected one of {sorted(_METRIC_LABELS)}"
        ) from None

# PINN vs non-PINN styling
PINN_COLOR = "#1f77b4"  # blue
NOPINN_COLOR = "#888888"  # grey
PINN_EDGE = "#0d4c92"
NOPINN_EDGE = "#444444"

# Filter-vs-PINN palette (filter line is purple to distinguish from PINN blue
# and Data-only grey/orange).
FILTER_COLOR = "#7B3F99"
CLEAN_COLOR = "#444444"
POISONED_DATA_COLOR = "#DE8F05"  # orange — matches companion_ramp_rate

# Per-w_phys palette (w_phys ∈ {0.0, 0.5, 1.0, 5.0, 10.0, 100.0} → 6 distinguishable colors).
# Sorted ascending so w_phys=0 reads as Non-PINN-grey-ish and larger weights
# walk toward warm colors.
W_PHYS_COLORS = {
    0.0: "#888888",  # grey — matches NOPINN_COLOR by convention
    0.5: "#1f77b4",  # blue
    1.0: "#2ca02c",  # green
    5.0: "#d62728",  # red
    10.0: "#9467bd",  # purple
    100.0: "#e377c2",  # pink — largest physics weight in the ablation grid
}

TOTAL_TRAIN_ROWS = 345_600  # v5 full training span (96 h at 1 Hz)


# ---------------------------------------------------------------------------
# Cache loading + aggregation
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: Path, exp_name: str, test_name: str) -> Path:
    """``grid__`` caches are stored under ``{exp}__grid__{test}.parquet``."""
    return cache_dir / f"{exp_name}__grid__{test_name}.parquet"


def _load_grid_cache(
    cache_dir: Path, exp_name: str, test_name: str
) -> pd.DataFrame | None:
    p = _cache_path(cache_dir, exp_name, test_name)
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_plot_data(df: pd.DataFrame, pdf_path: Path):
    """Write the long-form data backing a figure to ``{pdf_path.stem}.csv``.

    Lets the user re-plot / re-style without re-aggregating from the cache parquets.
    Schema: long format — every row is one plotted datum (bar, line point, or
    heatmap cell), columns capture the grouping variables and the value(s).
    """
    csv_path = pdf_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Dose-response figures (3, 4)
# ---------------------------------------------------------------------------
def _plot_dose_response_lines(
    ax,
    df: pd.DataFrame,
    exp_var: str,
    detector: str,
    x_transform=None,
    title: str | None = None,
    metric: str = "recall",
    df_filter: pd.DataFrame | None = None,
):
    """Aggregate df over archs+seeds per (pinn_flag, exp_var); plot lines ± std.

    ``metric`` selects the column family: "recall" → ``avg_recall_{det}``
    (episode-uniform), "timestep_recall" → ``timestep_recall_{det}`` (pooled over
    steps, so long attacks dominate), "fpr" → ``fpr_{det}``. The y-axis is
    clamped to [0, 1.05] for every case.

    If ``df_filter`` is given, its data-only rows (pinn_flag==0) are drawn as a
    third "Filter" line (data-only training + hard physics filter). Filter rows
    are tagged ``pinn_flag = -1`` in the returned data.

    Returns a long-form DataFrame with the plotted points (for data export).
    """
    if metric == "recall":
        rc = f"avg_recall_{detector}"
        value_label = "recall"
    elif metric == "timestep_recall":
        rc = f"timestep_recall_{detector}"
        value_label = "timestep_recall"
    elif metric == "fpr":
        rc = f"fpr_{detector}"
        value_label = "fpr"
    else:
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'recall', 'timestep_recall' or 'fpr'"
        )
    out_rows: list[dict] = []
    if rc not in df.columns:
        ax.set_title(f"(no data)  {title or ''}")
        return pd.DataFrame(out_rows)

    lines = [
        (1, df, PINN_COLOR, "-", "o", "PINN"),
        (0, df, NOPINN_COLOR, "--", "o", "Non-PINN"),
    ]
    if df_filter is not None and rc in df_filter.columns:
        lines.append((0, df_filter, FILTER_COLOR, ":", "s", "Filter"))

    for pinn_flag, src, color, ls, marker, lbl in lines:
        sub = src[src["pinn_flag"] == pinn_flag]
        if sub.empty:
            continue
        g = sub.groupby(exp_var)[rc].agg(["mean", "std", "count"]).reset_index()
        g["std"] = g["std"].fillna(0)
        x_raw = g[exp_var].to_numpy(dtype=float)
        x = np.array([x_transform(v) for v in x_raw]) if x_transform else x_raw.copy()
        order = np.argsort(x)
        x = x[order]
        x_raw_ord = x_raw[order]
        m = g["mean"].to_numpy()[order]
        s = g["std"].to_numpy()[order]
        n = g["count"].to_numpy()[order]
        ax.plot(x, m, color=color, ls=ls, marker=marker, ms=4, label=lbl)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        # Tag the Filter line with pinn_flag=-1 in the exported data.
        out_flag = -1 if lbl == "Filter" else pinn_flag
        for xi_raw, xi, mi, si, ni in zip(x_raw_ord, x, m, s, n):
            out_rows.append(
                {
                    "exp_var": exp_var,
                    f"{exp_var}_raw": float(xi_raw),
                    f"{exp_var}_plotted": float(xi),
                    "pinn_flag": out_flag,
                    f"mean_{value_label}": float(mi),
                    f"std_{value_label}": float(si),
                    "n": int(ni),
                }
            )

    if title:
        ax.set_title(title, fontsize=10)
    # Recall is bounded in [0, 1] so clamp; FPR varies in arbitrary range
    if metric == "recall":
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return pd.DataFrame(out_rows)


def _plot_poison_dose_response(
    cache_dir: Path,
    plot_dir: Path,
    *,
    left_exp: str,
    right_exp: str,
    left_test: str,
    right_test: str,
    title: str,
    out_name: str,
    detector: str = HERO_DETECTOR,
    metric: str = "recall",
    left_filter_exp: str | None = None,
    right_filter_exp: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    data_rows: list[pd.DataFrame] = []
    for ax, exp, test, filt_exp, lbl in (
        (axes[0], left_exp, left_test, left_filter_exp, left_exp),
        (axes[1], right_exp, right_test, right_filter_exp, right_exp),
    ):
        df = _load_grid_cache(cache_dir, exp, test)
        if df is None:
            ax.set_title(f"{lbl} (no data)")
            continue
        df_filter = _load_grid_cache(cache_dir, filt_exp, test) if filt_exp else None
        df_panel = _plot_dose_response_lines(
            ax,
            df,
            "corrupt_param",
            detector,
            title=f"{REGIME_LABELS.get(lbl, lbl)}",
            metric=metric,
            df_filter=df_filter,
        )
        df_panel["training_exp"] = exp
        df_panel["test"] = test
        data_rows.append(df_panel)
        ax.set_xlabel("Poison fraction")
    y_label = _metric_label(metric)
    axes[0].set_ylabel(f"{y_label} ({DETECTOR_LABELS[detector]})")
    fig.suptitle(title, fontsize=12, y=1.03)
    out_path = plot_dir / out_name
    _savefig(fig, out_path)
    if data_rows:
        df_data = pd.concat(data_rows, ignore_index=True)
        df_data["detector"] = detector
        _save_plot_data(df_data, out_path)


def fig3_inconsistent_poison_dose_response(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """Dose-response for inconsistent poisoning (paper figure)."""
    _plot_poison_dose_response(
        cache_dir,
        plot_dir,
        left_exp="v5_poison_lit",
        left_test="inconsistent_lit_test",
        right_exp="v5_poison_fit",
        right_test="inconsistent_fit_test",
        left_filter_exp="v5_poison_lit_filter",
        right_filter_exp="v5_poison_fit_filter",
        title="Inconsistent poisoning: PINN advantage vs. poisoning strength",
        out_name="inconsistent_poisoning.pdf",
        detector=detector,
    )


def fig3_inconsistent_poison_fpr_dose_response(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """FPR companion to :func:`fig3_inconsistent_poison_dose_response`.

    Same axes layout (LIT | FIT) but plots ``fpr_{detector}`` instead of
    recall. Used to sanity-check that the recall gains aren't bought with
    higher false-alarm rates as poison fraction grows.
    """
    _plot_poison_dose_response(
        cache_dir,
        plot_dir,
        left_exp="v5_poison_lit",
        left_test="inconsistent_lit_test",
        right_exp="v5_poison_fit",
        right_test="inconsistent_fit_test",
        left_filter_exp="v5_poison_lit_filter",
        right_filter_exp="v5_poison_fit_filter",
        title="Inconsistent poisoning: FPR vs. poisoning strength",
        out_name="inconsistent_poisoning_fpr.pdf",
        detector=detector,
        metric="fpr",
    )


def _plot_dose_response_lines_by_wphys(
    ax,
    df: pd.DataFrame,
    exp_var: str,
    detector: str,
    title: str | None = None,
    metric: str = "recall",
):
    """Like :func:`_plot_dose_response_lines` but one line per w_phys value.

    Aggregates over (model, seed) within each (w_phys, exp_var) cell. Returns
    a long-form DataFrame of plotted points.

    ``metric`` selects the column family: "recall" → ``avg_recall_{det}``,
    "fpr" → ``fpr_{det}``.
    """
    if metric == "recall":
        rc = f"avg_recall_{detector}"
        value_label = "recall"
    elif metric == "fpr":
        rc = f"fpr_{detector}"
        value_label = "fpr"
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected 'recall' or 'fpr'")
    out_rows: list[dict] = []
    if rc not in df.columns or "w_phys" not in df.columns:
        ax.set_title(f"(no data)  {title or ''}")
        return pd.DataFrame(out_rows)

    wphys_values = sorted(df["w_phys"].dropna().unique().tolist())
    for wp in wphys_values:
        sub = df[np.isclose(df["w_phys"], wp)]
        if sub.empty:
            continue
        color = W_PHYS_COLORS.get(float(wp), "#444444")
        g = sub.groupby(exp_var)[rc].agg(["mean", "std", "count"]).reset_index()
        g["std"] = g["std"].fillna(0)
        x = g[exp_var].to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        m = g["mean"].to_numpy()[order]
        s = g["std"].to_numpy()[order]
        n = g["count"].to_numpy()[order]
        ax.plot(
            x,
            m,
            color=color,
            ls="-",
            marker="o",
            ms=4,
            label=f"w_phys={wp:g}",
        )
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        for xi, mi, si, ni in zip(x, m, s, n):
            out_rows.append(
                {
                    "exp_var": exp_var,
                    f"{exp_var}_raw": float(xi),
                    "w_phys": float(wp),
                    f"mean_{value_label}": float(mi),
                    f"std_{value_label}": float(si),
                    "n": int(ni),
                }
            )

    if title:
        ax.set_title(title, fontsize=10)
    # Recall is bounded in [0, 1] so clamp; FPR varies in arbitrary range
    if metric == "recall":
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return pd.DataFrame(out_rows)


def fig3_inconsistent_lit_recall_per_wphys(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """One line per w_phys on the inconsistent-LIT dose-response.

    Reads the cache produced by the ``v5_poison_lit_per_wphys`` selection
    (which keeps w_phys as a grouping key during model selection). Single
    panel; if the cache is missing the function writes a "(no data)"
    placeholder and returns.
    """
    df = _load_grid_cache(cache_dir, "v5_poison_lit_per_wphys", "inconsistent_lit_test")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    if df is None:
        ax.set_title("v5_poison_lit_per_wphys (no data)")
        out_path = plot_dir / "wphys_ablation.pdf"
        _savefig(fig, out_path)
        return

    df_panel = _plot_dose_response_lines_by_wphys(
        ax,
        df,
        "corrupt_param",
        detector,
        title="Poison LIT (inconsistent)",
    )
    ax.set_xlabel("Poison fraction")
    ax.set_ylabel(f"Recall ({DETECTOR_LABELS[detector]})")
    fig.suptitle(
        "Inconsistent LIT poisoning: recall by physics weight",
        fontsize=12,
        y=1.02,
    )
    out_path = plot_dir / "wphys_ablation.pdf"
    _savefig(fig, out_path)
    if not df_panel.empty:
        df_panel["training_exp"] = "v5_poison_lit_per_wphys"
        df_panel["test"] = "inconsistent_lit_test"
        df_panel["detector"] = detector
        _save_plot_data(df_panel, out_path)


def fig3_inconsistent_lit_fpr_per_wphys(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """FPR companion to :func:`fig3_inconsistent_lit_recall_per_wphys`.

    Same cache and axes layout, but plots ``fpr_{detector}`` instead of
    recall. Catches false-alarm trade-offs across the four w_phys values.
    """
    df = _load_grid_cache(cache_dir, "v5_poison_lit_per_wphys", "inconsistent_lit_test")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    if df is None:
        ax.set_title("v5_poison_lit_per_wphys (no data)")
        out_path = plot_dir / "wphys_ablation_fpr.pdf"
        _savefig(fig, out_path)
        return

    df_panel = _plot_dose_response_lines_by_wphys(
        ax,
        df,
        "corrupt_param",
        detector,
        title="Poison LIT (inconsistent)",
        metric="fpr",
    )
    ax.set_xlabel("Poison fraction")
    ax.set_ylabel(f"FPR ({DETECTOR_LABELS[detector]})")
    fig.suptitle(
        "Inconsistent LIT poisoning: FPR by physics weight",
        fontsize=12,
        y=1.02,
    )
    out_path = plot_dir / "wphys_ablation_fpr.pdf"
    _savefig(fig, out_path)
    if not df_panel.empty:
        df_panel["training_exp"] = "v5_poison_lit_per_wphys"
        df_panel["test"] = "inconsistent_lit_test"
        df_panel["detector"] = detector
        _save_plot_data(df_panel, out_path)


def fig3_inconsistent_poison_dose_response_mse(cache_dir: Path, plot_dir: Path):
    """Same as :func:`fig3_inconsistent_poison_dose_response` but evaluated
    with the MSE detector.  Used in the paper to motivate why we report the
    per-sensor detector instead — MSE aggregates per-channel residuals into a
    single score and loses sensitivity to LIT-specific anomalies as the
    poison fraction grows.
    """
    _plot_poison_dose_response(
        cache_dir,
        plot_dir,
        left_exp="v5_poison_lit",
        left_test="inconsistent_lit_test",
        right_exp="v5_poison_fit",
        right_test="inconsistent_fit_test",
        left_filter_exp="v5_poison_lit_filter",
        right_filter_exp="v5_poison_fit_filter",
        title="Inconsistent poisoning — MSE detector",
        out_name="inconsistent_poisoning_mse.pdf",
        detector="mse",
    )


def fig4_consistent_poison_failure(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """Consistent-poisoning failure case (honest reporting, paper figure)."""
    _plot_poison_dose_response(
        cache_dir,
        plot_dir,
        # LIT side uses the sinusoidal (physics-exact-at-borders) variant.
        left_exp="v5_consistent_lit_sin",
        left_test="consistent_lit_test",
        right_exp="v5_consistent_fit",
        right_test="consistent_fit_test",
        left_filter_exp="v5_consistent_lit_sin_filter",
        right_filter_exp="v5_consistent_fit_filter",
        title="Consistent poisoning: PINN has no signal when physics is preserved",
        out_name="consistent_poisoning.pdf",
        detector=detector,
    )


# ---------------------------------------------------------------------------
# Companion plots (A–G) — read per-experiment caches, write into plot_dir
# ---------------------------------------------------------------------------


def _aggregate_lines(
    df: pd.DataFrame,
    exp_var: str,
    detector: str,
) -> pd.DataFrame:
    """Mean ± std over (architecture, seed) per (pinn_flag, exp_var)."""
    rc = f"avg_recall_{detector}"
    fc = f"fpr_{detector}"
    rows: list[dict] = []
    if rc not in df.columns or exp_var not in df.columns:
        return pd.DataFrame(rows)
    for pinn_flag in (1, 0):
        sub = df[df["pinn_flag"] == pinn_flag]
        if sub.empty:
            continue
        for v, g in sub.groupby(exp_var):
            rows.append(
                {
                    exp_var: float(v) if not pd.isna(v) else np.nan,
                    "pinn_flag": pinn_flag,
                    "mean_recall": float(g[rc].mean()),
                    "std_recall": float(g[rc].std() if len(g) > 1 else 0.0),
                    "mean_fpr": float(g[fc].mean()) if fc in g.columns else np.nan,
                    "std_fpr": float(g[fc].std() if len(g) > 1 else 0.0)
                    if fc in g.columns
                    else np.nan,
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def _line_panel(
    ax,
    df_agg: pd.DataFrame,
    exp_var: str,
    x_transform=None,
    y_col: str = "mean_recall",
    err_col: str = "std_recall",
):
    """Plot PINN / Data-only (and Filter, pinn_flag=-1, if present) lines."""
    for pinn_flag, color, ls, lbl in (
        (1, PINN_COLOR, "-", "PINN"),
        (0, NOPINN_COLOR, "--", "Data-only"),
        (-1, FILTER_COLOR, ":", "Filter"),
    ):
        sub = df_agg[df_agg["pinn_flag"] == pinn_flag]
        if sub.empty:
            continue
        x_raw = sub[exp_var].to_numpy(dtype=float)
        x = np.array([x_transform(v) for v in x_raw]) if x_transform else x_raw
        order = np.argsort(x)
        x = x[order]
        m = sub[y_col].to_numpy()[order]
        s = sub[err_col].to_numpy()[order]
        ax.plot(x, m, color=color, ls=ls, marker="o", ms=5, label=lbl)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)


# ----- A) Clean Data: per-architecture × {all, phys-valid, phys-invalid} ----
def companion_clean_data(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """Per-architecture grouped bars on clean SWaT (clean training data).

    For each architecture: 3 grouped bar triples (All attacks / Phys-valid /
    Phys-invalid) × {PINN, Data-only, Filter}. Three PDFs: episode-uniform
    recall, timestep-pooled recall, and FPR.

    Clean-data baseline is sourced from ``v5_poison_lit`` at ``corrupt_param == 0``
    (the unpoisoned level of the LIT-poisoning grid) instead of a dedicated
    scarcity run, so it needs no separate experiment. The Filter bar comes from
    ``v5_poison_lit_filter`` at corrupt_param == 0 (data-only + physics filter).
    Reads the grid caches on all_swat_attacks / phys_valid_swat / phys_invalid_swat.

    In the exported CSV the three series are encoded by ``pinn_flag``:
    1 = PINN, 0 = Data-only, -1 = Filter.
    """
    rc = f"avg_recall_{detector}"
    fc = f"fpr_{detector}"
    SUBSETS = ["All attacks", "Phys-valid", "Phys-invalid"]
    TESTS = ["all_swat_attacks", "phys_valid_swat", "phys_invalid_swat"]

    def _clean0(d):
        if d is not None and "corrupt_param" in d.columns:
            return d[d["corrupt_param"] == 0.0]
        return d

    # PINN/Data-only from v5_poison_lit; Filter from v5_poison_lit_filter — both
    # at the clean (corrupt_param == 0) level, per test subset.
    main = {
        lbl: _clean0(_load_grid_cache(cache_dir, "v5_poison_lit", t))
        for lbl, t in zip(SUBSETS, TESTS)
    }
    filt = {
        lbl: _clean0(_load_grid_cache(cache_dir, "v5_poison_lit_filter", t))
        for lbl, t in zip(SUBSETS, TESTS)
    }
    # (output pinn_flag, source dict, pinn_flag to select, bar label).
    SERIES = [
        (1, main, 1, "PINN"),
        (0, main, 0, "Data-only"),
        (-1, filt, 0, "Filter"),
    ]

    def _collect(metric_col):
        rows: list[dict] = []
        for arch in ARCHS:
            for subset_lbl in SUBSETS:
                for out_flag, src, pf, _lbl in SERIES:
                    d = src.get(subset_lbl)
                    if d is None or metric_col not in d.columns:
                        continue
                    g = d[(d["model"] == arch) & (d["pinn_flag"] == pf)]
                    if g.empty:
                        continue
                    rows.append(
                        {
                            "arch": arch,
                            "subset": subset_lbl,
                            "pinn_flag": out_flag,
                            "mean": float(g[metric_col].mean()),
                            "std": float(g[metric_col].std() if len(g) > 1 else 0.0),
                            "n": int(len(g)),
                        }
                    )
        return pd.DataFrame(rows)

    def _draw(metric_col, y_label, vmax, suffix):
        df_data = _collect(metric_col)
        if df_data.empty:
            df_data = pd.DataFrame(
                columns=["arch", "subset", "pinn_flag", "mean", "std", "n"]
            )
        fig, axes = plt.subplots(
            1, len(ARCHS), figsize=(4 * len(ARCHS), 4.5), sharey=True
        )
        x = np.arange(len(SUBSETS))
        width = 0.26
        for ax, arch in zip(axes, ARCHS):
            for offset, pinn_flag, color, hatch, edge, lbl in (
                (-width, 1, PINN_COLOR, None, PINN_EDGE, "PINN"),
                (0.0, 0, "white", "///", NOPINN_EDGE, "Data-only"),
                (width, -1, FILTER_COLOR, None, FILTER_COLOR, "Filter"),
            ):
                means = []
                stds = []
                for s in SUBSETS:
                    row = df_data[
                        (df_data["arch"] == arch)
                        & (df_data["subset"] == s)
                        & (df_data["pinn_flag"] == pinn_flag)
                    ]
                    if row.empty:
                        means.append(np.nan)
                        stds.append(0.0)
                    else:
                        means.append(row.iloc[0]["mean"])
                        stds.append(row.iloc[0]["std"])
                ax.bar(
                    x + offset,
                    means,
                    width,
                    yerr=stds,
                    color=color,
                    edgecolor=edge,
                    hatch=hatch,
                    capsize=3,
                    label=lbl,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(SUBSETS, rotation=15, ha="right", fontsize=8)
            ax.set_title(ARCH_LABELS[arch], fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            ax.set_ylim(0, vmax)
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f"Clean Data (rows=100%) — {y_label} ({DETECTOR_LABELS[detector]})",
            fontsize=12,
            y=1.03,
        )
        out = plot_dir / f"clean_data{suffix}.pdf"
        _savefig(fig, out)
        _save_plot_data(df_data.assign(detector=detector, metric=metric_col), out)

    _draw(rc, "Mean recall", 1.05, "")
    # Timestep-pooled companion to the episode-uniform recall above. Sourced from
    # the same grid caches, so a subset whose cache predates the timestep_recall
    # column is skipped by _collect and its bars simply do not appear.
    _draw(
        f"timestep_recall_{detector}",
        "Mean timestep recall",
        1.05,
        "_timestep_recall",
    )
    _draw(fc, "False positive rate", 0.30, "_fpr")


# ----- Appendix: timestep-wise table (companion to clean_data) --------------
_APPENDIX_TAG = "appendix_timestep"


_APPENDIX_TAG_FULL = "appendix_timestep_full"
# The paper's operating point (evaluation.py DETECT_CFG q_fused); the two headline
# full-SWaT tables report this quantile, the sweep table reports all of them.
_APPENDIX_BASE_Q = 0.99

_APPENDIX_METRICS = [
    "recall",
    "precision",
    "f1",
    "fpr_clean",
    "fpr_attackset",
    # the main figure's attack-averaged recall, for cross-reference
    "avg_recall",
]


def _appendix_table_from_caches(
    cache_dir: Path,
    plot_dir: Path,
    *,
    main_cache: str,
    filter_cache: str,
    out_base: str,
    detector: str,
    metrics: list[str] | None = None,
    extra_cols: tuple[str, ...] = (),
    quantile: float | None = None,
):
    """Write one appendix table (raw + aggregated) from a pair of caches.

    ``detector`` selects the rows to score. Caches may hold several detector
    variants per model, so without this filter the aggregation would average them
    together and report an inflated model count.
    """
    metrics = _APPENDIX_METRICS if metrics is None else metrics

    main = _load_cache(cache_dir, main_cache)
    filt = _load_cache(cache_dir, filter_cache)
    if main is None and filt is None:
        print(f"  [SKIP] {out_base}: no cache in {cache_dir}")
        return

    # (output pinn_flag, source frame, pinn_flag to select, row label).
    SERIES = [
        (1, main, 1, "PINN"),
        (0, main, 0, "Data-only"),
        (-1, filt, 0, "Filter"),
    ]

    raw_frames: list[pd.DataFrame] = []
    agg_rows: list[dict] = []
    for out_flag, src, pf, variant in SERIES:
        if src is None or "pinn_flag" not in src.columns:
            continue
        g = src[src["pinn_flag"] == pf]
        # Caches written while the windowed protocols existed carry a `protocol`
        # column; keep only the full-trace rows so they aggregate correctly.
        if "protocol" in g.columns:
            g = g[g["protocol"] == "full"]
        if "detector" in g.columns:
            g = g[g["detector"] == detector]
        # The full-SWaT cache holds one row per (model, detector, quantile).
        if quantile is not None and "q" in g.columns:
            g = g[np.isclose(g["q"], quantile)]
        if g.empty:
            continue
        # One row per model after filtering, or the mean ± std below is meaningless.
        if "run_dir" in g.columns and g["run_dir"].duplicated().any():
            raise ValueError(
                f"{out_base}/{variant}: duplicate run_dir rows after filtering on "
                f"detector={detector!r} — the aggregate would mix variants."
            )

        # alarm_bits is a packed per-step blob kept in the parquet cache for
        # offline re-scoring; it must not reach the CSV.
        raw = g.drop(columns=["alarm_bits"], errors="ignore").rename(
            columns={"model": "arch"}
        )
        raw["pinn_flag"] = out_flag
        raw.insert(0, "variant", variant)
        raw_frames.append(raw)

        row = {
            "variant": variant,
            "pinn_flag": out_flag,
            "detector": detector,
            "n": int(len(g)),
        }
        if "k_ratio" in g.columns:
            row["k_ratio"] = float(g["k_ratio"].mean())
            row["n_pos"] = int(g["n_pos"].iloc[0])
            row["n_neg"] = int(g["n_neg"].iloc[0])
        for c in extra_cols:
            if c in g.columns:
                row[c] = float(g[c].mean())
        for c in metrics:
            if c not in g.columns:
                continue
            row[f"mean_{c}"] = float(g[c].mean())
            row[f"std_{c}"] = float(g[c].std() if len(g) > 1 else 0.0)
        agg_rows.append(row)

    if not agg_rows:
        print(f"  [SKIP] {out_base}: caches present but hold no models")
        return

    out_dir = plot_dir / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{out_base}_raw.csv"
    pd.concat(raw_frames, ignore_index=True).to_csv(raw_path, index=False)
    print(f"  Saved: {raw_path}")

    agg_path = out_dir / f"{out_base}.csv"
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    print(f"  Saved: {agg_path}")


def appendix_timestep_table(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """Camera-ready appendix table: timestep-wise recall/precision/F1 + two FPRs.

    Companion to ``companion_clean_data`` — same models (clean training,
    ``corrupt_param == 0``), same test set (original SWaT attacks, PLC1 scope),
    same detector — but scored with the conventions a reader assumes:

    * recall/precision/F1 are timestep-wise;
    * the PLC1 scope is a "don't care" mask over other stages' attack rows;
    * two false-positive rates. ``fpr_clean`` is measured on the attack-free
      clean-normal slice, so it carries no boundary contamination and reproduces
      the headline figure's FPR exactly. ``fpr_attackset`` is measured on the
      normal-labelled timesteps of the evaluation set itself — the same negatives
      the precision denominator uses, so
      ``precision = TP / (TP + fpr_attackset * (FP + TN))`` closes.

    Two CSVs under ``plot_dir/appendix``: one raw row per model,
    and the aggregated three-row table (Data-only / PINN / Filter, mean ± std over
    4 architectures × 10 seeds).
    """
    _appendix_table_from_caches(
        cache_dir,
        plot_dir,
        main_cache=f"v5_poison_lit__{_APPENDIX_TAG}",
        filter_cache=f"v5_poison_lit_filter__{_APPENDIX_TAG}",
        out_base="appendix_timestep",
        detector=detector,
    )


_CHANNEL_JSON_COLS = {
    "channel_share_N": "share_N",
    "channel_share_A": "share_A",
    "channel_share_fp": "share_fp",
    "channel_ratio_A_N": "ratio_A_N",
    "channel_loo_fpr": "loo_fpr",
    "channel_loo_recall": "loo_recall",
}


def appendix_timestep_full_quantile_sweep(cache_dir: Path, plot_dir: Path):
    """MSE detection quantile sweep for the full-SWaT models.

    One row per (variant, detector, q), written to
    ``appendix/appendix_timestep_full_qsweep.csv``. This is the table that says
    whether any operating point of the MSE detector gives both a usable
    false-positive rate on the attack set and a recall worth reporting.

    Read ``mean_fpr_attackset`` against ``mean_recall``: the sweep is a
    recall/FPR trade-off curve, not a search for a good-looking number. Note the
    estimator limit — the fused threshold slice yields ~8635 score samples, so
    q=0.999 rests on ~8.6 of them and q=0.9995 on ~4; ``std_fpr_attackset`` across
    the models is the honest read on how stable each point is.
    """
    main = _load_cache(cache_dir, f"v5_full_poison_lit__{_APPENDIX_TAG_FULL}")
    filt = _load_cache(cache_dir, f"v5_full_poison_lit_filter__{_APPENDIX_TAG_FULL}")
    if main is None and filt is None:
        print(f"  [SKIP] appendix_timestep_full_qsweep: no cache in {cache_dir}")
        return
    if main is not None and "q" not in main.columns:
        print(
            "  [SKIP] appendix_timestep_full_qsweep: cache predates the quantile "
            "sweep — delete it and re-run to populate"
        )
        return

    SERIES = [(main, 1, "PINN"), (main, 0, "Data-only"), (filt, 0, "Filter")]
    METRICS = ["recall", "precision", "f1", "fpr_clean", "fpr_attackset", "avg_recall"]

    rows: list[dict] = []
    for src, pf, variant in SERIES:
        if src is None or "q" not in src.columns:
            continue
        for det in sorted(src["detector"].unique()):
            for q in sorted(src["q"].unique()):
                g = src[
                    (src["pinn_flag"] == pf)
                    & (src["detector"] == det)
                    & (np.isclose(src["q"], q))
                ]
                if g.empty:
                    continue
                row = {"variant": variant, "detector": det, "q": float(q),
                       "n": int(len(g))}
                if "k_ratio" in g.columns:
                    row["k_ratio"] = float(g["k_ratio"].mean())
                for c in METRICS:
                    if c in g.columns:
                        row[f"mean_{c}"] = float(g[c].mean())
                        row[f"std_{c}"] = float(g[c].std() if len(g) > 1 else 0.0)
                rows.append(row)

    if not rows:
        print("  [SKIP] appendix_timestep_full_qsweep: caches hold no models")
        return

    out_dir = plot_dir / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "appendix_timestep_full_qsweep.csv"
    pd.DataFrame(rows).sort_values(["detector", "variant", "q"]).to_csv(
        out, index=False
    )
    print(f"  Saved: {out}")


def appendix_timestep_channel_table(cache_dir: Path, plot_dir: Path):
    """Per-residual-channel attribution for the full-SWaT MSE detector.

    One row per (variant, channel), averaged over that variant's models. Written
    to ``appendix/appendix_timestep_full_channels.csv``, sorted by ``share_fp``
    descending — so the channels driving the false positives are at the top.

    Columns to read, in order of usefulness:

    * ``share_fp`` — fraction of the MSE score this channel contributes on the
      steps where the detector wrongly fires. The direct culprit ranking.
    * ``ratio_A_N`` — how much the channel's mean squared residual grows from the
      clean threshold slice to the attack recording. Separates "always noisy"
      from "broke on this dataset".
    * ``loo_fpr`` / ``loo_recall`` — what the detector would score with this one
      channel excluded and the threshold refit, next to ``base_fpr`` /
      ``base_recall``. ``delta_fpr`` is ``loo_fpr - base_fpr``: large negative
      means dropping this channel fixes the false-positive rate.
    """
    main = _load_cache(cache_dir, f"v5_full_poison_lit__{_APPENDIX_TAG_FULL}")
    filt = _load_cache(cache_dir, f"v5_full_poison_lit_filter__{_APPENDIX_TAG_FULL}")
    if main is None and filt is None:
        print(f"  [SKIP] appendix_timestep_full_channels: no cache in {cache_dir}")
        return

    SERIES = [(main, 1, "PINN"), (main, 0, "Data-only"), (filt, 0, "Filter")]

    rows: list[dict] = []
    for src, pf, variant in SERIES:
        if src is None or "channel_share_fp" not in src.columns:
            continue
        # Channel attribution rides on the baseline-MSE row only.
        g = src[(src["pinn_flag"] == pf) & (src["detector"] == "mse")]
        g = g[g["channel_share_fp"].astype(str).str.len() > 0]
        if g.empty:
            continue

        acc: dict[str, dict[str, list]] = {}
        for _, r in g.iterrows():
            for col, short in _CHANNEL_JSON_COLS.items():
                for ch, v in json.loads(r[col]).items():
                    acc.setdefault(ch, {}).setdefault(short, []).append(v)

        base_fpr = float(g["fpr_attackset"].mean())
        base_recall = float(g["recall"].mean())
        for ch, stats in acc.items():
            row = {"variant": variant, "channel": ch, "n": int(len(g))}
            for short in _CHANNEL_JSON_COLS.values():
                row[short] = float(np.nanmean(stats.get(short, [np.nan])))
            row["base_fpr"] = base_fpr
            row["base_recall"] = base_recall
            row["delta_fpr"] = row["loo_fpr"] - base_fpr
            row["delta_recall"] = row["loo_recall"] - base_recall
            rows.append(row)

    if not rows:
        print(
            "  [SKIP] appendix_timestep_full_channels: cache has no channel "
            "diagnostics (written before they were added — delete it and re-run)"
        )
        return

    df = pd.DataFrame(rows).sort_values(
        ["variant", "share_fp"], ascending=[True, False]
    )
    out_dir = plot_dir / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "appendix_timestep_full_channels.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")


def appendix_timestep_table_full(cache_dir: Path, plot_dir: Path):
    """Full-SWaT companion to ``appendix_timestep_table``, MSE detector.

    Same clean-training setting on the original SWaT attacks, but every catalogued
    attack counts as a positive (full-SWaT models are in scope everywhere, so there
    is no don't-care class) and the reported detector is MSE.

    One table, ``appendix_timestep_full``, over every residual column.

    Reading the numbers: ``k_ratio`` is far smaller here than in the PLC1 table, so
    the precision columns of the two tables are NOT comparable. ``mean_recall`` is
    dominated by attack 28 (~2/3 of all attack rows) — ``recall_ex28`` in the raw
    CSV holds it out. ``n_targets`` is the residual width, i.e. the check that the
    column drop actually took effect.
    """
    _appendix_table_from_caches(
        cache_dir,
        plot_dir,
        main_cache=f"v5_full_poison_lit__{_APPENDIX_TAG_FULL}",
        filter_cache=f"v5_full_poison_lit_filter__{_APPENDIX_TAG_FULL}",
        out_base="appendix_timestep_full",
        detector="mse",
        quantile=_APPENDIX_BASE_Q,
        extra_cols=(
            "recall_ex28",
            "n_label_only_pos",
            "n_targets",
        ),
    )


# ----- B/C/D/G common: single-panel line plot averaging over arches ---------
def _companion_single_line(
    cache_dir: Path,
    plot_dir: Path,
    *,
    cache_name: str,
    exp_var: str,
    x_label: str,
    title: str,
    out_name: str,
    detector: str = HERO_DETECTOR,
    x_transform=None,
    x_log: bool = False,
    filter_corrupt_param_max: bool = False,
    filter_cache: str | None = None,
):
    """Single-panel PINN vs Data-only line plot averaged over architectures.

    Loads `{cache_name}.parquet` from the cache dir.  If the cache contains a
    `corrupt_param` column with the baseline (0.0) level included and the user
    only wants the worst-case sweep, set ``filter_corrupt_param_max=True``.

    If ``filter_cache`` is given, its data-only models (pinn_flag==0) are drawn
    as a third "Filter" line (tagged pinn_flag=-1).
    """

    df = _load_cache(cache_dir, cache_name)
    if df is None:
        print(f"  [SKIP {out_name}] cache {cache_name}.parquet not found")
        return

    if filter_corrupt_param_max and "corrupt_param" in df.columns:
        df = df[df["corrupt_param"] == df["corrupt_param"].max()]

    df_agg = _aggregate_lines(df, exp_var, detector)
    if df_agg.empty:
        print(f"  [SKIP {out_name}] no rows after aggregation")
        return

    if filter_cache is not None:
        df_f = _load_cache(cache_dir, filter_cache)
        if df_f is not None:
            if filter_corrupt_param_max and "corrupt_param" in df_f.columns:
                df_f = df_f[df_f["corrupt_param"] == df_f["corrupt_param"].max()]
            agg_f = _aggregate_lines(df_f, exp_var, detector)
            agg_f = agg_f[agg_f["pinn_flag"] == 0].copy()
            agg_f["pinn_flag"] = -1  # tag as Filter
            df_agg = pd.concat([df_agg, agg_f], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _line_panel(
        axes[0],
        df_agg,
        exp_var,
        x_transform=x_transform,
        y_col="mean_recall",
        err_col="std_recall",
    )
    _line_panel(
        axes[1],
        df_agg,
        exp_var,
        x_transform=x_transform,
        y_col="mean_fpr",
        err_col="std_fpr",
    )

    for ax, ylabel, ylim in (
        (axes[0], f"Mean recall ({DETECTOR_LABELS[detector]})", (0, 1.05)),
        (axes[1], "False positive rate", (0, 0.30)),
    ):
        if x_log:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"{title}  (avg over architectures, {DETECTOR_LABELS[detector]})",
        fontsize=12,
        y=1.03,
    )
    out = plot_dir / out_name
    _savefig(fig, out)

    df_agg["detector"] = detector
    df_agg["source_cache"] = cache_name
    _save_plot_data(df_agg, out)


def companion_scarcity(cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR):
    """B) Scarcity: x = training rows (log fraction), avg over arches."""
    _companion_single_line(
        cache_dir,
        plot_dir,
        cache_name="v5_scarcity",
        filter_cache="v5_scarcity_filter",
        exp_var="rows",
        x_label="Training fraction",
        title="Scarcity: PINN vs Data-only",
        out_name="scarcity.pdf",
        detector=detector,
        x_transform=lambda v: v / TOTAL_TRAIN_ROWS,
        x_log=True,
    )


def companion_gaussian_noise(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """C) Faulty-Gaussian: x = noise std fraction, avg over arches."""
    _companion_single_line(
        cache_dir,
        plot_dir,
        cache_name="v5_faulty_gaussian",
        filter_cache="v5_faulty_gaussian_filter",
        exp_var="corrupt_param",
        x_label="Gaussian noise std (fraction)",
        title="Gaussian Sensor Noise: PINN vs Data-only",
        out_name="gaussian_noise.pdf",
        detector=detector,
    )


def companion_stuck(cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR):
    """D) Faulty-stuck: x = stuck fraction, avg over arches."""
    _companion_single_line(
        cache_dir,
        plot_dir,
        cache_name="v5_faulty_stuck",
        filter_cache="v5_faulty_stuck_filter",
        exp_var="corrupt_param",
        x_label="Stuck fraction",
        title="Stuck-At Faults: PINN vs Data-only",
        out_name="stale_episodes.pdf",
        detector=detector,
    )


# ----- E) Cross-Sensor Transfer (avg over arches) ----------------------------
def companion_cross_sensor(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """E) Cross-Sensor: 2×2 grid of single bar pairs (training × test sensor),
    averaged over architectures (and the highest poison fraction)."""
    rc = f"avg_recall_{detector}"

    # (row_label_long, cache_key) pairs.  rows = trained-on, cols = tested-on.
    cells = {
        ("LIT", "LIT"): "v5_poison_lit",  # same-sensor LIT
        ("LIT", "FIT"): "v5_poison_lit__on_fit101_linear",  # cross-sensor
        ("FIT", "LIT"): "v5_poison_fit__on_lit101_linear",  # cross-sensor
        ("FIT", "FIT"): "v5_poison_fit",  # same-sensor FIT
    }

    # Aggregate each cell to (pinn_flag) -> (mean, std)
    agg: dict = {}
    rows_for_csv: list[dict] = []
    for (train, test), name in cells.items():
        df = _load_cache(cache_dir, name)
        if df is None or rc not in df.columns:
            agg[(train, test)] = None
            continue
        if "corrupt_param" in df.columns:
            df = df[df["corrupt_param"] == df["corrupt_param"].max()]
        slot = {}
        for pinn_flag in (1, 0):
            sub = df[df["pinn_flag"] == pinn_flag]
            if sub.empty:
                continue
            slot[pinn_flag] = (
                float(sub[rc].mean()),
                float(sub[rc].std() if len(sub) > 1 else 0.0),
                int(len(sub)),
            )
            rows_for_csv.append(
                {
                    "trained_on": train,
                    "tested_on": test,
                    "source_cache": name,
                    "pinn_flag": pinn_flag,
                    "mean_recall": slot[pinn_flag][0],
                    "std_recall": slot[pinn_flag][1],
                    "n": slot[pinn_flag][2],
                }
            )
        # Filter baseline (data-only + physics filter) — pinn_flag = -1.
        fname = _filter_cache_name(name)
        fdf = _load_cache(cache_dir, fname)
        if fdf is not None and rc in fdf.columns:
            if "corrupt_param" in fdf.columns:
                fdf = fdf[fdf["corrupt_param"] == fdf["corrupt_param"].max()]
            fsub = fdf[fdf["pinn_flag"] == 0]
            if not fsub.empty:
                slot[-1] = (
                    float(fsub[rc].mean()),
                    float(fsub[rc].std() if len(fsub) > 1 else 0.0),
                    int(len(fsub)),
                )
                rows_for_csv.append(
                    {
                        "trained_on": train,
                        "tested_on": test,
                        "source_cache": fname,
                        "pinn_flag": -1,
                        "mean_recall": slot[-1][0],
                        "std_recall": slot[-1][1],
                        "n": slot[-1][2],
                    }
                )
        agg[(train, test)] = slot

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    train_order = ["LIT", "FIT"]
    test_order = ["LIT", "FIT"]
    for r, train in enumerate(train_order):
        for c, test in enumerate(test_order):
            ax = axes[r, c]
            slot = agg[(train, test)]
            x = np.arange(3)
            width = 0.6
            # (x position, slot key, color, edge, hatch, label)
            bar_specs = [
                (0, 1, PINN_COLOR, PINN_EDGE, None, "PINN"),
                (1, 0, "white", NOPINN_EDGE, "///", "Data-only"),
                (2, -1, FILTER_COLOR, FILTER_COLOR, None, "Filter"),
            ]
            if slot is None or 1 not in slot or 0 not in slot:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="gray",
                )
            else:
                for xi, key, color, edge, hatch, lbl in bar_specs:
                    if key not in slot:
                        continue  # Filter may be absent until its config is trained
                    ax.bar(
                        x[xi],
                        slot[key][0],
                        width,
                        yerr=slot[key][1],
                        color=color,
                        edgecolor=edge,
                        hatch=hatch,
                        capsize=3,
                        label=lbl,
                    )
                    ax.text(
                        x[xi],
                        slot[key][0] + 0.02,
                        f"{slot[key][0]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
            same = "same-sensor" if train == test else "cross-sensor"
            ax.set_title(
                f"Trained on {train}101 → Tested on {test}101 ({same})", fontsize=10
            )
            ax.set_xticks(x)
            ax.set_xticklabels(["PINN", "Data-only", "Filter"], fontsize=9)
            ax.set_ylim(0, 1.10)
            ax.grid(True, axis="y", alpha=0.3)
    axes[0, 0].set_ylabel(f"Mean recall ({DETECTOR_LABELS[detector]})")
    axes[1, 0].set_ylabel(f"Mean recall ({DETECTOR_LABELS[detector]})")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Cross-Sensor Transfer (corrupt_param=0.3, avg over arch)", fontsize=12, y=1.02
    )
    out = plot_dir / "cross_sensor.pdf"
    _savefig(fig, out)
    df_csv = pd.DataFrame(rows_for_csv)
    df_csv["detector"] = detector
    _save_plot_data(df_csv, out)


# ----- F) Cross-Shape Transfer (avg over arches, dashed split linear|rest) ---
def companion_cross_shape(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """F) Cross-Shape: 1×2 panels (LIT/FIT), bars per attack shape averaged
    over architectures, vertical dashed line between linear (training shape)
    and the held-out shapes."""
    rc = f"avg_recall_{detector}"

    shape_cfg = {
        "lit": {
            "source": "v5_poison_lit",
            "shapes": [
                ("Linear", None),  # main cache
                ("Staircase", "v5_poison_lit__on_lit_staircase"),
                ("Sigmoid", "v5_poison_lit__on_lit_sigmoid"),
                ("Noisy Ramp", "v5_poison_lit__on_lit_noisy_ramp"),
            ],
            "label": "LIT101 (poison_lit)",
        },
        "fit": {
            "source": "v5_poison_fit",
            "shapes": [
                ("Linear", None),
                ("Staircase", "v5_poison_fit__on_fit_staircase"),
                ("Sigmoid", "v5_poison_fit__on_fit_sigmoid"),
                ("Noisy Ramp", "v5_poison_fit__on_fit_noisy_ramp"),
            ],
            "label": "FIT101 (poison_fit)",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    rows_for_csv: list[dict] = []
    for ax, (sensor_key, cfg) in zip(axes, shape_cfg.items()):
        x_pos = np.arange(len(cfg["shapes"]))
        width = 0.26
        # (offset, out pinn_flag, use filter cache, pinn_flag to select, color,
        #  hatch, edge, label). Filter = data-only models trained with the filter.
        for offset, out_flag, is_filter, sel_pinn, color, hatch, edge, lbl in (
            (-width, 1, False, 1, PINN_COLOR, None, PINN_EDGE, "PINN"),
            (0.0, 0, False, 0, "white", "///", NOPINN_EDGE, "Data-only"),
            (width, -1, True, 0, FILTER_COLOR, None, FILTER_COLOR, "Filter"),
        ):
            means, stds = [], []
            for shape_label, alt_cache in cfg["shapes"]:
                base = alt_cache or cfg["source"]
                cache_name = _filter_cache_name(base) if is_filter else base
                df = _load_cache(cache_dir, cache_name)
                if df is None or rc not in df.columns:
                    means.append(np.nan)
                    stds.append(0.0)
                    continue
                if "corrupt_param" in df.columns:
                    df = df[df["corrupt_param"] == df["corrupt_param"].max()]
                sub = df[df["pinn_flag"] == sel_pinn]
                if sub.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                    continue
                means.append(float(sub[rc].mean()))
                stds.append(float(sub[rc].std() if len(sub) > 1 else 0.0))
                rows_for_csv.append(
                    {
                        "sensor": sensor_key,
                        "shape": shape_label,
                        "source_cache": cache_name,
                        "pinn_flag": out_flag,
                        "mean_recall": means[-1],
                        "std_recall": stds[-1],
                        "n": int(len(sub)),
                    }
                )
            ax.bar(
                x_pos + offset,
                means,
                width,
                yerr=stds,
                color=color,
                edgecolor=edge,
                hatch=hatch,
                capsize=3,
                label=lbl,
            )
        # Dashed line between Linear (idx 0) and the rest (idx>=1)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            0.5,
            1.01,
            "trained shape │ held-out shapes",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0] for s in cfg["shapes"]])
        ax.set_title(cfg["label"], fontweight="bold")
        ax.set_ylim(0, 1.10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel(f"Mean recall ({DETECTOR_LABELS[detector]})")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Cross-Shape Transfer (corrupt_param=0.3, avg over arch)", fontsize=12, y=1.03
    )
    out = plot_dir / "cross_shape.pdf"
    _savefig(fig, out)
    df_csv = pd.DataFrame(rows_for_csv)
    df_csv["detector"] = detector
    _save_plot_data(df_csv, out)


# ----- G) Ramp-rate Generalization (avg over arches) ------------------------
def companion_ramp_rate(cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR):
    """G) Ramp-rate: x = LIT101 ramp rate (mm/s, log scale), avg over arches."""
    rc = f"avg_recall_{detector}"
    fc = f"fpr_{detector}"

    from scripts.evaluation import (
        RAMP_RATES_LIT,
        V3_TRAINING_RATE_LIT,
        _rate_tag,
    )

    rows_for_csv: list[dict] = []
    for rate in RAMP_RATES_LIT:
        # The training-rate point reads the headline v5_poison_lit cache
        # (no separate per-rate test parquet exists for it — see
        # _run_ramp_rates).  Every other rate uses its per-rate cache.
        if rate == V3_TRAINING_RATE_LIT:
            cache_name = "v5_poison_lit"
        else:
            cache_name = f"v5_poison_lit__on_lit_rate_{_rate_tag(rate)}"
        # Main (PINN / Data-only) from v5_poison_lit; Filter from
        # v5_poison_lit_filter (data-only pinn_flag=0 → output pinn_flag=-1).
        for src_cache, sel_pinn, out_flag in (
            (cache_name, 1, 1),
            (cache_name, 0, 0),
            (_filter_cache_name(cache_name), 0, -1),
        ):
            df = _load_cache(cache_dir, src_cache)
            if df is None or rc not in df.columns:
                continue
            if "corrupt_param" in df.columns:
                df = df[df["corrupt_param"] == df["corrupt_param"].max()]
            sub = df[df["pinn_flag"] == sel_pinn]
            if sub.empty:
                continue
            rows_for_csv.append(
                {
                    "rate_m_per_s": float(rate),
                    "rate_mm_per_s": float(rate * 1000),
                    "is_training_rate": rate == V3_TRAINING_RATE_LIT,
                    "pinn_flag": out_flag,
                    "mean_recall": float(sub[rc].mean()),
                    "std_recall": float(sub[rc].std() if len(sub) > 1 else 0.0),
                    "mean_fpr": float(sub[fc].mean()) if fc in sub.columns else np.nan,
                    "std_fpr": float(sub[fc].std() if len(sub) > 1 else 0.0)
                    if fc in sub.columns
                    else np.nan,
                    "n": int(len(sub)),
                }
            )
    df_csv = pd.DataFrame(rows_for_csv)
    if df_csv.empty:
        print("  [SKIP ramp_rate] no caches found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax_idx, (y_col, err_col, ylabel, ylim) in enumerate(
        (
            (
                "mean_recall",
                "std_recall",
                f"Mean recall ({DETECTOR_LABELS[detector]})",
                (0, 1.05),
            ),
            ("mean_fpr", "std_fpr", "False positive rate", (0, 0.30)),
        )
    ):
        ax = axes[ax_idx]
        for pinn_flag, color, ls, lbl in (
            (1, PINN_COLOR, "-", "PINN"),
            (0, NOPINN_COLOR, "--", "Data-only"),
            (-1, FILTER_COLOR, ":", "Filter"),
        ):
            sub = df_csv[df_csv["pinn_flag"] == pinn_flag].sort_values("rate_mm_per_s")
            if sub.empty:
                continue
            ax.plot(
                sub["rate_mm_per_s"],
                sub[y_col],
                color=color,
                ls=ls,
                marker="o",
                ms=5,
                label=lbl,
            )
            ax.fill_between(
                sub["rate_mm_per_s"],
                sub[y_col] - sub[err_col],
                sub[y_col] + sub[err_col],
                color=color,
                alpha=0.15,
            )
        # Mark the training rate (read from headline cache).
        train_mm = V3_TRAINING_RATE_LIT * 1000
        ax.axvline(train_mm, color="black", linestyle=":", alpha=0.6)
        ax.text(
            train_mm,
            ylim[1] * 0.95,
            f"  trained @ {train_mm:.3f} mm/s",
            fontsize=8,
            color="gray",
            va="top",
        )
        ax.set_xscale("log")
        ax.set_xlabel("LIT101 ramp rate (mm/s)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Ramp-Rate Generalization (poison_lit @ corrupt_param=0.3, "
        f"avg over arch, {DETECTOR_LABELS[detector]})",
        fontsize=11,
        y=1.03,
    )
    out = plot_dir / "ramp_rate.pdf"
    _savefig(fig, out)
    df_csv["detector"] = detector
    _save_plot_data(df_csv, out)


# ----- H) Rate-sweep PINN improvement (training rate = testing rate) -------


# Local helper for companion plots that read per-experiment caches
def _load_cache(cache_dir: Path, name: str) -> pd.DataFrame | None:
    p = cache_dir / f"{name}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _filter_cache_name(name: str) -> str:
    """Map a poison cache name to its filter-baseline counterpart, e.g.
    ``v5_poison_lit`` → ``v5_poison_lit_filter`` and
    ``v5_poison_fit__on_lit101_linear`` → ``v5_poison_fit_filter__on_lit101_linear``."""
    if "__" in name:
        base, rest = name.split("__", 1)
        return f"{base}_filter__{rest}"
    return f"{name}_filter"


# ---------------------------------------------------------------------------
# Filter vs PINN: hard-constraint baseline against the soft-constraint PINN.
# ---------------------------------------------------------------------------
def _fig_filter_vs_pinn(
    cache_dir: Path,
    plot_dir: Path,
    detector: str,
    *,
    metric: str,
):
    """Backbone for the filter-vs-PINN ramp-rate plot.

    ``metric`` ∈ {"recall", "fpr"} selects which column family to plot. Emits
    one PDF per metric. Reads from the rate-matched caches:

      v5_poison_lit_rate_sweep__rate_matched_train_<tag>           — PINN + data-only
      v5_poison_lit_rate_sweep_filter__rate_matched_train_<tag>    — filter baseline

    The clean reference (rate=0, data-only, evaluated against the canonical v5
    attack) is drawn as a horizontal dashed line.
    """
    from scripts.evaluation import _rate_tag

    if metric == "recall":
        value_col = f"avg_recall_{detector}"
        y_label = f"Recall ({DETECTOR_LABELS[detector]})"
        y_lim = (0, 1.05)
        out_name = "rate_sweep.pdf"
    elif metric == "fpr":
        value_col = f"fpr_{detector}"
        y_label = "False positive rate"
        y_lim = None
        out_name = "rate_sweep_fpr.pdf"
    else:
        raise ValueError(f"Unknown metric {metric!r}")

    # Six dynamic x positions (training ramp rates). The clean baseline at
    # rate=0 is loaded separately and drawn as a horizontal line.
    rates_mm = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    rows: list[dict] = []

    def _collect(exp_name: str, pinn_flag_filter: int, label: str):
        for rate_mm in rates_mm:
            tag = _rate_tag(rate_mm / 1000.0)
            cache_name = f"{exp_name}__rate_matched_train_{tag}"
            df = _load_cache(cache_dir, cache_name)
            if df is None or value_col not in df.columns:
                continue
            sub = df[df["pinn_flag"] == pinn_flag_filter]
            if sub.empty:
                continue
            rows.append(
                {
                    "line": label,
                    "rate_mm_per_s": float(rate_mm),
                    "mean": float(sub[value_col].mean()),
                    "std": float(sub[value_col].std() if len(sub) > 1 else 0.0),
                    "n": int(len(sub)),
                }
            )

    _collect("v5_poison_lit_rate_sweep", 0, "Poisoned Data-Only")
    _collect("v5_poison_lit_rate_sweep", 1, "Poisoned PINN")
    _collect("v5_poison_lit_rate_sweep_filter", 0, "Poisoned Data-Only + filter")

    # Clean reference: rate=0 model evaluated against the canonical attack.
    # Use the data-only (pinn_flag=0) rows so the reference is comparable to
    # the "Poisoned Data-Only" line.
    clean_ref_mean = np.nan
    clean_ref_std = np.nan
    clean_df = _load_cache(
        cache_dir, "v5_poison_lit_rate_sweep__rate_matched_train_0mm"
    )
    if clean_df is not None and value_col in clean_df.columns:
        clean_sub = clean_df[clean_df["pinn_flag"] == 0]
        if not clean_sub.empty:
            clean_ref_mean = float(clean_sub[value_col].mean())
            clean_ref_std = float(
                clean_sub[value_col].std() if len(clean_sub) > 1 else 0.0
            )

    df_csv = pd.DataFrame(rows)
    if df_csv.empty and not np.isfinite(clean_ref_mean):
        print(
            f"  [SKIP fig_filter_vs_pinn_{metric}] no rate-matched caches found "
            "for v5_poison_lit_rate_sweep[_filter] — re-run "
            "`evaluation.py --rate-sweep` after selection."
        )
        return

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))

    # Clean Data-Only horizontal reference (drawn first so it sits behind).
    if np.isfinite(clean_ref_mean):
        ax.axhline(
            clean_ref_mean,
            color=CLEAN_COLOR,
            ls=":",
            lw=1.5,
            label=f"Clean Data-Only ({clean_ref_mean:.2f})",
        )
        if clean_ref_std > 0:
            ax.axhspan(
                clean_ref_mean - clean_ref_std,
                clean_ref_mean + clean_ref_std,
                color=CLEAN_COLOR,
                alpha=0.08,
            )

    line_style = {
        "Poisoned Data-Only": (POISONED_DATA_COLOR, "--", "o"),
        "Poisoned Data-Only + filter": (FILTER_COLOR, "-", "s"),
        "Poisoned PINN": (PINN_COLOR, "-", "o"),
    }
    for label, (color, ls, marker) in line_style.items():
        sub = df_csv[df_csv["line"] == label].sort_values("rate_mm_per_s")
        if sub.empty:
            continue
        ax.plot(
            sub["rate_mm_per_s"],
            sub["mean"],
            color=color,
            ls=ls,
            marker=marker,
            ms=5,
            label=label,
        )
        ax.fill_between(
            sub["rate_mm_per_s"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Training ramp rate (mm/s)")
    ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        f"Filtering vs. physics-informing — LIT poisoning ({DETECTOR_LABELS[detector]})",
        fontsize=12,
        y=1.02,
    )
    out_path = plot_dir / out_name
    _savefig(fig, out_path)

    # Save underlying data alongside the PDF
    if not df_csv.empty:
        df_csv["detector"] = detector
        df_csv["metric"] = metric
        # Append the clean reference as a single row (rate_mm=NaN signals "ref").
        if np.isfinite(clean_ref_mean):
            df_csv = pd.concat(
                [
                    df_csv,
                    pd.DataFrame(
                        [
                            {
                                "line": "Clean Data-Only",
                                "rate_mm_per_s": np.nan,
                                "mean": clean_ref_mean,
                                "std": clean_ref_std,
                                "n": int(len(clean_sub)),
                                "detector": detector,
                                "metric": metric,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        _save_plot_data(df_csv, out_path)


def fig_filter_vs_pinn_rate_sweep(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """Recall variant of the filter-vs-PINN ramp-rate plot (paper figure)."""
    _fig_filter_vs_pinn(cache_dir, plot_dir, detector, metric="recall")


def fig_filter_vs_pinn_rate_sweep_fpr(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """FPR companion to :func:`fig_filter_vs_pinn_rate_sweep`."""
    _fig_filter_vs_pinn(cache_dir, plot_dir, detector, metric="fpr")


# ---------------------------------------------------------------------------
# Full-SWaT multi-tank LIT poisoning, and filter-vs-PINN on the real SWaT
# attacks (PLC1 and full-SWaT). Synthetic-attack caches use the grid tag
# convention {exp}__grid__{test}.parquet.
# ---------------------------------------------------------------------------
FULL_SWAT_MULTISTAGE_TEST = "inconsistent_multistage_lit_test"


def _agg_line(sub: pd.DataFrame, exp_var: str, rc: str):
    """Mean/std/count of column ``rc`` grouped by ``exp_var``, x-sorted.

    Returns (x, mean, std, n) numpy arrays, or None if empty / column missing.
    """
    if rc not in sub.columns or sub.empty:
        return None
    g = sub.groupby(exp_var)[rc].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0)
    x = g[exp_var].to_numpy(dtype=float)
    order = np.argsort(x)
    return (
        x[order],
        g["mean"].to_numpy()[order],
        g["std"].to_numpy()[order],
        g["count"].to_numpy()[order],
    )


def _full_swat_dose_response(
    cache_dir: Path,
    plot_dir: Path,
    *,
    detector: str,
    metric: str,
    out_name: str,
    title: str,
):
    """Single-panel dose-response on the multistage LIT attack.

    3 lines: Data-only (pinn_flag=0) + PINN (pinn_flag=1) from the
    v5_full_poison_lit cache, plus a Filter line from the
    v5_full_poison_lit_filter cache (pinn_flag=0). Writes to
    ``plot_dir/{out_name}``.
    """
    _COLS = {
        "recall": ("recall", f"avg_recall_{detector}"),
        "fpr": ("fpr", f"fpr_{detector}"),
    }
    if metric not in _COLS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(_COLS)}")
    value_label, rc = _COLS[metric]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    data_rows: list[pd.DataFrame] = []

    df = _load_grid_cache(cache_dir, "v5_full_poison_lit", FULL_SWAT_MULTISTAGE_TEST)
    if df is not None:
        dfp = _plot_dose_response_lines(
            ax, df, "corrupt_param", detector, metric=metric
        )
        if not dfp.empty:
            dfp["line"] = dfp["pinn_flag"].map({1: "PINN", 0: "Data-only"})
            data_rows.append(dfp)

    df_f = _load_grid_cache(
        cache_dir, "v5_full_poison_lit_filter", FULL_SWAT_MULTISTAGE_TEST
    )
    if df_f is not None:
        agg = _agg_line(df_f[df_f["pinn_flag"] == 0], "corrupt_param", rc)
        if agg is not None:
            x, m, s, n = agg
            ax.plot(x, m, color=FILTER_COLOR, ls="-", marker="s", ms=4, label="Filter")
            ax.fill_between(x, m - s, m + s, color=FILTER_COLOR, alpha=0.15)
            data_rows.append(
                pd.DataFrame(
                    {
                        "exp_var": "corrupt_param",
                        "corrupt_param_raw": x,
                        "corrupt_param_plotted": x,
                        "pinn_flag": -1,
                        "line": "Filter",
                        f"mean_{value_label}": m,
                        f"std_{value_label}": s,
                        "n": n.astype(int),
                    }
                )
            )

    ax.set_xlabel("Poison fraction")
    ax.set_ylabel(
        f"{_metric_label(metric)} ({DETECTOR_LABELS[detector]})"
    )
    if metric == "recall":
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)

    out_path = plot_dir / out_name
    _savefig(fig, out_path)
    if data_rows:
        df_data = pd.concat(data_rows, ignore_index=True)
        df_data["detector"] = detector
        _save_plot_data(df_data, out_path)


def fig3_full_inconsistent_poison_dose_response(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """Full-SWaT dose-response (recall) on the multistage LIT attack."""
    _full_swat_dose_response(
        cache_dir,
        plot_dir,
        detector=detector,
        metric="recall",
        out_name="full_swat_inconsistent_poisoning.pdf",
        title="Full-SWaT multi-tank poisoning: recall vs. poison fraction",
    )


def fig3_full_inconsistent_poison_fpr_dose_response(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """FPR companion to :func:`fig3_full_inconsistent_poison_dose_response`."""
    _full_swat_dose_response(
        cache_dir,
        plot_dir,
        detector=detector,
        metric="fpr",
        out_name="full_swat_inconsistent_poisoning_fpr.pdf",
        title="Full-SWaT multi-tank poisoning: FPR vs. poison fraction",
    )


def _filter_vs_pinn_dose_response(
    df_main,
    df_filter,
    plot_dir: Path,
    *,
    detector: str,
    metric: str,
    out_name: str,
    title: str,
):
    """Dose-response over corrupt_param with 3 lines: Data-only & PINN (from
    ``df_main`` via pinn_flag) + Filter (from ``df_filter``, pinn_flag=0).

    ``df_main`` / ``df_filter`` are pre-loaded so the caller picks the scope
    (PLC1 grid caches on all_swat_attacks, or full-SWaT headline caches).
    """
    _COLS = {
        "recall": ("recall", f"avg_recall_{detector}"),
        "timestep_recall": ("timestep_recall", f"timestep_recall_{detector}"),
        "fpr": ("fpr", f"fpr_{detector}"),
    }
    if metric not in _COLS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(_COLS)}")
    value_label, rc = _COLS[metric]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    data_rows: list[pd.DataFrame] = []

    if df_main is not None and rc in df_main.columns:
        dfp = _plot_dose_response_lines(
            ax, df_main, "corrupt_param", detector, metric=metric
        )
        if not dfp.empty:
            dfp["line"] = dfp["pinn_flag"].map({1: "PINN", 0: "Data-only"})
            data_rows.append(dfp)

    if df_filter is not None and rc in df_filter.columns:
        agg = _agg_line(df_filter[df_filter["pinn_flag"] == 0], "corrupt_param", rc)
        if agg is not None:
            x, m, s, n = agg
            ax.plot(x, m, color=FILTER_COLOR, ls="-", marker="s", ms=4, label="Filter")
            ax.fill_between(x, m - s, m + s, color=FILTER_COLOR, alpha=0.15)
            data_rows.append(
                pd.DataFrame(
                    {
                        "exp_var": "corrupt_param",
                        "corrupt_param_raw": x,
                        "corrupt_param_plotted": x,
                        "pinn_flag": -1,
                        "line": "Filter",
                        f"mean_{value_label}": m,
                        f"std_{value_label}": s,
                        "n": n.astype(int),
                    }
                )
            )

    ax.set_xlabel("Poison fraction")
    ax.set_ylabel(
        f"{_metric_label(metric)} ({DETECTOR_LABELS[detector]})"
    )
    if metric == "recall":
        ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)

    out_path = plot_dir / out_name
    _savefig(fig, out_path)
    if data_rows:
        df_data = pd.concat(data_rows, ignore_index=True)
        df_data["detector"] = detector
        _save_plot_data(df_data, out_path)


def fig_filter_vs_pinn_swat(
    cache_dir: Path,
    plot_dir: Path,
    detector: str = HERO_DETECTOR,
    metric: str = "recall",
):
    """PLC1 filter-vs-PINN on the real SWaT attacks (clean→poisoned).

    Reads the grid caches on the combined ``all_swat_attacks`` test set.
    """
    df_main = _load_grid_cache(cache_dir, "v5_poison_lit", "all_swat_attacks")
    df_filter = _load_grid_cache(cache_dir, "v5_poison_lit_filter", "all_swat_attacks")
    suffix = {"recall": "", "timestep_recall": "_timestep_recall", "fpr": "_fpr"}[metric]
    _mlabel = {"recall": "recall", "timestep_recall": "timestep recall", "fpr": "FPR"}[
        metric
    ]
    _filter_vs_pinn_dose_response(
        df_main,
        df_filter,
        plot_dir,
        detector=detector,
        metric=metric,
        out_name=f"plc1_poison_on_original{suffix}.pdf",
        title=(f"PLC1 poisoning on real SWaT attacks: {_mlabel} vs. poison fraction"),
    )


def fig_filter_vs_pinn_full_swat_attacks(
    cache_dir: Path,
    plot_dir: Path,
    metric: str = "recall",
):
    """Full-SWaT filter-vs-PINN on the real SWaT attacks (clean→poisoned), MSE
    detector.

    Reads the HEADLINE caches (``v5_full_poison_lit`` / ``v5_full_poison_lit_filter``)
    which already evaluate on the real SWaT attacks across every corrupt_param.
    """
    df_main = _load_cache(cache_dir, "v5_full_poison_lit")
    df_filter = _load_cache(cache_dir, "v5_full_poison_lit_filter")
    suffix = {"recall": "", "timestep_recall": "_timestep_recall", "fpr": "_fpr"}[metric]
    _mlabel = {"recall": "recall", "timestep_recall": "timestep recall", "fpr": "FPR"}[
        metric
    ]
    _filter_vs_pinn_dose_response(
        df_main,
        df_filter,
        plot_dir,
        detector="mse",
        metric=metric,
        out_name=f"full_swat_poison_on_original{suffix}_mse.pdf",
        title=(
            f"Full-SWaT poisoning on real SWaT attacks: {_mlabel} vs. poison fraction"
        ),
    )


# ---------------------------------------------------------------------------
# Per-architecture plots
# ---------------------------------------------------------------------------
MODELS = ARCHS
PINN_STYLE = {
    0: ("Data-only", "tab:orange", "--"),
    1: ("PINN", "tab:blue", "-"),
    -1: (
        "Filter",
        FILTER_COLOR,
        "-.",
    ),  # physics filter series (only present when merged in)
}


def _rc(det: str) -> str:
    """Recall column name for detector."""
    return f"avg_recall_{det}"


def _pc(det: str) -> str:
    """FPR column name for detector."""
    return f"fpr_{det}"


def _plot_experiment_lines_metric(
    df: pd.DataFrame,
    exp_var: str,
    x_label: str,
    title: str,
    save_path: Path,
    metric_col: str,
    y_label: str,
    x_transform=None,
):
    """2x2 subplot grid (one per architecture), PINN vs data-only lines with +-1std."""
    if metric_col not in df.columns:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()

    for ax_idx, model_name in enumerate(MODELS):
        ax = axes[ax_idx]
        mdf = df[df["model"] == model_name]

        for pinn_flag, (label, color, ls) in PINN_STYLE.items():
            sub = mdf[mdf["pinn_flag"] == pinn_flag]
            if sub.empty:
                continue
            grouped = sub.groupby(exp_var)[metric_col]
            mean = grouped.mean()
            std = grouped.std().fillna(0)
            x = mean.index.values
            if x_transform:
                x_plot = np.array([x_transform(v) for v in x])
            else:
                x_plot = x.astype(float)

            order = np.argsort(x_plot)
            x_plot = x_plot[order]
            m = mean.values[order]
            s = std.values[order]

            ax.plot(x_plot, m, color=color, ls=ls, marker="o", ms=4, label=label)
            ax.fill_between(x_plot, m - s, m + s, color=color, alpha=0.15)

        ax.set_title(model_name, fontweight="bold")
        ax.set_ylabel(y_label)
        if not metric_col.startswith("fpr"):
            ax.set_ylim(0, 1.05)
        ax.set_xlabel(x_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _savefig(fig, save_path)


def plot_experiment_lines(
    df: pd.DataFrame,
    exp_var: str,
    x_label: str,
    title: str,
    save_path: Path,
    x_transform=None,
    detector: str = HERO_DETECTOR,
):
    """2x2 subplot grid (one per architecture), PINN vs data-only lines with +-1std.

    Writes a recall PDF at ``save_path``, an FPR PDF with the ``_fpr`` suffix, and
    a long-form companion CSV (per model x pinn_flag x exp_var, recall + FPR).
    """
    det_label = DETECTOR_LABELS.get(detector, detector)

    _plot_experiment_lines_metric(
        df,
        exp_var,
        x_label,
        f"{title} [{det_label}]",
        save_path,
        metric_col=_rc(detector),
        y_label="Avg Recall",
        x_transform=x_transform,
    )
    fpr_path = save_path.with_name(save_path.stem + "_fpr" + save_path.suffix)
    _plot_experiment_lines_metric(
        df,
        exp_var,
        x_label,
        f"{title} [{det_label}] - FPR",
        fpr_path,
        metric_col=_pc(detector),
        y_label="False Positive Rate",
        x_transform=x_transform,
    )

    # Companion CSV: aggregated mean/std per (model, pinn_flag, exp_var).
    rows: list[dict] = []
    for model_name in MODELS:
        mdf = df[df["model"] == model_name]
        for pinn_flag in (0, 1, -1):
            sub = mdf[mdf["pinn_flag"] == pinn_flag]
            if sub.empty:
                continue
            for metric_col, mname in (
                (_rc(detector), "recall"),
                (_pc(detector), "fpr"),
            ):
                if metric_col not in sub.columns:
                    continue
                g = (
                    sub.groupby(exp_var)[metric_col]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                for _, r in g.iterrows():
                    rows.append(
                        {
                            "model": model_name,
                            "pinn_flag": pinn_flag,
                            exp_var: float(r[exp_var]),
                            "metric": mname,
                            "mean": float(r["mean"]),
                            "std": float(r["std"]) if pd.notna(r["std"]) else 0.0,
                            "n": int(r["count"]),
                            "detector": detector,
                        }
                    )
    if rows:
        csv_path = save_path.with_suffix(".csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------
def companion_full_swat_clean(
    cache_dir: Path,
    plot_dir: Path,
    *,
    main_exp: str,
    filter_exp: str,
    out_base: str,
    title: str,
    detector: str = HERO_DETECTOR,
):
    """Full-SWaT per-architecture bars on ALL real SWaT attacks at clean training
    (corrupt_param == 0): PINN / Data-only / Filter. No phys-valid/invalid split.

    Reads the headline caches ``{main_exp}`` / ``{filter_exp}``. Writes recall and
    FPR; ``_mse`` suffix when ``detector == 'mse'``. Filter is pinn_flag = -1.
    """
    df_main = _load_cache(cache_dir, main_exp)
    if df_main is None:
        print(f"  [SKIP {out_base}] {main_exp} cache not found")
        return
    df_filter = _load_cache(cache_dir, filter_exp)

    def _clean0(d):
        if d is not None and "corrupt_param" in d.columns:
            return d[d["corrupt_param"] == 0.0]
        return d

    df_main, df_filter = _clean0(df_main), _clean0(df_filter)
    # (output pinn_flag, source df, pinn_flag to select).
    series = [(1, df_main, 1), (0, df_main, 0), (-1, df_filter, 0)]
    dsfx = "_mse" if detector == "mse" else ""

    def _draw(metric_col, y_label, vmax, suffix):
        rows = []
        for arch in ARCHS:
            for out_flag, src, pf in series:
                if src is None or metric_col not in src.columns:
                    continue
                g = src[(src["model"] == arch) & (src["pinn_flag"] == pf)]
                if g.empty:
                    continue
                rows.append(
                    {
                        "arch": arch,
                        "pinn_flag": out_flag,
                        "mean": float(g[metric_col].mean()),
                        "std": float(g[metric_col].std() if len(g) > 1 else 0.0),
                        "n": int(len(g)),
                    }
                )
        df_data = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(ARCHS))
        width = 0.26
        for offset, flag, color, hatch, edge, lbl in (
            (-width, 1, PINN_COLOR, None, PINN_EDGE, "PINN"),
            (0.0, 0, "white", "///", NOPINN_EDGE, "Data-only"),
            (width, -1, FILTER_COLOR, None, FILTER_COLOR, "Filter"),
        ):
            means, stds = [], []
            for arch in ARCHS:
                r = (
                    df_data[(df_data["arch"] == arch) & (df_data["pinn_flag"] == flag)]
                    if not df_data.empty
                    else df_data
                )
                if r.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(r.iloc[0]["mean"])
                    stds.append(r.iloc[0]["std"])
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                color=color,
                edgecolor=edge,
                hatch=hatch,
                capsize=3,
                label=lbl,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
        ax.set_ylabel(y_label)
        ax.set_ylim(0, vmax)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{title} — {y_label} ({DETECTOR_LABELS[detector]})", fontsize=12)
        out = plot_dir / f"{out_base}{suffix}{dsfx}.pdf"
        _savefig(fig, out)
        if not df_data.empty:
            _save_plot_data(
                df_data.assign(
                    subset="All attacks", detector=detector, metric=metric_col
                ),
                out,
            )

    _draw(f"avg_recall_{detector}", "Mean recall", 1.05, "")
    _draw(f"fpr_{detector}", "False positive rate", 0.30, "_fpr")


def fig_invariant_vs_dataonly(
    cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR
):
    """(A) Data-only vs invariant-filter vs physics-filter recall over poison
    fraction on the synthetic inconsistent-LIT test. Shows static invariants
    under-regularise relative to the original mass-balance physics filter."""
    rc = f"avg_recall_{detector}"
    df_do = _load_grid_cache(cache_dir, "v5_poison_lit", "inconsistent_lit_test")
    df_inv = _load_grid_cache(
        cache_dir, "v5_poison_lit_invariant", "inconsistent_lit_test"
    )
    df_filt = _load_grid_cache(
        cache_dir, "v5_poison_lit_filter", "inconsistent_lit_test"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    rows: list[dict] = []
    for src, color, ls, lbl in (
        (df_do, NOPINN_COLOR, "--", "Data-only"),
        (df_inv, "#2ca02c", "-.", "Invariant filter"),
        (df_filt, FILTER_COLOR, "-", "Filter"),
    ):
        if src is None or rc not in src.columns:
            continue
        sub = src[src["pinn_flag"] == 0]
        if sub.empty:
            continue
        g = sub.groupby("corrupt_param")[rc].agg(["mean", "std", "count"]).reset_index()
        x = g["corrupt_param"].to_numpy(dtype=float)
        m = g["mean"].to_numpy()
        s = g["std"].fillna(0).to_numpy()
        ax.plot(x, m, color=color, ls=ls, marker="o", ms=5, label=lbl)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
        for cp, mm, ss, n in zip(x, m, s, g["count"]):
            rows.append(
                {
                    "corrupt_param": float(cp),
                    "line": lbl,
                    "mean_recall": float(mm),
                    "std_recall": float(ss),
                    "n": int(n),
                }
            )
    ax.set_xlabel("Poison fraction")
    ax.set_ylabel(f"Recall ({DETECTOR_LABELS[detector]})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    ax.set_title("Invariant filter vs Data-only — inconsistent LIT poisoning")
    out = plot_dir / "invariant_vs_dataonly.pdf"
    _savefig(fig, out)
    if rows:
        _save_plot_data(pd.DataFrame(rows).assign(detector=detector), out)


def fig_filter_qsweep(cache_dir: Path, plot_dir: Path, detector: str = HERO_DETECTOR):
    """(B) Physics-filter quantile sensitivity on the RATE-MATCHED (0.5 mm/s)
    inconsistent-LIT test at 30% poison: recall vs filter quantile q (the filter
    curve), with the no-filter Data-only and PINN baselines drawn as horizontal
    reference lines. Both baselines come from the rate-sweep experiment evaluated
    rate-matched at 0.5 mm/s (``v5_poison_lit_rate_sweep__rate_matched_train_0p5mm``,
    corrupt_param=0.3), so all three lines share the same rate and test set.
    Writes ``filter_qsweep_recall_vs_q.{pdf,csv}``."""
    rc = f"avg_recall_{detector}"
    df = _load_grid_cache(
        cache_dir, "v5_poison_lit_filter_qsweep", "inconsistent_lit_test_0p5mm"
    )
    if df is None or rc not in df.columns or "physics_filter" not in df.columns:
        print(
            "  [SKIP filter_qsweep] cache or physics_filter/corrupt_param columns missing"
        )
        return
    agg = (
        df.groupby(["physics_filter", "corrupt_param"])[rc]
        .agg(["mean", "std"])
        .reset_index()
    )
    cps = sorted(agg["corrupt_param"].unique())
    poison_cp = 0.3 if 0.3 in cps else cps[-1]
    sub = agg[agg["corrupt_param"] == poison_cp].sort_values("physics_filter")
    if sub.empty:
        print("  [SKIP filter_qsweep] no poison-level rows")
        return
    dfp = (
        pd.DataFrame(
            {
                "q": sub["physics_filter"].astype(float).to_numpy(),
                "poison_mean": sub["mean"].to_numpy(),
                "poison_std": sub["std"].fillna(0).to_numpy(),
            }
        )
        .sort_values("q")
        .reset_index(drop=True)
    )

    # No-filter Data-only / PINN baselines at 0.5 mm/s, 30% poison (rate-matched,
    # same test). Constant in q -> drawn as horizontal reference lines + bands.
    base = _load_cache(cache_dir, "v5_poison_lit_rate_sweep__rate_matched_train_0p5mm")
    baselines: dict[str, tuple[float, float]] = {}
    if base is not None and rc in base.columns and "pinn_flag" in base.columns:
        for flag, key in ((0, "dataonly"), (1, "pinn")):
            b = base.loc[base["pinn_flag"] == flag, rc].dropna()
            if not b.empty:
                baselines[key] = (float(b.mean()), float(b.std(ddof=0)))
                dfp[f"{key}_mean"] = baselines[key][0]
                dfp[f"{key}_std"] = baselines[key][1]
    else:
        print("  [filter_qsweep] no rate-sweep 0p5mm cache -> baselines skipped")

    # Recall vs filter quantile q at 30% poison (+ baseline reference lines).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, color, ls, lbl in (
        ("dataonly", NOPINN_COLOR, "--", "Data-only"),
        ("pinn", PINN_COLOR, "-.", "PINN"),
    ):
        if key in baselines:
            m, s = baselines[key]
            ax.plot(dfp["q"], [m] * len(dfp), color=color, ls=ls, lw=1.3, label=lbl)
            ax.fill_between(dfp["q"], m - s, m + s, color=color, alpha=0.12)
    ax.plot(
        dfp["q"],
        dfp["poison_mean"],
        color=FILTER_COLOR,
        marker="o",
        ms=5,
        label="Filter",
    )
    ax.fill_between(
        dfp["q"],
        dfp["poison_mean"] - dfp["poison_std"],
        dfp["poison_mean"] + dfp["poison_std"],
        color=FILTER_COLOR,
        alpha=0.15,
    )
    ax.set_xlabel("Filter quantile q")
    ax.set_ylabel(f"Mean recall, 30% poison ({DETECTOR_LABELS[detector]})")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    ax.set_title(
        "Robustness vs filter quantile (30% inconsistent LIT poison, 0.5 mm/s)"
    )
    out = plot_dir / "filter_qsweep_recall_vs_q.pdf"
    _savefig(fig, out)
    _save_plot_data(dfp.assign(detector=detector), out)


def generate_grid_plots(cache_dir: Path, plot_dir: Path):
    """Generate the full v5 plot set into a single flat directory ``plot_dir``.

    Per-sensor is the headline detector; MSE variants carry a ``_mse`` suffix.
    Each function prints ``[SKIP]`` and continues if its cache is missing.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[plots] -> {plot_dir}")
    H = HERO_DETECTOR

    # Data-condition companions (per-sensor).
    companion_clean_data(cache_dir, plot_dir, detector=H)
    appendix_timestep_table(cache_dir, plot_dir, detector=H)
    appendix_timestep_table_full(cache_dir, plot_dir)
    appendix_timestep_channel_table(cache_dir, plot_dir)
    appendix_timestep_full_quantile_sweep(cache_dir, plot_dir)
    companion_scarcity(cache_dir, plot_dir, detector=H)
    companion_gaussian_noise(cache_dir, plot_dir, detector=H)
    companion_stuck(cache_dir, plot_dir, detector=H)
    companion_cross_sensor(cache_dir, plot_dir, detector=H)
    companion_cross_shape(cache_dir, plot_dir, detector=H)
    companion_ramp_rate(cache_dir, plot_dir, detector=H)

    # Poisoning dose-response (per-sensor) + MSE companion for inconsistent.
    fig3_inconsistent_poison_dose_response(cache_dir, plot_dir, detector=H)
    fig3_inconsistent_poison_fpr_dose_response(cache_dir, plot_dir, detector=H)
    fig3_inconsistent_poison_dose_response_mse(cache_dir, plot_dir)
    fig4_consistent_poison_failure(cache_dir, plot_dir, detector=H)

    # Physics-weight ablation (per-sensor).
    fig3_inconsistent_lit_recall_per_wphys(cache_dir, plot_dir, detector=H)
    fig3_inconsistent_lit_fpr_per_wphys(cache_dir, plot_dir, detector=H)

    # Ramp-rate sweep (per-sensor).
    fig_filter_vs_pinn_rate_sweep(cache_dir, plot_dir, detector=H)
    fig_filter_vs_pinn_rate_sweep_fpr(cache_dir, plot_dir, detector=H)

    # Real-SWaT-attack story: PLC1 (per-sensor) and full-SWaT (MSE). The
    # timestep_recall variant is the pooled-over-steps companion to the
    # episode-uniform recall.
    for metric in ("recall", "timestep_recall", "fpr"):
        fig_filter_vs_pinn_swat(cache_dir, plot_dir, detector=H, metric=metric)
        fig_filter_vs_pinn_full_swat_attacks(cache_dir, plot_dir, metric=metric)
    fig3_full_inconsistent_poison_dose_response(cache_dir, plot_dir, detector=H)
    fig3_full_inconsistent_poison_fpr_dose_response(cache_dir, plot_dir, detector=H)

    # (A) invariant-filter vs data-only; (B) physics-filter quantile sensitivity.
    fig_invariant_vs_dataonly(cache_dir, plot_dir, detector=H)
    fig_filter_qsweep(cache_dir, plot_dir, detector=H)

    # Full-SWaT clean-training bar charts on ALL real SWaT attacks (MSE).
    companion_full_swat_clean(
        cache_dir,
        plot_dir,
        main_exp="v5_full_poison_lit",
        filter_exp="v5_full_poison_lit_filter",
        out_base="full_swat_clean_data",
        title="Full-SWaT clean training on real SWaT attacks",
        detector="mse",
    )

    # Per-architecture inconsistent poisoning (per-sensor) + CSV.
    df_pa = _load_cache(cache_dir, "v5_poison_lit")
    if df_pa is not None:
        # Splice in the physics-filter models as the Filter series (pinn_flag=-1)
        # so the per-architecture plot/CSVs carry a third line alongside
        # Data-only and PINN.
        df_filt = _load_cache(cache_dir, "v5_poison_lit_filter")
        if df_filt is not None:
            df_pa = pd.concat([df_pa, df_filt.assign(pinn_flag=-1)], ignore_index=True)
        plot_experiment_lines(
            df_pa,
            "corrupt_param",
            "Poison Fraction",
            "Inconsistent LIT poisoning: PINN vs Data-only vs Filter",
            plot_dir / "inconsistent_poisoning_per_architecture.pdf",
            detector=H,
        )

    print("[plots] Done.")


def main():
    ap = argparse.ArgumentParser(description="Generate the v5 plot set (flat dir).")
    ap.add_argument("--cache-dir", default="outputs/v5-eval-cache")
    ap.add_argument("--plot-dir", default="outputs/v5-plots")
    args = ap.parse_args()
    generate_grid_plots(Path(args.cache_dir), Path(args.plot_dir))


if __name__ == "__main__":
    main()
