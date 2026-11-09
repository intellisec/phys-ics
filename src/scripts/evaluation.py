"""Evaluate v5 experiment grid: inference, caching, and plotting.

Usage
-----
# Run inference for all v5 experiments:
python -m src.scripts.evaluation --device cuda

# Run only specific experiments:
python -m src.scripts.evaluation --device cuda --experiments v5_poison_lit v5_poison_fit

# Skip inference, regenerate plots from cache:
python -m src.scripts.evaluation --plots-only

# Also run cross-sensor and cross-shape transfer evaluations:
python -m src.scripts.evaluation --device cuda --cross-sensor --cross-shape
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pin to GPU 0 before any CUDA import
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# v5 slice setup: 96 h training + 3 × 12 h disjoint held-out slices.
ATTACK_SLICE_V3 = slice(367_200, 410_400)  # synthetic attacks injected here
THRESHOLD_SLICE_V3 = slice(410_400, 453_600)  # quantile thresholds fit here (PLC1)
SELECTION_SLICE_V3 = slice(453_600, 496_800)  # in-training test, drives test_rmse
# The full-SWaT experiments fuse the last two into FULL_SWAT_THRESHOLD_SLICE below;
# these two constants describe the PLC1 layout only.

DATA_ROOT = Path("data/processed")
NORMAL_PARQUET = DATA_ROOT / "normal_si.parquet"
SWAT_ATTACK_PARQUET = DATA_ROOT / "swat_attack.parquet"
SWAT_LABELS_PARQUET = DATA_ROOT / "swat_attack_labels.parquet"

# Clean-normal eval set for an attack-independent FPR: the un-attacked attack
# background (normal_si[ATTACK_SLICE_V3], all labels 0), so the false-alarm rate
# depends only on the model, not on attack count/placement. Auto-generated on first use.
CLEAN_NORMAL_DIR = DATA_ROOT / "clean_normal_v3"
CLEAN_NORMAL_PARQUET = CLEAN_NORMAL_DIR / "clean_attack.parquet"
CLEAN_NORMAL_LABELS = CLEAN_NORMAL_DIR / "clean_labels.parquet"

DETECT_CFG = dict(
    q_fused=0.99,  # used by the MSE detector
    q_per_sensor=0.999,  # used by the per-sensor detector (decoupled from MSE)
)

# PLC1 attack slices (row indices in the SWaT attack parquet)
# Boundaries refined to better match the real attack start/end.
SWAT_ATTACK_SLICES = {
    1: {"Start_Row": 1738, "End_Row": 2673},
    2: {"Start_Row": 3046, "End_Row": 3491},
    3: {"Start_Row": 4901, "End_Row": 5283},
    21: {"Start_Row": 116999, "End_Row": 117701},
    26: {"Start_Row": 198273, "End_Row": 199716},
    30: {"Start_Row": 280023, "End_Row": 281185},
    33: {"Start_Row": 361243, "End_Row": 361674},
    34: {"Start_Row": 371519, "End_Row": 371618},
    35: {"Start_Row": 371893, "End_Row": 372374},
    36: {"Start_Row": 389746, "End_Row": 390262},
}

# Physics-validity split of the PLC1 attacks.
# "phys_valid"   = attack preserves the PLC1 mass balance.
# "phys_invalid" = attack violates mass balance.
SWAT_PHYS_VALID_SLICES = {
    aid: SWAT_ATTACK_SLICES[aid] for aid in (1, 2, 26, 30, 34, 35)
}
SWAT_PHYS_INVALID_SLICES = {aid: SWAT_ATTACK_SLICES[aid] for aid in (3, 21, 33, 36)}

# Attacks outside PLC1 scope — dropped from evaluation data.
# Aligned with Clement Fung's corrected/fused labelling (Bauer & Fung 2020):
# https://github.com/pwwl/ics-anomaly-attribution
SWAT_ATTACK_SLICES_DROPPED = {
    6: {"Start_Row": 7233, "End_Row": 7432},
    7: {"Start_Row": 7685, "End_Row": 8113},
    8: {"Start_Row": 11385, "End_Row": 12355},
    10: {"Start_Row": 15361, "End_Row": 16084},  # Doc 10+11 fused
    13: {"Start_Row": 90662, "End_Row": 90917},
    16: {"Start_Row": 93424, "End_Row": 93705},
    17: {"Start_Row": 103092, "End_Row": 103797},
    19: {"Start_Row": 115822, "End_Row": 116080},
    20: {"Start_Row": 116123, "End_Row": 116515},
    22: {"Start_Row": 132896, "End_Row": 133362},
    23: {"Start_Row": 142927, "End_Row": 143611},
    24: {"Start_Row": 172268, "End_Row": 172588},
    25: {"Start_Row": 172892, "End_Row": 173499},
    27: {"Start_Row": 227828, "End_Row": 228362},
    28: {"Start_Row": 229519, "End_Row": 263727},
    31: {"Start_Row": 302653, "End_Row": 303020},
    32: {"Start_Row": 347718, "End_Row": 348315},
    37: {"Start_Row": 436672, "End_Row": 437046},
    38: {"Start_Row": 437455, "End_Row": 437734},
    39: {"Start_Row": 438184, "End_Row": 438584},
    40: {"Start_Row": 438659, "End_Row": 438955},
    41: {"Start_Row": 443540, "End_Row": 445191},
}

# Full-SWaT attack scope: every SWaT attack (PLC1 + non-PLC1).
# Used by full-SWaT experiments (drop_non_plc1=False).
SWAT_ATTACK_SLICES_FULL = {**SWAT_ATTACK_SLICES, **SWAT_ATTACK_SLICES_DROPPED}


# ---------------------------------------------------------------------------
# Experiment registry — v5 only.
# Synthetic test parquets are generated with --validation_start 367_200
# --validation_end 410_400 (= ATTACK_SLICE_V3) and live in *_v3 dirs.
# ---------------------------------------------------------------------------
# Fused threshold+selection region for the full-SWaT experiments. The 43,200-row
# THRESHOLD_SLICE_V3 left only ~4,315 score samples, so q=0.999 rested on ~4 points
# and was effectively the sample maximum; fusing it with SELECTION_SLICE_V3 doubles
# that to ~8,635. Matches --test_rows 86_400 in the full-SWaT gridsearch configs,
# i.e. exactly the region train.py holds out.
# NB: tau is now fitted on the same rows model selection optimises over, so tau is
# marginally tight and the measured FPR marginally inflated (the conservative
# direction). The clean-normal FPR set (ATTACK_SLICE_V3 background) stays disjoint
# from both roles.
FULL_SWAT_THRESHOLD_SLICE = slice(410_400, 496_800)

EXPERIMENTS = {
    "v5_scarcity": dict(
        selected_dir="outputs/selected_models/v5-scarcity",
        exp_var="rows",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_faulty_gaussian": dict(
        selected_dir="outputs/selected_models/v5-faulty_gaussian",
        exp_var="corrupt_param",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_faulty_stuck": dict(
        selected_dir="outputs/selected_models/v5-faulty_stuck",
        exp_var="corrupt_param",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── Filter baselines for scarcity / faulty-Gaussian / faulty-stuck. ──
    "v5_scarcity_filter": dict(
        selected_dir="outputs/selected_models/v5-scarcity_filter",
        exp_var="rows",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_faulty_gaussian_filter": dict(
        selected_dir="outputs/selected_models/v5-faulty_gaussian_filter",
        exp_var="corrupt_param",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_faulty_stuck_filter": dict(
        selected_dir="outputs/selected_models/v5-faulty_stuck_filter",
        exp_var="corrupt_param",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_poison_lit": dict(
        selected_dir="outputs/selected_models/v5-poison_lit",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # Same training grid as v5_poison_lit but model selection keeps w_phys
    # as a grouping key (one selected model per (arch, corrupt_param, w_phys,
    # seed) cell). Used by fig3_inconsistent_lit_recall_per_wphys.
    "v5_poison_lit_per_wphys": dict(
        selected_dir="outputs/selected_models/v5-poison_lit_per_wphys",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_poison_fit": dict(
        selected_dir="outputs/selected_models/v5-poison_fit",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "ramp_attack_fit_v3"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_fit_v3"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_consistent_lit_sin": dict(
        selected_dir="outputs/selected_models/v5-consistent_lit_sin",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "consistent_ramp_attack_lit_sin_v3"
        / "synthetic_consistent_lit_sin_attack.parquet",
        labels_path=DATA_ROOT
        / "consistent_ramp_attack_lit_sin_v3"
        / "synthetic_consistent_lit_sin_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_consistent_fit": dict(
        selected_dir="outputs/selected_models/v5-consistent_fit",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "consistent_ramp_attack_fit_v3"
        / "synthetic_consistent_fit_attack.parquet",
        labels_path=DATA_ROOT
        / "consistent_ramp_attack_fit_v3"
        / "synthetic_consistent_fit_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── v5 LIT poisoning ramp-rate sweep (poison fraction fixed at 0.3) ──
    "v5_poison_lit_rate_sweep": dict(
        selected_dir="outputs/selected_models/v5-poison_lit_rate_sweep",
        exp_var="ramp_rate",
        # Default attack used as a placeholder; rate-matched eval is dispatched
        # via --rate-sweep, which selects per-rate parquets at runtime
        # (see RATE_SWEEP_TRAIN_TO_TEST + _run_rate_sweep_matched).
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3_3mm"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3_3mm"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── Hard-constraint baseline: data-only + physics filter (q=0.99). ──
    # Mirrors v5_poison_lit_rate_sweep; rate-matched eval works the same way.
    "v5_poison_lit_rate_sweep_filter": dict(
        selected_dir="outputs/selected_models/v5-poison_lit_rate_sweep_filter",
        exp_var="ramp_rate",
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3_3mm"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3_3mm"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── PLC1 LIT poisoning FILTER baseline across poison fractions. ──
    # Same attack parquet as v5_poison_lit (synthetic LIT101 ramps); provides the
    # Filter line of the filter-vs-PINN-on-poison-fraction plots. Grid caches on
    # all test sets (incl. all_swat_attacks) via GRID_TRAINING_EXPERIMENTS.
    "v5_poison_lit_filter": dict(
        selected_dir="outputs/selected_models/v5-poison_lit_filter",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── (A) Invariant-filter run (data-only + invariant window drop). Same
    # synthetic inconsistent-LIT attack as v5_poison_lit. ──
    "v5_poison_lit_invariant": dict(
        selected_dir="outputs/selected_models/v5-poison_lit_invariant",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── (B) Physics-filter quantile sweep. exp_var = physics_filter (q); the
    # cache also carries corrupt_param via extra_cols so clean (0.0) and 30%
    # poison can be separated in the q-sensitivity plots. Trained at the stealthy
    # 0.5 mm/s ramp, so the test attack must be the RATE-MATCHED 0.5 mm/s parquet
    # (ramp_attack_lit_v3_0p5mm) — NOT the 1.943 mm/s default, which would grade
    # the stealthy q-sweep against a steep attack and wash out the q-sensitivity.
    "v5_poison_lit_filter_qsweep": dict(
        selected_dir="outputs/selected_models/v5-poison_lit_filter_qsweep",
        exp_var="physics_filter",
        extra_cols=["corrupt_param"],
        attack_path=DATA_ROOT
        / "ramp_attack_lit_v3_0p5mm"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_lit_v3_0p5mm"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── Filter baselines for the remaining PLC1 poisoning families. ──
    # Same attack parquets as their non-filter siblings (the filter changes the
    # training set, not the test attack); provide the Filter line of the
    # inconsistent/consistent dose-response plots.
    "v5_poison_fit_filter": dict(
        selected_dir="outputs/selected_models/v5-poison_fit_filter",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "ramp_attack_fit_v3"
        / "synthetic_linear_attack.parquet",
        labels_path=DATA_ROOT
        / "ramp_attack_fit_v3"
        / "synthetic_linear_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_consistent_lit_sin_filter": dict(
        selected_dir="outputs/selected_models/v5-consistent_lit_sin_filter",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "consistent_ramp_attack_lit_sin_v3"
        / "synthetic_consistent_lit_sin_attack.parquet",
        labels_path=DATA_ROOT
        / "consistent_ramp_attack_lit_sin_v3"
        / "synthetic_consistent_lit_sin_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    "v5_consistent_fit_filter": dict(
        selected_dir="outputs/selected_models/v5-consistent_fit_filter",
        exp_var="corrupt_param",
        attack_path=DATA_ROOT
        / "consistent_ramp_attack_fit_v3"
        / "synthetic_consistent_fit_attack.parquet",
        labels_path=DATA_ROOT
        / "consistent_ramp_attack_fit_v3"
        / "synthetic_consistent_fit_labels.parquet",
        use_swat_slices=False,
        threshold_slice=THRESHOLD_SLICE_V3,
    ),
    # ── Full-SWaT multi-tank LIT poisoning. ──
    # Headline pass evaluates on the original SWaT attacks;
    # use drop_non_plc1=False to keeps the non-PLC1 attacks in scope.
    "v5_full_poison_lit": dict(
        selected_dir="outputs/selected_models/v5-full_poison_lit",
        exp_var="corrupt_param",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        drop_non_plc1=False,
        threshold_slice=FULL_SWAT_THRESHOLD_SLICE,
    ),
    # Same as above but data-only (w_phys=0) with the multi-equation physics
    # filter enabled — the "Filter" baseline for the full-SWaT plots.
    "v5_full_poison_lit_filter": dict(
        selected_dir="outputs/selected_models/v5-full_poison_lit_filter",
        exp_var="corrupt_param",
        attack_path=SWAT_ATTACK_PARQUET,
        labels_path=SWAT_LABELS_PARQUET,
        use_swat_slices=True,
        drop_non_plc1=False,
        threshold_slice=FULL_SWAT_THRESHOLD_SLICE,
    ),
}

# Experiments whose HEADLINE cache ({exp}.parquet) a figure reads; the default
# inference loop runs only these. The remaining experiments are consumed through
# their grid / rate-matched caches (--grid, --rate-sweep) and selected summaries.
HEADLINE_EXPERIMENTS = [
    "v5_scarcity",
    "v5_scarcity_filter",
    "v5_faulty_gaussian",
    "v5_faulty_gaussian_filter",
    "v5_faulty_stuck",
    "v5_faulty_stuck_filter",
    "v5_poison_lit",
    "v5_poison_lit_filter",
    "v5_poison_fit",
    "v5_poison_fit_filter",
    "v5_full_poison_lit",
    "v5_full_poison_lit_filter",
]

# Cross-shape attack datasets (for transfer evaluation, v5 boundaries).
# *_v3 dirs: regenerated with --validation_start 367_200 --validation_end 410_400.
# linear paths reuse the v5 headline parquets (same data as v5_poison_*); the
# other shapes live in their own _v3 dirs.
CROSS_SHAPE_ATTACKS = {
    "lit_linear": DATA_ROOT / "ramp_attack_lit_v3" / "synthetic_linear_attack.parquet",
    "lit_staircase": DATA_ROOT
    / "staircase_attack_lit_v3"
    / "synthetic_staircase_attack.parquet",
    "lit_sigmoid": DATA_ROOT
    / "sigmoid_attack_lit_v3"
    / "synthetic_sigmoid_attack.parquet",
    "lit_noisy_ramp": DATA_ROOT
    / "noisy_ramp_attack_lit_v3"
    / "synthetic_noisy_ramp_attack.parquet",
    "fit_linear": DATA_ROOT / "ramp_attack_fit_v3" / "synthetic_linear_attack.parquet",
    "fit_staircase": DATA_ROOT
    / "staircase_attack_fit_v3"
    / "synthetic_staircase_attack.parquet",
    "fit_sigmoid": DATA_ROOT
    / "sigmoid_attack_fit_v3"
    / "synthetic_sigmoid_attack.parquet",
    "fit_noisy_ramp": DATA_ROOT
    / "noisy_ramp_attack_fit_v3"
    / "synthetic_noisy_ramp_attack.parquet",
}

CROSS_SHAPE_LABELS = {
    "lit_linear": DATA_ROOT / "ramp_attack_lit_v3" / "synthetic_linear_labels.parquet",
    "lit_staircase": DATA_ROOT
    / "staircase_attack_lit_v3"
    / "synthetic_staircase_labels.parquet",
    "lit_sigmoid": DATA_ROOT
    / "sigmoid_attack_lit_v3"
    / "synthetic_sigmoid_labels.parquet",
    "lit_noisy_ramp": DATA_ROOT
    / "noisy_ramp_attack_lit_v3"
    / "synthetic_noisy_ramp_labels.parquet",
    "fit_linear": DATA_ROOT / "ramp_attack_fit_v3" / "synthetic_linear_labels.parquet",
    "fit_staircase": DATA_ROOT
    / "staircase_attack_fit_v3"
    / "synthetic_staircase_labels.parquet",
    "fit_sigmoid": DATA_ROOT
    / "sigmoid_attack_fit_v3"
    / "synthetic_sigmoid_labels.parquet",
    "fit_noisy_ramp": DATA_ROOT
    / "noisy_ramp_attack_fit_v3"
    / "synthetic_noisy_ramp_labels.parquet",
}


def _rate_tag(rate: float) -> str:
    """Human-readable directory tag for a ramp rate, e.g. 0.003 → '3mm', 0.0005 → '0p5mm'."""
    mm = rate * 1000  # m → mm
    if mm == int(mm):
        return f"{int(mm)}mm"
    return f"{mm:.1f}mm".replace(".", "p")


# LIT101 ramp rate that v5_poison_lit was trained at.  In the generalization
# sweep below, this rate is the train==test reference point and is read
# straight from the headline ``v5_poison_lit`` cache (no separate per-rate
# test parquet needed).
V3_TRAINING_RATE_LIT = 0.001943

# Multi-rate ramp attacks on LIT101 for generalization study (v5 boundaries).
# Includes the training rate (read from the headline cache) plus 5 off-rate
# generalization probes — 2 below and 3 above the training rate.
RAMP_RATES_LIT = [0.0005, 0.001, V3_TRAINING_RATE_LIT, 0.005, 0.01, 0.02]
RAMP_RATE_ATTACKS_LIT = {
    rate: {
        "attack": DATA_ROOT
        / f"ramp_attack_lit_v3_{_rate_tag(rate)}"
        / "synthetic_linear_attack.parquet",
        "labels": DATA_ROOT
        / f"ramp_attack_lit_v3_{_rate_tag(rate)}"
        / "synthetic_linear_labels.parquet",
    }
    for rate in RAMP_RATES_LIT
    if rate != V3_TRAINING_RATE_LIT  # training-rate cache lives elsewhere
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_episodes(labels: np.ndarray) -> dict[int, dict]:
    """Convert binary label array → attack_slices dict compatible with attackwise_recall."""
    slices = {}
    in_ep = False
    start = 0
    ep_id = 1
    for i in range(len(labels)):
        if labels[i] == 1 and not in_ep:
            start = i
            in_ep = True
        elif labels[i] == 0 and in_ep:
            slices[ep_id] = {"Start_Row": start, "End_Row": i}
            ep_id += 1
            in_ep = False
    if in_ep:
        slices[ep_id] = {"Start_Row": start, "End_Row": len(labels)}
    return slices


def _remap_slices(attack_slices: dict, dropped: dict) -> dict:
    """Re-index attack slices after dropping rows for non-PLC1 attacks."""
    drop_ranges = sorted(
        ((int(v["Start_Row"]), int(v["End_Row"])) for v in dropped.values()),
        key=lambda x: x[0],
    )

    def _new_row(row: int) -> int:
        offset = 0
        for s, e in drop_ranges:
            if row >= e:
                offset += e - s
            elif row >= s:
                offset += row - s
                break
        return row - offset

    remapped = {}
    for k, v in attack_slices.items():
        remapped[k] = {
            "Start_Row": _new_row(int(v["Start_Row"])),
            "End_Row": _new_row(int(v["End_Row"])),
        }
    return remapped


def _remap_slices_subset(keep_slices: dict) -> tuple[dict, dict]:
    """Remap a subset of SWAT_ATTACK_SLICES while dropping the non-kept PLC1 attacks.

    Returns (remapped_keep_slices, full_dropped) where full_dropped combines
    SWAT_ATTACK_SLICES_DROPPED with any PLC1 attacks absent from keep_slices.
    """
    extra_dropped = {
        aid: sl for aid, sl in SWAT_ATTACK_SLICES.items() if aid not in keep_slices
    }
    full_dropped = {**SWAT_ATTACK_SLICES_DROPPED, **extra_dropped}
    return _remap_slices(keep_slices, full_dropped), full_dropped


def _guard_nonempty(results: list, label: str, cache_path: Path) -> None:
    """Refuse to cache an empty result frame.

    Every cache here is existence-gated, so a 0-row parquet written after a pass in
    which every model raised would short-circuit all later runs and make the failure
    look permanent. Downstream, plots.py treats an empty frame as a missing series
    rather than an error, so the whole thing renders as a silently absent figure.
    """
    if not results:
        raise RuntimeError(
            f"{label}: no model produced results; refusing to write an empty "
            f"cache to {cache_path}. Check the [ERROR] lines above."
        )


def load_summary(exp_name: str) -> pd.DataFrame:
    """Load the summary.csv for a selected experiment."""
    cfg = EXPERIMENTS[exp_name]
    csv_path = Path(cfg["selected_dir"]) / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Attack-independent (clean-normal) FPR
# ---------------------------------------------------------------------------
def _ensure_clean_normal():
    """Create the clean-normal eval set (un-attacked ATTACK_SLICE_V3 + zero labels)
    from normal_si.parquet if it doesn't exist yet."""
    if CLEAN_NORMAL_PARQUET.exists() and CLEAN_NORMAL_LABELS.exists():
        return
    CLEAN_NORMAL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(NORMAL_PARQUET)
    sl = df.iloc[ATTACK_SLICE_V3].reset_index(drop=True)
    sl.to_parquet(CLEAN_NORMAL_PARQUET, index=False)
    pd.DataFrame({"attack": np.zeros(len(sl), dtype=np.int8)}).to_parquet(
        CLEAN_NORMAL_LABELS, index=False
    )
    print(f"  [clean-fpr] generated {CLEAN_NORMAL_PARQUET} ({len(sl)} rows)")


# Memoize clean FPR per selected model so the many transfer passes don't recompute
# (clean FPR depends only on the model, not the attack set). Cleared per process.
_CLEAN_FPR_CACHE: dict = {}


def _clean_eval(
    model_dict,
    detect_cfg,
    threshold_slice,
    keep_cols,
    device,
    *,
    cache_key=None,
):
    """Per-detector FPR on the clean-normal slice (all windows genuinely normal).

    Returns ``{"fpr_mse": ..., "fpr_per_sensor": ...}``, memoized by ``cache_key``.
    ``detect_cfg`` is deliberately not part of the key: the quantiles are set once
    from --q-data/--q-per-sensor before any pass runs.
    """
    if cache_key is not None and cache_key in _CLEAN_FPR_CACHE:
        return _CLEAN_FPR_CACHE[cache_key]

    _ensure_clean_normal()
    from inference.metrics import compute_metrics

    m = compute_metrics(
        model_dict,
        detect_cfg,
        attack_dir=CLEAN_NORMAL_PARQUET,
        device=device,
        labels_parquet=CLEAN_NORMAL_LABELS,
        normal_dir=NORMAL_PARQUET,
        threshold_slice=threshold_slice,
        attack_slices_dropped=None,
        keep_cols=keep_cols,
    )
    fpr = {}
    for det in ("mse", "per_sensor"):
        fp = int(m[det]["FP"])
        tn = int(m[det]["TN"])
        fpr[f"fpr_{det}"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if cache_key is not None:
        _CLEAN_FPR_CACHE[cache_key] = fpr
    return fpr


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference_for_experiment(
    exp_name: str,
    *,
    device: str = "cpu",
    cache_dir: Path = Path("outputs/v5-eval-cache"),
) -> pd.DataFrame:
    """Run inference for all selected models in one experiment. Returns and caches results."""
    cache_path = cache_dir / f"{exp_name}.parquet"
    if cache_path.exists():
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return pd.read_parquet(cache_path)

    # Lazy imports so the script can be imported without torch/inference deps
    sys.path.insert(0, str(Path("src").resolve()))
    from models.config import load_model_bundle
    from inference.metrics import compute_metrics, attackwise_recall

    cfg = EXPERIMENTS[exp_name]
    summary = load_summary(exp_name)
    print(f"\n[{exp_name}] {len(summary)} models to evaluate")

    # --- Load attack data & build attack_slices ---
    if cfg["use_swat_slices"]:
        if cfg.get("drop_non_plc1", True):
            attack_slices = _remap_slices(
                SWAT_ATTACK_SLICES, SWAT_ATTACK_SLICES_DROPPED
            )
            attack_slices_dropped = SWAT_ATTACK_SLICES_DROPPED
        else:
            # Full-SWaT: keep every catalogued attack, drop nothing.
            attack_slices = _remap_slices(SWAT_ATTACK_SLICES_FULL, {})
            attack_slices_dropped = None
    else:
        labels_arr = pd.read_parquet(cfg["labels_path"])["attack"].to_numpy()
        attack_slices = _find_episodes(labels_arr)
        attack_slices_dropped = None

    results = []
    selected_dir = Path(cfg["selected_dir"])

    for idx, row in summary.iterrows():
        run_dir = selected_dir / row["run_dir"]
        ckpt_path = next(run_dir.glob("*.pt"), None)
        if ckpt_path is None:
            print(f"  [SKIP] No .pt in {run_dir}")
            continue

        try:
            model_dict = load_model_bundle(ckpt_path, device=torch.device(device))

            # Derive keep_cols from checkpoint
            ckpt_keep_cols = model_dict["input_cols"]

            threshold_slice = cfg["threshold_slice"]
            detect_cfg = DETECT_CFG.copy()

            # Run detection
            m = compute_metrics(
                model_dict,
                detect_cfg,
                attack_dir=cfg["attack_path"],
                device=device,
                labels_parquet=cfg["labels_path"],
                normal_dir=NORMAL_PARQUET,
                threshold_slice=threshold_slice,
                attack_slices_dropped=attack_slices_dropped,
                keep_cols=ckpt_keep_cols,
            )

            # Attack-INDEPENDENT FPR on the clean-normal slice.
            clean_fpr = _clean_eval(
                model_dict,
                detect_cfg,
                threshold_slice,
                ckpt_keep_cols,
                device,
                cache_key=(str(selected_dir), row["run_dir"]),
            )

            # Compute per-attack metrics for all detectors
            rec = {
                "model": row.get("model", ""),
                cfg["exp_var"]: row[cfg["exp_var"]],
                "pinn_flag": row.get("pinn_flag", 0),
                "seed": row.get("seed", 0),
                "w_phys": row.get("w_phys", 0.0),
                "lr": row.get("lr", 0.0),
                "test_rmse": row.get("test_rmse", None),
                "run_dir": row["run_dir"],
            }
            for det_name in ("mse", "per_sensor"):
                ar, sr, _, ts = attackwise_recall(
                    model_dict,
                    m[det_name]["alarm_mask"],
                    attack_slices=attack_slices,
                )
                rec[f"avg_recall_{det_name}"] = ar
                rec[f"std_recall_{det_name}"] = sr
                # Timestep-wise recall over the SAME catalogued episodes as
                # avg_recall, but pooled over steps rather than averaged uniformly
                # over episodes -- so long attacks dominate it. tp/pos are stored
                # too, keeping the ratio auditable and poolable from the cache.
                rec[f"timestep_recall_{det_name}"] = ts[0]
                rec[f"timestep_tp_{det_name}"] = ts[1]
                rec[f"timestep_pos_{det_name}"] = ts[2]
                rec[f"fpr_{det_name}"] = clean_fpr[f"fpr_{det_name}"]

            results.append(rec)
            print(
                f"  [{idx + 1}/{len(summary)}] {row['run_dir']}: "
                f"mse={rec['avg_recall_mse']:.3f} "
                f"per_sensor={rec['avg_recall_per_sensor']:.3f}"
            )

        except Exception as e:
            print(f"  [ERROR] {row['run_dir']}: {e}")

    cache_path = cache_dir / f"{exp_name}.parquet"
    _guard_nonempty(results, exp_name, cache_path)
    df = pd.DataFrame(results)

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df)} results → {cache_path}")

    return df


def run_transfer_inference(
    source_exp: str,
    attack_path: Path,
    labels_path: Path,
    tag: str,
    *,
    device: str = "cpu",
    cache_dir: Path = Path("outputs/v5-eval-cache"),
    use_swat_slices: bool = False,
    swat_slice_subset: dict | None = None,
    summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate models from source_exp on an alternative attack dataset.

    If ``use_swat_slices`` is True, optionally pass ``swat_slice_subset`` (a dict
    {attack_id: slice_info}) to evaluate only on that subset of SWaT PLC1 attacks;
    the remaining PLC1 attacks are treated as dropped.

    If ``summary`` is provided, use it instead of loading the full summary CSV
    for ``source_exp`` — useful when the caller wants to evaluate only a subset
    of models (e.g. a single ramp_rate group in the rate-sweep experiment).
    """
    cache_path = cache_dir / f"{source_exp}__{tag}.parquet"
    if cache_path.exists():
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return pd.read_parquet(cache_path)

    sys.path.insert(0, str(Path("src").resolve()))
    from models.config import load_model_bundle
    from inference.metrics import compute_metrics, attackwise_recall

    cfg = EXPERIMENTS[source_exp]
    if summary is None:
        summary = load_summary(source_exp)
    print(f"\n[{source_exp} → {tag}] {len(summary)} models")

    if use_swat_slices:
        if swat_slice_subset is not None:
            attack_slices, attack_slices_dropped = _remap_slices_subset(
                swat_slice_subset
            )
        elif cfg.get("drop_non_plc1", True):
            attack_slices = _remap_slices(
                SWAT_ATTACK_SLICES, SWAT_ATTACK_SLICES_DROPPED
            )
            attack_slices_dropped = SWAT_ATTACK_SLICES_DROPPED
        else:
            # Full-SWaT: all catalogued attacks are in scope, drop nothing.
            attack_slices = _remap_slices(SWAT_ATTACK_SLICES_FULL, {})
            attack_slices_dropped = None
    else:
        labels_arr = pd.read_parquet(labels_path)["attack"].to_numpy()
        attack_slices = _find_episodes(labels_arr)
        attack_slices_dropped = None

    results = []
    selected_dir = Path(cfg["selected_dir"])

    for idx, row in summary.iterrows():
        run_dir = selected_dir / row["run_dir"]
        ckpt_path = next(run_dir.glob("*.pt"), None)
        if ckpt_path is None:
            continue

        try:
            model_dict = load_model_bundle(ckpt_path, device=torch.device(device))

            ckpt_keep_cols = model_dict["input_cols"]

            threshold_slice = cfg["threshold_slice"]
            detect_cfg = DETECT_CFG.copy()

            m = compute_metrics(
                model_dict,
                detect_cfg,
                attack_dir=attack_path,
                device=device,
                labels_parquet=labels_path,
                normal_dir=NORMAL_PARQUET,
                threshold_slice=threshold_slice,
                attack_slices_dropped=attack_slices_dropped,
                keep_cols=ckpt_keep_cols,
            )

            # Attack-INDEPENDENT FPR on the clean-normal slice.
            clean_fpr = _clean_eval(
                model_dict,
                detect_cfg,
                threshold_slice,
                ckpt_keep_cols,
                device,
                cache_key=(str(selected_dir), row["run_dir"]),
            )

            rec = {
                "model": row.get("model", ""),
                cfg["exp_var"]: row[cfg["exp_var"]],
                "pinn_flag": row.get("pinn_flag", 0),
                "seed": row.get("seed", 0),
                "w_phys": row.get("w_phys", 0.0),
                "lr": row.get("lr", 0.0),
                "run_dir": row["run_dir"],
            }
            # Extra grouping columns (e.g. corrupt_param when exp_var=physics_filter).
            for _c in cfg.get("extra_cols", []):
                if _c in row and _c not in rec:
                    rec[_c] = row[_c]
            for det_name in ("mse", "per_sensor"):
                ar, sr, _, ts = attackwise_recall(
                    model_dict,
                    m[det_name]["alarm_mask"],
                    attack_slices=attack_slices,
                )
                rec[f"avg_recall_{det_name}"] = ar
                rec[f"std_recall_{det_name}"] = sr
                # Timestep-wise recall over the SAME catalogued episodes as
                # avg_recall, but pooled over steps rather than averaged uniformly
                # over episodes -- so long attacks dominate it. tp/pos are stored
                # too, keeping the ratio auditable and poolable from the cache.
                rec[f"timestep_recall_{det_name}"] = ts[0]
                rec[f"timestep_tp_{det_name}"] = ts[1]
                rec[f"timestep_pos_{det_name}"] = ts[2]
                rec[f"fpr_{det_name}"] = clean_fpr[f"fpr_{det_name}"]

            results.append(rec)
            print(
                f"  [{idx + 1}/{len(summary)}] {row['run_dir']}: "
                f"mse={rec['avg_recall_mse']:.3f} "
                f"ps={rec['avg_recall_per_sensor']:.3f}"
            )
        except Exception as e:
            print(f"  [ERROR] {row['run_dir']}: {e}")

    _guard_nonempty(results, f"{source_exp}/{tag}", cache_path)
    df = pd.DataFrame(results)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df)} results → {cache_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate the experiment grid.")
    ap.add_argument(
        "--plot-dir", default="outputs/v5-plots", help="Output directory for plots."
    )
    ap.add_argument(
        "--cache-dir",
        default="outputs/v5-eval-cache",
        help="Cache directory for inference results.",
    )
    ap.add_argument("--device", default="cpu", help="Torch device (cpu/cuda).")
    ap.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment names to run (default: HEADLINE_EXPERIMENTS for the "
        "headline pass, every grid experiment for --grid).",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip inference, generate plots from cached results.",
    )
    ap.add_argument(
        "--cross-sensor",
        action="store_true",
        help="Run cross-sensor transfer evaluation.",
    )
    ap.add_argument(
        "--cross-shape",
        action="store_true",
        help="Run cross-shape transfer evaluation.",
    )
    ap.add_argument(
        "--ramp-rates",
        action="store_true",
        help="Evaluate poison_lit models across multiple LIT101 ramp rates.",
    )
    ap.add_argument(
        "--rate-sweep",
        action="store_true",
        help="Rate-matched evaluation of v5_poison_lit_rate_sweep "
        "(each training rate evaluated on the same-rate LIT "
        "ramp test parquet).",
    )
    ap.add_argument(
        "--grid",
        action="store_true",
        help="Evaluate the 5×6 cross-regime grid "
        "(training experiments × test datasets).",
    )
    ap.add_argument(
        "--full-swat-synthetic",
        action="store_true",
        help="Evaluate the full-SWaT models (v5_full_poison_lit[_filter]) on the "
        "multi-pool synthetic LIT attack parquet. Feeds the full-SWaT "
        "dose-response plot. Kept off the default --grid path so it doesn't "
        "slow it down.",
    )
    ap.add_argument(
        "--appendix-table",
        action="store_true",
        help="Appendix tables: timestep-wise recall/precision/F1 for the "
        "clean-trained models on the original SWaT attacks. PLC1 models are "
        "scored on the unspliced timeline with non-PLC1 attacks masked as don't "
        "care (per-sensor detector); full-SWaT models are scored over all 32 "
        "catalogued attacks with the MSE detector over a sweep of detection "
        "quantiles. Writes CSVs under "
        "{plot-dir}/appendix; changes no existing numbers.",
    )
    ap.add_argument(
        "--q-data",
        type=float,
        default=None,
        help="Override q_fused in DETECT_CFG. Used by the MSE detector. Default: 0.99.",
    )
    ap.add_argument(
        "--q-per-sensor",
        type=float,
        default=None,
        help="Override q_per_sensor in DETECT_CFG. "
        "Used by the per-sensor detector. Default: 0.999.",
    )
    args = ap.parse_args()

    # --- Override detection quantiles if requested ---
    if args.q_data is not None:
        DETECT_CFG["q_fused"] = args.q_data
    if args.q_per_sensor is not None:
        DETECT_CFG["q_per_sensor"] = args.q_per_sensor

    # No auto-suffix on cache/plot dirs — pass --cache-dir / --plot-dir explicitly
    # if you want isolation between runs.
    cache_dir = Path(args.cache_dir)
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_names = args.experiments or HEADLINE_EXPERIMENTS

    # --- Inference ---
    if not args.plots_only:
        for exp_name in exp_names:
            if exp_name not in EXPERIMENTS:
                print(f"[SKIP] Unknown experiment: {exp_name}")
                continue
            try:
                run_inference_for_experiment(
                    exp_name,
                    device=args.device,
                    cache_dir=cache_dir,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {exp_name}: {e}")

        # Cross-sensor transfer
        if args.cross_sensor:
            _run_cross_sensor(args.device, cache_dir)

        # Cross-shape transfer
        if args.cross_shape:
            _run_cross_shape(args.device, cache_dir)

        # Multi-rate ramp evaluation
        if args.ramp_rates:
            _run_ramp_rates(args.device, cache_dir)

        # Rate-matched evaluation for the LIT ramp-rate sweep.
        if args.rate_sweep:
            _run_rate_sweep_matched(args.device, cache_dir)

        # Cross-regime evaluation grid (supervisor)
        if args.grid:
            # Restrict to the subset of user-specified experiments if any overlap
            grid_exps = [
                e for e in (args.experiments or []) if e in GRID_TRAINING_EXPERIMENTS
            ] or None
            _run_grid(args.device, cache_dir, experiments=grid_exps)

        # Full-SWaT synthetic LIT attacks (multi-pool dose-response).
        if args.full_swat_synthetic:
            _run_full_swat_synthetic_transfers(args.device, cache_dir)

        # Appendix: timestep-wise table on the unspliced SWaT timeline.
        if args.appendix_table:
            _run_appendix_timestep(args.device, cache_dir)
            _run_appendix_timestep_full(args.device, cache_dir)

    # --- Plots ---
    # Single entry point, single flat output directory (per-sensor headline,
    # MSE variants suffixed _mse).
    try:
        from plotting.plots import generate_grid_plots

        generate_grid_plots(cache_dir, plot_dir)
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")


def _run_cross_sensor(device: str, cache_dir: Path):
    """Evaluate v5_poison_lit models on FIT101 ramps and vice versa.

    The ``*_filter`` experiments are evaluated on the same transfer sets so the
    cross-sensor plot can show a Filter bar alongside PINN / Data-only.
    """
    transfers = [
        (
            "v5_poison_lit",
            CROSS_SHAPE_ATTACKS["fit_linear"],
            CROSS_SHAPE_LABELS["fit_linear"],
            "on_fit101_linear",
        ),
        (
            "v5_poison_fit",
            CROSS_SHAPE_ATTACKS["lit_linear"],
            CROSS_SHAPE_LABELS["lit_linear"],
            "on_lit101_linear",
        ),
        (
            "v5_poison_lit_filter",
            CROSS_SHAPE_ATTACKS["fit_linear"],
            CROSS_SHAPE_LABELS["fit_linear"],
            "on_fit101_linear",
        ),
        (
            "v5_poison_fit_filter",
            CROSS_SHAPE_ATTACKS["lit_linear"],
            CROSS_SHAPE_LABELS["lit_linear"],
            "on_lit101_linear",
        ),
    ]
    for source_exp, attack_path, labels_path, tag in transfers:
        if source_exp not in EXPERIMENTS:
            continue
        try:
            run_transfer_inference(
                source_exp,
                attack_path,
                labels_path,
                tag,
                device=device,
                cache_dir=cache_dir,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {source_exp} → {tag}: {e}")


def _run_cross_shape(device: str, cache_dir: Path):
    """Evaluate v5 poison models on staircase, sigmoid, and noisy-ramp attacks.

    The ``*_filter`` experiments are evaluated on the same shapes so the
    cross-shape plot can show a Filter bar alongside PINN / Data-only.
    """
    transfers = [
        ("v5_poison_lit", "lit_staircase"),
        ("v5_poison_lit", "lit_sigmoid"),
        ("v5_poison_lit", "lit_noisy_ramp"),
        ("v5_poison_fit", "fit_staircase"),
        ("v5_poison_fit", "fit_sigmoid"),
        ("v5_poison_fit", "fit_noisy_ramp"),
        ("v5_poison_lit_filter", "lit_staircase"),
        ("v5_poison_lit_filter", "lit_sigmoid"),
        ("v5_poison_lit_filter", "lit_noisy_ramp"),
        ("v5_poison_fit_filter", "fit_staircase"),
        ("v5_poison_fit_filter", "fit_sigmoid"),
        ("v5_poison_fit_filter", "fit_noisy_ramp"),
    ]
    for source_exp, shape_key in transfers:
        if source_exp not in EXPERIMENTS:
            continue
        try:
            run_transfer_inference(
                source_exp,
                CROSS_SHAPE_ATTACKS[shape_key],
                CROSS_SHAPE_LABELS[shape_key],
                f"on_{shape_key}",
                device=device,
                cache_dir=cache_dir,
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {source_exp} → {shape_key}: {e}")


def _run_ramp_rates(device: str, cache_dir: Path):
    """Evaluate v5_poison_lit models across multiple LIT101 ramp rates.

    The training-rate point (``V3_TRAINING_RATE_LIT``) is excluded because its
    cache is the headline ``v5_poison_lit.parquet`` already produced by
    ``run_inference_for_experiment`` — the plot routines splice it in.
    """
    # PINN/Data-only from v5_poison_lit; Filter baseline from v5_poison_lit_filter
    # (its headline cache supplies the training-rate point).
    for source_exp in ("v5_poison_lit", "v5_poison_lit_filter"):
        if source_exp not in EXPERIMENTS:
            print(f"[SKIP] {source_exp} not available for ramp-rate sweep")
            continue
        for rate, paths in RAMP_RATE_ATTACKS_LIT.items():
            tag = f"on_lit_rate_{_rate_tag(rate)}"
            try:
                run_transfer_inference(
                    source_exp,
                    paths["attack"],
                    paths["labels"],
                    tag,
                    device=device,
                    cache_dir=cache_dir,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {source_exp} → rate={rate}: {e}")


# ---------------------------------------------------------------------------
# LIT poisoning ramp-rate sweep — rate-matched evaluation
# ---------------------------------------------------------------------------
# Each model in v5_poison_lit_rate_sweep was trained at one of these rates;
# we evaluate it on the LIT-ramp test parquet at the same rate.
RATE_SWEEP_TRAIN_TO_TEST = {
    0.0005: 0.0005,
    0.001: 0.001,
    0.0015: 0.0015,
    0.002: 0.002,
    0.0025: 0.0025,
    0.003: 0.003,
}


def _run_rate_sweep_matched_for(
    exp_name: str,
    device: str,
    cache_dir: Path,
):
    """Rate-matched evaluation for a rate-sweep experiment.

    For each training rate, picks the matching LIT-ramp test parquet and
    evaluates only the models trained at that rate.  Produces one cache file
    per training rate at ``{exp_name}__rate_matched_train_<tag>.parquet``.
    """
    if exp_name not in EXPERIMENTS:
        print(f"[SKIP rate-sweep] {exp_name} not registered in EXPERIMENTS")
        return
    try:
        full_summary = load_summary(exp_name)
    except FileNotFoundError as e:
        print(f"[SKIP rate-sweep] {e}")
        return
    if "ramp_rate" not in full_summary.columns:
        print(
            f"[SKIP rate-sweep] {exp_name}: summary lacks 'ramp_rate' column "
            "(re-run select_best_models with the updated script)"
        )
        return

    for train_rate, test_rate in RATE_SWEEP_TRAIN_TO_TEST.items():
        sub = full_summary[np.isclose(full_summary["ramp_rate"], train_rate)]
        if sub.empty:
            print(f"  [SKIP] {exp_name} rate={train_rate}: no models in summary")
            continue

        test_tag = _rate_tag(test_rate)
        attack_path = (
            DATA_ROOT
            / f"ramp_attack_lit_v3_{test_tag}"
            / "synthetic_linear_attack.parquet"
        )
        labels_path = (
            DATA_ROOT
            / f"ramp_attack_lit_v3_{test_tag}"
            / "synthetic_linear_labels.parquet"
        )
        if not attack_path.exists():
            print(f"  [SKIP] {attack_path} not found — generate the test parquet first")
            continue

        train_tag = _rate_tag(train_rate)
        try:
            run_transfer_inference(
                exp_name,
                attack_path,
                labels_path,
                tag=f"rate_matched_train_{train_tag}",
                device=device,
                cache_dir=cache_dir,
                use_swat_slices=False,
                summary=sub.reset_index(drop=True),
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {exp_name} → rate_matched train={train_tag}: {e}")


def _run_rate_sweep_matched(device: str, cache_dir: Path):
    """Convenience: run rate-matched eval for both rate-sweep experiments
    (PINN/data-only and the filter baseline). Either is silently skipped if
    its selected-models summary is missing.
    """
    _run_rate_sweep_matched_for("v5_poison_lit_rate_sweep", device, cache_dir)
    _run_rate_sweep_matched_for("v5_poison_lit_rate_sweep_filter", device, cache_dir)


# ---------------------------------------------------------------------------
# Cross-regime evaluation grid
# ---------------------------------------------------------------------------
# Experiments that participate in the supervisor's grid.  All levels of each
# experiment are kept in the resulting cache (no summary filtering here) so the
# same files serve both the overview (plot-time level filter) and dose-response
# figures (use all levels directly).
GRID_TRAINING_EXPERIMENTS = [
    "v5_poison_lit",
    "v5_poison_fit",
    "v5_consistent_lit_sin",  # physics-consistent sinusoidal LIT poisoning
    "v5_consistent_fit",
    # Per-w_phys selection on the LIT-poisoning ablation grid — feeds
    # fig3_inconsistent_lit_recall_per_wphys + its FPR sibling (one line per
    # w_phys value). Cache file:
    # v5_poison_lit_per_wphys__grid__inconsistent_lit_test.parquet
    "v5_poison_lit_per_wphys",
    # PLC1 LIT poisoning filter baseline — Filter line of filter-vs-PINN-on-
    # poison-fraction plots (needs grid caches incl. all_swat_attacks).
    "v5_poison_lit_filter",
    # Filter baselines for the other PLC1 poisoning families — Filter line of the
    # inconsistent (FIT) and consistent (LIT-sin / FIT) dose-response plots.
    "v5_poison_fit_filter",
    "v5_consistent_lit_sin_filter",
    "v5_consistent_fit_filter",
    # (A) invariant-filter run — evaluated on the synthetic inconsistent-LIT
    # test (grid cache); feeds the data-only vs invariant comparison.
    "v5_poison_lit_invariant",
    # (B) physics-filter quantile sweep — evaluated on the RATE-MATCHED 0.5 mm/s
    # inconsistent-LIT test (the rate the q-sweep poison was trained at).
    "v5_poison_lit_filter_qsweep",
]

# (attack_path, labels_path, use_swat_slices, swat_slice_subset)
GRID_TEST_DATASETS = {
    "phys_valid_swat": (
        SWAT_ATTACK_PARQUET,
        SWAT_LABELS_PARQUET,
        True,
        SWAT_PHYS_VALID_SLICES,
    ),
    "phys_invalid_swat": (
        SWAT_ATTACK_PARQUET,
        SWAT_LABELS_PARQUET,
        True,
        SWAT_PHYS_INVALID_SLICES,
    ),
    # All 10 PLC1 SWaT attacks combined (no phys-valid/invalid split). Feeds the
    # wphys-ablation-on-real-attacks and filter-vs-PINN-on-real-attacks plots.
    "all_swat_attacks": (
        SWAT_ATTACK_PARQUET,
        SWAT_LABELS_PARQUET,
        True,
        SWAT_ATTACK_SLICES,
    ),
    "inconsistent_lit_test": (
        DATA_ROOT / "ramp_attack_lit_v3" / "synthetic_linear_attack.parquet",
        DATA_ROOT / "ramp_attack_lit_v3" / "synthetic_linear_labels.parquet",
        False,
        None,
    ),
    # Rate-matched (0.5 mm/s) inconsistent-LIT test for the stealthy q-sweep,
    # which trains its poison at 0.0005 m/s. Grading the q-sweep on the steep
    # 1.943 mm/s inconsistent_lit_test washes out the q-sensitivity.
    "inconsistent_lit_test_0p5mm": (
        DATA_ROOT / "ramp_attack_lit_v3_0p5mm" / "synthetic_linear_attack.parquet",
        DATA_ROOT / "ramp_attack_lit_v3_0p5mm" / "synthetic_linear_labels.parquet",
        False,
        None,
    ),
    "inconsistent_fit_test": (
        DATA_ROOT / "ramp_attack_fit_v3" / "synthetic_linear_attack.parquet",
        DATA_ROOT / "ramp_attack_fit_v3" / "synthetic_linear_labels.parquet",
        False,
        None,
    ),
    "consistent_lit_test": (
        DATA_ROOT
        / "consistent_ramp_attack_lit_sin_v3"
        / "synthetic_consistent_lit_sin_attack.parquet",
        DATA_ROOT
        / "consistent_ramp_attack_lit_sin_v3"
        / "synthetic_consistent_lit_sin_labels.parquet",
        False,
        None,
    ),
    "consistent_fit_test": (
        DATA_ROOT
        / "consistent_ramp_attack_fit_v3"
        / "synthetic_consistent_fit_attack.parquet",
        DATA_ROOT
        / "consistent_ramp_attack_fit_v3"
        / "synthetic_consistent_fit_labels.parquet",
        False,
        None,
    ),
    # Full-SWaT multi-pool synthetic LIT attack. Consumed by the full-SWaT
    # dispatcher (_run_full_swat_synthetic_transfers), not the cross-regime grid.
    "inconsistent_multistage_lit_test": (
        DATA_ROOT
        / "ramp_attack_multistage_lit_v3"
        / "synthetic_multistage_attack.parquet",
        DATA_ROOT
        / "ramp_attack_multistage_lit_v3"
        / "synthetic_multistage_labels.parquet",
        False,
        None,
    ),
}


# Per-experiment grid test-set restriction: each experiment is only evaluated on
# the subset of GRID_TEST_DATASETS its plots read; experiments absent fall back
# to the full set.
GRID_TEST_SUBSET = {
    # fig3 LIT panel + invariant data-only line (inconsistent_lit_test),
    # plc1_poison_on_original + clean_data (all_swat_attacks / phys_valid /
    # phys_invalid). Never the FIT / consistent / multistage sets.
    "v5_poison_lit": [
        "inconsistent_lit_test",
        "all_swat_attacks",
        "phys_valid_swat",
        "phys_invalid_swat",
    ],
    "v5_poison_lit_filter": [
        "inconsistent_lit_test",
        "all_swat_attacks",
        "phys_valid_swat",
        "phys_invalid_swat",
    ],
    # fig3 FIT panel / fig4 consistent panels (model + Filter line).
    "v5_poison_fit": ["inconsistent_fit_test"],
    "v5_poison_fit_filter": ["inconsistent_fit_test"],
    "v5_consistent_lit_sin": ["consistent_lit_test"],
    "v5_consistent_lit_sin_filter": ["consistent_lit_test"],
    "v5_consistent_fit": ["consistent_fit_test"],
    "v5_consistent_fit_filter": ["consistent_fit_test"],
    # One line per w_phys on the inconsistent-LIT dose-response.
    "v5_poison_lit_per_wphys": ["inconsistent_lit_test"],
    "v5_poison_lit_invariant": ["inconsistent_lit_test"],
    "v5_poison_lit_filter_qsweep": ["inconsistent_lit_test_0p5mm"],
}


def _run_grid(device: str, cache_dir: Path, experiments: list[str] | None = None):
    """Evaluate the cross-regime grid (training experiments × test datasets).

    Produces cache files named ``grid__{exp_name}__{test_name}.parquet``.
    Each cache holds all levels of the experiment so downstream plotting can
    either filter to a representative level (overview) or use all levels
    (dose-response). Experiments in GRID_TEST_SUBSET only run their named tests.
    """
    exps = experiments or GRID_TRAINING_EXPERIMENTS
    for exp_name in exps:
        if exp_name not in GRID_TRAINING_EXPERIMENTS:
            print(f"[SKIP grid] {exp_name} not in GRID_TRAINING_EXPERIMENTS")
            continue
        test_names = GRID_TEST_SUBSET.get(exp_name, list(GRID_TEST_DATASETS))
        for test_name in test_names:
            apath, lpath, use_swat, subset = GRID_TEST_DATASETS[test_name]
            tag = f"grid__{test_name}"
            try:
                run_transfer_inference(
                    exp_name,
                    apath,
                    lpath,
                    tag,
                    device=device,
                    cache_dir=cache_dir,
                    use_swat_slices=use_swat,
                    swat_slice_subset=subset,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {exp_name} → {test_name}: {e}")


# ---------------------------------------------------------------------------
# Appendix: timestep-wise table (unspliced timeline + don't-care mask)
# ---------------------------------------------------------------------------
# The headline numbers average recall over attack episodes and enforce the
# PLC1 scope by DELETING the 22 non-PLC1 attack ranges, which splices the timeline
# and fabricates a discontinuity at every drop boundary. The appendix reports the
# conventional timestep-wise recall/precision/F1 instead: every row is kept, PLC1
# attack rows are positives, other stages' attack rows are "don't care" (excluded
# from the confusion matrix), and everything else is a negative.
#
# FPR is not taken from this pass at all -- it reuses _clean_eval's attack-free
# slice of the normal data, so it carries no boundary contamination whatsoever.
APPENDIX_EXPERIMENTS = ("v5_poison_lit", "v5_poison_lit_filter")
APPENDIX_TAG = "appendix_timestep"
APPENDIX_DETECTOR = "per_sensor"


def _appendix_masks(
    model_dict: dict,
    n_steps: int,
    n_rows: int,
    *,
    pos_slices: dict | None = None,
    ignore_slices: dict | None = None,
):
    """Positive / don't-care masks per prediction step on the unspliced timeline.

    Returns ``(pos, ignore)``, both length ``n_steps`` and aligned to ``alarm_mask``.
    Positives are the catalogued PLC1 attacks, don't-care the catalogued non-PLC1
    attacks, everything else a negative. The full-SWaT companion overrides both:
    every catalogued attack is a positive and nothing is don't-care.

    The classes come from the slice catalogue in this module, which is the
    authoritative source -- its boundaries were refined to match the real attack
    start/end. ``swat_attack_labels.parquet`` is deliberately NOT consulted: it
    disagrees with the catalogue on ~3.7k rows (20-40 row tails/lead-ins on 31 of
    the 32 episodes, plus four episodes the catalogue omits), and those looser
    boundaries are exactly what the refinement corrected.
    """
    from inference.detectors import Detector

    pos_slices = SWAT_ATTACK_SLICES if pos_slices is None else pos_slices
    ignore_slices = (
        SWAT_ATTACK_SLICES_DROPPED if ignore_slices is None else ignore_slices
    )

    def _cover(slices):
        m = np.zeros(int(n_rows), dtype=bool)
        for info in slices.values():
            m[int(info["Start_Row"]) : int(info["End_Row"])] = True
        return m

    pos_1hz = _cover(pos_slices)
    ignore_1hz = _cover(ignore_slices)

    # Same window -> 1 Hz row mapping attackwise_recall uses, so the appendix and
    # the headline share one coordinate system.
    idx_rows = Detector._downsampled_time(
        n_steps,
        0,
        int(model_dict["window"]),
        int(model_dict["temporal_stride"]),
        str(model_dict["downsample_agg"]),
    )
    idx_rows = np.clip(idx_rows, 0, int(n_rows) - 1)
    return pos_1hz[idx_rows], ignore_1hz[idx_rows]


def _appendix_label_steps(
    model_dict: dict, attack_label_1hz: np.ndarray, n_steps: int
) -> np.ndarray:
    """The published SWaT labels mapped onto prediction steps.

    Same window -> 1 Hz mapping as ``_appendix_masks``, so the result can be
    compared against the catalogue-derived positives step for step.
    """
    from inference.detectors import Detector

    idx_rows = Detector._downsampled_time(
        n_steps,
        0,
        int(model_dict["window"]),
        int(model_dict["temporal_stride"]),
        str(model_dict["downsample_agg"]),
    )
    idx_rows = np.clip(idx_rows, 0, len(attack_label_1hz) - 1)
    return attack_label_1hz[idx_rows].astype(bool)


def _timestep_scores(
    alarm_mask: np.ndarray, pos: np.ndarray, ignore: np.ndarray
) -> dict:
    """Timestep-wise confusion matrix and rates, don't-care steps excluded.

    Scored exactly like Detector.calculate_pointwise_metrics: same sklearn calls,
    same ``labels=[0, 1]`` guard so the matrix stays 2x2 when only one class is
    present, same ``zero_division=0``. The only difference is that steps falling
    inside a non-PLC1 attack are removed from both vectors first.
    """
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
    )

    keep = ~ignore
    y = pos[keep].astype(int)
    y_hat = np.asarray(alarm_mask, dtype=bool)[keep].astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()

    return dict(
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        n_scored=int(np.count_nonzero(keep)),
        n_ignored=int(np.count_nonzero(ignore)),
        recall=recall_score(y, y_hat, zero_division=0),
        precision=precision_score(y, y_hat, zero_division=0),
        f1=f1_score(y, y_hat, zero_division=0),
        # In-situ FPR over the normal-labelled steps of THIS evaluation set, i.e.
        # the same negatives the precision denominator uses. Unlike fpr_clean it
        # sits next to attacks, so it carries whatever boundary effects the
        # evaluation set has -- but it is the one that reconciles with precision:
        #   precision = TP / (TP + fpr_attackset * (FP + TN)).
        fpr_attackset=float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0,
    )


def run_appendix_timestep_inference(
    exp_name: str,
    *,
    device: str = "cpu",
    cache_dir: Path = Path("outputs/v5-eval-cache"),
) -> pd.DataFrame:
    """Timestep-wise metrics for clean-trained models on the original SWaT attacks.

    Evaluates the ``corrupt_param == 0`` models of ``exp_name`` against the full,
    UNSPLICED SWaT attack parquet and scores every prediction step except those
    falling inside a non-PLC1 attack. Caches to ``{exp}__appendix_timestep.parquet``.
    """
    cache_path = cache_dir / f"{exp_name}__{APPENDIX_TAG}.parquet"
    if cache_path.exists():
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return pd.read_parquet(cache_path)

    sys.path.insert(0, str(Path("src").resolve()))
    from models.config import load_model_bundle
    from inference.metrics import compute_metrics, attackwise_recall

    cfg = EXPERIMENTS[exp_name]
    summary = load_summary(exp_name)
    summary = summary[summary["corrupt_param"] == 0.0].reset_index(drop=True)
    print(f"\n[{exp_name}/{APPENDIX_TAG}] {len(summary)} clean models to evaluate")

    # Loaded only to size the masks to the attack parquet; the class definitions
    # come from the slice catalogue, not from these labels (see _appendix_masks).
    attack_label_1hz = (
        pd.read_parquet(SWAT_LABELS_PARQUET)["attack"].to_numpy().astype(np.int8)
    )
    # Nothing is dropped here, so the catalogued PLC1 slices keep raw coordinates.
    attack_slices = _remap_slices(SWAT_ATTACK_SLICES, {})

    results = []
    selected_dir = Path(cfg["selected_dir"])
    threshold_slice = cfg["threshold_slice"]

    for idx, row in summary.iterrows():
        run_dir = selected_dir / row["run_dir"]
        ckpt_path = next(run_dir.glob("*.pt"), None)
        if ckpt_path is None:
            print(f"  [SKIP] No .pt in {run_dir}")
            continue

        try:
            model_dict = load_model_bundle(ckpt_path, device=torch.device(device))
            ckpt_keep_cols = model_dict["input_cols"]
            detect_cfg = DETECT_CFG.copy()

            # attack_slices_dropped=None keeps every row -> no splice, no seams.
            m = compute_metrics(
                model_dict,
                detect_cfg,
                attack_dir=SWAT_ATTACK_PARQUET,
                device=device,
                labels_parquet=SWAT_LABELS_PARQUET,
                normal_dir=NORMAL_PARQUET,
                threshold_slice=threshold_slice,
                attack_slices_dropped=None,
                keep_cols=ckpt_keep_cols,
            )
            alarm_mask = m[APPENDIX_DETECTOR]["alarm_mask"]

            # Attack-independent FPR on the clean-normal slice (same as headline).
            clean_fpr = _clean_eval(
                model_dict,
                detect_cfg,
                threshold_slice,
                ckpt_keep_cols,
                device,
                cache_key=(str(selected_dir), row["run_dir"]),
            )

            # Diagnostic only: the headline attack-averaged recall
            # (avg_recall_* in the other caches) recomputed on this unspliced
            # pass. Sanity-checks the coordinate mapping; not reported.
            avg_recall = attackwise_recall(
                model_dict, alarm_mask, attack_slices=attack_slices
            )[0]

            base = {
                "exp": exp_name,
                "model": row.get("model", ""),
                "pinn_flag": row.get("pinn_flag", 0),
                "seed": row.get("seed", 0),
                "w_phys": row.get("w_phys", 0.0),
                "lr": row.get("lr", 0.0),
                "corrupt_param": row.get("corrupt_param", 0.0),
                "run_dir": row["run_dir"],
                "detector": APPENDIX_DETECTOR,
                "fpr_clean": clean_fpr[f"fpr_{APPENDIX_DETECTOR}"],
                "avg_recall": avg_recall,
            }

            pos, ignore = _appendix_masks(
                model_dict, len(alarm_mask), len(attack_label_1hz)
            )
            scores = _timestep_scores(alarm_mask, pos, ignore)
            n_pos = scores["tp"] + scores["fn"]
            n_neg = scores["fp"] + scores["tn"]
            rec = {
                **base,
                "n_pos": n_pos,
                "n_neg": n_neg,
                # k = N/P. Precision and F1 scale with it; recall does not, so the
                # ratio has to be reported for the precision column to be readable.
                "k_ratio": (n_neg / n_pos) if n_pos else float("nan"),
                **scores,
                # Packed per-step alarm mask (~5.6 kB/model). Lets any alternative
                # scoring be evaluated offline from the cache, with no re-run.
                "n_steps": int(len(alarm_mask)),
                "alarm_bits": np.packbits(np.asarray(alarm_mask, dtype=bool)).tobytes(),
            }
            results.append(rec)
            print(
                f"  [{idx + 1}/{len(summary)}] {row['run_dir']}: "
                f"recall={rec['recall']:.3f} prec={rec['precision']:.3f} "
                f"f1={rec['f1']:.3f} fpr_clean={rec['fpr_clean']:.4f} "
                f"fpr_eval={rec['fpr_attackset']:.4f}"
            )
        except Exception as e:
            print(f"  [ERROR] {row['run_dir']}: {e}")

    _guard_nonempty(results, f"{exp_name}/{APPENDIX_TAG}", cache_path)
    df = pd.DataFrame(results)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df)} results -> {cache_path}")

    return df


def _run_appendix_timestep(device: str, cache_dir: Path):
    """Run the appendix timestep-wise pass for every appendix experiment."""
    for exp_name in APPENDIX_EXPERIMENTS:
        cfg = EXPERIMENTS.get(exp_name)
        if cfg is None:
            print(f"[SKIP appendix] unknown experiment {exp_name}")
            continue
        if not (Path(cfg["selected_dir"]) / "summary.csv").exists():
            print(
                f"[SKIP appendix] {exp_name}: missing {cfg['selected_dir']}/summary.csv"
            )
            continue
        try:
            run_appendix_timestep_inference(
                exp_name, device=device, cache_dir=cache_dir
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {exp_name} -> {APPENDIX_TAG}: {e}")


# ---------------------------------------------------------------------------
# Appendix companion for the full-SWaT models: MSE detector, quantile sweep
# ---------------------------------------------------------------------------
# Same clean-training setting as the PLC1 appendix, same original SWaT attack
# parquet -- but every catalogued attack is in scope. Full-SWaT models are trained
# on all stages, so nothing is "don't care" and the headline pass already runs on
# the unspliced timeline (drop_non_plc1=False). What this table adds over the
# headline is therefore not the splice, but conventional timestep-wise
# recall/precision/F1 instead of the episode-uniform average recall, over a sweep
# of detection quantiles, plus a per-channel attribution of the MSE score.
APPENDIX_FULL_EXPERIMENTS = ("v5_full_poison_lit", "v5_full_poison_lit_filter")
APPENDIX_FULL_TAG = "appendix_timestep_full"
APPENDIX_FULL_BASE_DETECTOR = "mse"
# Attack 28 spans ~34k of the ~52k catalogued attack rows, so timestep-wise recall
# over the full positive set is largely "recall on attack 28". Recall with 28 held
# out as don't-care is reported next to it so the two can be told apart.
APPENDIX_FULL_DOMINANT_ATTACK = 28


# Detection quantiles swept for the MSE detector. With the fused full-SWaT
# threshold slice (86 400 rows) at stride 10 and a 50-step window there are ~8 635
# score samples to fit tau from, so ~86 sit above q=0.99 and ~8.6 above q=0.999.
# Beyond q=0.9995 (~4 samples) the "quantile" approaches the sample maximum and
# starts to swing with the seed, which is why the sweep stops there.
APPENDIX_FULL_QUANTILES = (0.99, 0.995, 0.999, 0.9995)


def _mse_alarm_masks(zN, zA, quantiles) -> dict:
    """Alarm mask per quantile from one pair of score tensors.

    ``zN``/``zA`` do not depend on q — only ``tau = quantile(mse_N, q)`` does — so
    a whole quantile sweep costs one forward pass, not one per q.
    """
    mse_n = zN.mean(dim=1)
    mse_a = zA.mean(dim=1)
    out = {}
    for q in quantiles:
        tau = torch.quantile(
            mse_n,
            q=torch.tensor(float(q), dtype=mse_n.dtype, device=mse_n.device),
            dim=0,
        )
        out[float(q)] = (mse_a > tau).cpu().numpy().astype(bool)
    return out


def _clean_fpr_sweep(
    model_dict, detect_cfg, threshold_slice, keep_cols, device, detectors, quantiles
) -> dict:
    """``{(detector, q): fpr}`` on the attack-free clean-normal slice.

    Deliberately not memoized through ``_CLEAN_FPR_CACHE``: each model is visited
    once per pass, and caching the score tensors this needs would hold ~50 MB per
    model. Costs the same single clean-set forward pass ``_clean_eval`` would.
    """
    _ensure_clean_normal()
    from inference.metrics import compute_metrics

    m = compute_metrics(
        model_dict,
        detect_cfg,
        attack_dir=CLEAN_NORMAL_PARQUET,
        device=device,
        labels_parquet=CLEAN_NORMAL_LABELS,
        normal_dir=NORMAL_PARQUET,
        threshold_slice=threshold_slice,
        attack_slices_dropped=None,
        keep_cols=keep_cols,
        detectors=detectors,
    )
    out = {}
    for det in detectors:
        masks = _mse_alarm_masks(m[det.name]["zN"], m[det.name]["zA"], quantiles)
        for q, alarm in masks.items():
            # Every step of the clean-normal set is a true negative by construction.
            out[(det.name, q)] = float(np.count_nonzero(alarm)) / float(len(alarm))
    return out


def _appendix_full_channel_report(
    m,
    model_dict,
    pos: np.ndarray,
    ignore: np.ndarray,
    q_fused: float,
    base_alarm: np.ndarray,
) -> dict:
    """Per-residual-channel attribution for the MSE detector.

    Answers "which sensor is driving the alarms" without a second forward pass:
    everything here comes off the baseline detector's full-width zN/zA, which
    ``calculate_pointwise_metrics`` already returns.

    Three families of columns, each a JSON dict keyed by channel name:

    ``share_*``  which channels carry the score mass — on the clean threshold
                 slice, on the attack set, and on the false-positive steps
                 specifically. The last one is the direct answer: it is the
                 composition of the score at the moments the detector wrongly
                 fires.
    ``ratio``    mean squared residual on the attack set over the same on the
                 clean slice, per channel. The domain-shift fingerprint: a
                 channel the model fits on the normal recording but not on the
                 attack recording shows up here as a large number, whether or not
                 it carries much absolute mass.
    ``loo_*``    leave-one-out FPR and recall: drop that one channel, refit the
                 quantile on the reduced mean, rescore. Turns "should we exclude
                 another sensor?" into a measurement rather than a guess.
    """
    base_det = APPENDIX_FULL_BASE_DETECTOR
    zN = m[base_det]["zN"]
    zA = m[base_det]["zA"]
    alarm = np.asarray(base_alarm, dtype=bool)
    cols = list(model_dict["target_cols"])
    n = int(zN.shape[1])
    if len(cols) != n:
        raise ValueError(f"target_cols ({len(cols)}) != residual width ({n})")

    neg = (~pos) & (~ignore)
    fp_steps = neg & alarm

    col_n = zN.mean(dim=0)
    col_a = zA.mean(dim=0)
    if fp_steps.any():
        col_fp = zA[torch.from_numpy(fp_steps).to(zA.device)].mean(dim=0)
    else:
        col_fp = torch.full_like(col_a, float("nan"))

    def _share(v):
        t = float(v.sum())
        return (v / t).cpu().numpy() if t > 0 else np.full(n, np.nan)

    eps = torch.finfo(col_n.dtype).tiny
    ratio = (col_a / col_n.clamp_min(eps)).cpu().numpy()

    # Leave-one-out, vectorised over channels: (sum - column) / (n - 1) is the
    # mean with that channel dropped, and the threshold is refit on it, so these
    # numbers predict what excluding that channel from the detector would produce.
    if n > 1:
        loo_n = (zN.sum(dim=1, keepdim=True) - zN) / (n - 1)
        loo_a = (zA.sum(dim=1, keepdim=True) - zA) / (n - 1)
        tau = torch.quantile(
            loo_n, q=torch.tensor(q_fused, dtype=loo_n.dtype, device=loo_n.device), dim=0
        )
        alarm_loo = (loo_a > tau).cpu().numpy()
        loo_fpr = alarm_loo[neg].mean(axis=0) if neg.any() else np.full(n, np.nan)
        loo_rec = alarm_loo[pos].mean(axis=0) if pos.any() else np.full(n, np.nan)
    else:
        loo_fpr = loo_rec = np.full(n, np.nan)

    def _js(vals):
        return json.dumps({c: float(v) for c, v in zip(cols, vals)})

    return {
        "channel_share_N": _js(_share(col_n)),
        "channel_share_A": _js(_share(col_a)),
        "channel_share_fp": _js(_share(col_fp)),
        "channel_ratio_A_N": _js(ratio),
        "channel_loo_fpr": _js(loo_fpr),
        "channel_loo_recall": _js(loo_rec),
    }


def run_appendix_timestep_full_inference(
    exp_name: str,
    *,
    device: str = "cpu",
    cache_dir: Path = Path("outputs/v5-eval-cache"),
) -> pd.DataFrame:
    """Timestep-wise MSE metrics for clean full-SWaT models on the SWaT attacks.

    One row per model per swept quantile, from a single forward pass, using the
    ``mse`` detector over every residual column. Caches to
    ``{exp}__appendix_timestep_full.parquet``.
    """
    cache_path = cache_dir / f"{exp_name}__{APPENDIX_FULL_TAG}.parquet"
    if cache_path.exists():
        print(f"  [CACHED] {cache_path} exists, skipping.")
        return pd.read_parquet(cache_path)

    sys.path.insert(0, str(Path("src").resolve()))
    from models.config import load_model_bundle
    from inference.metrics import compute_metrics, attackwise_recall
    from inference.detectors import MSEDetector

    cfg = EXPERIMENTS[exp_name]
    summary = load_summary(exp_name)
    summary = summary[summary["corrupt_param"] == 0.0].reset_index(drop=True)
    print(f"\n[{exp_name}/{APPENDIX_FULL_TAG}] {len(summary)} clean models to evaluate")

    # Sizes the masks only; the classes come from the slice catalogue. Also used to
    # count rows the published labels call attacks but the catalogue does not.
    attack_label_1hz = (
        pd.read_parquet(SWAT_LABELS_PARQUET)["attack"].to_numpy().astype(np.int8)
    )
    # Identity remap (nothing is dropped) — kept for symmetry with the headline pass.
    attack_slices = _remap_slices(SWAT_ATTACK_SLICES_FULL, {})
    slices_ex28 = {
        k: v
        for k, v in SWAT_ATTACK_SLICES_FULL.items()
        if k != APPENDIX_FULL_DOMINANT_ATTACK
    }
    slices_28 = {
        k: v
        for k, v in SWAT_ATTACK_SLICES_FULL.items()
        if k == APPENDIX_FULL_DOMINANT_ATTACK
    }

    detectors = [MSEDetector()]

    results = []
    selected_dir = Path(cfg["selected_dir"])
    threshold_slice = cfg["threshold_slice"]

    for idx, row in summary.iterrows():
        run_dir = selected_dir / row["run_dir"]
        ckpt_path = next(run_dir.glob("*.pt"), None)
        if ckpt_path is None:
            print(f"  [SKIP] No .pt in {run_dir}")
            continue

        try:
            model_dict = load_model_bundle(ckpt_path, device=torch.device(device))
            ckpt_keep_cols = model_dict["input_cols"]
            detect_cfg = DETECT_CFG.copy()

            # attack_slices_dropped=None keeps every row: full-SWaT scope.
            m = compute_metrics(
                model_dict,
                detect_cfg,
                attack_dir=SWAT_ATTACK_PARQUET,
                device=device,
                labels_parquet=SWAT_LABELS_PARQUET,
                normal_dir=NORMAL_PARQUET,
                threshold_slice=threshold_slice,
                attack_slices_dropped=None,
                keep_cols=ckpt_keep_cols,
                detectors=detectors,
            )

            # Attack-independent FPR on the clean-normal slice (same as headline),
            # for every swept quantile.
            clean_fpr = _clean_fpr_sweep(
                model_dict,
                detect_cfg,
                threshold_slice,
                ckpt_keep_cols,
                device,
                detectors,
                APPENDIX_FULL_QUANTILES,
            )

            n_steps = len(m[APPENDIX_FULL_BASE_DETECTOR]["alarm_mask"])
            n_rows = len(attack_label_1hz)
            pos, ignore = _appendix_masks(
                model_dict,
                n_steps,
                n_rows,
                pos_slices=SWAT_ATTACK_SLICES_FULL,
                ignore_slices={},
            )
            pos_ex28, ignore_ex28 = _appendix_masks(
                model_dict,
                n_steps,
                n_rows,
                pos_slices=slices_ex28,
                ignore_slices=slices_28,
            )
            # Rows the published labels mark as attack but the refined catalogue
            # does not cover (episode tails/lead-ins and the episodes it omits).
            # They are scored as negatives here, so they land in FP for any working
            # detector — a systematic, bounded precision penalty worth reporting.
            label_steps = _appendix_label_steps(model_dict, attack_label_1hz, n_steps)
            n_label_only_pos = int(np.count_nonzero(label_steps & ~pos))

            # One sweep, reused for the diagnostics below so every reported number
            # refers to the same quantile grid regardless of what
            # DETECT_CFG/--q-data happens to be set to.
            det_name = APPENDIX_FULL_BASE_DETECTOR
            sweep = _mse_alarm_masks(
                m[det_name]["zN"], m[det_name]["zA"], APPENDIX_FULL_QUANTILES
            )
            q0 = float(APPENDIX_FULL_QUANTILES[0])
            channels = _appendix_full_channel_report(
                m, model_dict, pos, ignore, q0, sweep[q0]
            )

            base = {
                "exp": exp_name,
                "model": row.get("model", ""),
                "pinn_flag": row.get("pinn_flag", 0),
                "seed": row.get("seed", 0),
                "w_phys": row.get("w_phys", 0.0),
                "lr": row.get("lr", 0.0),
                "corrupt_param": row.get("corrupt_param", 0.0),
                "run_dir": row["run_dir"],
                "n_label_only_pos": n_label_only_pos,
                # Residual width, i.e. how many channels the MSE mean runs over:
                # the check that --drop_cols actually took effect.
                "n_targets": int(m[det_name]["zN"].shape[1]),
            }

            recs = []
            for q, alarm_mask in sweep.items():
                avg_recall, _, per_attack, _ = attackwise_recall(
                    model_dict, alarm_mask, attack_slices=attack_slices
                )
                scores = _timestep_scores(alarm_mask, pos, ignore)
                n_pos = scores["tp"] + scores["fn"]
                n_neg = scores["fp"] + scores["tn"]
                recs.append(
                    {
                        **base,
                        "q": float(q),
                        "detector": det_name,
                        # Channel attribution is a property of the model, not
                        # of the quantile, so it rides on the baseline q only;
                        # blanks (not NaN) keep the parquet column a string.
                        **(
                            channels
                            if q == APPENDIX_FULL_QUANTILES[0]
                            else {k: "" for k in channels}
                        ),
                        "fpr_clean": clean_fpr[(det_name, float(q))],
                        "avg_recall": avg_recall,
                        "per_attack_recall": json.dumps(
                            {str(k): float(v) for k, v in per_attack.items()}
                        ),
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                        # k = N/P. Precision and F1 scale with it; recall does
                        # not, so the ratio has to be reported for the
                        # precision column to be readable — and it differs
                        # sharply from the PLC1 table's, so the two precisions
                        # are not comparable.
                        "k_ratio": (n_neg / n_pos) if n_pos else float("nan"),
                        **scores,
                        "recall_ex28": _timestep_scores(
                            alarm_mask, pos_ex28, ignore_ex28
                        )["recall"],
                        "n_steps": int(n_steps),
                        "alarm_bits": np.packbits(
                            np.asarray(alarm_mask, dtype=bool)
                        ).tobytes(),
                    }
                )

            # Extend atomically so the per-model row set cannot break halfway
            # through a model.
            results.extend(recs)
            # Print the quantile sweep — one row per q, too wide to print whole.
            print(
                f"  [{idx + 1}/{len(summary)}] {row['run_dir']}: "
                + "  ".join(
                    f"q={r['q']:g} rec={r['recall']:.3f} prec={r['precision']:.3f} "
                    f"fpr_a={r['fpr_attackset']:.3f} fpr_c={r['fpr_clean']:.4f}"
                    for r in recs
                )
            )
        except Exception as e:
            print(f"  [ERROR] {row['run_dir']}: {e}")

    _guard_nonempty(results, f"{exp_name}/{APPENDIX_FULL_TAG}", cache_path)
    expected = len(APPENDIX_FULL_QUANTILES) * len(summary)
    if len(results) != expected:
        print(
            f"  [WARN] {len(results)} rows for {len(summary)} models "
            f"(expected {expected} = {len(APPENDIX_FULL_QUANTILES)} quantiles) "
            "— some models failed above."
        )

    df = pd.DataFrame(results)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df)} rows -> {cache_path}")

    return df


def _run_appendix_timestep_full(device: str, cache_dir: Path):
    """Run the full-SWaT appendix timestep-wise pass for every full experiment."""
    for exp_name in APPENDIX_FULL_EXPERIMENTS:
        cfg = EXPERIMENTS.get(exp_name)
        if cfg is None:
            print(f"[SKIP appendix-full] unknown experiment {exp_name}")
            continue
        if not (Path(cfg["selected_dir"]) / "summary.csv").exists():
            print(
                f"[SKIP appendix-full] {exp_name}: "
                f"missing {cfg['selected_dir']}/summary.csv"
            )
            continue
        try:
            run_appendix_timestep_full_inference(
                exp_name, device=device, cache_dir=cache_dir
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[SKIP] {exp_name} -> {APPENDIX_FULL_TAG}: {e}")


# Full-SWaT synthetic LIT attacks evaluated for the two full-SWaT experiments.
# Feeds the full-SWaT inconsistent-poisoning dose-response plot (multi-pool LIT).
FULL_SWAT_SYNTHETIC_EXPERIMENTS = ("v5_full_poison_lit", "v5_full_poison_lit_filter")
FULL_SWAT_SYNTHETIC_TESTS = ("inconsistent_multistage_lit_test",)


def _run_full_swat_synthetic_transfers(device: str, cache_dir: Path):
    """Evaluate the full-SWaT models on the multi-pool synthetic LIT attack.

    Produces caches ``{exp}__grid__{test_name}.parquet`` (matching the grid tag
    convention so the plotting helpers can read them uniformly). Skips silently
    if an experiment's selected_dir/summary is missing.
    """
    for exp_name in FULL_SWAT_SYNTHETIC_EXPERIMENTS:
        cfg = EXPERIMENTS.get(exp_name)
        if cfg is None:
            print(f"[SKIP full-swat] unknown experiment {exp_name}")
            continue
        if not (Path(cfg["selected_dir"]) / "summary.csv").exists():
            print(
                f"[SKIP full-swat] {exp_name}: missing "
                f"{cfg['selected_dir']}/summary.csv"
            )
            continue
        for test_name in FULL_SWAT_SYNTHETIC_TESTS:
            apath, lpath, use_swat, subset = GRID_TEST_DATASETS[test_name]
            tag = f"grid__{test_name}"
            try:
                run_transfer_inference(
                    exp_name,
                    apath,
                    lpath,
                    tag,
                    device=device,
                    cache_dir=cache_dir,
                    use_swat_slices=use_swat,
                    swat_slice_subset=subset,
                )
            except FileNotFoundError as e:
                print(f"[SKIP] {exp_name} → {test_name}: {e}")


if __name__ == "__main__":
    main()
