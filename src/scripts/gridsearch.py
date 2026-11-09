"""Hyperparameter-sweep driver: expand a config block into train.py runs.

Each block is a Cartesian product over architecture, w_phys (0=Data-Only, 1=PINN),
learning rate, and seed; the Filtering variant adds the physics-filter flag.
"""

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Edit only the config blocks and ACTIVE_CONFIG below.

# ═══════════════════════════════════════════════════════════════════════════
# v5 fixed defaults — 4 disjoint slices for clean train / attack-eval /
# threshold-fit / selection separation (96 h training + 3 × 12 h held-out
# slices on the 138-h normal_si.parquet).
#
#   warmup (skipped):  0       →  21,600   ( 6 h)
#   TRAIN:             21,600  → 367,200   (96 h training, 345,600 rows)
#   ATTACK_SLICE_V3:  367,200  → 410,400   (12 h synthetic-attack injection)
#   THRESHOLD_SLICE_V3: 410,400 → 453,600  (12 h quantile threshold fit)
#   SELECTION_SLICE_V3: 453,600 → 496,800  (12 h in-training test → test_rmse)
#
# The full-SWaT configs fuse the last two into one 24 h region (--test_rows 86_400,
# i.e. 410,400 → 496,800) so the detector quantile has twice the samples behind it;
# the training region is unaffected because training_slice_end still caps it.
#
# ``training_slice_end`` caps where train.py is allowed to draw training rows,
# so they never reach into ATTACK_SLICE_V3 / THRESHOLD_SLICE_V3.
# ═══════════════════════════════════════════════════════════════════════════
_V5_FIXED = {
    "device": "cuda",
    "epochs": 250,
    "min_epochs": 80,
    "window": 50,
    "batch_size": 512,
    "workers": 0,
    "temporal_stride": 10,
    "row_start": 21_600,
    "rows": 345_600,  # 96 h training
    # 12 h SELECTION_SLICE. The full-SWaT configs override this to 86_400 to fuse
    # the threshold and selection slices (see CONFIG_FULL_POISON_LIT).
    "test_rows": 43_200,
    "training_slice_end": 367_200,  # caps training region upper bound
    "keep_seconds": True,
    "save_artifacts": True,
    # Every v5 config fits the physics bias on the clean held-out slice.
    "bias_source": "test",
}

# Base grid shared by every config: 4 architectures × {Data-Only, PINN}
# × 10 seeds × 4 learning rates. The headline ablation is w_phys ∈ {0, 1};
# the broader w_phys sweep lives in CONFIG_V5_POISONING_LIT_WPHYS_ABLATION
# (used only by the per-w_phys plots).
_V5_GRID_BASE = {
    "w_phys": [0.0, 1.0],
    "model": ["mlp", "cnn1d", "lstm", "transformer"],
    "seed": [42, 123, 456, 7, 11, 99, 314, 2024, 31415, 65535],
    "lr": [2e-4, 5e-4, 7e-4, 1e-3],
}
# ═══════════════════════════════════════════════════════════════════════════
# v5 retraining set — six core experiments + a LIT ramp-rate sweep, all on
# the same train/attack/threshold/selection slice layout (_V5_FIXED) so plots
# regenerate end-to-end on the cleaner data split.
#
# Ramp rates are pinned to the stealth working point (between the p99 and
# p99.99 percentiles of |Δ/s| on the normal slice):
#   LIT 0.001943 m/s
#   FIT 1.085e-4 m³/s²
# ═══════════════════════════════════════════════════════════════════════════

CONFIG_V5_SCARCITY = {
    "grid": {
        **_V5_GRID_BASE,
        # 2.5%, 10%, 20%, 50%, 100% of the training span (345_600 rows).
        # Each level uses the LAST `rows` rows of the training slice
        # (most recent data).
        "rows": [8_640, 34_560, 69_120, 172_800, 345_600],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-scarcity-s10w50",
    },
}

CONFIG_V5_FAULTY_GAUSSIAN = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 5.0, 10.0, 20.0, 50.0],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-faulty-gauss-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "gaussian",
    },
}

CONFIG_V5_FAULTY_STUCK = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-faulty-stuck-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "stuck_at",
        "stuck_min_dur": 50,
        "stuck_max_dur": 300,
    },
}

CONFIG_V5_POISONING_LIT = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 0.001943,
    },
}

CONFIG_V5_POISONING_FIT = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-fit-s10w50",
        "corrupt_sensor": "FIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 1.085e-4,
    },
}

CONFIG_V5_CONSISTENT_LIT_SIN = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-consistent-lit-sin-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp_consistent_sin",
        "corrupt_flow_col": "FIT101",
        "corrupt_outflow_col": "FIT201",
        "ramp_rate": 0.001943,
    },
}

CONFIG_V5_CONSISTENT_FIT = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-consistent-fit-s10w50",
        "corrupt_sensor": "FIT101",
        "corrupt_type": "ramp_flow_consistent",
        "corrupt_flow_col": "FIT201",
        "ramp_rate": 1.085e-4,
    },
}

# ---------------------------------------------------------------------------
# v5 LIT poisoning ramp-rate sweep (poison fraction fixed at 0.3).
# Used to plot ramp-rate vs PINN benefit.
# ---------------------------------------------------------------------------
CONFIG_V5_POISONING_LIT_RATE_SWEEP = {
    "grid": {
        **_V5_GRID_BASE,
        # 0.0 → 3 mm/s in 0.5 mm/s steps
        "ramp_rate": [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-rate-sweep-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "corrupt_param": 0.3,  # fixed poison fraction
    },
}

# ---------------------------------------------------------------------------
# v5 LIT poisoning ramp-rate sweep — hard-constraint baseline (physics filter).
# Same grid as RATE_SWEEP but data-only (w_phys=0) and with the q=0.99
# residual filter enabled. Provides the "Poisoned Data-Only + filter" line
# of the filter-vs-PINN comparison plot.
# ---------------------------------------------------------------------------
CONFIG_V5_POISONING_LIT_RATE_SWEEP_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],  # Data-only only — physics enters via the filter
        "ramp_rate": [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-rate-sweep-filter-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "corrupt_param": 0.3,
        "physics_filter": 0.99,
        # THRESHOLD_SLICE_V3 from evaluation.py — clean held-out, never
        # touched by training corruption. Filter τ is fit on |LIT101 mass-
        # balance residual| over this slice.
        "threshold_slice_start": 410_400,
        "threshold_slice_end": 453_600,
    },
}

# ---------------------------------------------------------------------------
# v5 LIT-poisoning FILTER baseline (hard constraint) across poison fractions.
# Mirrors CONFIG_V5_POISONING_LIT but data-only (w_phys=0) with the q=0.99
# LIT101 residual filter enabled. Provides the "Filter" line of the
# filter-vs-PINN-on-poison-fraction plots (PLC1 analogue of the full-SWaT
# CONFIG_FULL_POISON_LIT_FILTER).
# ---------------------------------------------------------------------------
CONFIG_V5_POISONING_LIT_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],  # Data-only only — physics enters via the filter
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-filter-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 0.001943,
        "physics_filter": 0.99,
        # THRESHOLD_SLICE_V3 — clean held-out slice the filter fits τ on.
        "threshold_slice_start": 410_400,
        "threshold_slice_end": 453_600,
    },
}

# ---------------------------------------------------------------------------
# (A) Invariant-rule window filter on the inconsistent LIT poison. Data-only
# (w_phys=0); training windows that violate the hand-written PLC1 invariants are
# dropped (NOT the mass-balance filter). Shows static invariants under-regularise.
# ---------------------------------------------------------------------------
CONFIG_V5_POISONING_LIT_INVARIANT = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-invariant-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 0.001943,
        "invariant_filter": True,
    },
}

# ---------------------------------------------------------------------------
# (B) Physics-filter quantile sensitivity: data-only + mass-balance filter with
# the threshold quantile q swept densely in the high tail [0.98, 1.0], where the
# only interesting behaviour lives (recall changes only past ~q=0.999). Spaced
# log-uniform in the tail mass (1-q): 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001,
# 0. Includes 0.99 (the quantile used elsewhere in the work). Trained at clean
# (0.0) and 30% poison so the poisoned-robustness-vs-q curve can be drawn.
# exp_var = q. NOTE: ramp_rate=0.0005 (stealthy 0.5 mm/s); the matching test
# attack at eval time is ramp_attack_lit_v3_0p5mm (NOT the 1.943 mm/s default),
# so the q-sweep is graded in the same stealthy regime it was trained on.
# ---------------------------------------------------------------------------
CONFIG_V5_POISONING_LIT_FILTER_QSWEEP = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 0.3],
        "physics_filter": [0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1.00],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-filter-qsweep-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 0.0005,  # 0.5 mm/s (stealthier ramp than the 1.943 mm/s default)
        "threshold_slice_start": 410_400,
        "threshold_slice_end": 453_600,
    },
}

# ---------------------------------------------------------------------------
# Filter baselines for the remaining PLC1 poisoning families (FIT poisoning,
# consistent LIT-sin, consistent FIT). Each mirrors its non-filter config but
# data-only (w_phys=0) with the q=0.99 LIT101 mass-balance residual filter, so
# every kept dose-response plot can show a Filter line alongside PINN/Data-only.
# (The wphys ablation deliberately has no filter variant.) The PLC1 filter
# always uses the LIT101 equation regardless of which sensor is poisoned.
# ---------------------------------------------------------------------------
_FILTER_DEFAULTS = {
    "physics_filter": 0.99,
    "threshold_slice_start": 410_400,  # THRESHOLD_SLICE_V3 (clean held-out)
    "threshold_slice_end": 453_600,
}

CONFIG_V5_POISONING_FIT_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        **_FILTER_DEFAULTS,
        "grid_name": "v5-poison-fit-filter-s10w50",
        "corrupt_sensor": "FIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 1.085e-4,
    },
}

CONFIG_V5_CONSISTENT_LIT_SIN_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        **_FILTER_DEFAULTS,
        "grid_name": "v5-consistent-lit-sin-filter-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp_consistent_sin",
        "corrupt_flow_col": "FIT101",
        "corrupt_outflow_col": "FIT201",
        "ramp_rate": 0.001943,
    },
}

CONFIG_V5_CONSISTENT_FIT_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        **_FILTER_DEFAULTS,
        "grid_name": "v5-consistent-fit-filter-s10w50",
        "corrupt_sensor": "FIT101",
        "corrupt_type": "ramp_flow_consistent",
        "corrupt_flow_col": "FIT201",
        "ramp_rate": 1.085e-4,
    },
}

# ---------------------------------------------------------------------------
# v5 LIT-poisoning w_phys ablation — broader sweep used only by the
# per-w_phys plots (fig3_inconsistent_lit_recall_per_wphys + its FPR sibling).
# Inherits clean-test bias from _V5_FIXED. Sweeps w_phys over several values.
# ---------------------------------------------------------------------------
CONFIG_V5_POISONING_LIT_WPHYS_ABLATION = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0, 0.5, 1.0, 5.0, 10.0, 100.0],
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-poison-lit-wphys-ablation-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "ramp",
        "ramp_rate": 0.001943,
    },
}

# ---------------------------------------------------------------------------
# Full-SWaT multi-tank LIT poisoning (first-class v5 config).
# Multistage ramp poisoning attacks LIT101/LIT301/LIT401 with equal prob; the
# PINN has objectives on all three tanks (full_swat + global_obj).
# ---------------------------------------------------------------------------
CONFIG_FULL_POISON_LIT = {
    "grid": {
        **_V5_GRID_BASE,
        "corrupt_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        # Own grid_name for the delta / 49-channel population: pick_best_v2
        # groups on (model, corrupt_param, pinn_flag, seed) only, so sharing a
        # directory with legacy absolute / 51-channel runs would let the two
        # compete on a test_rmse that is not comparable across them
        # (_assert_homogeneous in select_best_models refuses such a mix).
        "grid_name": "v5-full-poison-lit-delta-s10w50",
        "full_swat": True,
        "global_obj": True,
        # Residual (delta) head: the backbone predicts the change from the last
        # observed step. Every tank objective's level term is
        # (pred_level - obs_last_level)/dt, so the backbone output becomes the
        # constrained quantity instead of being recovered by cancelling two large
        # near-equal numbers.
        "wrapper": "delta",
        # Train/attack distributions differ sharply on these two: AIT201 shifts
        # 9.6 train-sd with only 25.1% of attack rows inside the training range;
        # AIT203 keeps 66.3% and its spread collapses (49.9 -> 16.6). AIT202
        # overlaps 100% and stays.
        "drop_cols": "AIT201,AIT203",
        # Fused threshold+selection region [410_400, 496_800), 86_400 rows:
        # ~8,635 score samples behind the detector quantile instead of the
        # ~4,315 a 43_200-row threshold slice gives (q=0.999 would rest on ~4).
        "test_rows": 86_400,
        "corrupt_type": "ramp_multistage",
        "corrupt_pool": "lit",  # {LIT101, LIT301, LIT401}
        "ramp_rate": 0.001943,
        # Skip diagnostic plots (parity/loss grid) — not needed for the full-SWaT
        # story, and the full-SWaT parity plot has an unrelated dim bug. The
        # pipeline-required artifacts (run_config.json, test_rmse CSV) are always
        # written regardless of this flag.
        "save_artifacts": False,
    },
}

# ---------------------------------------------------------------------------
# Full-SWaT multi-tank LIT poisoning — hard-constraint baseline (physics filter).
# Same grid as CONFIG_FULL_POISON_LIT but data-only (w_phys=0) with the q=0.99
# multi-equation residual filter (LIT101/LIT301/LIT401 + global balance).
# ---------------------------------------------------------------------------
CONFIG_FULL_POISON_LIT_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],  # Data-only only — physics enters via the filter
        "corrupt_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        # Own grid_name for the delta / 49-channel population: pick_best_v2
        # groups on (model, corrupt_param, pinn_flag, seed) only, so sharing a
        # directory with legacy absolute / 51-channel runs would let the two
        # compete on a test_rmse that is not comparable across them
        # (_assert_homogeneous in select_best_models refuses such a mix).
        "grid_name": "v5-full-poison-lit-delta-filter-s10w50",
        "full_swat": True,
        "global_obj": True,
        # Residual (delta) head: the backbone predicts the change from the last
        # observed step. Every tank objective's level term is
        # (pred_level - obs_last_level)/dt, so the backbone output becomes the
        # constrained quantity instead of being recovered by cancelling two large
        # near-equal numbers.
        "wrapper": "delta",
        # Train/attack distributions differ sharply on these two: AIT201 shifts
        # 9.6 train-sd with only 25.1% of attack rows inside the training range;
        # AIT203 keeps 66.3% and its spread collapses (49.9 -> 16.6). AIT202
        # overlaps 100% and stays.
        "drop_cols": "AIT201,AIT203",
        # Fused threshold+selection region [410_400, 496_800), 86_400 rows:
        # ~8,635 score samples behind the detector quantile instead of the
        # ~4,315 a 43_200-row threshold slice gives (q=0.999 would rest on ~4).
        "test_rows": 86_400,
        "corrupt_type": "ramp_multistage",
        "corrupt_pool": "lit",  # {LIT101, LIT301, LIT401}
        "ramp_rate": 0.001943,
        "physics_filter": 0.99,
        # THRESHOLD_SLICE_V3 from evaluation.py — clean held-out slice the
        # multi-equation filter fits τ on.
        "threshold_slice_start": 410_400,
        "threshold_slice_end": 453_600,
        # Skip diagnostic plots (see CONFIG_FULL_POISON_LIT). Required artifacts
        # (run_config.json, test_rmse CSV) are still written.
        "save_artifacts": False,
    },
}

# ---------------------------------------------------------------------------
# Filter baselines for the scarcity / faulty-Gaussian / faulty-stuck families:
# data-only (w_phys=0) + q=0.99 LIT101 mass-balance residual filter, so those
# companion plots can show a Filter line alongside PINN / Data-only.
# ---------------------------------------------------------------------------
CONFIG_V5_SCARCITY_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "rows": [8_640, 34_560, 69_120, 172_800, 345_600],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        **_FILTER_DEFAULTS,
        "grid_name": "v5-scarcity-filter-s10w50",
    },
}

CONFIG_V5_FAULTY_GAUSSIAN_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 5.0, 10.0, 20.0, 50.0],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        **_FILTER_DEFAULTS,
        "grid_name": "v5-faulty-gauss-filter-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "gaussian",
    },
}

CONFIG_V5_FAULTY_STUCK_FILTER = {
    "grid": {
        **_V5_GRID_BASE,
        "w_phys": [0.0],
        "corrupt_param": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        **_FILTER_DEFAULTS,
        "grid_name": "v5-faulty-stuck-filter-s10w50",
        "corrupt_sensor": "LIT101",
        "corrupt_type": "stuck_at",
        "stuck_min_dur": 50,
        "stuck_max_dur": 300,
    },
}

# ---------------------------------------------------------------------------
# Training-cost benchmark: data-only vs PINN compute, on CLEAN PLC1 data.
# Just the base ablation (w_phys ∈ {0,1}) × 4 archs × seeds × lr, no corruption.
# Early stopping is left ON (inherited from _V5_FIXED) so total wall-time reflects
# the real end-to-end cost incl. convergence; the LR sweep is averaged over so the
# cost claim isn't tied to one learning rate. run_config.json records wall-time +
# epochs, so training_time_report.py can decompose total = per-epoch × epochs.
# Run SERIALLY (xargs -P 1) so wall-times aren't inflated by GPU contention.
# ---------------------------------------------------------------------------
CONFIG_V5_TIMING = {
    "grid": {
        **_V5_GRID_BASE,
        "seed": [42, 123, 456],
    },
    "fixed_defaults": {
        **_V5_FIXED,
        "grid_name": "v5-timing-s10w50",
        "save_artifacts": False,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# Active config registry
# ═══════════════════════════════════════════════════════════════════════════
CONFIGS = {
    "v5_scarcity": CONFIG_V5_SCARCITY,
    "v5_faulty_gaussian": CONFIG_V5_FAULTY_GAUSSIAN,
    "v5_faulty_stuck": CONFIG_V5_FAULTY_STUCK,
    "v5_poisoning_lit": CONFIG_V5_POISONING_LIT,
    "v5_poisoning_fit": CONFIG_V5_POISONING_FIT,
    "v5_consistent_lit_sin": CONFIG_V5_CONSISTENT_LIT_SIN,
    "v5_consistent_fit": CONFIG_V5_CONSISTENT_FIT,
    # LIT poisoning ramp-rate sweep (poison fraction fixed at 0.3)
    "v5_poisoning_lit_rate_sweep": CONFIG_V5_POISONING_LIT_RATE_SWEEP,
    # Same sweep but data-only + physics filter (hard-constraint baseline)
    "v5_poisoning_lit_rate_sweep_filter": CONFIG_V5_POISONING_LIT_RATE_SWEEP_FILTER,
    # PLC1 LIT poisoning filter baseline across poison fractions (Filter line of
    # the filter-vs-PINN-on-poison-fraction plot)
    "v5_poisoning_lit_filter": CONFIG_V5_POISONING_LIT_FILTER,
    "v5_poisoning_lit_invariant": CONFIG_V5_POISONING_LIT_INVARIANT,
    "v5_poisoning_lit_filter_qsweep": CONFIG_V5_POISONING_LIT_FILTER_QSWEEP,
    "v5_poisoning_fit_filter": CONFIG_V5_POISONING_FIT_FILTER,
    "v5_consistent_lit_sin_filter": CONFIG_V5_CONSISTENT_LIT_SIN_FILTER,
    "v5_consistent_fit_filter": CONFIG_V5_CONSISTENT_FIT_FILTER,
    # Full-SWaT multi-tank LIT poisoning (PINN/data-only) + filter baseline
    "v5_full_poison_lit": CONFIG_FULL_POISON_LIT,
    "v5_full_poison_lit_filter": CONFIG_FULL_POISON_LIT_FILTER,
    "v5_scarcity_filter": CONFIG_V5_SCARCITY_FILTER,
    "v5_faulty_gaussian_filter": CONFIG_V5_FAULTY_GAUSSIAN_FILTER,
    "v5_faulty_stuck_filter": CONFIG_V5_FAULTY_STUCK_FILTER,
    # Per-w_phys ablation for the inconsistent-LIT plot (5 w_phys values)
    "v5_poisoning_lit_wphys_ablation": CONFIG_V5_POISONING_LIT_WPHYS_ABLATION,
    # Training-cost benchmark (data-only vs PINN wall-time + epochs); run -P 1.
    "v5_timing": CONFIG_V5_TIMING,
}

ACTIVE_CONFIG = "v5_scarcity"


def infer_workers(workers_arg: int) -> int:
    if workers_arg == 0:
        return 0
    if workers_arg > 0:
        return workers_arg

    n = os.cpu_count()
    if n is None or n <= 0:
        raise RuntimeError("Could not infer CPU count; pass --workers explicitly.")
    return int(n)


def grid_configs(grid: dict) -> list[dict]:
    """Expand a {param: [values]} grid into the list of all value combinations."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]

    out = []
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        out.append(cfg)
    return out


def run_tag(merged: dict, fixed: dict, cfg: dict) -> str:
    """Build a unique run name encoding the hyperparameters of one job."""
    model = cfg.get("model", fixed.get("model", "?"))
    lr = cfg.get("lr", fixed.get("lr", "?"))
    w_phys = cfg.get("w_phys", fixed.get("w_phys", 0.0))

    size = "full_swat" if bool(merged.get("full_swat", False)) else "plc1"
    tag = (
        f"w{merged['window']}_s{merged['temporal_stride']}_b{merged['batch_size']}"
        f"_lr{lr}_{model}_{size}_wp{w_phys}"
    )

    if "rows" in cfg:
        tag += f"_r{int(cfg['rows'])}"
    if "seed" in cfg:
        tag += f"_sd{int(cfg['seed'])}"
    if "ramp_rate" in cfg:
        tag += f"_rr{cfg['ramp_rate']}"
    if merged.get("corrupt_sensor"):
        tag += f"_c{merged['corrupt_type']}_{merged['corrupt_param']}"
    elif str(merged.get("corrupt_type")) == "ramp_multistage":
        tag += f"_c{merged['corrupt_type']}-{merged['corrupt_pool']}_{merged['corrupt_param']}"

    if str(merged.get("bias_source", "train")) != "train":
        tag += f"_bs{merged['bias_source']}"

    # physics_filter only when it is a swept grid value (B's q-sensitivity run),
    # so each quantile gets a distinct run dir. Fixed-default filters are unchanged.
    if "physics_filter" in cfg:
        tag += f"_pf{cfg['physics_filter']}"

    # Output head, dropped columns and test-region size change what the checkpoint
    # IS, but not any key select_best_models groups on -- and main() skips any
    # outdir that already holds a *.pt. Without these suffixes a delta/49-channel
    # run would silently reuse an absolute/51-channel checkpoint. Each is emitted
    # only when it departs from the v5 default, so PLC1 tags are unchanged.
    if str(merged.get("wrapper", "absolute")) != "absolute":
        tag += f"_{merged['wrapper']}"
    if str(merged.get("drop_cols", "")).strip():
        tag += "_drop" + str(merged["drop_cols"]).replace(",", "-")
    if int(merged.get("test_rows", 43_200)) != 43_200:
        tag += f"_tr{int(merged['test_rows'])}"

    return tag


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True, help="Parquet path")
    ap.add_argument(
        "--active_config",
        choices=list(CONFIGS.keys()),
        default=ACTIVE_CONFIG,
        help="Which config block to run when --all_blocks is not set.",
    )
    ap.add_argument(
        "--all_blocks",
        action="store_true",
        help="Run all config blocks in CONFIGS.",
    )
    ap.add_argument(
        "--task_id",
        type=int,
        default=-1,
        help="Local config index within a single config block. -1 = run all.",
    )
    ap.add_argument(
        "--global_task_id",
        type=int,
        default=-1,
        help="Global task index across ALL config blocks (for --all_blocks). "
        "Maps to (config_block, local_index) automatically. "
        "-1 = not used (fall back to --task_id). "
        "Use with: seq 0 $((N-1)) | parallel -j 10 ... --global_task_id {}",
    )

    ap.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to round-robin across in --global_task_id dispatch "
        "(each job pinned to global_task_id mod num_gpus). Default 1 = GPU 0 only.",
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which single GPU to pin to when NOT using --global_task_id "
        "(serial / single-process mode). Default 0.",
    )
    ap.add_argument(
        "--outbase",
        type=str,
        default=None,
        required=True,
        help="Shared output root directory. Write into this directory. "
        "Required for parallel dispatch with --global_task_id.",
    )

    ap.add_argument("--dry_run", action="store_true", help="Print what would run")
    ap.add_argument(
        "--count",
        action="store_true",
        help="Print total number of runs and per-config counts, then exit.",
    )

    return ap.parse_args()


def _build_flat_jobs(
    run_configs: List[Tuple[str, dict]],
) -> List[Tuple[str, dict, dict, int]]:
    """Build a flat list of (config_name, fixed_defaults, grid_cfg, local_idx)."""
    jobs = []
    for config_name, config in run_configs:
        cfgs = grid_configs(config["grid"])
        fixed = config["fixed_defaults"]
        for local_idx, cfg in enumerate(cfgs):
            jobs.append((config_name, fixed, cfg, local_idx))
    return jobs


def main() -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    args = parse_args()

    # GPU pinning. In parallel dispatch (--global_task_id) each invocation runs
    # one job pinned to a single GPU, round-robined across --num_gpus so
    # concurrent workers spread over all cards. In single-process mode (no
    # --global_task_id) everything runs on --gpu.
    gpu = args.global_task_id % args.num_gpus if args.global_task_id >= 0 else args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["NVIDIA_VISIBLE_DEVICES"] = str(gpu)

    data_path = Path(args.data)
    root_outbase = Path(args.outbase)

    if not data_path.exists():
        raise FileNotFoundError(f"--data not found: {data_path}")

    if args.all_blocks:
        run_configs = list(CONFIGS.items())
    else:
        config = CONFIGS[args.active_config]
        run_configs = [(args.active_config, config)]

    # Build flat job list across all selected config blocks
    all_jobs = _build_flat_jobs(run_configs)

    # --count: just print totals and exit
    if args.count:
        print(f"Total runs: {len(all_jobs)}")
        from collections import Counter

        counts = Counter(name for name, _, _, _ in all_jobs)
        for name, cnt in counts.items():
            print(f"  {name}: {cnt}")
        return

    # Determine which jobs to run
    if args.global_task_id >= 0:
        # Global task ID mode: single job from the flat list
        if args.global_task_id >= len(all_jobs):
            raise ValueError(
                f"--global_task_id {args.global_task_id} out of range "
                f"[0, {len(all_jobs) - 1}] ({len(all_jobs)} total jobs)"
            )
        selected_jobs = [all_jobs[args.global_task_id]]
    elif args.task_id >= 0:
        # Per-config task_id: index within each config block
        selected_jobs = []
        for config_name, config in run_configs:
            cfgs = grid_configs(config["grid"])
            fixed = config["fixed_defaults"]
            if args.task_id >= len(cfgs):
                raise ValueError(
                    f"--task_id {args.task_id} out of range for '{config_name}' "
                    f"[0, {len(cfgs) - 1}]"
                )
            selected_jobs.append((config_name, fixed, cfgs[args.task_id], args.task_id))
    else:
        # Run everything
        selected_jobs = all_jobs

    root_outbase.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["MPLBACKEND"] = "Agg"

    for active_config, fixed, cfg, idx in selected_jobs:
        # Grid values override fixed defaults so params can live in either.
        merged = {**fixed, **cfg}

        if str(merged["device"]).lower() == "mps":
            child_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

        outbase = root_outbase / fixed["grid_name"]
        outbase.mkdir(parents=True, exist_ok=True)

        workers = infer_workers(int(merged["workers"]))

        tag = run_tag(merged, fixed, cfg)

        outdir = outbase / tag
        if outdir.exists() and any(outdir.glob("*.pt")):
            print(f"[skip] {active_config} {idx} -> {outdir} (checkpoint exists)")
            continue
        outdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "src.scripts.train",
            "--data",
            str(data_path),
            "--outdir",
            str(outdir),
            "--window",
            str(int(merged["window"])),
            "--batch_size",
            str(int(merged["batch_size"])),
            "--workers",
            str(int(workers)),
            "--epochs",
            str(int(merged["epochs"])),
            "--min_epochs",
            str(int(merged["min_epochs"])),
            "--device",
            str(merged["device"]),
            "--seed",
            str(int(merged["seed"])),
            "--w_phys",
            str(merged["w_phys"]),
            "--lr",
            str(merged["lr"]),
            "--model",
            str(merged["model"]),
            "--row_start",
            str(int(merged["row_start"])),
            "--rows",
            str(int(merged["rows"])),
            "--test_rows",
            str(int(merged["test_rows"])),
            "--temporal_stride",
            str(int(merged["temporal_stride"])),
            "--name",
            tag,
        ]

        # v5 caps the training region to keep held-out slices clean.
        if int(merged.get("training_slice_end", 0)) > 0:
            cmd += ["--training_slice_end", str(int(merged["training_slice_end"]))]

        if bool(merged.get("keep_seconds", False)):
            cmd.append("--keep_seconds")
        if bool(merged.get("save_artifacts", False)):
            cmd.append("--save_artifacts")

        if bool(merged.get("full_swat", False)):
            cmd.append("--full_swat")
        if bool(merged.get("global_obj", False)):
            cmd.append("--global_obj")

        if str(merged.get("bias_source", "train")) != "train":
            cmd += ["--bias_source", str(merged["bias_source"])]

        # Must stay in sync with run_tag above: a flag added to one and not the
        # other yields runs NAMED delta but TRAINED absolute.
        if str(merged.get("wrapper", "absolute")) != "absolute":
            cmd += ["--wrapper", str(merged["wrapper"])]
        if str(merged.get("drop_cols", "")).strip():
            cmd += ["--drop_cols", str(merged["drop_cols"])]

        _is_multistage = str(merged.get("corrupt_type")) == "ramp_multistage"
        if merged.get("corrupt_sensor") or _is_multistage:
            cmd += ["--corrupt_type", str(merged["corrupt_type"])]
            cmd += ["--corrupt_param", str(merged["corrupt_param"])]
            if merged.get("corrupt_sensor"):
                cmd += ["--corrupt_sensor", str(merged["corrupt_sensor"])]
            if _is_multistage:
                cmd += ["--corrupt_pool", str(merged["corrupt_pool"])]
                cmd += ["--ramp_rate", str(float(merged.get("ramp_rate", 0.003)))]
            if str(merged.get("corrupt_type")) in (
                "ramp",
                "ramp_consistent_sin",
                "ramp_flow_consistent",
            ):
                cmd += ["--ramp_rate", str(float(merged.get("ramp_rate", 0.003)))]
            if str(merged.get("corrupt_type")) in (
                "ramp_consistent_sin",
                "ramp_flow_consistent",
            ):
                cmd += ["--corrupt_flow_col", str(merged["corrupt_flow_col"])]
            if str(merged.get("corrupt_type")) == "ramp_consistent_sin":
                cmd += ["--corrupt_outflow_col", str(merged["corrupt_outflow_col"])]
            if str(merged.get("corrupt_type")) == "stuck_at":
                if "stuck_min_dur" in merged:
                    cmd += ["--stuck_min_dur", str(int(merged["stuck_min_dur"]))]
                if "stuck_max_dur" in merged:
                    cmd += ["--stuck_max_dur", str(int(merged["stuck_max_dur"]))]

        # Physics-filter baseline (hard-constraint comparison to PINN). The τ is
        # fitted on the threshold slice, so it must be forwarded with the filter.
        if float(merged.get("physics_filter", 0.0)) > 0.0:
            cmd += ["--physics_filter", str(float(merged["physics_filter"]))]
            cmd += [
                "--threshold_slice_start",
                str(int(merged["threshold_slice_start"])),
            ]
            cmd += ["--threshold_slice_end", str(int(merged["threshold_slice_end"]))]
        # Invariant-rule filter (no threshold slice — uses fixed bounds).
        if bool(merged.get("invariant_filter", False)):
            cmd.append("--invariant_filter")

        (outdir / "cmd.txt").write_text(shlex.join(cmd) + "\n")
        (outdir / "config.json").write_text(
            json.dumps(
                {
                    "active_config": active_config,
                    "all_blocks": bool(args.all_blocks),
                    "index": idx,
                    "outdir": str(outdir),
                    "cfg": cfg,
                    "fixed_defaults": fixed,
                    "merged": merged,
                },
                indent=2,
            )
            + "\n"
        )

        if args.dry_run:
            print(outdir)
            print(shlex.join(cmd))
            continue

        try:
            with (
                (outdir / "stdout.log").open("w") as fo,
                (outdir / "stderr.log").open("w") as fe,
            ):
                subprocess.run(cmd, env=child_env, stdout=fo, stderr=fe, check=True)
        except subprocess.CalledProcessError:
            err_path = outdir / "stderr.log"
            print(
                f"\n[train failed] config={active_config} idx={idx} see: {err_path}\n"
            )
            if err_path.exists():
                lines = err_path.read_text().splitlines()
                print("\n".join(lines[-200:]))
            raise

        print(f"[ok] {active_config} {idx} -> {outdir}")


if __name__ == "__main__":
    main()
