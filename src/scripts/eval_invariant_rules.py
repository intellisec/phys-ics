"""Diagnostic: do hand-written PLC1 invariant rules safeguard against the
inconsistent LIT101 ramp poisoning?

Loads the training slice, applies the *inconsistent* LIT101 ramp poison (the
exact ``ramp_poison`` train.py uses for v5_poison_lit — only LIT101 is ramped,
MV101/FIT101 stay clean), then checks each **timestep** (not windows) against a
set of invariant rules. Reports what fraction of the poisoned timesteps the
rules flag (= poison removed if you filtered on these rules) and the
false-positive rate on the clean timesteps.

Invariant rules (a timestep is FLAGGED if it violates any):
    R1: LIT101 < 490        ->  MV101 open  AND FIT101 > 0.1
    R2: LIT101 > 810        ->  MV101 closed AND FIT101 < 1
    R3: MV101 open          ->  FIT101 > 0.1
    R4: LIT101 < 490        ->  FIT101 > 0
with MV101 open == 2, closed == 1.

Units: the rule thresholds are in RAW (non-SI) units — LIT101 in mm, FIT101 in
m^3/h — while normal_si.parquet stores SI (LIT in m, FIT in m^3/s). The script
converts LIT*1000 and FIT*3600 before applying the rules, and prints the clean
value ranges so you can confirm the scale.

Usage::

    python -m src.scripts.eval_invariant_rules --poison-fracs 0.3 0.5 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path("src").resolve()))

import numpy as np
import pandas as pd
import torch

from data.transforms import ramp_poison

COLS = ["LIT101", "FIT101", "MV101"]
TRAIN_START, TRAIN_END = 21_600, 367_200  # row_start, row_start + rows
RAMP_RATE = 0.001943  # SI m/timestep (v5_poison_lit)
# The 10 training seeds (_V5_GRID_BASE). train.py seeds the corruption with
# args.seed, so these are the actual per-run poisoning episodes; the selected
# models span all of them. ramp_poison(seed=s) here reproduces run-s exactly.
TRAIN_SEEDS = [42, 123, 456, 7, 11, 99, 314, 2024, 31415, 65535]
LIT_TO_MM = 1000.0
FIT_TO_M3H = 3600.0
MV_OPEN, MV_CLOSED = 2, 1


def _rule_violations(lit_mm: np.ndarray, fit_m3h: np.ndarray, mv: np.ndarray) -> dict:
    """Per-timestep boolean violation of each rule (in RAW units)."""
    is_open = mv == MV_OPEN
    is_closed = mv == MV_CLOSED
    r1 = (lit_mm < 490) & ~(is_open & (fit_m3h > 0.1))
    r2 = (lit_mm > 810) & ~(is_closed & (fit_m3h < 1))
    r3 = is_open & ~(fit_m3h > 0.1)
    r4 = (lit_mm < 490) & ~(fit_m3h > 0)
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4}


def _downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    """Temporal downsample [LIT101, FIT101, MV101] like the training pipeline:
    block-mean for the continuous LIT/FIT columns, block-mode for MV101."""
    if stride <= 1:
        return arr
    T = (arr.shape[0] // stride) * stride
    x = arr[:T].reshape(T // stride, stride, arr.shape[1])
    lit = x[:, :, 0].mean(axis=1)
    fit = x[:, :, 1].mean(axis=1)
    mv_int = x[:, :, 2].astype(int)  # SWaT MV states 0/1/2
    counts = np.stack([(mv_int == v).sum(1) for v in (0, 1, 2)], axis=1)
    mv = counts.argmax(1).astype(float)  # mode per block
    return np.stack([lit, fit, mv], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normal-parquet", default="data/processed/normal_si.parquet")
    ap.add_argument("--poison-fracs", type=float, nargs="+", default=[0.3])
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=TRAIN_SEEDS,
        help="Corruption seeds to evaluate (default: the 10 training seeds). "
        "Each reproduces the exact poisoning episodes of the run with that seed.",
    )
    ap.add_argument("--ramp-rate", type=float, default=RAMP_RATE)
    ap.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Temporal downsample stride applied AFTER poisoning, as in training "
        "(LIT/FIT block-mean, MV mode). 1 = raw 1 Hz, 10 = the trained resolution.",
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.normal_parquet)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"normal parquet missing required columns {missing}")
    arr = torch.tensor(df[COLS].to_numpy()[TRAIN_START:TRAIN_END], dtype=torch.float64)
    clean = arr.clone()
    stride = args.stride

    # Downsample the clean stream once (LIT/FIT mean, MV mode). The LIT poison
    # leaves FIT/MV untouched, so the downsampled FIT/MV are reused per seed.
    clean_ds = _downsample(clean.numpy(), stride)
    clean_lit_ds = clean_ds[:, 0]

    # ---- scale sanity on CLEAN data (raw units) ----
    clit = clean_ds[:, 0] * LIT_TO_MM
    cfit = clean_ds[:, 1] * FIT_TO_M3H
    cmv = clean_ds[:, 2]
    res = "1 Hz raw" if stride <= 1 else f"downsampled (stride={stride})"
    print(
        f"training slice: {clean_ds.shape[0]} timesteps  ({res}, rows {TRAIN_START}:{TRAIN_END})"
    )
    print("--- clean ranges (raw units, confirm thresholds sit in here) ---")
    print(
        f"  LIT101 [mm]   min/mean/max = {clit.min():.1f} / {clit.mean():.1f} / {clit.max():.1f}"
    )
    print(
        f"  FIT101 [m^3/h] min/mean/max = {cfit.min():.3f} / {cfit.mean():.3f} / {cfit.max():.3f}"
    )
    print(
        f"  MV101 unique values        = {np.unique(cmv)}  (open={MV_OPEN}, closed={MV_CLOSED})"
    )

    # baseline: rule false positives on the fully-clean slice
    clean_v = _rule_violations(clit, cfit, cmv)
    clean_flag = np.logical_or.reduce(list(clean_v.values()))
    print(f"\nClean-data false positives (any rule): {clean_flag.mean():.2%}")
    for k, v in clean_v.items():
        print(f"    {k}: {v.mean():.2%}")

    fit_clean = cfit  # MV/FIT unchanged by the LIT poison (downsampled clean)
    mv_clean = cmv

    # ---- per poison fraction, averaged over the (training) seeds ----
    for frac in args.poison_fracs:
        per_seed = []  # (caught, clean_fp, per_rule_dict, n_pois)
        for s in args.seeds:
            pois = ramp_poison(
                arr.clone(),
                col_idx=0,
                poison_frac=frac,
                seed=s,
                ramp_rate=args.ramp_rate,
            )
            # Downsample the poisoned stream the same way training does, then
            # compare LIT to the clean downsampled LIT to mark poisoned steps.
            pois_lit_ds = _downsample(pois.numpy(), stride)[:, 0]
            poison_mask = np.abs(pois_lit_ds - clean_lit_ds) > 1e-9
            if poison_mask.sum() == 0:
                continue
            lit = pois_lit_ds * LIT_TO_MM
            viol = _rule_violations(lit, fit_clean, mv_clean)
            flagged = np.logical_or.reduce(list(viol.values()))
            per_seed.append(
                (
                    flagged[poison_mask].mean(),
                    flagged[~poison_mask].mean(),
                    {k: v[poison_mask].mean() for k, v in viol.items()},
                    int(poison_mask.sum()),
                )
            )

        print(
            f"\n=== poison_frac = {frac}  (mean ± std over {len(per_seed)} seeds) ==="
        )
        if not per_seed:
            print("  no poisoned timesteps")
            continue
        caught = np.array([p[0] for p in per_seed])
        fp = np.array([p[1] for p in per_seed])
        npois = np.array([p[3] for p in per_seed])
        print(
            f"  poisoned timesteps:        {npois.mean():.0f}  ({npois.mean() / clean_ds.shape[0]:.2%} of slice)"
        )
        print(
            f"  FLAGGED (removed if filtered): {caught.mean():.2%} ± {caught.std():.2%}"
        )
        print(f"  SLIPS THROUGH:                 {1 - caught.mean():.2%}")
        print(f"  clean-timestep false positives: {fp.mean():.2%} ± {fp.std():.2%}")
        print("  per-rule catch on poisoned timesteps (mean over seeds):")
        for k in ("R1", "R2", "R3", "R4"):
            vals = np.array([p[2][k] for p in per_seed])
            print(f"    {k}: {vals.mean():.2%}")


if __name__ == "__main__":
    main()
