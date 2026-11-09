"""CLI entry point for a single training run.

Loads data, optionally poisons it and/or applies the physics filter, builds and
trains the forecaster, and writes the checkpoint + metrics. Variants by flag:
Data-Only (--w_phys 0), PINN (--w_phys 1), Filtering (--physics_filter q).
"""

import argparse
import pathlib as P
import math
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend for SLURM nodes
from torch.optim import Adam
import pandas as pd
import time
from datetime import datetime, timezone


from training.trainer import Trainer
from data.registry_norm import build_registry_from_columns
from data.dataset import (
    make_windows,
    make_targets,
    make_dataloaders,
    temporal_downsample,
)
from data.data_utils import (
    load_parquet,
    select_targets,
    fit_normalizer,
    fit_all_biases,
    zero_biases,
    _balance_residual,
    fit_scales_on_target_baseline,
    TANK_AREA,
)
from models.wrappers import NextStepForecaster
from models.config import build_model
from physics.build_objectives import build_objectives
from plotting.plot_training import (
    save_parity_and_rmse,
    save_loss_grid,
    save_eval_csvs,
    save_run_json,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", required=True, help="Parquet file produced by convert_swat.py"
    )
    ap.add_argument(
        "--outdir", default="outputs", help="Where to write checkpoints/plots"
    )
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--w_phys", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=12,
        help="Stop if no dev MSE improvement for this many epochs.",
    )
    ap.add_argument(
        "--min_epochs", type=int, default=0, help="Do not stop before this many epochs."
    )
    ap.add_argument(
        "--early_stop_delta",
        type=float,
        default=0.0,
        help="Minimum improvement (absolute) to reset patience.",
    )
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--name", type=str, default="cnn_pinn")
    ap.add_argument(
        "--save_artifacts",
        action="store_true",
        help="Save model, parity plots, RMSE table, loss grid after training.",
    )
    ap.add_argument(
        "--row_start",
        type=int,
        default=0,
        help="Start row (0-based) before windowing (1 Hz ⇒ seconds).",
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=0,
        help="How many rows to use for train/val region (0=auto up to test start).",
    )
    ap.add_argument(
        "--test_rows",
        type=int,
        default=75_000,
        help="Fixed test size in raw rows taken from the END of the dataset.",
    )
    ap.add_argument(
        "--training_slice_end",
        type=int,
        default=0,
        help="Upper-bound row for training-window scattering. Defaults to 0 = the "
        "test region's start row. Set it so windows stay inside "
        "[row_start, training_slice_end) and do not leak into the held-out slices.",
    )
    ap.add_argument("--temporal_stride", type=int, default=1)
    ap.add_argument(
        "--downsample_agg",
        type=str,
        default="block_mean",
        choices=["block_mean", "pick_last", "pick_first"],
    )
    ap.add_argument(
        "--full_swat",
        action="store_true",
        help="Use every column in the parquet (all SWaT stages) instead of the "
        "PLC1-only 6. Combine with --drop_cols to exclude individual channels.",
    )
    ap.add_argument(
        "--invariant_filter",
        action="store_true",
        help="Drop training windows that violate hand-written PLC1 invariant rules "
        "(LIT/MV/FIT bounds, raw units). Distinct from --physics_filter; used to "
        "show static invariants under-regularise the inconsistent LIT poison.",
    )
    ap.add_argument(
        "--global_obj",
        action="store_true",
        help="Include global (sum-of-3-tanks) balance objective (only with --full_swat).",
    )
    ap.add_argument(
        "--bias_source",
        choices=["train", "test", "zero"],
        default="train",
        help="Source for physics bias fitting. "
        "'train': fit on (possibly poisoned) training blocks (default). "
        "'test': fit on clean held-out test slice. "
        "'zero': skip bias entirely (b=0). Used for bias-robustness experiments.",
    )
    ap.add_argument(
        "--physics_filter",
        type=float,
        default=0.0,
        help="If > 0, fit q-th quantile of |LIT101 mass-balance residual| on the "
        "clean threshold slice and DROP training windows whose sample residual "
        "exceeds it. 0 = disabled. Typical: 0.99. Hard-constraint baseline that "
        "isolates the PINN's soft constraint contribution.",
    )
    ap.add_argument(
        "--threshold_slice_start",
        type=int,
        default=0,
        help="Start row (in the raw 1 Hz parquet) of the clean held-out slice "
        "used to fit the physics_filter threshold. Required when --physics_filter > 0.",
    )
    ap.add_argument(
        "--threshold_slice_end",
        type=int,
        default=0,
        help="End row (exclusive) of the clean threshold slice. "
        "Required when --physics_filter > 0.",
    )
    ap.add_argument("--keep_seconds", action="store_true")
    ap.add_argument(
        "--model",
        choices=["cnn1d", "lstm", "mlp", "transformer", "copy-last"],
        default="cnn1d",
    )
    ap.add_argument(
        "--wrapper",
        choices=["absolute", "delta"],
        default="absolute",
        help="Output parameterisation of the forecaster. 'absolute': predict the "
        "next state directly. 'delta': predict the change from the last observed "
        "step and add it back (ResNet-style residual head). Recorded in the "
        "checkpoint meta -- both modes share a state_dict layout, so inference "
        "MUST pass the same value back (see models.config.load_model_bundle).",
    )
    ap.add_argument(
        "--drop_cols",
        type=str,
        default="",
        help="Comma-separated columns to remove from BOTH model input and "
        "prediction, before the registry is built (e.g. 'AIT201,AIT203' for "
        "channels whose train/attack distributions differ). Applied after the "
        "PLC1/full-SWaT column selection.",
    )
    ap.add_argument(
        "--corrupt_sensor",
        type=str,
        default=None,
        help="Sensor name to corrupt in training data (e.g. 'LIT101'). None = no corruption.",
    )
    ap.add_argument(
        "--corrupt_type",
        type=str,
        default="gaussian",
        choices=[
            "gaussian",
            "stuck_at",
            "ramp",
            "ramp_consistent_sin",
            "ramp_flow_consistent",
            "ramp_multistage",
        ],
        help="Type of sensor fault to apply (default: gaussian). "
        "ramp_consistent_sin: smooth cosine-bump LIT attack with half-sine "
        "inflow/outflow compensation (physics-exact, no FIT discontinuities). "
        "ramp_multistage samples a target column from --corrupt_pool per episode "
        "and requires --full_swat.",
    )
    ap.add_argument(
        "--corrupt_pool",
        type=str,
        default=None,
        choices=["lit", "fit", "fit_ro"],
        help="Sensor pool for corrupt_type=ramp_multistage. "
        "'lit' = {LIT101, LIT301, LIT401}; "
        "'fit' = {FIT101, FIT201, FIT301, FIT401}; "
        "'fit_ro' = {FIT401, FIT501, FIT502, FIT503, FIT504}.",
    )
    ap.add_argument(
        "--corrupt_param",
        type=float,
        default=0.1,
        help="Fault parameter: std_frac for gaussian, stuck_frac for stuck_at, poison_frac for ramp.",
    )
    ap.add_argument(
        "--ramp_rate",
        type=float,
        default=0.003,
        help="Ramp rate in data units/timestep for corrupt_type=ramp "
        "(0.003 m/s = 3 mm/s for LIT101 in SI). Ignored for other types.",
    )
    ap.add_argument(
        "--corrupt_flow_col",
        type=str,
        default=None,
        help="Inflow sensor to co-adjust for corrupt_type=ramp_consistent_sin "
        "(e.g. 'FIT101'), or the second flow sensor for ramp_flow_consistent (e.g. 'FIT201').",
    )
    ap.add_argument(
        "--corrupt_outflow_col",
        type=str,
        default=None,
        help="Outflow sensor to spoof during the falling half for "
        "corrupt_type=ramp_consistent_sin (e.g. 'FIT201').",
    )
    ap.add_argument(
        "--stuck_min_dur",
        type=int,
        default=5,
        help="Min stuck-at episode duration in timesteps (default: 5).",
    )
    ap.add_argument(
        "--stuck_max_dur",
        type=int,
        default=100,
        help="Max stuck-at episode duration in timesteps (default: 100).",
    )

    return ap.parse_args()


# Pools used by --corrupt_type=ramp_multistage (full-SWaT multi-stage poisoning).
# Module scope so the --drop_cols guard can consult it before corruption runs.
MULTISTAGE_POOLS = {
    "lit": ["LIT101", "LIT301", "LIT401"],
    "fit": ["FIT101", "FIT201", "FIT301", "FIT401"],
    "fit_ro": ["FIT401", "FIT501", "FIT502", "FIT503", "FIT504"],
}


def set_repro(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


def main():
    args = parse_args()
    set_repro(args.seed)

    def pick_device(arg: str):
        if arg in ("cuda", "cpu", "mps"):
            if arg == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if (
                arg == "mps"
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            if arg == "cpu":
                return torch.device("cpu")
        # auto/fallback
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = pick_device(getattr(args, "device", "auto"))
    # ---------- data ----------
    _, input_cols, data_full_phys = load_parquet(
        args.data, device="cpu"
    )  # full table (1 Hz)

    if not args.full_swat:
        # PLC1 only: 6 sensors + mode masks
        keep_cols = ["FIT101", "LIT101", "MV101", "P101", "P102", "FIT201"]

        keep_idx = torch.tensor(
            [input_cols.index(c) for c in keep_cols], dtype=torch.long
        )

        data_full_phys = data_full_phys.index_select(dim=1, index=keep_idx)
        input_cols = keep_cols
    # else: full SWaT — use all columns from parquet (no filtering)

    # --- optional column drop (applies to model input AND prediction) ---
    # MUST happen here: before the shape is read and before any row slicing or
    # corruption, because every later `input_cols.index(...)` lookup (corruption
    # targets, physics-filter equations) indexes into data_full_phys. Trimming
    # input_cols without index_select-ing the tensor in the same step silently
    # shifts those indices and corrupts the wrong sensor.
    drop_cols = [c.strip() for c in str(args.drop_cols).split(",") if c.strip()]
    if drop_cols:
        missing = [c for c in drop_cols if c not in input_cols]
        if missing:
            raise ValueError(
                f"--drop_cols names column(s) not present: {missing}. "
                f"Available: {input_cols}"
            )
        # Never drop a column the corruption or the physics needs.
        needed_corruption = set()
        if args.corrupt_sensor:
            needed_corruption.add(args.corrupt_sensor)
        if args.corrupt_flow_col:
            needed_corruption.add(args.corrupt_flow_col)
        if args.corrupt_outflow_col:
            needed_corruption.add(args.corrupt_outflow_col)
        if args.corrupt_pool:
            needed_corruption |= set(MULTISTAGE_POOLS.get(args.corrupt_pool, []))
        # Columns read by the mass-balance / flow-split objectives and their biases.
        needed_physics = {"LIT101", "FIT101", "FIT201"}
        if args.full_swat:
            needed_physics |= {
                "LIT301", "LIT401", "FIT301", "FIT401",
                "FIT501", "FIT502", "FIT503", "FIT504", "UF_REJECT_MASK",
            }
        clash = sorted(set(drop_cols) & (needed_corruption | needed_physics))
        if clash:
            raise ValueError(
                f"--drop_cols would remove column(s) still required: {clash} "
                f"(corruption needs {sorted(needed_corruption)}, "
                f"physics needs {sorted(needed_physics)})"
            )
        keep = [c for c in input_cols if c not in set(drop_cols)]
        keep_idx = torch.tensor(
            [input_cols.index(c) for c in keep], dtype=torch.long
        )
        data_full_phys = data_full_phys.index_select(dim=1, index=keep_idx)
        input_cols = keep
        print(f"[drop_cols] removed {drop_cols} -> {len(input_cols)} columns")

    N_total = data_full_phys.shape[0]

    # ---- fixed test region: last test_rows rows of the dataset ----
    test_rows = max(1, int(args.test_rows))
    if test_rows >= N_total:
        raise ValueError(f"--test_rows={test_rows} >= total rows {N_total}")
    test_start_row = N_total - test_rows
    test_phys_raw = data_full_phys[test_start_row:].clone()

    # ---- training region: row_start .. train_end ----
    s_raw = max(0, int(args.row_start))
    available_end = test_start_row  # must not overlap test
    # Optional tighter cap so training stays inside [row_start, training_slice_end)
    # and doesn't leak into the held-out slices. 0 = no extra cap.
    tse = int(args.training_slice_end) if int(args.training_slice_end) > 0 else 0
    train_end = min(available_end, tse) if tse > 0 else available_end
    if train_end <= s_raw:
        raise ValueError(f"--training_slice_end={tse} must be > --row_start={s_raw}")
    if args.rows and args.rows > 0:
        total_train_rows = min(int(args.rows), train_end - s_raw)
    else:
        total_train_rows = train_end - s_raw

    if total_train_rows < args.window:
        raise ValueError(
            f"Training region {total_train_rows} too small for window "
            f"({args.window}). Increase --rows."
        )

    # Single contiguous block taking the LAST total_train_rows rows of the
    # training region (most recent data — relevant for scarcity sweeps).
    train_start = train_end - total_train_rows
    train_phys_raw = data_full_phys[train_start:train_end].clone()
    print(
        f"[data] {total_train_rows} train rows "
        f"[{train_start}:{train_end}], "
        f"test/threshold region [{test_start_row}:{N_total}) of {N_total} rows"
    )
    if N_total != 496_800:
        print(
            f"[data] NOTE: parquet has {N_total} rows, not the 496,800 the v5 "
            "eval slices (evaluation.py THRESHOLD/SELECTION_SLICE_V3) assume -- "
            "the hardcoded eval slices will not line up with this test region."
        )

    # --- optional sensor corruption (Experiments B & C) ---
    corruption_meta = None
    is_multistage = args.corrupt_type == "ramp_multistage"
    if args.corrupt_sensor or is_multistage:
        from data.transforms import apply_sensor_fault

        if is_multistage:
            if not args.full_swat:
                raise ValueError(
                    "corrupt_type=ramp_multistage requires --full_swat "
                    "(multi-stage pools span sensors outside PLC1)"
                )
            if not args.corrupt_pool:
                raise ValueError(
                    "--corrupt_pool is required for corrupt_type=ramp_multistage "
                    f"(choose from {sorted(MULTISTAGE_POOLS)})"
                )
            pool_names = MULTISTAGE_POOLS[args.corrupt_pool]
            target_cols = [input_cols.index(n) for n in pool_names]
            col_idx = -1  # unused; placeholder for dispatcher signature
        else:
            col_idx = input_cols.index(args.corrupt_sensor)

        extra_kwargs = {}
        if (
            args.corrupt_type
            in (
                "ramp",
                "ramp_consistent_sin",
                "ramp_flow_consistent",
            )
            or is_multistage
        ):
            extra_kwargs["ramp_rate"] = args.ramp_rate
        if args.corrupt_type in (
            "ramp_consistent_sin",
            "ramp_flow_consistent",
        ):
            if not args.corrupt_flow_col:
                raise ValueError(
                    f"--corrupt_flow_col required for corrupt_type={args.corrupt_type}"
                )
            extra_kwargs["flow_col_idx"] = input_cols.index(args.corrupt_flow_col)
        if args.corrupt_type == "ramp_consistent_sin":
            if not args.corrupt_outflow_col:
                raise ValueError(
                    f"--corrupt_outflow_col required for corrupt_type={args.corrupt_type} "
                    "(outflow spoofed during the falling half of each episode)"
                )
            extra_kwargs["outflow_col_idx"] = input_cols.index(args.corrupt_outflow_col)
        if args.corrupt_type == "stuck_at":
            extra_kwargs["min_dur"] = args.stuck_min_dur
            extra_kwargs["max_dur"] = args.stuck_max_dur
        if is_multistage:
            extra_kwargs["target_cols"] = target_cols

        train_phys_raw = apply_sensor_fault(
            train_phys_raw,
            col_idx,
            args.corrupt_type,
            args.corrupt_param,
            seed=args.seed,
            **extra_kwargs,
        )
        # --- sanity summary ---
        if is_multistage:
            print(
                f"[corruption] ramp_multistage: rate={args.ramp_rate}, "
                f"poison_frac={args.corrupt_param}, "
                f"pool={args.corrupt_pool}: {pool_names}"
            )
            print(
                f"[corruption] {args.corrupt_type}({args.corrupt_param}) applied to "
                f"{len(pool_names)} sensors"
            )
        else:
            corrupted_col = train_phys_raw[:, col_idx]
            clean_ref_col = data_full_phys[train_start:train_end, col_idx]
            if args.corrupt_type == "gaussian":
                diff = corrupted_col[: len(clean_ref_col)] - clean_ref_col
                print(
                    f"[corruption] gaussian: realized noise std = {diff.std().item():.6f}"
                )
            elif args.corrupt_type == "stuck_at":
                # count timesteps where value equals previous timestep (proxy for stuck)
                diffs = corrupted_col[1:] - corrupted_col[:-1]
                stuck_count = int((diffs == 0).sum().item())
                print(
                    f"[corruption] stuck_at: ~{stuck_count}/{len(corrupted_col)} "
                    f"timesteps stuck ({100 * stuck_count / len(corrupted_col):.1f}%)"
                )
            elif args.corrupt_type == "ramp":
                print(
                    f"[corruption] ramp: rate={args.ramp_rate}, "
                    f"poison_frac={args.corrupt_param}"
                )
            elif args.corrupt_type == "ramp_consistent_sin":
                print(
                    f"[corruption] ramp_consistent_sin (cosine LIT, half-sine FITs): "
                    f"max_slope={args.ramp_rate}, poison_frac={args.corrupt_param}, "
                    f"inflow={args.corrupt_flow_col}, outflow={args.corrupt_outflow_col}"
                )
            elif args.corrupt_type == "ramp_flow_consistent":
                print(
                    f"[corruption] ramp_flow_consistent: rate={args.ramp_rate}, "
                    f"poison_frac={args.corrupt_param}, flow_col={args.corrupt_flow_col}"
                )

            print(
                f"[corruption] {args.corrupt_type}({args.corrupt_param}) applied to "
                f"{args.corrupt_sensor} (col {col_idx})"
            )

        corruption_meta = {
            "sensor": args.corrupt_sensor,
            "type": args.corrupt_type,
            "pool": args.corrupt_pool if is_multistage else None,
            "pool_sensors": pool_names if is_multistage else None,
            "param": args.corrupt_param,
            "ramp_rate": args.ramp_rate
            if args.corrupt_type
            in (
                "ramp",
                "ramp_consistent_sin",
                "ramp_flow_consistent",
                "ramp_multistage",
            )
            else None,
            "seed": args.seed,
        }

    # --- registry/targets ---
    registry = build_registry_from_columns(input_cols)
    target_indices, target_cols = select_targets(registry)

    # --- Downsample + window per block, then concatenate ---
    stride = max(1, int(args.temporal_stride))

    W = int(args.window)
    if args.keep_seconds and stride > 1:
        W = max(1, math.ceil(W / stride))

    ds_kwargs = dict(
        temporal_stride=stride,
        registry=registry,
        agg=args.downsample_agg,
    )

    train_phys_ds = temporal_downsample(train_phys_raw, **ds_kwargs)
    x_phys_tr = make_windows(train_phys_ds, W, stride=1)
    y_phys_tr = make_targets(train_phys_ds, target_indices, W, stride=1)

    # --- optional physics filter: hard-constraint baseline ---
    # Fit q-th quantile of |LIT101 mass-balance residual| on the clean
    # threshold slice, then drop training windows whose sample residual
    # exceeds the threshold. No bias subtraction — the quantile absorbs
    # steady-state offset. Filtering happens on windowed samples so window
    # continuity within surviving samples is preserved.
    filter_meta = None
    tau = None  # LIT101 filter threshold, retained in filter_meta
    if args.physics_filter > 0.0:
        # q is the quantile of |residual| on the CLEAN threshold slice. q=1.0 →
        # τ = max clean residual; training windows whose residual exceeds it are
        # still dropped (poison residuals can exceed any clean one), so q=1.0 is
        # the loosest clean-calibrated threshold, NOT "filter off".
        if args.physics_filter > 1.0:
            raise ValueError(
                f"--physics_filter must be in (0, 1], got {args.physics_filter}"
            )
        if args.threshold_slice_end <= args.threshold_slice_start:
            raise ValueError(
                "--threshold_slice_start/end must be set and non-empty when "
                f"--physics_filter > 0 (got [{args.threshold_slice_start}, "
                f"{args.threshold_slice_end}))"
            )

        # Mass-balance equations the filter enforces. PLC1 (default) uses the
        # single LIT101 equation. Full-SWaT additionally constrains LIT301 and
        # LIT401, plus an optional global 3-tank balance when --global_obj.
        # A window survives only if it is clean under ALL equations (AND).
        EQUATIONS = [("LIT101", ["FIT101"], ["FIT201"])]
        if args.full_swat:
            EQUATIONS += [
                ("LIT301", ["FIT201"], ["FIT301"]),
                ("LIT401", ["FIT301"], ["FIT401"]),
            ]
            if args.global_obj:
                EQUATIONS.append(
                    (["LIT101", "LIT301", "LIT401"], ["FIT101"], ["FIT501"])
                )

        # Threshold + training residual sources are shared across equations;
        # downsample once and fit each equation's τ off the same frames.
        thr_phys_raw = data_full_phys[
            args.threshold_slice_start : args.threshold_slice_end
        ].clone()
        thr_phys_ds = temporal_downsample(thr_phys_raw, **ds_kwargs)
        thr_df = pd.DataFrame(thr_phys_ds.cpu().numpy(), columns=input_cols)
        train_df = pd.DataFrame(train_phys_ds.cpu().numpy(), columns=input_cols)
        Ntr_pre = x_phys_tr.shape[0]

        keep = np.ones(Ntr_pre, dtype=bool)
        eq_metas: list[dict] = []
        for level, inflows, outflows in EQUATIONS:
            level_cols = [level] if isinstance(level, str) else list(level)
            needed = [*level_cols, *inflows, *outflows]
            if not all(c in input_cols for c in needed):
                raise ValueError(
                    f"--physics_filter requires columns {needed} "
                    f"in input_cols={input_cols}"
                )

            # Fit tau_i for this equation on the (clean) threshold slice.
            r_thr = _balance_residual(
                thr_df, level, inflows, outflows, dt=float(stride), area=TANK_AREA
            )
            r_thr = r_thr[np.isfinite(r_thr)]
            if r_thr.size == 0:
                raise ValueError(
                    f"Threshold slice produced no finite residuals for {level} — "
                    "check --threshold_slice_start/end."
                )
            tau_i = float(np.quantile(np.abs(r_thr), args.physics_filter))
            if tau is None:
                tau = tau_i  # keep the first (LIT101) τ for filter_meta

            # Training residuals on the (possibly corrupted) downsampled tensor.
            r_train = _balance_residual(
                train_df, level, inflows, outflows, dt=float(stride), area=TANK_AREA
            )
            # Each window-sample i covers rows [i, i+W-1]; reject the window if
            # ANY of its W residuals exceeds tau_i (same logic as single-eq).
            abs_r = np.abs(r_train)
            if abs_r.shape[0] < W + Ntr_pre - 1:
                raise RuntimeError(
                    f"residual length {abs_r.shape[0]} too small for "
                    f"W={W} and Ntr={Ntr_pre}"
                )
            sw = np.lib.stride_tricks.sliding_window_view(abs_r, W)[:Ntr_pre]
            if sw.shape[0] != Ntr_pre:
                raise RuntimeError(
                    f"sliding-window count {sw.shape[0]} != Ntr={Ntr_pre}"
                )
            finite_per_sample = np.all(np.isfinite(sw), axis=1)
            max_resid_per_sample = sw.max(axis=1)
            keep_i = (max_resid_per_sample <= tau_i) & finite_per_sample
            eq_metas.append(
                {
                    "level": level,
                    "tau": tau_i,
                    "kept_per_eq": int(keep_i.sum()),
                }
            )
            keep &= keep_i

        kept = int(keep.sum())
        total = int(keep.size)
        if kept == 0:
            raise RuntimeError(
                "physics_filter dropped ALL samples across "
                f"{len(EQUATIONS)} equation(s); check threshold slice and quantile."
            )
        keep_t = torch.as_tensor(keep, dtype=torch.bool)
        x_phys_tr = x_phys_tr[keep_t]
        y_phys_tr = y_phys_tr[keep_t]
        pct = 100.0 * kept / max(1, total)
        print(
            f"[filter] {len(EQUATIONS)} eq(s) kept {kept}/{total} ({pct:.1f}%) "
            f"q={args.physics_filter} "
            + " ".join(f"tau[{m['level']}]={m['tau']:.4e}" for m in eq_metas)
        )
        filter_meta = {
            "q": float(args.physics_filter),
            "tau": tau,  # LIT101 τ retained for backward compatibility
            "equations": eq_metas,
            "kept": kept,
            "total": total,
            "threshold_slice": [
                int(args.threshold_slice_start),
                int(args.threshold_slice_end),
            ],
        }

    # --- Invariant-rule window filter (alternative to --physics_filter) ---
    # PLC1 invariants from Wolsing et al. 2025 as a training-window drop: a window is dropped
    # if ANY of its timesteps violates ANY rule. Rules use RAW units (LIT mm,
    # FIT m^3/h, MV 1/2); the downsampled training tensor is SI, so convert.
    if getattr(args, "invariant_filter", False):
        need = ["LIT101", "MV101", "FIT101"]
        if not all(c in input_cols for c in need):
            raise ValueError(
                f"--invariant_filter requires {need} in input_cols={input_cols}"
            )
        inv_df = pd.DataFrame(train_phys_ds.cpu().numpy(), columns=input_cols)
        lit = inv_df["LIT101"].to_numpy() * 1000.0  # m -> mm
        fit = inv_df["FIT101"].to_numpy() * 3600.0  # m^3/s -> m^3/h
        mv = inv_df["MV101"].to_numpy()
        is_open = mv == 2
        is_closed = mv == 1
        viol = (
            ((lit < 490) & ~(is_open & (fit > 0.1)))  # R1
            | ((lit > 810) & ~(is_closed & (fit < 1)))  # R2
            | (is_open & ~(fit > 0.1))  # R3
            | ((lit < 490) & ~(fit > 0))  # R4
        )
        Ntr_pre = x_phys_tr.shape[0]
        sw = np.lib.stride_tricks.sliding_window_view(viol, W)[:Ntr_pre]
        if sw.shape[0] != Ntr_pre:
            raise RuntimeError(
                f"invariant_filter window count {sw.shape[0]} != Ntr={Ntr_pre}"
            )
        keep = ~sw.any(axis=1)
        kept, total = int(keep.sum()), int(keep.size)
        if kept == 0:
            raise RuntimeError("invariant_filter dropped ALL training windows")
        keep_t = torch.as_tensor(keep, dtype=torch.bool)
        x_phys_tr = x_phys_tr[keep_t]
        y_phys_tr = y_phys_tr[keep_t]
        print(
            f"[invariant-filter] kept {kept}/{total} "
            f"({100.0 * kept / max(1, total):.1f}%) windows"
        )
        filter_meta = {"type": "invariant", "kept": kept, "total": total}

    # test (always a single contiguous block at the end)
    test_phys = temporal_downsample(test_phys_raw, **ds_kwargs)
    x_phys_te = make_windows(test_phys, W, stride=1)
    y_phys_te = make_targets(test_phys, target_indices, W, stride=1)

    Ntr = x_phys_tr.shape[0]
    Nte = x_phys_te.shape[0]
    if Ntr <= 0 or Nte <= 0:
        raise ValueError(
            f"Need positive sequences for both train/val and test (got Ntr={Ntr}, Nte={Nte})"
        )

    # --- choose train/dev split INSIDE trainval sequences (ratios as before) ---
    train_ratio = 0.85
    dev_ratio = 0.15
    n_train = int(train_ratio * Ntr)
    n_dev = int(dev_ratio * Ntr)
    n_train = max(1, min(n_train, Ntr - 1))
    n_dev = max(1, min(n_dev, Ntr - n_train))

    # --- normalizer: fit on the training rows ---
    normalizer = fit_normalizer(train_phys_ds, registry, device=torch.device("cpu"))

    # --- normalize inputs & targets for BOTH splits with SAME stats ---
    mean_full = normalizer.mean.view(-1)
    std_full = normalizer.std.view(-1)
    inputs_norm_tr = normalizer(x_phys_tr.cpu())
    inputs_norm_te = normalizer(x_phys_te.cpu())
    y_norm_tr = (y_phys_tr.cpu() - mean_full[target_indices]) / std_full[target_indices]
    y_norm_te = (y_phys_te.cpu() - mean_full[target_indices]) / std_full[target_indices]

    # --- concatenate to use a single make_dataloaders() call with explicit slices ---
    x_phys_all = torch.cat([x_phys_tr.cpu(), x_phys_te.cpu()], dim=0)
    inputs_all = torch.cat([inputs_norm_tr, inputs_norm_te], dim=0)
    y_norm_all = torch.cat([y_norm_tr, y_norm_te], dim=0)
    y_phys_all = torch.cat([y_phys_tr.cpu(), y_phys_te.cpu()], dim=0)

    train_sl = slice(0, n_train)
    dev_sl = slice(n_train, n_train + n_dev)
    test_sl = slice(Ntr, Ntr + Nte)

    normalizer = normalizer.to(device)
    pin_mem = device.type == "cuda"
    train_loader, dev_loader, test_loader, _ = make_dataloaders(
        inputs_all,
        x_phys_all,
        y_norm_all,
        y_phys_all,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_mem,
        train_sl=train_sl,
        dev_sl=dev_sl,
        test_sl=test_sl,
    )

    # ---------- physics biases: median fit on training rows ----------
    dt_sec = float(stride)
    if args.bias_source == "test":
        # Fit on clean held-out test slice (never touched by sensor corruption).
        bias_df = pd.DataFrame(test_phys.cpu().numpy(), columns=input_cols)
    else:
        bias_df = pd.DataFrame(train_phys_ds.cpu().numpy(), columns=input_cols)
    fitted = fit_all_biases(
        bias_df,
        dt=dt_sec,
        full_swat=args.full_swat,
    )
    if args.bias_source == "zero":
        fitted = zero_biases(fitted)
    print(f"[bias] source={args.bias_source} fitted={fitted}")

    # ---------- model ----------
    F_in_model = len(target_indices)
    F_out_model = len(target_indices)

    size = "full_swat" if args.full_swat else "plc1"
    base = build_model(args.model, size, F_in_model, F_out_model, W, args.dropout)

    if args.model == "copy-last":
        model = base.to(device)
    else:
        model = NextStepForecaster(
            base,
            target_indices=target_indices,
            residual=(args.wrapper == "delta"),
        ).to(device)

    # ---------- objectives ----------
    objective_set_phys = build_objectives(
        fitted,
        scale_mode="online",
        scale_exclude=(),
        dt=dt_sec,
        full_swat=args.full_swat,
        global_obj=args.global_obj,
    )
    # Calibrate physics-objective scales on the CLEAN held-out region (test_loader =
    # the last --test_rows rows: [453_600:496_800] for PLC1, [410_400:496_800] for
    # full-SWaT), the same poison-free slice the bias uses.
    if args.w_phys > 0:
        _ = fit_scales_on_target_baseline(
            loader=test_loader,
            objective_set_phys=objective_set_phys,
            registry=registry,
            normalizer=normalizer,
            device=device,
            target_indices=target_indices,
        )

    # Lock scales so they don't drift
    objective_set_phys.set_scale_mode("frozen")

    optimizer = Adam(
        [
            {"params": model.parameters(), "lr": args.lr},
        ]
    )
    # ---------- trainer ----------
    outdir = P.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        objective_set=objective_set_phys,
        optimizer=optimizer,
        scheduler=None,
        loaders={"train": train_loader, "dev": dev_loader, "test": test_loader},
        registry=registry,
        normalizer=normalizer,
        device=device,
        print_every=15,
        target_indices=target_indices,
        w_phys=args.w_phys,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        early_stop_delta=args.early_stop_delta,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        objective_set_phys.load_state_dict(ckpt["objective_state"])
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])

    # record training timing and store it on trainer.state so save_run_json can pick it up
    train_start_utc = datetime.now(timezone.utc).isoformat()
    _t0 = time.perf_counter()

    best_ckpt = trainer.train(epochs=args.epochs)

    elapsed_sec = time.perf_counter() - _t0
    train_end_utc = datetime.now(timezone.utc).isoformat()
    trainer.state = getattr(trainer, "state", {})
    trainer.state["timing"] = {
        "train_start_utc": train_start_utc,
        "train_end_utc": train_end_utc,
        "train_wall_time_sec": elapsed_sec,
    }
    print(f"[timing] training took {elapsed_sec:.2f}s")

    # ---------- save checkpoint ----------
    # put the best weights back into the live objects
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    objective_set_phys.load_state_dict(best_ckpt["objective_state"])
    optimizer.load_state_dict(best_ckpt["optimizer_state"])
    trainer.state["epoch"] = best_ckpt["epoch"]  # keeps bookkeeping consistent

    # model checkpoint
    ckpt = {
        "epoch": trainer.state["epoch"],
        "model_state": model.state_dict(),
        "objective_state": objective_set_phys.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "registry": registry,
        "normalizer": normalizer.state_dict(),
        "meta": {
            "args": vars(args).copy(),
            "effective_window": W,
            "temporal_stride": args.temporal_stride,
            "downsample_agg": args.downsample_agg,
            "dt_sec": 1.0 * args.temporal_stride,
            "input_cols": input_cols,
            "target_cols": target_cols,
            "fitted_biases": fitted,
            "train_meta": {
                "train_start": train_start,
                "train_end": train_end,
                "total_train_rows": total_train_rows,
                "rows_arg": args.rows,
                "row_start": args.row_start,
                "test_start_row": test_start_row,
            },
            "corruption": corruption_meta,
            "physics_filter": filter_meta,
        },
    }
    ckpt_path = outdir / f"{args.name}.pt"
    torch.save(ckpt, ckpt_path)
    print(f"saved -> {ckpt_path}")

    # --- REQUIRED artifacts (consumed by select_best_models) ---
    # ALWAYS written, independent of --save_artifacts, so every run is usable by
    # the selection/eval pipeline.
    objective_set_phys.eval()
    if hasattr(objective_set_phys, "set_scale_mode"):
        objective_set_phys.set_scale_mode("frozen")

    save_eval_csvs(
        model=model,
        registry=registry,
        normalizer=normalizer,
        test_loader=test_loader,
        objective_set_phys=objective_set_phys,
        run_name=args.name,
        metrics_dir=outdir / "metrics",
        target_indices=target_indices,
    )

    save_run_json(
        outdir=outdir,
        args=args,
        model=model,
        trainer=trainer,
        registry=registry,
        objective_set=objective_set_phys,
        include_model_repr=False,
        pretty=True,
    )

    # --- DIAGNOSTIC-only artifacts (parity plots, loss grid) ---
    if args.save_artifacts:

        def _safe_artifact(label, fn):
            try:
                fn()
            except Exception as e:
                print(f"[artifact] {label} failed (non-fatal): {e}")

        _safe_artifact(
            "save_parity_and_rmse",
            lambda: save_parity_and_rmse(
                model,
                registry,
                normalizer,
                test_loader,
                target_indices=target_indices,
                target_cols=target_cols,
                run_name=args.name,
                plot_dir=outdir / "plots",
                table_dir=outdir / "metrics",
            ),
        )

        _safe_artifact(
            "save_loss_grid",
            lambda: save_loss_grid(
                trainer,
                run_name=args.name,
                out_dir=outdir / "losses",
            ),
        )


if __name__ == "__main__":
    main()
