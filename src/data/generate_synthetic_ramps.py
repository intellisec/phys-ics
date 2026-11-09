"""Generate synthetic attack test data for Experiment C evaluation.

Supports linear ramp, staircase, sigmoid, noisy ramp, physics-consistent
(LIT sinusoidal and FIT), and multistage shapes. Loads a slice of normal data,
injects attacks on a sensor column, and saves the result as parquet files
compatible with ``compute_metrics()``.

Also generates a PDF visualization showing original vs spoofed traces
for LIT101, FIT101 and FIT201 across a few representative episodes.

Usage
-----
# Linear ramp (default):
python -m src.data.generate_synthetic_ramps \
    --normal_parquet data/processed/normal_si.parquet \
    --out_dir data/processed/ramp_attack_3mm \
    --shape linear --ramp_rate 0.003

# Staircase (envelope = linear ramp at the same rate; --step_interval is the
# only shape knob, controls how many discrete steps fit in each episode):
python -m src.data.generate_synthetic_ramps \
    --normal_parquet data/processed/normal_si.parquet \
    --out_dir data/processed/staircase_attack \
    --shape staircase --ramp_rate 0.001943 --step_interval 30

# Sigmoid (envelope = linear ramp at the same rate; --steepness is the only
# shape knob, controls how sharp the inflection is at midpoint):
python -m src.data.generate_synthetic_ramps \
    --normal_parquet data/processed/normal_si.parquet \
    --out_dir data/processed/sigmoid_attack \
    --shape sigmoid --ramp_rate 0.001943 --steepness 8

# All shapes derive their per-episode max-spoof envelope from --ramp_rate, so
# differently-shaped attacks reach the same peak offset per episode.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from data.data_utils import TANK_AREA

import numpy as np
import pandas as pd
import torch


def _find_episodes(labels: torch.Tensor) -> list[tuple[int, int]]:
    """Return (start, end) pairs for contiguous label==1 regions."""
    episodes: list[tuple[int, int]] = []
    in_ep = False
    start = 0
    for i in range(len(labels)):
        if labels[i] == 1 and not in_ep:
            start = i
            in_ep = True
        elif labels[i] == 0 and in_ep:
            episodes.append((start, i))
            in_ep = False
    if in_ep:
        episodes.append((start, len(labels)))
    return episodes


def plot_attack_episodes(
    tensor_clean: torch.Tensor,
    tensor_attacked: torch.Tensor,
    labels: torch.Tensor,
    cols: list[str],
    *,
    vis_sensors: list[str] = ("LIT101", "FIT101", "FIT201"),
    num_episodes: int = 4,
    context_pad: int = 50,
    desc: str = "",
    save_path: Path | None = None,
) -> None:
    """Plot original (dashed) vs spoofed (solid) traces for selected sensors.

    Creates a grid: rows = episodes, cols = sensors.
    """
    import matplotlib.pyplot as plt

    episodes = _find_episodes(labels)
    if not episodes:
        print("[WARN] No episodes found, skipping plot.")
        return

    # Pick evenly spaced episodes
    indices = np.linspace(
        0, len(episodes) - 1, min(num_episodes, len(episodes)), dtype=int
    )

    # Resolve sensor column indices (skip sensors not in the data)
    sensor_idxs = []
    sensor_names = []
    for s in vis_sensors:
        if s in cols:
            sensor_idxs.append(cols.index(s))
            sensor_names.append(s)
    if not sensor_idxs:
        print(
            "[WARN] None of the visualisation sensors found in columns, skipping plot."
        )
        return

    clean_np = tensor_clean.numpy()
    attacked_np = tensor_attacked.numpy()
    T = len(labels)

    n_rows = len(indices)
    n_cols = len(sensor_names)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), squeeze=False, sharey=False
    )

    for row, ep_idx in enumerate(indices):
        ep_start, ep_end = episodes[ep_idx]
        lo = max(0, ep_start - context_pad)
        hi = min(T, ep_end + context_pad)
        t = np.arange(lo, hi)

        for col, (s_name, s_idx) in enumerate(zip(sensor_names, sensor_idxs)):
            ax = axes[row, col]

            ax.plot(
                t,
                clean_np[lo:hi, s_idx],
                color="tab:blue",
                ls="--",
                lw=1.2,
                alpha=0.8,
                label="Original",
            )
            ax.plot(
                t,
                attacked_np[lo:hi, s_idx],
                color="tab:red",
                ls="-",
                lw=1.2,
                alpha=0.9,
                label="Spoofed",
            )

            ax.axvspan(ep_start, ep_end, color="red", alpha=0.07)
            ax.axvline(ep_start, color="gray", ls=":", lw=0.6)
            ax.axvline(ep_end, color="gray", ls=":", lw=0.6)

            ax.set_xlabel("Timestep")
            if col == 0:
                ax.set_ylabel(f"Ep {ep_idx + 1}\n(t={ep_start}–{ep_end})")
            if row == 0:
                ax.set_title(s_name, fontsize=11, fontweight="bold")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.25)

    fig.suptitle(f"Synthetic Attack Visualisation — {desc}", fontsize=13, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate synthetic attack parquet + labels."
    )
    ap.add_argument(
        "--normal_parquet",
        required=True,
        help="Path to normal_si.parquet (SI units).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for synthetic parquet files.",
    )
    ap.add_argument(
        "--sensor",
        default="LIT101",
        help="Sensor column to attack (default: LIT101).",
    )
    ap.add_argument(
        "--shape",
        default="linear",
        choices=[
            "linear",
            "staircase",
            "sigmoid",
            "noisy_ramp",
            "consistent_lit_sin",
            "consistent_fit",
            "multistage_lit",
        ],
        help="Attack shape type (default: linear). "
        "consistent_lit_sin: cosine-bump on LIT with half-sine inflow/outflow "
        "compensation (physics residual is exactly zero throughout; smooth, "
        "everywhere-differentiable consistent LIT attack). "
        "multistage_lit: linear ramps spread deterministically across the "
        "sensors named by --pool (test-time analogue of training-time "
        "corrupt_type=ramp_multistage).",
    )
    ap.add_argument(
        "--pool",
        default="lit",
        choices=["lit", "fit", "fit_ro"],
        help="Sensor pool for shape=multistage_lit (mirrors training-time "
        "--corrupt_pool). 'lit' = {LIT101, LIT301, LIT401}; "
        "'fit' = {FIT101, FIT201, FIT301, FIT401}; "
        "'fit_ro' = {FIT401, FIT501, FIT502, FIT503, FIT504}.",
    )

    # --- shared ---
    ap.add_argument(
        "--num_attacks", type=int, default=20, help="Number of attack episodes."
    )
    ap.add_argument(
        "--min_dur", type=int, default=60, help="Min episode duration (timesteps)."
    )
    ap.add_argument(
        "--max_dur", type=int, default=375, help="Max episode duration (timesteps)."
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed.")
    # Defaults match the attack-injection slice in evaluation.py
    # (= [367_200, 410_400), 12 h at 1 Hz).
    ap.add_argument(
        "--validation_start",
        type=int,
        default=367_200,
        help="Start row of the slice attacks are injected into "
        "(default: ATTACK_SLICE_V3.start = 367_200).",
    )
    ap.add_argument(
        "--validation_end",
        type=int,
        default=410_400,
        help="End row of the slice attacks are injected into "
        "(default: ATTACK_SLICE_V3.stop = 410_400).",
    )

    # --- shared envelope (all shapes share the same per-episode max-spoof
    #     target = ramp_rate * actual_episode_duration) ---
    ap.add_argument(
        "--ramp_rate",
        type=float,
        default=0.003,
        help="Linear-equivalent slope (data units / timestep). "
        "All shapes derive their per-episode max from this.",
    )

    # --- staircase shape knob ---
    ap.add_argument(
        "--step_interval",
        type=int,
        default=30,
        help="Timesteps between staircase steps "
        "(step_height = ramp_rate * step_interval).",
    )

    # --- sigmoid shape knob ---
    ap.add_argument(
        "--steepness",
        type=float,
        default=8.0,
        help="Logistic steepness for shape=sigmoid (unitless). "
        "Higher → sharper inflection at midpoint. "
        "Default 8.0 reaches ≈99%% of M by t = 0.9·d.",
    )

    # --- noisy ramp shape knob ---
    ap.add_argument(
        "--noise_std",
        type=float,
        default=None,
        help="Gaussian noise std for noisy_ramp. Default: 1Hz change-std of target column.",
    )

    # --- consistent attacks ---
    ap.add_argument(
        "--flow_sensor",
        type=str,
        default=None,
        help="Inflow sensor for consistent_lit_sin (default FIT101) or secondary "
        "flow for consistent_fit (default FIT201).",
    )
    ap.add_argument(
        "--outflow_sensor",
        type=str,
        default=None,
        help="Outflow sensor for consistent_lit_sin (default FIT201). Spoofed during the falling half.",
    )
    ap.add_argument(
        "--area",
        type=float,
        default=TANK_AREA,
        help="Tank cross-section area in m² (for consistent_lit_sin). Default: TANK_AREA.",
    )

    args = ap.parse_args()

    # --- load normal data (validation slice) ---
    df_full = pd.read_parquet(args.normal_parquet)
    df_slice = df_full.iloc[args.validation_start : args.validation_end].reset_index(
        drop=True
    )
    print(
        f"Loaded VALIDATION_SLICE [{args.validation_start}:{args.validation_end}] "
        f"-> {len(df_slice)} rows"
    )

    cols = list(df_slice.columns)
    col_idx = cols.index(args.sensor)
    tensor = torch.tensor(df_slice.values, dtype=torch.float32)

    # --- inject attacks ---
    from data.transforms import (
        inject_test_ramps,
        inject_test_ramps_multistage,
        inject_test_staircase,
        inject_test_sigmoid,
        inject_test_noisy_ramp,
        inject_test_ramps_consistent_sinusoidal,
        inject_test_ramps_flow_consistent,
    )

    # Sensor pools for shape=multistage_lit (mirror train.py MULTISTAGE_POOLS).
    MULTISTAGE_POOLS = {
        "lit": ["LIT101", "LIT301", "LIT401"],
        "fit": ["FIT101", "FIT201", "FIT301", "FIT401"],
        "fit_ro": ["FIT401", "FIT501", "FIT502", "FIT503", "FIT504"],
    }

    if args.shape == "linear":
        corrupted, labels = inject_test_ramps(
            tensor,
            col_idx=col_idx,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_ramps=args.num_attacks,
        )
        desc = f"linear ramp (rate={args.ramp_rate})"

    elif args.shape == "staircase":
        corrupted, labels = inject_test_staircase(
            tensor,
            col_idx=col_idx,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            step_interval=args.step_interval,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_attacks=args.num_attacks,
        )
        derived_step_height = args.ramp_rate * args.step_interval
        desc = (
            f"staircase (rate={args.ramp_rate}, interval={args.step_interval}, "
            f"step_height={derived_step_height:.6g})"
        )

    elif args.shape == "sigmoid":
        corrupted, labels = inject_test_sigmoid(
            tensor,
            col_idx=col_idx,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            steepness=args.steepness,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_attacks=args.num_attacks,
        )
        desc = f"sigmoid (rate={args.ramp_rate}, steepness={args.steepness})"

    elif args.shape == "noisy_ramp":
        corrupted, labels = inject_test_noisy_ramp(
            tensor,
            col_idx=col_idx,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            noise_std=args.noise_std,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_attacks=args.num_attacks,
        )
        noise_str = f"{args.noise_std}" if args.noise_std else "auto(change_std)"
        desc = f"noisy ramp (rate={args.ramp_rate}, noise_std={noise_str})"

    elif args.shape == "consistent_lit_sin":
        # Physics-exact sinusoidal LIT attack: cosine bump on LIT; inflow
        # spoofed (half-sine) while LIT rises, outflow while it falls.
        flow_sensor = args.flow_sensor or "FIT101"
        outflow_sensor = args.outflow_sensor or "FIT201"
        flow_col_idx = cols.index(flow_sensor)
        outflow_col_idx = cols.index(outflow_sensor)
        corrupted, labels = inject_test_ramps_consistent_sinusoidal(
            tensor,
            col_idx=col_idx,
            flow_col_idx=flow_col_idx,
            outflow_col_idx=outflow_col_idx,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            area=args.area,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_attacks=args.num_attacks,
        )
        desc = (
            f"sinusoidal consistent LIT (max-slope={args.ramp_rate}, "
            f"inflow={flow_sensor}, outflow={outflow_sensor}, A={args.area})"
        )

    elif args.shape == "consistent_fit":
        # Consistent flow attack: ramp FIT101 + same offset on FIT201
        flow_sensor = args.flow_sensor or "FIT201"
        flow_col_idx = cols.index(flow_sensor)
        corrupted, labels = inject_test_ramps_flow_consistent(
            tensor,
            col_idx=col_idx,
            flow_col_idx=flow_col_idx,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_attacks=args.num_attacks,
        )
        desc = f"consistent FIT ramp (rate={args.ramp_rate}, secondary={flow_sensor})"

    elif args.shape == "multistage_lit":
        # Deterministic round-robin ramps across the --pool sensors.
        pool_names = MULTISTAGE_POOLS[args.pool]
        target_cols = [cols.index(n) for n in pool_names]
        corrupted, labels = inject_test_ramps_multistage(
            tensor,
            target_cols=target_cols,
            seed=args.seed,
            ramp_rate=args.ramp_rate,
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            num_ramps=args.num_attacks,
        )
        desc = (
            f"multistage ramp (rate={args.ramp_rate}, pool={args.pool}: {pool_names})"
        )

    is_multistage = args.shape == "multistage_lit"
    target_str = f"pool {args.pool}" if is_multistage else args.sensor
    n_attack = int(labels.sum().item())
    print(
        f"Injected {args.num_attacks} {desc} episodes on {target_str}. "
        f"Attack timesteps: {n_attack}/{len(labels)} "
        f"({100 * n_attack / len(labels):.1f}%)"
    )

    # --- save as parquet ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # multistage_lit writes the canonical "synthetic_multistage_*" names the
    # eval pipeline expects (GRID_TEST_DATASETS in evaluation.py).
    suffix = "multistage" if is_multistage else args.shape
    df_attack = pd.DataFrame(corrupted.numpy(), columns=cols)
    attack_path = out_dir / f"synthetic_{suffix}_attack.parquet"
    df_attack.to_parquet(attack_path, index=False)

    df_labels = pd.DataFrame({"attack": labels.numpy()})
    labels_path = out_dir / f"synthetic_{suffix}_labels.parquet"
    df_labels.to_parquet(labels_path, index=False)

    print(f"Saved: {attack_path}")
    print(f"Saved: {labels_path}")

    # --- visualisation PDF ---
    if is_multistage:
        vis_sensors = MULTISTAGE_POOLS[args.pool]
    else:
        vis_sensors = ["LIT101", "FIT101", "FIT201"]
    plot_path = out_dir / f"synthetic_{suffix}_visualisation.pdf"
    plot_attack_episodes(
        tensor_clean=tensor,
        tensor_attacked=corrupted,
        labels=labels,
        cols=cols,
        vis_sensors=vis_sensors,
        num_episodes=4,
        context_pad=50,
        desc=desc,
        save_path=plot_path,
    )


if __name__ == "__main__":
    main()
