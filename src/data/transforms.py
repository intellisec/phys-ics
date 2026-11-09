"""
Training-data transforms for degraded-data experiments.

Experiment B: sensor faults (gaussian noise, stuck-at)
Experiment C: dataset poisoning (ramp attacks)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor
from data.data_utils import TANK_AREA


# ---------------------------------------------------------------------------
# Experiment B: sensor faults
# ---------------------------------------------------------------------------


def gaussian_noise(
    tensor: Tensor,
    col_idx: int,
    std_frac: float,
    seed: int,
) -> Tensor:
    """Add i.i.d. Gaussian noise to a single column.

    The noise magnitude is calibrated relative to the column's 1 Hz
    **change standard deviation** (std of consecutive differences), so
    ``std_frac=1.0`` means the added noise has the same magnitude as a
    typical one-second level change.  This makes the parameter
    physically interpretable and comparable across sensors.

    Parameters
    ----------
    tensor : (T, F) tensor — modified **in-place**.
    col_idx : column index to corrupt.
    std_frac : noise std as a multiple of the column's 1 Hz change std.
    seed : RNG seed for reproducibility.
    """
    col = tensor[:, col_idx]
    change_std = (col[1:] - col[:-1]).std().item()
    sigma = std_frac * change_std
    gen = torch.Generator(device=tensor.device).manual_seed(seed)
    noise = (
        torch.randn(col.shape, generator=gen, device=tensor.device, dtype=tensor.dtype)
        * sigma
    )
    tensor[:, col_idx] = col + noise
    return tensor


def stuck_at(
    tensor: Tensor,
    col_idx: int,
    stuck_frac: float,
    seed: int,
    min_dur: int = 5,
    max_dur: int = 100,
) -> Tensor:
    """Scatter multiple short stuck-at episodes across the block.

    Each episode freezes the sensor at its value at episode onset for a
    random duration drawn from U(min_dur, max_dur) timesteps (at 1 Hz
    before downsampling, so 5–100 seconds by default).

    Parameters
    ----------
    tensor : (T, F) tensor — modified **in-place**.
    col_idx : column index to corrupt.
    stuck_frac : approximate fraction of total timesteps that will be stuck.
    seed : RNG seed for reproducibility.
    min_dur, max_dur : episode duration bounds in timesteps.
    """
    T = tensor.shape[0]
    if stuck_frac <= 0.0 or T < 2:
        return tensor

    total_stuck = int(T * stuck_frac)
    gen = torch.Generator(device=tensor.device).manual_seed(seed)

    # --- sample non-overlapping episodes until budget is spent ---
    occupied = torch.zeros(T, dtype=torch.bool)
    budget = total_stuck
    max_attempts = total_stuck * 10  # avoid infinite loop on dense packing

    attempts = 0
    while budget > 0 and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        dur = min(dur, budget)  # don't exceed remaining budget
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)

        # skip if overlaps an existing episode
        if occupied[start:end].any():
            continue

        frozen_value = tensor[start, col_idx].item()
        tensor[start:end, col_idx] = frozen_value
        occupied[start:end] = True
        budget -= end - start

    return tensor


# ---------------------------------------------------------------------------
# Experiment C: ramp poisoning
# ---------------------------------------------------------------------------


def ramp_poison(
    tensor: Tensor,
    col_idx: int,
    poison_frac: float,
    seed: int,
    ramp_rate: float = 0.003,
    min_dur: int = 60,
    max_dur: int = 375,
) -> Tensor:
    """Scatter ramp-attack episodes across the block.

    Each episode replaces the sensor column with a linear ramp starting
    from the clean value at onset.  Direction alternates (up / down).
    At the end of each episode the sensor snaps back to the clean value
    (realistic: attacker stops spoofing, sensor returns to ground truth).

    Parameters
    ----------
    tensor : (T, F) tensor — modified **in-place**.
    col_idx : column index to corrupt.
    poison_frac : fraction of total timesteps under ramp episodes.
    seed : RNG seed for reproducibility.
    ramp_rate : absolute ramp slope in data units per timestep
        (e.g. 0.003 m/s = 3 mm/s for LIT101 in SI).
    min_dur, max_dur : episode duration bounds in timesteps.
    """
    T = tensor.shape[0]
    if poison_frac <= 0.0 or T < 2:
        return tensor

    total_budget = int(T * poison_frac)
    gen = torch.Generator(device=tensor.device).manual_seed(seed)

    clean_col = tensor[:, col_idx].clone()

    occupied = torch.zeros(T, dtype=torch.bool)
    budget = total_budget
    max_attempts = total_budget * 10
    episode_idx = 0

    attempts = 0
    while budget > 0 and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        dur = min(dur, budget)
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        actual_dur = end - start

        if occupied[start:end].any():
            continue

        # alternating direction: even episodes ramp up, odd ramp down
        sign = 1.0 if episode_idx % 2 == 0 else -1.0
        base_val = clean_col[start].item()

        # pure linear ramp for the entire episode
        t_ramp = torch.arange(actual_dur, dtype=tensor.dtype, device=tensor.device)
        ramp_vals = base_val + sign * ramp_rate * t_ramp

        tensor[start:end, col_idx] = ramp_vals
        occupied[start:end] = True
        budget -= actual_dur
        episode_idx += 1

    return tensor


def ramp_poison_multistage(
    tensor: Tensor,
    target_cols: list[int],
    poison_frac: float,
    seed: int,
    ramp_rate: float = 0.003,
    min_dur: int = 60,
    max_dur: int = 375,
) -> Tensor:
    """Scatter linear ramp episodes across a pool of target columns.

    Identical shape to :func:`ramp_poison` (pure linear ramp from the
    clean value at onset, direction alternating up/down across episodes,
    snap-back at episode end). The only difference: for each episode
    the target column is sampled **uniformly** from ``target_cols``.

    Intended for full-SWaT multi-stage poisoning experiments where a
    single training run contains attack episodes across multiple
    sensors (e.g. all 3 LITs or all 4 FITs).

    Parameters
    ----------
    tensor : (T, F) tensor — modified **in-place**.
    target_cols : list of column indices; each episode picks one uniformly.
    poison_frac : fraction of total timesteps under ramp episodes.
    seed : RNG seed for reproducibility.
    ramp_rate : absolute ramp slope in data units per timestep.
    min_dur, max_dur : episode duration bounds in timesteps.
    """
    if not target_cols:
        raise ValueError("target_cols must be non-empty")

    T = tensor.shape[0]
    if poison_frac <= 0.0 or T < 2:
        return tensor

    total_budget = int(T * poison_frac)
    gen = torch.Generator(device=tensor.device).manual_seed(seed)

    # Snapshot clean values for every pool column so the episode start
    # value is referenced from the unpoisoned trace even if the same
    # column is revisited by a later episode.
    clean_cols = {c: tensor[:, c].clone() for c in target_cols}

    occupied = torch.zeros(T, dtype=torch.bool)
    budget = total_budget
    max_attempts = max(1, total_budget) * 10
    episode_idx = 0

    attempts = 0
    while budget > 0 and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        dur = min(dur, budget)
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        actual_dur = end - start

        if occupied[start:end].any():
            continue

        # Sample the primary column for this episode
        col_idx = int(
            target_cols[
                int(torch.randint(0, len(target_cols), (1,), generator=gen).item())
            ]
        )

        # alternating direction: even episodes ramp up, odd ramp down
        sign = 1.0 if episode_idx % 2 == 0 else -1.0
        base_val = clean_cols[col_idx][start].item()

        t_ramp = torch.arange(actual_dur, dtype=tensor.dtype, device=tensor.device)
        ramp_vals = base_val + sign * ramp_rate * t_ramp

        tensor[start:end, col_idx] = ramp_vals
        occupied[start:end] = True
        budget -= actual_dur
        episode_idx += 1

    return tensor


def inject_test_ramps(
    tensor: Tensor,
    col_idx: int,
    seed: int,
    ramp_rate: float = 0.003,
    min_dur: int = 60,
    max_dur: int = 375,
    num_ramps: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Inject a fixed number of ramp attacks into test data.

    Returns (corrupted_tensor, labels) where labels is a 1-D binary
    tensor (1 during ramp, 0 otherwise).  Pure linear ramp with
    snap-back to clean value at episode end (same style as
    :func:`ramp_poison`).

    Parameters
    ----------
    tensor : (T, F) — cloned internally, original is not modified.
    col_idx : column index to attack.
    seed : RNG seed.
    ramp_rate : ramp slope in data units / timestep.
    min_dur, max_dur : episode duration bounds.
    num_ramps : how many ramp episodes to inject.
    """
    out = tensor.clone()
    T = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    clean_col = out[:, col_idx].clone()
    labels = torch.zeros(T, dtype=torch.long, device=out.device)
    occupied = torch.zeros(T, dtype=torch.bool)

    placed = 0
    max_attempts = num_ramps * 20

    attempts = 0
    while placed < num_ramps and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        actual_dur = end - start

        if occupied[start:end].any():
            continue

        sign = 1.0 if placed % 2 == 0 else -1.0
        base_val = clean_col[start].item()

        t_ramp = torch.arange(actual_dur, dtype=out.dtype, device=out.device)
        ramp_vals = base_val + sign * ramp_rate * t_ramp

        out[start:end, col_idx] = ramp_vals
        labels[start:end] = 1
        occupied[start:end] = True
        placed += 1

    return out, labels


def inject_test_ramps_multistage(
    tensor: Tensor,
    target_cols: list[int],
    seed: int,
    ramp_rate: float = 0.003,
    min_dur: int = 60,
    max_dur: int = 375,
    num_ramps: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Inject a fixed number of ramp attacks across a pool of columns.

    Test-time analogue of :func:`ramp_poison_multistage`: each episode
    attacks one sensor drawn from ``target_cols``.  Unlike the training
    helper (which samples the column **randomly**), this generator cycles
    through ``target_cols`` **deterministically**
    (``col = target_cols[episode_idx % len(target_cols)]``) so that a
    fixed ``(seed, num_ramps)`` produces a byte-reproducible attack/labels
    parquet that can be regenerated identically across runs.

    Episode placement (non-overlapping, alternating up/down sign, uniform
    duration, snap-back at episode end) matches :func:`inject_test_ramps`.

    Returns ``(corrupted_tensor, labels)`` where ``labels`` is a 1-D binary
    tensor (1 inside any episode) — same shape as :func:`inject_test_ramps`
    output so downstream label-loading is unchanged.
    """
    if not target_cols:
        raise ValueError("target_cols must be non-empty")

    out = tensor.clone()
    T = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    # Snapshot clean values per pool column so episode-start values reference
    # the unpoisoned trace even if a column is revisited by a later episode.
    clean_cols = {c: out[:, c].clone() for c in target_cols}
    labels = torch.zeros(T, dtype=torch.long, device=out.device)
    occupied = torch.zeros(T, dtype=torch.bool)

    placed = 0
    max_attempts = num_ramps * 20

    attempts = 0
    while placed < num_ramps and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        actual_dur = end - start

        if occupied[start:end].any():
            continue

        # Deterministic target selection over the target pool.
        col_idx = int(target_cols[placed % len(target_cols)])

        sign = 1.0 if placed % 2 == 0 else -1.0
        base_val = clean_cols[col_idx][start].item()

        t_ramp = torch.arange(actual_dur, dtype=out.dtype, device=out.device)
        ramp_vals = base_val + sign * ramp_rate * t_ramp

        out[start:end, col_idx] = ramp_vals
        labels[start:end] = 1
        occupied[start:end] = True
        placed += 1

    return out, labels


# ---------------------------------------------------------------------------
# Experiment C variant: physics-consistent ramp poisoning
# ---------------------------------------------------------------------------


def ramp_poison_consistent_sinusoidal(
    tensor: Tensor,
    col_idx: int,
    poison_frac: float,
    seed: int,
    ramp_rate: float = 0.003,
    min_dur: int = 60,
    max_dur: int = 375,
    flow_col_idx: int | None = None,
    outflow_col_idx: int | None = None,
    area: float = TANK_AREA,
) -> Tensor:
    """Smooth (cosine) physics-consistent LIT poison; training-time variant of
    :func:`inject_test_ramps_consistent_sinusoidal`.

    LIT offset: ``ΔLIT(t) = sign * (A_lit / 2) * (1 - cos(2π t / dur))``,
    where ``A_lit = ramp_rate * dur / π`` so the **maximum** slope of ΔLIT
    equals ``ramp_rate``.

    Flow compensations are positive half-sines that are physics-exact at every
    timestep — there are no FIT discontinuities at episode start, midpoint,
    or end:

        dΔLIT/dt = sign * ramp_rate * sin(2π t / dur)
        ΔFIT_in  = area * max(0, +dΔLIT/dt)
        ΔFIT_out = area * max(0, -dΔLIT/dt)

    Peak flow spoof is ``area * ramp_rate``.
    """
    if flow_col_idx is None:
        raise ValueError(
            "flow_col_idx (inflow) must be provided for ramp_consistent_sin"
        )
    if outflow_col_idx is None:
        raise ValueError("outflow_col_idx must be provided for ramp_consistent_sin")

    T = tensor.shape[0]
    if poison_frac <= 0.0 or T < 2:
        return tensor

    total_budget = int(T * poison_frac)
    gen = torch.Generator(device=tensor.device).manual_seed(seed)

    clean_level = tensor[:, col_idx].clone()
    clean_inflow = tensor[:, flow_col_idx].clone()
    clean_outflow = tensor[:, outflow_col_idx].clone()

    occupied = torch.zeros(T, dtype=torch.bool)
    budget = total_budget
    max_attempts = total_budget * 10
    episode_idx = 0

    attempts = 0
    while budget > 0 and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        dur = min(dur, budget)
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        actual_dur = end - start

        if occupied[start:end].any():
            continue

        sign = 1.0 if episode_idx % 2 == 0 else -1.0

        if actual_dur >= 2:
            dur_f = float(actual_dur)
            t = torch.arange(actual_dur, dtype=tensor.dtype, device=tensor.device)
            phase = (2.0 * math.pi / dur_f) * t

            A_lit = ramp_rate * dur_f / math.pi
            level_offset = sign * (A_lit / 2.0) * (1.0 - torch.cos(phase))

            dLdt = sign * ramp_rate * torch.sin(phase)
            inflow_offset = area * torch.clamp(dLdt, min=0.0)
            outflow_offset = area * torch.clamp(-dLdt, min=0.0)
        else:
            t = torch.arange(actual_dur, dtype=tensor.dtype, device=tensor.device)
            level_offset = torch.zeros_like(t)
            inflow_offset = torch.zeros_like(t)
            outflow_offset = torch.zeros_like(t)

        tensor[start:end, col_idx] = clean_level[start:end] + level_offset
        tensor[start:end, flow_col_idx] = clean_inflow[start:end] + inflow_offset
        tensor[start:end, outflow_col_idx] = clean_outflow[start:end] + outflow_offset
        occupied[start:end] = True
        budget -= actual_dur
        episode_idx += 1

    return tensor


def ramp_poison_flow_consistent(
    tensor: Tensor,
    col_idx: int,
    poison_frac: float,
    seed: int,
    ramp_rate: float = 2e-5,
    min_dur: int = 50,
    max_dur: int = 300,
    flow_col_idx: int | None = None,
) -> Tensor:
    """Ramp FIT101 AND apply the same offset to FIT201 so net flow is unchanged.

    During each ramp episode both the primary flow sensor (``col_idx``,
    typically FIT101) and a second flow sensor (``flow_col_idx``, typically
    FIT201) receive the same linear ramp offset.  Because the mass-balance
    equation depends on ``(FIT101 − FIT201) / A``, the net difference is
    unchanged and the physics residual stays near zero.

    At episode end both sensors snap back to their clean values.

    Parameters
    ----------
    tensor : (T, F) — modified **in-place**.
    col_idx : primary flow sensor column (FIT101).
    poison_frac : fraction of timesteps under ramp episodes.
    seed : RNG seed.
    ramp_rate : ramp slope in flow units / timestep.
    min_dur, max_dur : episode duration bounds.
    flow_col_idx : second flow sensor column to co-adjust (FIT201).
        Must be provided; raises if None.
    """
    if flow_col_idx is None:
        raise ValueError("flow_col_idx must be provided for ramp_flow_consistent")

    T = tensor.shape[0]
    if poison_frac <= 0.0 or T < 2:
        return tensor

    total_budget = int(T * poison_frac)
    gen = torch.Generator(device=tensor.device).manual_seed(seed)

    clean_primary = tensor[:, col_idx].clone()
    clean_secondary = tensor[:, flow_col_idx].clone()

    occupied = torch.zeros(T, dtype=torch.bool)
    budget = total_budget
    max_attempts = total_budget * 10
    episode_idx = 0

    attempts = 0
    while budget > 0 and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        dur = min(dur, budget)
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        actual_dur = end - start

        if occupied[start:end].any():
            continue

        sign = 1.0 if episode_idx % 2 == 0 else -1.0

        # same linear offset applied to BOTH flow sensors
        t_ramp = torch.arange(actual_dur, dtype=tensor.dtype, device=tensor.device)
        offset = sign * ramp_rate * t_ramp

        tensor[start:end, col_idx] = clean_primary[start:end] + offset
        tensor[start:end, flow_col_idx] = clean_secondary[start:end] + offset
        occupied[start:end] = True
        budget -= actual_dur
        episode_idx += 1

    return tensor


# ---------------------------------------------------------------------------
# Test-time consistent attacks (for presentation / comparison)
# ---------------------------------------------------------------------------


def inject_test_ramps_consistent_sinusoidal(
    tensor: Tensor,
    col_idx: int,
    flow_col_idx: int,
    outflow_col_idx: int,
    seed: int,
    ramp_rate: float = 0.003,
    area: float = TANK_AREA,
    min_dur: int = 60,
    max_dur: int = 375,
    num_attacks: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Sinusoidal physics-consistent LIT attack.

    LIT offset is a cosine bump (zero at both endpoints, peak at midpoint):

        ΔLIT(t) = sign * (A_lit / 2) * (1 - cos(2π t / T))

    where ``A_lit = ramp_rate * T / π`` is chosen so that the maximum
    slope magnitude of ΔLIT equals ``ramp_rate``. The peak amplitude is
    therefore ``T·ramp_rate / π``.

    FIT compensations preserve the mass balance exactly. Let

        dΔLIT/dt = sign * ramp_rate * sin(2π t / T)

    Then the spoofs are split by sign of dΔLIT/dt:

        ΔFIT_in(t)  = area * max(0, +dΔLIT/dt)   (nonzero while LIT rises)
        ΔFIT_out(t) = area * max(0, -dΔLIT/dt)   (nonzero while LIT falls)

    Each flow spoof is a half-sine on its half, peaking at ``area * ramp_rate``;
    both flows start and end at 0 and meet at 0 at the apex — no boundary
    discontinuities anywhere.

    Result: physics residual ``dLIT/dt − (FIT_in − FIT_out)/A`` is zero
    at every timestep (up to float precision). Returns (corrupted, labels).
    """
    out = tensor.clone()
    T_total = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    clean_level = out[:, col_idx].clone()
    clean_inflow = out[:, flow_col_idx].clone()
    clean_outflow = out[:, outflow_col_idx].clone()
    labels = torch.zeros(T_total, dtype=torch.long, device=out.device)

    episodes = _place_episodes(T_total, num_attacks, min_dur, max_dur, gen)

    for i, (start, end) in enumerate(episodes):
        actual_dur = end - start
        if actual_dur < 2:
            continue
        sign = 1.0 if i % 2 == 0 else -1.0

        dur = float(actual_dur)
        t = torch.arange(actual_dur, dtype=out.dtype, device=out.device)
        phase = (2.0 * math.pi / dur) * t

        # LIT cosine bump; amplitude chosen so max |dΔLIT/dt| = ramp_rate.
        A_lit = ramp_rate * dur / math.pi
        level_offset = sign * (A_lit / 2.0) * (1.0 - torch.cos(phase))

        # Physics-exact flow split from dΔLIT/dt:
        dLdt = sign * ramp_rate * torch.sin(phase)
        inflow_offset = area * torch.clamp(dLdt, min=0.0)
        outflow_offset = area * torch.clamp(-dLdt, min=0.0)

        out[start:end, col_idx] = clean_level[start:end] + level_offset
        out[start:end, flow_col_idx] = clean_inflow[start:end] + inflow_offset
        out[start:end, outflow_col_idx] = clean_outflow[start:end] + outflow_offset
        labels[start:end] = 1

    return out, labels


def inject_test_ramps_flow_consistent(
    tensor: Tensor,
    col_idx: int,
    flow_col_idx: int,
    seed: int,
    ramp_rate: float = 2e-5,
    min_dur: int = 60,
    max_dur: int = 375,
    num_attacks: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Inject consistent flow ramp: spoof FIT101 + apply same offset to FIT201.

    Both flow sensors get the same linear offset → (FIT101 − FIT201)/A unchanged
    → physics residual stays near zero.  Returns (corrupted_tensor, labels).
    """
    out = tensor.clone()
    T = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    clean_primary = out[:, col_idx].clone()
    clean_secondary = out[:, flow_col_idx].clone()
    labels = torch.zeros(T, dtype=torch.long, device=out.device)

    episodes = _place_episodes(T, num_attacks, min_dur, max_dur, gen)

    for i, (start, end) in enumerate(episodes):
        actual_dur = end - start
        sign = 1.0 if i % 2 == 0 else -1.0

        t = torch.arange(actual_dur, dtype=out.dtype, device=out.device)
        offset = sign * ramp_rate * t

        out[start:end, col_idx] = clean_primary[start:end] + offset
        out[start:end, flow_col_idx] = clean_secondary[start:end] + offset
        labels[start:end] = 1

    return out, labels


# ---------------------------------------------------------------------------
# Test-time attack shapes (cross-shape transfer evaluation)
# ---------------------------------------------------------------------------


def _place_episodes(
    T: int,
    num_attacks: int,
    min_dur: int,
    max_dur: int,
    gen: torch.Generator,
) -> list[tuple[int, int]]:
    """Sample non-overlapping episode (start, end) pairs."""
    occupied = torch.zeros(T, dtype=torch.bool)
    episodes = []
    max_attempts = num_attacks * 20
    attempts = 0
    while len(episodes) < num_attacks and attempts < max_attempts:
        attempts += 1
        dur = int(torch.randint(min_dur, max_dur + 1, (1,), generator=gen).item())
        start = int(torch.randint(0, max(1, T - dur + 1), (1,), generator=gen).item())
        end = min(start + dur, T)
        if occupied[start:end].any():
            continue
        occupied[start:end] = True
        episodes.append((start, end))
    return episodes


def inject_test_staircase(
    tensor: Tensor,
    col_idx: int,
    seed: int,
    ramp_rate: float = 0.003,
    step_interval: int = 30,
    min_dur: int = 60,
    max_dur: int = 375,
    num_attacks: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Inject staircase spoof episodes that match the linear-ramp envelope.

    Each episode of duration ``d`` (≥ 2 samples) climbs in ``n_steps`` equal
    discrete jumps from the clean value at ``t=0`` to exactly
    ``base_val ± ramp_rate · (d-1)`` at ``t=d-1`` — the same endpoint a linear
    ramp at ``ramp_rate`` would hit.

    ``step_interval`` is interpreted as the *target* timesteps between steps:

        n_steps = max(1, round((d - 1) / step_interval))
        step_height = ramp_rate · (d - 1) / n_steps        # per-episode
        boundary_k  = round(k · (d - 1) / n_steps)         # k = 1..n_steps

    Step heights and boundaries adjust per episode so the final sample lands
    exactly on the linear envelope.  ``step_interval`` therefore controls
    average step spacing (not exact spacing) — its visual character (≈ d /
    step_interval discrete jumps per episode) is preserved.

    Returns (corrupted_tensor, binary_labels).
    """
    out = tensor.clone()
    T = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    clean_col = out[:, col_idx].clone()
    labels = torch.zeros(T, dtype=torch.long, device=out.device)

    episodes = _place_episodes(T, num_attacks, min_dur, max_dur, gen)

    for i, (start, end) in enumerate(episodes):
        actual_dur = end - start
        if actual_dur < 2:
            labels[start:end] = 1
            continue
        sign = 1.0 if i % 2 == 0 else -1.0
        base_val = clean_col[start].item()

        # Per-episode step count and step height.  M is the linear-ramp
        # endpoint so step_height * n_steps = M exactly.
        n_steps = max(1, round((actual_dur - 1) / int(step_interval)))
        M = float(ramp_rate) * (actual_dur - 1)
        step_height = M / n_steps

        # Boundaries at evenly-spaced rounded positions in [1, d-1].  The
        # k-th boundary is the time at which the level transitions to k.
        boundaries = torch.tensor(
            [round(k * (actual_dur - 1) / n_steps) for k in range(1, n_steps + 1)],
            dtype=torch.long,
            device=out.device,
        )
        t = torch.arange(actual_dur, dtype=torch.long, device=out.device)
        # levels[i] = number of boundaries ≤ t[i]  ∈ {0, 1, …, n_steps}
        levels = (t.unsqueeze(1) >= boundaries.unsqueeze(0)).sum(dim=1).to(out.dtype)
        episode_vals = base_val + sign * step_height * levels

        out[start:end, col_idx] = episode_vals
        labels[start:end] = 1

    return out, labels


def inject_test_sigmoid(
    tensor: Tensor,
    col_idx: int,
    seed: int,
    ramp_rate: float = 0.003,
    steepness: float = 8.0,
    min_dur: int = 60,
    max_dur: int = 375,
    num_attacks: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Inject smooth S-curve spoof episodes whose envelope matches a linear ramp.

    For each episode of duration ``d`` (≥ 2 samples), the spoof traces a
    *rescaled* logistic that starts at exactly the clean value at ``t=0`` and
    ends at exactly ``base_val ± ramp_rate · (d-1)`` at the last sample
    ``t=d-1`` — the same endpoint a linear ramp at ``ramp_rate`` would hit:

        f(t) = base_val ± M · (σ(k·(τ − 1/2)) − σ(−k/2))
                              / (σ(k/2) − σ(−k/2))
        where τ = t / (d-1),  M = ramp_rate · (d-1),  k = steepness.

    Both endpoints are exact (not asymptotic).  ``steepness`` is a unitless
    shape knob: higher → sharper inflection at midpoint; lower → near-linear.
    Default 8.0 gives ≈ 99 % of M by t = 0.9·(d-1).

    Returns (corrupted_tensor, binary_labels).
    """
    out = tensor.clone()
    T = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    clean_col = out[:, col_idx].clone()
    labels = torch.zeros(T, dtype=torch.long, device=out.device)

    episodes = _place_episodes(T, num_attacks, min_dur, max_dur, gen)

    k = float(steepness)
    sigma_min = float(torch.sigmoid(torch.tensor(-k / 2.0)))
    sigma_max = float(torch.sigmoid(torch.tensor(k / 2.0)))
    sigma_span = sigma_max - sigma_min

    for i, (start, end) in enumerate(episodes):
        actual_dur = end - start
        if actual_dur < 2:
            # Degenerate single-sample episode: no slope to inject.
            labels[start:end] = 1
            continue
        sign = 1.0 if i % 2 == 0 else -1.0
        base_val = clean_col[start].item()

        t = torch.arange(actual_dur, dtype=out.dtype, device=out.device)
        tau = t / float(actual_dur - 1)
        sig = torch.sigmoid(k * (tau - 0.5))
        sig_normalised = (sig - sigma_min) / sigma_span
        M = ramp_rate * (actual_dur - 1)
        episode_vals = base_val + sign * M * sig_normalised

        out[start:end, col_idx] = episode_vals
        labels[start:end] = 1

    return out, labels


def inject_test_noisy_ramp(
    tensor: Tensor,
    col_idx: int,
    seed: int,
    ramp_rate: float = 0.003,
    noise_std: float | None = None,
    min_dur: int = 60,
    max_dur: int = 375,
    num_attacks: int = 20,
) -> Tuple[Tensor, Tensor]:
    """Inject ramp attacks camouflaged with additive Gaussian noise.

    The attacker overlays a linear ramp with i.i.d. Gaussian noise to make
    instantaneous changes indistinguishable from measurement noise.  The
    physics violation (cumulative drift) still accumulates.

    If ``noise_std`` is None, it defaults to the 1 Hz change-std of the
    target column (std of consecutive differences), which is the natural
    measurement noise level.

    Returns (corrupted_tensor, binary_labels).
    """
    out = tensor.clone()
    T = out.shape[0]
    gen = torch.Generator(device=out.device).manual_seed(seed)
    clean_col = out[:, col_idx].clone()
    labels = torch.zeros(T, dtype=torch.long, device=out.device)

    # Default noise: 1 Hz change-std of the target column
    if noise_std is None:
        diffs = clean_col[1:] - clean_col[:-1]
        noise_std = float(diffs.std().item())

    episodes = _place_episodes(T, num_attacks, min_dur, max_dur, gen)

    for i, (start, end) in enumerate(episodes):
        actual_dur = end - start
        sign = 1.0 if i % 2 == 0 else -1.0
        base_val = clean_col[start].item()

        t = torch.arange(actual_dur, dtype=out.dtype, device=out.device)
        ramp = sign * ramp_rate * t
        noise = (
            torch.randn(actual_dur, generator=gen, dtype=out.dtype, device=out.device)
            * noise_std
        )
        episode_vals = base_val + ramp + noise

        out[start:end, col_idx] = episode_vals
        labels[start:end] = 1

    return out, labels


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FAULT_REGISTRY = {
    "gaussian": gaussian_noise,
    "stuck_at": stuck_at,
    "ramp": ramp_poison,
    "ramp_consistent_sin": ramp_poison_consistent_sinusoidal,
    "ramp_flow_consistent": ramp_poison_flow_consistent,
}

# Multistage faults take a pool of target columns instead of a single col_idx
_MULTISTAGE_FAULTS = {
    "ramp_multistage": ramp_poison_multistage,  # expects target_cols in kwargs
}


def apply_sensor_fault(
    tensor: Tensor,
    col_idx: int,
    fault_type: str,
    fault_param: float,
    seed: int,
    **kwargs,
) -> Tensor:
    """Clone *tensor*, apply the named fault in-place on the clone, return it.

    Single-stage faults use ``col_idx`` as the target column. Multistage
    faults (``ramp_multistage``) ignore ``col_idx`` and expect
    ``target_cols=[...]`` in kwargs.

    Parameters
    ----------
    tensor : (T, F) raw block tensor.
    col_idx : which column (sensor) to corrupt (ignored for multistage).
    fault_type : key into ``_FAULT_REGISTRY`` or ``_MULTISTAGE_FAULTS``.
    fault_param : interpretation depends on *fault_type*
        - ``"gaussian"`` → ``std_frac``
        - ``"stuck_at"`` → ``stuck_frac``
        - ``"ramp"`` / ``"ramp_multistage"`` → ``poison_frac``
    seed : RNG seed (each block should use a different seed).
    **kwargs : forwarded to the fault function (e.g. ``ramp_rate``, or
        ``target_cols`` for multistage).
    """
    out = tensor.clone()
    if fault_type in _MULTISTAGE_FAULTS:
        target_cols = kwargs.pop("target_cols")
        _MULTISTAGE_FAULTS[fault_type](out, target_cols, fault_param, seed, **kwargs)
        return out
    if fault_type not in _FAULT_REGISTRY:
        raise ValueError(
            f"Unknown fault type {fault_type!r}. "
            f"Available: {sorted(list(_FAULT_REGISTRY) + list(_MULTISTAGE_FAULTS))}"
        )
    _FAULT_REGISTRY[fault_type](out, col_idx, fault_param, seed, **kwargs)
    return out
