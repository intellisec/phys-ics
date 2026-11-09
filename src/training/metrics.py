"""Training and evaluation metrics: the per-batch data + physics loss used by the
training loop, and per-sensor / per-equation residual errors used for model
selection and diagnostics."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def compute_batch_metrics(
    model,
    batch,
    objective_set,
    registry,
    normalizer,
    device,
    target_indices,
):
    b = {k: v.to(device) for k, v in batch.items()}
    preds = model(b["x_norm"])

    assert preds.ndim == 2
    y = b["y_norm"]
    preds_tgt = preds[:, target_indices]
    y_tgt = y if y.shape[-1] == len(target_indices) else y[:, target_indices]
    loss_data = F.mse_loss(preds_tgt, y_tgt)

    _, metrics = objective_set(b, preds, registry, normalizer)

    total = None
    for name, d in metrics.items():
        if name in ("total", "data"):
            continue
        if "loss" in d:
            total = d["loss"] if total is None else total + d["loss"]
    loss_phys = total if total is not None else torch.zeros((), device=preds.device)

    return {
        "loss_data": loss_data,
        "loss_phys": loss_phys,
        "metrics_phys": metrics,
    }


# Compute metrics for model evaluation/ comparison
@torch.no_grad()
def compute_one_step_data_errors(
    model,
    registry,
    normalizer,
    test_loader,
    *,
    target_indices: Optional[List[int]] = None,
) -> Tuple[List[dict], dict]:
    """
    Per-sensor data errors on the first predicted step in normalized (z) space.

    Works with model outputs:
      - (B, F)
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    model.eval()

    if target_indices is None:
        target_indices = sorted(
            v.index for v in registry._vars.values() if v.vtype != "aux"
        )

    sse: Dict[int, float] = {i: 0.0 for i in target_indices}
    sae: Dict[int, float] = {i: 0.0 for i in target_indices}
    nobs = 0

    def _group_of(idx: int) -> str:
        v = registry._by_index[idx]
        g = getattr(v, "group", None)
        if g:
            return str(g)
        name = v.name
        for pfx in ("LIT", "FIT", "DPIT", "PIT", "AIT"):
            if name.startswith(pfx):
                return pfx
        return "other"

    for batch in test_loader:
        b = {k: v.to(device) for k, v in batch.items()}
        preds = model(b["x_norm"])

        if preds.ndim != 2:
            raise ValueError(f"Expected preds ndim 2, got shape {tuple(preds.shape)}")

        pred_norm = (
            preds[:, target_indices]
            if preds.shape[-1] != len(target_indices)
            else preds
        )

        y_full = b["y_norm"]
        y_norm = (
            y_full
            if y_full.shape[-1] == pred_norm.shape[-1]
            else y_full[:, target_indices]
        )

        err = pred_norm - y_norm
        B = err.size(0)
        nobs += B

        for j, idx in enumerate(target_indices):
            e = err[:, j]
            sse[idx] += float(torch.nansum(e * e).item())
            sae[idx] += float(torch.nansum(e.abs()).item())

    rows_data: List[dict] = []
    rmse_pairs: List[Tuple[int, float]] = []
    per_group_acc: Dict[str, List[float]] = {}

    for i in target_indices:
        mse = sse[i] / max(1, nobs)
        rmse = math.sqrt(mse)
        mae = sae[i] / max(1, nobs)
        name = registry._by_index[i].name
        group = _group_of(i)

        rmse_pairs.append((i, rmse))
        per_group_acc.setdefault(group, []).append(rmse)

        rows_data.append(
            {
                "kind": "data",
                "name": name,
                "group": group,
                "mse_z": f"{mse:.6e}",
                "rmse_z": f"{rmse:.6e}",
                "mae_z": f"{mae:.6e}",
                "rms_si": "",
                "n": nobs,
            }
        )

    rmse_vals = [rmse for _, rmse in rmse_pairs]
    data_overall_rmse = float(
        torch.nanmean(torch.as_tensor(rmse_vals, dtype=torch.float32)).item()
    )

    cont_rmse_vals = [
        rmse
        for idx, rmse in rmse_pairs
        if registry._by_index[idx].vtype == "continuous"
    ]
    data_cont_rmse_z = float(
        torch.nanmean(torch.as_tensor(cont_rmse_vals, dtype=torch.float32)).item()
    )

    per_group = {
        g: float(torch.nanmean(torch.as_tensor(v, dtype=torch.float32)).item())
        for g, v in per_group_acc.items()
    }

    summary = {
        "data_overall_rmse_z": data_overall_rmse,
        "per_group": per_group,
        "data_cont_rmse_z": data_cont_rmse_z,
    }
    return rows_data, summary


@torch.no_grad()
def compute_one_step_physics_errors(
    model,
    objective_set_phys,
    registry,
    normalizer,
    test_loader,
) -> Tuple[List[dict], dict]:
    """
    Computes raw physics residual RMS per objective in SI units.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    model.eval()
    objective_set_phys.eval()
    if hasattr(objective_set_phys, "set_scale_mode"):
        objective_set_phys.set_scale_mode("frozen")

    sse: Dict[str, float] = {}
    cnt: Dict[str, int] = {}

    for batch in test_loader:
        b = {k: v.to(device) for k, v in batch.items()}
        preds = model(b["x_norm"])  # (B,F)

        _, metrics = objective_set_phys(b, preds, registry, normalizer)
        for name, d in metrics.items():
            r = d["raw_resid"]
            finite = torch.isfinite(r)
            sse[name] = sse.get(name, 0.0) + float(
                torch.nansum((r * r) * finite).item()
            )
            cnt[name] = cnt.get(name, 0) + int(finite.sum().item())

    def _phys_group(name: str) -> Optional[str]:
        if name.startswith("dLIT"):
            return "tank"
        if "split" in name or name.startswith("FIT"):
            return "split"
        return None

    rows_phys: List[dict] = []
    tank_vals, split_vals = [], []

    for name in sorted(sse.keys()):
        if cnt[name] <= 0:
            continue
        rms = math.sqrt(sse[name] / cnt[name])
        g = _phys_group(name) or "other"

        rows_phys.append(
            {
                "kind": "physics",
                "name": name,
                "group": g,
                "mse_z": "",
                "rmse_z": "",
                "mae_z": "",
                "rms_si": f"{rms:.6e}",
                "n": cnt[name],
            }
        )

        if g == "tank":
            tank_vals.append(rms)
        if g == "split":
            split_vals.append(rms)

    summary = {
        "tank_mean_rms_si": float(
            torch.nanmean(torch.as_tensor(tank_vals, dtype=torch.float32)).item()
        ),
        "split_mean_rms_si": float(
            torch.nanmean(torch.as_tensor(split_vals, dtype=torch.float32)).item()
        ),
    }
    return rows_phys, summary
