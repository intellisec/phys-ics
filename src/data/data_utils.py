"""Data helpers: parquet loading, normalizer fitting, and physics calibration —
the mass-balance residual (SI units), the static bias `b`, and the residual scales `s_k`."""

import torch
import numpy as np
import pandas as pd
from typing import Sequence, Dict, List, Union
from data.registry_norm import build_normalizer


TANK_AREA = 1.496
INV_A = 1.0 / TANK_AREA


# ====== data helpers ======
def load_parquet(path: str, device: torch.device):
    df = pd.read_parquet(path)
    cols = list(df.columns)
    data_phys = torch.tensor(df.values, dtype=torch.float32, device="cpu")
    return df, cols, data_phys


def select_targets(registry):
    # only continuous and categorical are predicted/compared
    target_indices = sorted(
        v.index for v in registry._vars.values() if v.vtype != "aux"
    )
    target_cols = [registry._by_index[i].name for i in target_indices]
    return target_indices, target_cols


def fit_normalizer(train_rows_tensor: torch.Tensor, registry, device):
    # normalize continuous sensors
    normalizer = build_normalizer(
        train_rows_tensor, registry, no_norm_groups=("actuator", "aux"), eps=1e-8
    )
    return normalizer.to(device)


# ====== bias fitting ======


def _balance_residual(
    df: pd.DataFrame,
    level: Union[str, Sequence[str]],
    inflows: Sequence[str],
    outflows: Sequence[str],
    dt: float,
    area: float,
) -> np.ndarray:
    if isinstance(level, (list, tuple)):
        dhdt = None
        for lvl in level:
            h_i = df[lvl].to_numpy()
            dh_i = (h_i[1:] - h_i[:-1]) / dt
            dhdt = dh_i if dhdt is None else (dhdt + dh_i)
    else:
        h = df[level].to_numpy()
        dhdt = (h[1:] - h[:-1]) / dt
    qin = df[list(inflows)].to_numpy().sum(1) if len(inflows) else 0.0
    qout = df[list(outflows)].to_numpy().sum(1) if len(outflows) else 0.0
    rhs = (qin if isinstance(qin, float) else qin[:-1]) / area - (
        qout if isinstance(qout, float) else qout[:-1]
    ) / area
    return dhdt - rhs


def fit_bias_terms(
    df_train: pd.DataFrame,
    level: Union[str, Sequence[str]],
    inflows: Sequence[str],
    outflows: Sequence[str],
    dt: float,
    area: float,
    params: List[Dict],
) -> Dict[str, float]:
    r = _balance_residual(df_train, level, inflows, outflows, dt, area)
    Xcols = []
    for p in params:
        if p.get("mask_col") is None:
            m = np.ones(len(r), dtype=float)
        else:
            m_full = df_train[p["mask_col"]].to_numpy().astype(float)
            m = m_full[:-1]
        Xcols.append(p["coeff"] * m)
    if not Xcols:
        return {}
    X = np.stack(Xcols, axis=1)
    ok = np.isfinite(r) & np.all(np.isfinite(X), axis=1)
    theta, *_ = np.linalg.lstsq(X[ok], r[ok], rcond=None)
    return {p["name"]: float(theta[i]) for i, p in enumerate(params)}


def _fit_bias_terms_median(
    df_train: pd.DataFrame,
    level: Union[str, Sequence[str]],
    inflows: Sequence[str],
    outflows: Sequence[str],
    dt: float,
    area: float,
    params: List[Dict],
) -> Dict[str, float]:
    """Like fit_bias_terms but uses median instead of lstsq — robust to outliers."""
    r = _balance_residual(df_train, level, inflows, outflows, dt, area)
    result = {}
    for p in params:
        coeff = p["coeff"]
        if p.get("mask_col") is None:
            mask = np.ones(len(r), dtype=bool)
        else:
            m_full = df_train[p["mask_col"]].to_numpy().astype(float)
            mask = m_full[:-1] > 0.5
        ok = mask & np.isfinite(r)
        if ok.sum() == 0:
            result[p["name"]] = 0.0
        else:
            result[p["name"]] = float(np.median(r[ok]) / coeff)
    return result


def _algebraic_residual(df: pd.DataFrame, coeffs: Dict[str, float]) -> np.ndarray:
    """Compute residual of a linear combination of columns: r = sum(coeff_i * col_i)."""
    cols = list(coeffs.keys())
    M = df[cols].to_numpy(dtype=float)
    c = np.array([coeffs[k] for k in cols], dtype=float)
    return M @ c


def fit_algebraic_median_bias(
    df_train: pd.DataFrame, coeffs: Dict[str, float], mask_col: str | None = None
) -> float:
    """Fit mean bias for an algebraic balance: mean(sum(coeff_i * col_i))."""
    r = _algebraic_residual(df_train, coeffs)
    if mask_col and mask_col in df_train.columns:
        r = r[df_train[mask_col].to_numpy().astype(float) > 0.5]
    r = r[np.isfinite(r)]
    return float(np.median(r)) if r.size else 0.0


def zero_biases(fitted: dict) -> dict:
    """Return a copy of a fitted-biases dict with every float leaf set to 0.0.

    Preserves nested structure so downstream consumers (e.g. build_objectives)
    see the same shape as fit_all_biases would produce.
    """
    out = {}
    for k, v in fitted.items():
        if isinstance(v, dict):
            out[k] = zero_biases(v)
        else:
            out[k] = 0.0
    return out


def fit_all_biases(
    df_train: pd.DataFrame,
    dt: float = 1.0,
    full_swat: bool = False,
) -> dict:
    """Fit physics biases as median residuals on the training data.

    full_swat=True: fit biases for LIT101, LIT301, LIT401, global balance,
    and algebraic flow splits (FIT501).
    """
    fit_LIT101 = _fit_bias_terms_median(
        df_train,
        "LIT101",
        inflows=["FIT101"],
        outflows=["FIT201"],
        dt=dt,
        area=TANK_AREA,
        params=[{"name": "biasQ101", "coeff": -INV_A, "mask_col": None}],
    )

    if not full_swat:
        return {"fit_LIT101": fit_LIT101}

    # --- Full SWaT: additional tank balances + flow splits ---

    # LIT301: dLIT301/dt = (FIT201 - FIT301) / A + FITchemP2
    fit_LIT301 = _fit_bias_terms_median(
        df_train,
        "LIT301",
        inflows=["FIT201"],
        outflows=["FIT301"],
        dt=dt,
        area=TANK_AREA,
        params=[{"name": "FITchemP2", "coeff": -INV_A, "mask_col": None}],
    )

    # LIT401: dLIT401/dt = (FIT301 - FIT401) / A + biasQ401 + FITreject_param * UF_REJECT_MASK
    fit_LIT401 = _fit_bias_terms_median(
        df_train,
        "LIT401",
        inflows=["FIT301"],
        outflows=["FIT401"],
        dt=dt,
        area=TANK_AREA,
        params=[
            {"name": "biasQ401", "coeff": -INV_A, "mask_col": None},
            {"name": "FITreject_param", "coeff": -INV_A, "mask_col": "UF_REJECT_MASK"},
        ],
    )

    # Global balance: d(LIT101+LIT301+LIT401)/dt = (FIT101 - FIT501) / A + bias + masked*MASK
    fit_GLOBAL = _fit_bias_terms_median(
        df_train,
        ["LIT101", "LIT301", "LIT401"],
        inflows=["FIT101"],
        outflows=["FIT501"],
        dt=dt,
        area=TANK_AREA,
        params=[
            {"name": "bias_global", "coeff": -INV_A, "mask_col": None},
            {"name": "masked_global", "coeff": -INV_A, "mask_col": "UF_REJECT_MASK"},
        ],
    )

    # Algebraic flow splits
    mean_split1 = fit_algebraic_median_bias(df_train, {"FIT501": +1.0, "FIT401": -1.0})
    mean_split2 = fit_algebraic_median_bias(
        df_train, {"FIT501": +1.0, "FIT502": -1.0, "FIT503": -1.0, "FIT504": +1.0}
    )

    return {
        "fit_LIT101": fit_LIT101,
        "fit_LIT301": fit_LIT301,
        "fit_LIT401": fit_LIT401,
        "global": fit_GLOBAL,
        "mean_split1": mean_split1,
        "mean_split2": mean_split2,
    }


# ====== per-equation scaling ======
@torch.no_grad()
def fit_scales_on_target_baseline(
    loader,
    objective_set_phys,
    registry,
    normalizer,
    device,
    target_indices: List[int],  # pass the SAME list used to build y_*_seq
    floor: float = 1e-8,
    print_summary: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
     - compare step-1 physics residuals for two constructed
        predictions (in SI units) and summarize per-objective stats.
     - set each objective's scale to the RMS of the
        raw residual under the 'target' baseline (with a floor), and optionally print a summary.

    Returns:
      res: Dict[str, Dict[str, float]] — per-objective metrics plus a '_step_change_' context block.
    """
    # freeze ObjectiveSet updates during this pass
    was_training = objective_set_phys.training
    objective_set_phys.eval()

    # prepare index tensor for target columns
    tgt = torch.as_tensor(sorted(target_indices), dtype=torch.long, device=device)

    # accumulators
    sums_raw_t, sums_raw_c = {}, {}
    sums_scl_t, sums_scl_c = {}, {}
    counts = {}
    step_sq_phys, step_sq_z, step_cnt = 0.0, 0.0, 0

    # expose per-objective 'scale' in output if available
    obj_by_name = {
        obj.name: obj for obj in getattr(objective_set_phys, "objectives", [])
    }

    # iterate over loader
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}

        # last observed (k) and true next (k+1) — y1_phys_tgt packed in target_indices order
        x_last_phys = b["x_phys"][:, -1, :]  # (B, F)
        y1_phys_tgt = b["y_phys"]

        # build physical predictions
        preds_target_phys = x_last_phys.clone()  # (B, F)
        preds_target_phys[:, tgt] = (
            y1_phys_tgt  # replace exactly target cols in full-F space
        )
        preds_copied_phys = x_last_phys  # pure copy-last

        # convert to z to satisfy ObjectiveSet API
        preds_target_norm = normalizer(preds_target_phys.unsqueeze(1)).squeeze(1)
        preds_copied_norm = normalizer(preds_copied_phys.unsqueeze(1)).squeeze(1)

        # build full-F next-step targets to match preds_* shapes
        y1_phys_full = x_last_phys.clone()  # (B, F_full)
        y1_phys_full[:, tgt] = y1_phys_tgt  # insert true next-step targets
        y1_norm_full = normalizer(y1_phys_full.unsqueeze(1)).squeeze(1)

        # expose single-step targets so DataObjective can use its simple residual
        b["y_phys"] = y1_phys_full  # (B, F_full)
        b["y_norm"] = y1_norm_full  # (B, F_full)

        # evaluate physics (raw residuals are in physical units)
        _, m_t = objective_set_phys(b, preds_target_norm, registry, normalizer)
        _, m_c = objective_set_phys(b, preds_copied_norm, registry, normalizer)

        for name, mt in m_t.items():
            mc = m_c[name]
            rT, sT = mt["raw_resid"], mt["scaled_resid"]
            rC, sC = mc["raw_resid"], mc["scaled_resid"]

            sums_raw_t[name] = sums_raw_t.get(name, 0.0) + float((rT**2).sum().item())
            sums_raw_c[name] = sums_raw_c.get(name, 0.0) + float((rC**2).sum().item())
            sums_scl_t[name] = sums_scl_t.get(name, 0.0) + float((sT**2).sum().item())
            sums_scl_c[name] = sums_scl_c.get(name, 0.0) + float((sC**2).sum().item())
            counts[name] = counts.get(name, 0) + rT.numel()

        # step-change context on target (continuous) dims ONLY
        std_full = normalizer.std.view(-1).to(device)
        std_safe = torch.where(std_full == 0, torch.ones_like(std_full), std_full)
        d_phys = y1_phys_tgt - x_last_phys[:, tgt]  # (B, F_out)
        d_z = d_phys / std_safe[tgt][None, :]

        step_sq_phys += float((d_phys**2).sum().item())
        step_sq_z += float((d_z**2).sum().item())
        step_cnt += d_phys.numel()

    # summarize metrics
    res: Dict[str, Dict[str, float]] = {}
    eps = 1e-12
    for name, cnt in counts.items():
        raw_T = (sums_raw_t[name] / cnt) ** 0.5
        raw_C = (sums_raw_c[name] / cnt) ** 0.5
        scl_T = (sums_scl_t[name] / cnt) ** 0.5
        scl_C = (sums_scl_c[name] / cnt) ** 0.5

        scale_val = None
        if name in obj_by_name and hasattr(obj_by_name[name], "scale"):
            try:
                scale_val = float(obj_by_name[name].scale.item())
            except Exception:
                scale_val = None

        res[name] = {
            "raw_rms_target": raw_T,
            "raw_rms_copied": raw_C,
            "scaled_rms_target": scl_T,
            "scaled_rms_copied": scl_C,
            "copied_over_target_raw": raw_C / max(raw_T, eps),
            "copied_over_target_scaled": scl_C / max(scl_T, eps),
            "n": int(cnt),
            **({"scale": scale_val} if scale_val is not None else {}),
        }

    res["_step_change_"] = {
        "rms_phys": (step_sq_phys / max(step_cnt, 1)) ** 0.5,
        "rms_z": (step_sq_z / max(step_cnt, 1)) ** 0.5,
    }

    # initialize objective scales from target-baseline RMS (with floor)
    for obj in getattr(objective_set_phys, "objectives", []):
        name = getattr(obj, "name", None)
        if not name or name == "_step_change_":
            continue
        s = max(res[name]["raw_rms_target"], floor)
        obj.update_scale(torch.tensor(s, device=obj._scale_buf.device))

    # Sampling-rate diagnostic: per equation, S_phys = copy-last residual (the
    # physical change the equation should capture), N_resid = true-next residual
    # (the noise floor), gamma = S_phys / N_resid. Prefer the fastest Δt whose
    # gamma > 1, i.e. physical signal above the noise floor.
    if print_summary:
        rows = []
        for name, d in res.items():
            if name == "_step_change_":
                continue
            gamma = d["raw_rms_copied"] / max(d["raw_rms_target"], 1e-12)
            rows.append((name, d["raw_rms_copied"], d["raw_rms_target"], gamma, d["n"]))
        rows.sort(key=lambda x: x[3], reverse=True)
        print("Objective\tS_phys\tN_resid\tgamma=S/N\t(n)")
        for name, s_phys, n_resid, gamma, cnt in rows:
            print(f"{name}\t{s_phys:.3g}\t{n_resid:.3g}\t{gamma:.2f}\t({cnt})")
    # restore original training state
    if was_training:
        objective_set_phys.train()

    return res
