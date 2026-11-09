"""Physics objectives: turn a next-step prediction into a physics residual (e.g.
tank mass balance), rescale it per equation, and square it into a loss. ObjectiveSet
aggregates the equation losses; the data loss is computed elsewhere."""

from typing import Optional, Sequence, Tuple, Mapping, Dict, List, Union

import torch
import torch.nn as nn
from data.registry_norm import VariableRegistry, Normalizer


def compute_rms(x: torch.Tensor) -> torch.Tensor:
    """Root-mean-square, used to scale each equation's residual to unit magnitude."""
    return torch.sqrt(torch.mean(x**2) + 1e-12)


def _preds_norm_to_phys(
    preds_norm: torch.Tensor, normalizer: "Normalizer"
) -> torch.Tensor:
    if preds_norm.ndim == 2:
        return normalizer.inverse(preds_norm.unsqueeze(1)).squeeze(1)
    if preds_norm.ndim == 3:
        return normalizer.inverse(preds_norm)
    raise ValueError(
        f"Expected preds_norm with ndim 2 or 3, got shape {tuple(preds_norm.shape)}"
    )


class BaseObjective(nn.Module):
    """
    Base class for objectives.
    Subclasses implement _compute_raw_residual(...) returning r (B,1) or (B,K) in SI (or chosen) units.
    Scaling and weighting handled here.
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        scale: Optional[float] = None,
        auto_scale: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.name = name
        self.weight = float(weight)
        self.auto_scale = bool(auto_scale)
        self.reduction = reduction
        self._scale_frozen = scale is not None
        init_scale = 1.0 if scale is None else float(scale)
        self.register_buffer("_scale_buf", torch.tensor(init_scale))

    def extra_repr(self) -> str:
        return f"name={self.name}, weight={self.weight}, scale={self._scale_buf.item():.3g}, frozen={self._scale_frozen}"

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return torch.mean(x)
        if self.reduction == "sum":
            return torch.sum(x)
        raise ValueError(self.reduction)

    def _compute_raw_residual(
        self,
        batch: Mapping[str, torch.Tensor],
        preds_norm: torch.Tensor,
        preds_phys: torch.Tensor,
        registry: VariableRegistry,
        normalizer: Normalizer,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def update_scale(self, r: torch.Tensor):
        if self._scale_frozen or not self.auto_scale:
            return
        rms = compute_rms(r.detach())
        if torch.isfinite(rms):
            self._scale_buf.copy_(rms.clamp_min(1e-12))

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        preds_norm: torch.Tensor,
        registry: VariableRegistry,
        normalizer: Normalizer,
    ) -> Dict[str, torch.Tensor]:
        preds_phys = _preds_norm_to_phys(preds_norm, normalizer)
        r = self._compute_raw_residual(
            batch, preds_norm, preds_phys, registry, normalizer
        )
        s = self._scale_buf.detach().clone()
        r_scaled = r / s
        loss_base = self._reduce(r_scaled**2)
        loss = loss_base * self.weight

        return {
            "loss": loss,
            "raw_resid": r.detach(),
            "scaled_resid": r_scaled.detach(),
            "scale": s.detach().clone(),
        }


class EquationObjective(BaseObjective):
    """
    General linear equation: r = sum_i coeff_i * value_i.

    terms: list of (var_name, coeff, source)
        source='pred' uses predicted next-step value
        source='obs_last' uses last observed step from batch['x_phys']

    params: dict pname -> (init_val, coeff) for learnable scalars
    """

    def __init__(
        self,
        name: str,
        terms: Sequence[Tuple[str, float, str]],
        params: Optional[
            Mapping[str, Tuple[float, Union[float, Tuple, List], bool]]
        ] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.terms = list(terms)
        self.param_dict = nn.ParameterDict()
        self.param_coeffs = {}
        if params:
            for pname, (init_val, coeff) in params.items():
                self.param_dict[pname] = nn.Parameter(torch.tensor(float(init_val)))
                self.param_coeffs[pname] = float(coeff)

    def _get_var(self, batch, preds_phys, registry, name, source):
        if source == "pred":
            idx = registry.index(name)
            return preds_phys[:, idx : idx + 1]
        if source == "obs_last":
            return registry.get_last_step(batch["x_phys"], name)
        raise ValueError(source)

    def _compute_raw_residual(
        self, batch, preds_norm, preds_phys, registry, normalizer
    ):
        B = preds_phys.size(0)
        device = preds_phys.device
        dtype = preds_phys.dtype
        r = torch.zeros(B, 1, device=device, dtype=dtype)
        for var_name, coeff, source in self.terms:
            v = self._get_var(batch, preds_phys, registry, var_name, source)
            r = r + coeff * v
        if len(self.param_dict) > 0:
            for pname, p in self.param_dict.items():
                coeff = self.param_coeffs[pname]
                if isinstance(coeff, tuple):
                    if len(coeff) == 2:
                        coeff_val, mask_name = coeff
                        source = "pred"  # default
                    elif len(coeff) == 3:
                        coeff_val, mask_name, source = coeff
                    else:
                        raise ValueError(f"param_coeffs[{pname}] expects 2 or 3 items.")

                    mask = self._get_var(
                        batch, preds_phys, registry, mask_name, source=source
                    )
                    r = r + coeff_val * p * mask

                else:  # unchanged path
                    r = r + coeff * p
        return r


class ObjectiveSet(nn.Module):
    """Container that aggregates the per-equation objective losses.

    ``set_scale_mode("online"|"frozen")`` controls whether residual scales are
    refreshed online (EMA) during training; runs freeze them after the one-time
    scale calibration.
    """

    def __init__(
        self,
        objectives: Sequence[BaseObjective],
        *,
        scale_mode: str = "online",  # "online" | "frozen"
        scale_exclude: Sequence[str] = (),
        online_alpha: float = 0.01,
        online_min_samples: int = 1024,
    ):
        super().__init__()
        self.objectives = nn.ModuleList(objectives)
        self.scale_mode = scale_mode
        self.scale_exclude = set(scale_exclude or ())
        self.online_alpha = float(online_alpha)
        self.online_min_samples = int(online_min_samples)
        self._online_states: Dict[str, Dict[str, torch.Tensor | int]] = {}

    def set_scale_mode(self, mode: str):
        if mode not in ("online", "frozen"):
            raise ValueError(mode)
        self.scale_mode = mode

    @torch.no_grad()
    def _maybe_update_scale(self, obj: BaseObjective, resid: torch.Tensor):
        if (
            (not self.training)
            or (obj.name in self.scale_exclude)
            or (self.scale_mode == "frozen")
        ):
            return
        state = self._online_states.get(obj.name)
        if state is None:
            state = {"n": 0, "ema": obj._scale_buf.detach().clone()}
        state["n"] += resid.numel()
        rms_now = torch.sqrt(torch.mean(resid**2) + 1e-12)
        if state["n"] < self.online_min_samples:
            obj.update_scale(rms_now)
        else:
            state["ema"] = (1 - self.online_alpha) * state[
                "ema"
            ] + self.online_alpha * rms_now
            obj.update_scale(state["ema"])
        self._online_states[obj.name] = state

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        preds_norm: torch.Tensor,
        registry: "VariableRegistry",
        normalizer: "Normalizer",
    ) -> Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
        metrics = {}
        total = torch.zeros((), device=preds_norm.device)
        for obj in self.objectives:
            out = obj(batch, preds_norm, registry, normalizer)
            metrics[obj.name] = out
            total = total + out["loss"]
            self._maybe_update_scale(obj, out["raw_resid"])
        return total, metrics


def make_tank_balance(
    name: str,
    level: Union[str, Sequence[str]],
    inflows: Sequence[str],
    outflows: Sequence[str],
    dt: float,
    area: float,
    unknown_in_params: Optional[Mapping[str, float]] = None,  # pname->init_val
    unknown_out_params: Optional[Mapping[str, float]] = None,  # pname->init_val
    extra_params: Optional[
        Mapping[str, Tuple[float, Union[float, Tuple[float, str, str]]]]
    ] = None,
    weight: float = 1.0,
    **kwargs,
) -> EquationObjective:
    inv_dt = 1.0 / dt
    inv_A = 1.0 / area
    levels: List[str] = [level] if isinstance(level, str) else list(level)

    terms: List[Tuple[str, float, str]] = []
    for lvl in levels:
        terms.append((lvl, +inv_dt, "pred"))
        terms.append((lvl, -inv_dt, "obs_last"))
    for f in inflows:
        terms.append((f, -inv_A, "pred"))
    for f in outflows:
        terms.append((f, +inv_A, "pred"))

    params: Dict[str, Tuple[float, Union[float, Tuple[float, str, str]]]] = {}
    if unknown_in_params:
        for pname, init_val in unknown_in_params.items():
            params[pname] = (init_val, -inv_A)
    if unknown_out_params:
        for pname, init_val in unknown_out_params.items():
            params[pname] = (init_val, +inv_A)
    if extra_params:
        params.update(extra_params)

    return EquationObjective(
        name=name, terms=terms, params=params, weight=weight, **kwargs
    )


def make_algebraic_balance(
    name: str,
    coeffs: Mapping[str, float],
    params: Optional[Mapping[str, Tuple[float, float]]] = None,
    source: str = "pred",
    weight: float = 1.0,
    **kwargs,
) -> EquationObjective:
    """Linear combination of predicted sensor values equals zero (plus bias params).

    coeffs: {sensor_name: coefficient} — all terms use the same source.
    params: {pname: (init_val, coeff)} — learnable scalar biases.
    """
    terms = [(v, float(c), source) for v, c in coeffs.items()]
    return EquationObjective(
        name=name, terms=terms, params=params, weight=weight, **kwargs
    )
