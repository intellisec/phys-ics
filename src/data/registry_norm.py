"""Variable registry (name<->index, groups) and z-score normalizer.

Continuous channels are standardized; categorical actuator/mask channels pass
through unchanged so their integer states stay intact.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict

import torch
import torch.nn as nn


@dataclass
class VariableMeta:
    name: str
    index: int
    unit: str = ""
    group: Optional[str] = None  # "level", "flow", "pressure", "actuator", ...
    vtype: str = "continuous"  # "continuous", "categorical", ...
    categories: Optional[Sequence[int]] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    state_map: Optional[Dict[int, str]] = None


class VariableRegistry:
    """
    Maps variable names to column indices and holds light metadata.
    Provides convenience accessors into batch tensors.
    """

    def __init__(self, variables: Sequence[VariableMeta]):
        self._vars = {v.name: v for v in variables}
        self._by_index = {v.index: v for v in variables}
        self.num_features = len(variables)

    def index(self, name: str) -> int:
        return self._vars[name].index

    def get_last_step(self, batch_x: torch.Tensor, name: str) -> torch.Tensor:
        idx = self.index(name)
        return batch_x[:, -1, idx : idx + 1]


def infer_group_from_name(name: str) -> Optional[str]:
    """
    Basic naming conventions:
      LIT* -> level
      FIT* -> flow
      (D)PIT* -> pressure
      AIT* -> other
      MV*  -> actuator
      P*   -> actuator (pump states), but skip PIT via earlier check
    Returns None if no match.
    """
    if name.startswith("LIT"):
        return "level"
    if name.startswith("FIT"):
        return "flow"
    if name.startswith("PIT"):
        return "pressure"
    if name.startswith("DPIT"):
        return "diff-pressure"
    if name.startswith("AIT"):
        return "AIT"
    if name.startswith("MV"):
        return "actuator"
    if name.startswith("UV"):
        return "actuator"
    if name.startswith("P") and not name.startswith("PIT"):
        return "actuator"
    if "MASK" in name or "MODE" in name:
        return "aux"
    return None


def build_registry_from_columns(columns: Sequence[str]) -> VariableRegistry:
    variables = []
    for i, c in enumerate(columns):
        g = infer_group_from_name(c)
        if g == "actuator":
            vtype = "categorical"
        elif g == "aux":
            vtype = "aux"
        else:
            vtype = "continuous"
        variables.append(VariableMeta(name=c, index=i, group=g, vtype=vtype))
    return VariableRegistry(variables)


class Normalizer(nn.Module):
    """
    Stores mean/std per feature (broadcastable). If a group is flagged no-normalize,
    mean=0 and std=1 so normalized==physical.
    """

    def __init__(self, means: torch.Tensor, stds: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", means.clone().view(1, 1, -1))
        self.register_buffer("std", stds.clone().clamp_min(eps).view(1, 1, -1))
        self.eps = eps

    def forward(self, x_phys: torch.Tensor) -> torch.Tensor:
        assert x_phys.dim() == 3
        return (x_phys - self.mean) / self.std

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.std + self.mean


def build_normalizer(
    train_phys: torch.Tensor,  # (N, num_features) in SI
    registry: VariableRegistry,
    no_norm_groups: Sequence[str] = ("actuator", "aux"),
    eps: float = 1e-8,
) -> Normalizer:
    """Compute per-sensor (per-column) mean/std on the training rows.

    Groups listed in `no_norm_groups` (actuators, masks) get identity scaling
    (mean=0, std=1) so they pass through unchanged.
    """
    device = train_phys.device
    dtype = train_phys.dtype
    num_features = train_phys.shape[1]

    means = torch.zeros(num_features, dtype=dtype, device=device)
    stds = torch.ones(num_features, dtype=dtype, device=device)

    # indices per group (for quick lookup)
    groups: Dict[str, List[int]] = {}
    for v in registry._vars.values():
        if v.group is not None:
            groups.setdefault(v.group, []).append(v.index)

    for v in registry._vars.values():
        idx = v.index
        g = v.group

        # 1) groups that MUST stay un‑normalised
        if g in no_norm_groups:
            means[idx] = 0.0
            stds[idx] = 1.0
            continue

        # 2) default: per‑sensor stats
        col = train_phys[:, idx]
        means[idx] = col.mean()
        stds[idx] = col.std(unbiased=False).clamp_min(eps)

    return Normalizer(means, stds, eps=eps)
