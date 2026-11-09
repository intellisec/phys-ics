"""Assembles the physics ObjectiveSet: the tank mass-balance equations (plus
flow-split and global balances in the full-SWaT setup) with their fitted bias terms."""

import torch
import torch.nn as nn
from physics.objectives import (
    ObjectiveSet,
    make_tank_balance,
    make_algebraic_balance,
)
from data.data_utils import TANK_AREA, INV_A


def build_objectives(
    fitted: dict,
    scale_mode="online",
    scale_exclude=(),
    dt: float = 1.0,
    full_swat: bool = False,
    global_obj: bool = False,
):
    """Build physics ObjectiveSet.

    full_swat=False (default): single dLIT101 tank balance (PLC1 only).
    full_swat=True: dLIT101 + dLIT301 + dLIT401 + FIT501 flow splits.
        Optionally global balance via global_obj=True.
    """

    # --- LIT101 ---
    eq_LIT101 = make_tank_balance(
        name="dLIT101",
        level="LIT101",
        inflows=["FIT101"],
        outflows=["FIT201"],
        dt=dt,
        area=TANK_AREA,
        unknown_in_params={"biasQ101": fitted["fit_LIT101"]["biasQ101"]},
    )

    if not full_swat:
        return ObjectiveSet(
            [eq_LIT101], scale_mode=scale_mode, scale_exclude=scale_exclude
        )

    # --- Full SWaT: additional objectives ---

    # LIT301: dLIT301/dt = (FIT201 - FIT301) / A + FITchemP2
    eq_LIT301 = make_tank_balance(
        name="dLIT301",
        level="LIT301",
        inflows=["FIT201"],
        outflows=["FIT301"],
        dt=dt,
        area=TANK_AREA,
        unknown_in_params={"FITchemP2": fitted["fit_LIT301"]["FITchemP2"]},
    )

    # LIT401: dLIT401/dt = (FIT301 - FIT401) / A + biasQ401 + FITreject_param * UF_REJECT_MASK
    eq_LIT401 = make_tank_balance(
        name="dLIT401",
        level="LIT401",
        inflows=["FIT301"],
        outflows=["FIT401"],
        dt=dt,
        area=TANK_AREA,
        unknown_out_params={"biasQ401": fitted["fit_LIT401"]["biasQ401"]},
    )
    # Add UF_REJECT_MASK-gated bias term for reject flow
    eq_LIT401.param_dict["FITreject_param"] = nn.Parameter(
        torch.tensor(float(fitted["fit_LIT401"]["FITreject_param"]))
    )
    eq_LIT401.param_coeffs["FITreject_param"] = (+INV_A, "UF_REJECT_MASK", "obs_last")

    # FIT501 flow split 1: FIT501 ≈ FIT401 + bias
    eq_FIT501_split1 = make_algebraic_balance(
        name="FIT501_split1",
        coeffs={"FIT501": +1.0, "FIT401": -1.0},
        params={"bias_FIT501_split1": (fitted["mean_split1"], -1.0)},
        weight=1.0,
    )

    # FIT501 flow split 2: FIT501 ≈ FIT502 + FIT503 - FIT504 + bias
    eq_FIT501_split2 = make_algebraic_balance(
        name="FIT501_split2",
        coeffs={"FIT501": +1.0, "FIT502": -1.0, "FIT503": -1.0, "FIT504": +1.0},
        params={"bias_FIT501_split2": (fitted["mean_split2"], -1.0)},
        weight=1.0,
    )

    objectives = [eq_LIT101, eq_LIT301, eq_LIT401, eq_FIT501_split1, eq_FIT501_split2]

    # Optional global balance: d(LIT101+LIT301+LIT401)/dt = (FIT101 - FIT501) / A + biases
    if global_obj and "global" in fitted:
        eq_GLOBAL = make_tank_balance(
            name="global_LITs",
            level=["LIT101", "LIT301", "LIT401"],
            inflows=["FIT101"],
            outflows=["FIT501"],
            dt=dt,
            area=TANK_AREA,
            weight=1.0,
        )
        # Add bias terms with UF_REJECT_MASK
        bias_val = float(fitted["global"]["bias_global"])
        masked_val = float(fitted["global"]["masked_global"])

        eq_GLOBAL.param_dict["bias_global"] = nn.Parameter(torch.tensor(bias_val))
        eq_GLOBAL.param_coeffs["bias_global"] = +INV_A

        eq_GLOBAL.param_dict["masked_global"] = nn.Parameter(torch.tensor(masked_val))
        eq_GLOBAL.param_coeffs["masked_global"] = (+INV_A, "UF_REJECT_MASK", "obs_last")

        objectives.append(eq_GLOBAL)

    return ObjectiveSet(objectives, scale_mode=scale_mode, scale_exclude=scale_exclude)
