"""Forecaster factory: builds the backbone for a given architecture and size.

Size configs (SIZE_CFG) hold the per-architecture hyperparameters; variants are tuned to roughly equal parameter counts for a fair comparison.
``load_model_bundle`` rebuilds a trained model plus its normalizer and physics
objectives from a checkpoint for inference.
"""

from enum import Enum
from models.cnn import TemporalCNN1d
from models.lstm import TimeSeriesLSTMForecaster
from models.mlp import MLPForecaster
from models.transformer import TimeSeriesTransformerForecaster
from models.copy_baseline import CopyLastBaseline1d

import torch
from pathlib import Path
from models.wrappers import NextStepForecaster
from physics.build_objectives import build_objectives
from data.registry_norm import Normalizer
from data.data_utils import select_targets


class Size(str, Enum):
    full_swat = "full_swat"
    plc1 = "plc1"


SIZE_CFG = {
    "copy-last": {
        Size.plc1: {},
        Size.full_swat: {},
    },
    # 4-block “wide-ish” CNN (Conv1d or Conv2d)
    "cnn1d": {
        Size.full_swat: {"channels": [32, 64, 128], "fc_size": 212},
        Size.plc1: {"channels": [32, 64, 128], "fc_size": 212},  # 50,162
    },
    "mlp": {
        Size.full_swat: {"hidden": 140},
        Size.plc1: {"hidden": 205},  # 49,881
    },
    "lstm": {
        Size.full_swat: {"h1": 200, "h2": 352},
        Size.plc1: {"h1": 52, "h2": 73},  # 50,008
    },
    "transformer": {
        Size.full_swat: {"d_model": 208, "nhead": 8, "dim_ff": 416},
        Size.plc1: {"d_model": 40, "nhead": 4, "dim_ff": 120},  # 50,046
    },
}


def build_model(name, size, F_in, F_out, W, dropout):
    """Instantiate a backbone (cnn1d/lstm/mlp/transformer/copy-last) for the given size."""
    cfg = SIZE_CFG[name][Size(size)]
    match name:
        case "copy-last":
            # size and dropout are ignored, but we keep the same signature
            return CopyLastBaseline1d(
                num_features=F_in,
                outsize=F_in,
                window=W,
            )
        case "cnn1d":
            return TemporalCNN1d(
                num_features=F_in, outsize=F_out, window=W, dropout_p=dropout, **cfg
            )
        case "lstm":
            return TimeSeriesLSTMForecaster(
                in_dim=F_in,
                out_dim=F_out,
                hidden_sizes=(cfg["h1"], cfg["h2"]),
                dropout=dropout,
            )
        case "mlp":
            return MLPForecaster(
                in_dim=F_in,
                out_dim=F_out,
                hidden=cfg["hidden"],
                dropout=dropout,
                window=W,
            )
        case "transformer":
            return TimeSeriesTransformerForecaster(
                in_dim=F_in,
                out_dim=F_out,
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                dim_feedforward=cfg["dim_ff"],
                num_layers=3,
                dropout=dropout,
            )
        case _:
            raise ValueError(f"unknown model {name}")


def load_model_bundle(ckpt_path: Path, *, device: torch.device):
    """Load everything needed for inference from a training checkpoint."""

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    registry = ckpt["registry"]

    norm_state = ckpt["normalizer"]
    mean_shape = tuple(norm_state["mean"].shape)
    if len(mean_shape) != 3 or mean_shape[0] != 1 or mean_shape[1] != 1:
        raise ValueError(f"Unexpected normalizer mean shape: {mean_shape}")

    num_features = int(mean_shape[2])

    normalizer = Normalizer(
        means=torch.zeros(num_features, dtype=torch.float32),
        stds=torch.ones(num_features, dtype=torch.float32),
        eps=1e-8,
    ).to(device)
    normalizer.load_state_dict(norm_state)

    meta = ckpt["meta"]
    args = dict(meta["args"])

    window = int(meta["effective_window"])
    dt_sec = float(meta["dt_sec"])

    input_cols = list(meta["input_cols"])
    target_indices, target_cols = select_targets(registry)

    F_in_model = len(target_indices)
    F_out_model = len(target_indices)

    model_type = str(args["model"])
    size = "full_swat" if bool(args.get("full_swat", False)) else "plc1"
    dropout = float(args["dropout"])

    base = build_model(
        model_type,
        size,
        F_in_model,
        F_out_model,
        window,
        dropout,
    )

    # Absolute and delta checkpoints share a state_dict layout, so strict=True
    # below cannot detect a mismatch -- the mode has to come from meta["args"].
    # Checkpoints written before --wrapper existed default to "absolute".
    residual = str(args.get("wrapper", "absolute")) == "delta"

    if model_type in ["copy-last"]:
        model = base.to(device)
    else:
        model = NextStepForecaster(
            base,
            target_indices=target_indices,
            residual=residual,
        ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    full_swat = bool(args.get("full_swat", False))
    global_obj = bool(args.get("global_obj", False))

    obj_phys = build_objectives(
        fitted=meta["fitted_biases"],
        scale_mode="frozen",
        scale_exclude=(),
        dt=dt_sec,
        full_swat=full_swat,
        global_obj=global_obj,
    )

    obj_phys.load_state_dict(ckpt["objective_state"], strict=True)
    obj_phys.set_scale_mode("frozen")
    obj_phys.eval()

    bundle = {
        "registry": registry,
        "normalizer": normalizer,
        "model": model,
        "obj_phys": obj_phys,
        "window": window,
        "input_cols": input_cols,
        "target_cols": target_cols,
        "target_indices": target_indices,
        "meta": meta,
        "temporal_stride": int(args["temporal_stride"]),
        "downsample_agg": str(args["downsample_agg"]),
        "dt_sec": dt_sec,
        "device": device,
        "wrapper": "delta" if residual else "absolute",
    }
    return bundle
