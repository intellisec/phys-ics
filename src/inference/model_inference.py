"""Run a trained forecaster over a dataset: window, normalize, predict next step."""

import torch
from data.dataset import make_windows, make_targets, temporal_downsample
from data.registry_norm import Normalizer
from typing import Sequence


def downsample(
    x_phys: torch.Tensor,
    *,
    stride: int = 1,
    registry,
    agg: str = "block_mean",
) -> torch.Tensor:
    """Down-sample the time axis if stride is not 1"""
    if stride <= 1:
        return x_phys
    else:
        return temporal_downsample(
            x_phys,
            temporal_stride=stride,
            registry=registry,
            agg=agg,
        )


@torch.no_grad()
def inputs(
    data_phys_rows: torch.Tensor,
    *,
    window: int,
    normalizer: Normalizer,
) -> torch.Tensor:
    x_phys = make_windows(data_phys_rows, window=window, stride=1)
    return normalizer.forward(x_phys)


@torch.no_grad()
def targets(
    data_phys_rows: torch.Tensor,
    target_indices: Sequence[int],
    *,
    window: int,
    normalizer: Normalizer,
) -> torch.Tensor:
    y_phys = make_targets(
        data_phys_rows,
        target_indices,
        window=window,
        stride=1,
    )

    mean_full = normalizer.mean.view(-1).to(y_phys.device, y_phys.dtype)
    std_full = normalizer.std.view(-1).to(y_phys.device, y_phys.dtype)
    tgt = list(target_indices)

    return (y_phys - mean_full[tgt].view(1, -1)) / std_full[tgt].view(1, -1)


@torch.no_grad()
def predictions(
    x_norm: torch.Tensor,
    *,
    model: torch.nn.Module,
    registry,
    target_indices: Sequence[int],
) -> torch.Tensor:
    model.eval()
    preds_norm = model(x_norm)

    full_dim = int(registry.num_features)
    target_dim = int(len(target_indices))

    if preds_norm.shape[-1] not in (full_dim, target_dim):
        raise ValueError(
            f"Unexpected prediction width {preds_norm.shape[-1]}. "
            f"Expected full registry width {full_dim} or target width {target_dim}."
        )

    return preds_norm


def make_predictions(model_dict: dict, df: torch.Tensor) -> dict:
    """Normalized targets and next-step predictions for every window of ``df``."""
    registry = model_dict["registry"]
    window = int(model_dict["window"])
    stride = int(model_dict["temporal_stride"])
    agg = str(model_dict["downsample_agg"])
    model = model_dict["model"]
    normalizer = model_dict["normalizer"]
    device = model_dict["device"]
    normalizer.to(device)
    target_indices = model_dict["target_indices"]

    data_downsampled = downsample(df, stride=stride, registry=registry, agg=agg)

    x_norm = inputs(
        data_downsampled,
        window=window,
        normalizer=normalizer,
    )
    y_norm = targets(
        data_downsampled,
        target_indices,
        window=window,
        normalizer=normalizer,
    )
    preds_norm = predictions(
        x_norm,
        model=model,
        registry=registry,
        target_indices=target_indices,
    )

    return {"y_norm": y_norm, "preds_norm": preds_norm}
