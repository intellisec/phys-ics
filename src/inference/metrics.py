"""Detection metrics: per-attack recall (then uniform mean over attacks), the
timestep-wise recall over the same catalogued episodes, and FPR."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from inference.utils import parquet_to_tensor
from inference.detectors import Detector, MSEDetector, PerSensorDetector
from sklearn.metrics import confusion_matrix, recall_score


@torch.no_grad()
def compute_metrics(
    model_dict: dict,
    detect_cfg: dict,
    *,
    device: str = "cpu",
    labels_parquet: Path = None,
    attack_dir: Path = None,
    normal_dir: Path = None,
    threshold_slice: slice = slice(0, None),
    attack_slices_dropped: dict | None = None,
    keep_cols: list[str] | None = None,
    detectors: list | None = None,
):
    registry = model_dict["registry"]

    # load data and labels
    data_phys_N = parquet_to_tensor(normal_dir, device=device, registry=registry)[
        threshold_slice, :
    ]
    data_phys_A = parquet_to_tensor(attack_dir, device=device, registry=registry)

    labels_df = pd.read_parquet(labels_parquet)

    attack_label_1hz = labels_df["attack"].to_numpy().astype(np.int8)
    assert len(attack_label_1hz) == data_phys_A.shape[0]

    # --- drop attack slices (rows) from BOTH attack data and labels ---
    if attack_slices_dropped:
        n = data_phys_A.shape[0]
        keep = torch.ones(n, dtype=torch.bool, device=data_phys_A.device)
        for _, info in attack_slices_dropped.items():
            s = int(info["Start_Row"])
            e = int(info["End_Row"])  # end-exclusive
            keep[s:e] = False
        data_phys_A = data_phys_A[keep]
        attack_label_1hz = attack_label_1hz[keep.cpu().numpy()]

    # --- keep only stage-1 cols (apply to N and A) ---
    if keep_cols is None:
        raise ValueError("keep_cols must be provided.")

    keep_idxs = [int(registry.index(c)) for c in keep_cols]
    data_phys_N = data_phys_N[:, keep_idxs]
    data_phys_A = data_phys_A[:, keep_idxs]

    # run predictions and calculate metrics
    # predictions for Normal.
    # Predictions are computed once and shared across the detector list.
    if detectors is None:
        detectors = [MSEDetector(), PerSensorDetector()]
    return Detector.calculate_pointwise_metrics(
        model_dict,
        detect_cfg,
        data_phys_N,
        data_phys_A,
        attack_label_1hz,
        detectors,
    )


def attackwise_recall(model_dict, alarm_mask, attack_slices):
    """Recall per catalogued attack episode, and timestep recall over their union.

    Returns ``(mean_recall, std_recall, per_attack_recall, (timestep_recall, tp,
    pos))``, where the mean/std are taken uniformly over episodes.
    """
    window = int(model_dict["window"])
    stride = int(model_dict["temporal_stride"])
    agg = str(model_dict["downsample_agg"])

    alarm_rows = Detector._downsampled_time(len(alarm_mask), 0, window, stride, agg)

    recall = []
    full_attack_mask = np.zeros_like(alarm_mask, dtype=bool)

    per_attack_recall = {}
    for key, row in attack_slices.items():
        start_row = int(row["Start_Row"])
        end_row = int(row["End_Row"])

        attack_mask = (alarm_rows >= start_row) & (alarm_rows <= end_row)
        full_attack_mask |= attack_mask

        attack_recall = recall_score(attack_mask, alarm_mask, zero_division=0)
        per_attack_recall[key] = attack_recall
        recall.append(attack_recall)

    # labels=[0,1] keeps the matrix 2x2 even if only one class is present.
    _, _, FN, TP = confusion_matrix(
        full_attack_mask, alarm_mask, labels=[0, 1]
    ).ravel()

    # Timestep-wise recall over the CATALOGUED positive set: full_attack_mask is
    # the union of the episode step-masks built above, so TP/(TP+FN) is
    # conventional pointwise recall on exactly the slices this function scores.
    # It differs from the episode-uniform np.mean(recall) returned first -- long
    # episodes dominate it -- and from the detector's own pointwise recall, whose
    # positives come from the published labels rather than the slice catalogue.
    ts_pos = int(TP + FN)
    timestep_recall = float(TP) / ts_pos if ts_pos else float("nan")

    return (
        np.mean(recall),
        np.std(recall),
        per_attack_recall,
        (timestep_recall, int(TP), ts_pos),
    )
