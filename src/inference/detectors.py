"""Residual-based anomaly detectors.

Scores a timestep by its squared next-step residual, thresholds it on the clean set,
and alarms above threshold. MSE aggregates over sensors; the per-sensor variant
(one threshold per sensor) is applied in evaluation.py.
"""

import torch
import pandas as pd
import numpy as np
import numpy.typing as npt

from inference.model_inference import make_predictions

from typing import List
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Detector:
    """Base detector: squared per-sensor residuals + pointwise metric helpers."""

    def __init__(self):
        pass

    @staticmethod
    def _downsampled_time(
        T: int,
        slice_start: int,
        window: int,
        stride: int,
        agg: str,
    ) -> np.ndarray:
        if agg == "pick_last":
            offset = stride - 1.0
        elif agg == "pick_first":
            offset = 0.0
        else:
            offset = (stride - 1.0) / 2.0

        j = window + np.arange(T, dtype=float)
        t_abs = slice_start + j * stride + offset

        total_rows = (T + window) * stride
        idx_rows = np.rint(t_abs).astype(int)
        idx_rows = np.clip(idx_rows, 0, total_rows - 1)
        return idx_rows

    @staticmethod
    def _data_residual(predictions: dict, model_dict: dict) -> torch.Tensor:
        y = predictions["y_norm"]
        preds = predictions["preds_norm"]
        target_indices = list(model_dict["target_indices"])

        if preds.shape[-1] == len(target_indices):
            preds_tgt = preds
        else:
            preds_tgt = preds[:, target_indices]

        if preds_tgt.shape != y.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: "
                f"preds={tuple(preds_tgt.shape)}, y={tuple(y.shape)}"
            )

        return y - preds_tgt

    @staticmethod
    def _data_sensor_scores(predictions: dict, model_dict: dict) -> torch.Tensor:
        resid = Detector._data_residual(predictions, model_dict)
        return resid**2

    @staticmethod
    def calculate_pointwise_metrics(
        model_dict: dict,
        detect_cfg: dict,
        data_phys_N: pd.DataFrame,
        data_phys_A: pd.DataFrame,
        attack_label_1hz: npt.NDArray,
        detectors: List["Detector"],
    ):
        window = int(model_dict["window"])
        stride = int(model_dict["temporal_stride"])
        agg = str(model_dict["downsample_agg"])

        q_fused = float(detect_cfg["q_fused"])

        # Results are keyed by detector.name, so a duplicate would silently
        # overwrite the earlier entry.
        names = [d.name for d in detectors]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate detector names: {names}")

        pointwise_metrics = dict()

        predictions_N = make_predictions(model_dict, data_phys_N)
        predictions_A = make_predictions(model_dict, data_phys_A)

        for detector in detectors:
            zN, zA, alarm_mask = detector.eval_func(
                model_dict,
                detect_cfg,
                predictions_N,
                predictions_A,
            )

            TA = int(zA.shape[0])
            idx_rows = Detector._downsampled_time(
                TA,
                0,
                window,
                stride,
                agg,
            )
            attack_mask = attack_label_1hz[idx_rows].astype(int)

            # Force both labels so the matrix is always 2x2 → 4 values, even when
            # only one class is present (e.g. the clean-normal FPR set has all
            # labels=0, and a clean model fires no alarms → single-class otherwise).
            TN, FP, FN, TP = confusion_matrix(
                attack_mask, alarm_mask, labels=[0, 1]
            ).ravel()

            accuracy = accuracy_score(attack_mask, alarm_mask)
            precision = precision_score(attack_mask, alarm_mask, zero_division=0)
            recall = recall_score(attack_mask, alarm_mask, zero_division=0)
            f1 = f1_score(attack_mask, alarm_mask, zero_division=0)

            pointwise_metrics[detector.name] = dict(
                TP=TP,
                FP=FP,
                TN=TN,
                FN=FN,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                q_fused=float(q_fused),
                zA=zA,
                zN=zN,
                alarm_mask=alarm_mask,
            )

        return pointwise_metrics

    @torch.no_grad()
    def eval_func(
        self,
        model_dict: dict,
        detect_cfg: dict,
        predictions_N: dict,
        predictions_A: dict,
    ):
        raise NotImplementedError()


class MSEDetector(Detector):
    """Aggregates residuals (mean over sensors) into one score with a single threshold."""

    name = "mse"

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def eval_func(
        self,
        model_dict: dict,
        detect_cfg: dict,
        predictions_N: dict,
        predictions_A: dict,
    ):
        q_fused = float(detect_cfg["q_fused"])

        zN = Detector._data_sensor_scores(predictions_N, model_dict)
        zA = Detector._data_sensor_scores(predictions_A, model_dict)

        mse_N = zN.mean(dim=1)
        tau_mse = torch.quantile(
            mse_N, q=torch.tensor(q_fused, device=mse_N.device), dim=0
        )

        mse_A = zA.mean(dim=1)

        alarm_mask = (mse_A > tau_mse).cpu().numpy().astype(bool)
        return zN, zA, alarm_mask


class PerSensorDetector(Detector):
    """One threshold per sensor; alarms when any sensor exceeds its threshold."""

    name = "per_sensor"

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def eval_func(
        self,
        model_dict: dict,
        detect_cfg: dict,
        predictions_N: dict,
        predictions_A: dict,
    ):
        q_per = float(detect_cfg["q_per_sensor"])

        zN = Detector._data_sensor_scores(predictions_N, model_dict)
        zA = Detector._data_sensor_scores(predictions_A, model_dict)
        tau = torch.quantile(
            zN, q=torch.tensor(q_per, dtype=zN.dtype, device=zN.device), dim=0
        )
        alarm_mask = (zA > tau).any(dim=1).cpu().numpy().astype(bool)
        return zN, zA, alarm_mask
