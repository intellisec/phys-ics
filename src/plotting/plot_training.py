"""Plots for inspecting training dynamics: parity, RMSE, and loss curves."""

import math
import numpy as np
import torch

import matplotlib.pyplot as plt
import csv
from typing import Optional, Iterable
from pathlib import Path
import os
import sys
import json
import datetime as _dt
from training.metrics import (
    compute_one_step_data_errors,
    compute_one_step_physics_errors,
)
from data.registry_norm import infer_group_from_name


@torch.no_grad()
def save_eval_csvs(
    model,
    registry,
    normalizer,
    test_loader,
    *,
    objective_set_phys,
    run_name: str = "run",
    metrics_dir: Path = Path("outputs/metrics"),
    target_indices: Optional[Iterable[int]] = None,
):
    """
    Writes {run}_group_summary_1step.csv -> one data-overall score, per sensor-group means, physics group means
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- DATA (per-sensor z-errors) ----

    data_rows, data_summary = compute_one_step_data_errors(
        model,
        registry,
        normalizer,
        test_loader,
        target_indices=target_indices,
    )

    # ---- PHYSICS via your ObjectiveSet (raw SI residuals, unscaled) ----
    phys_rows, phys_summary = compute_one_step_physics_errors(
        model, objective_set_phys, registry, normalizer, test_loader
    )

    # ---- Group summary CSV ----
    group_path = metrics_dir / f"{run_name}_group_summary_1step.csv"
    with open(group_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["type", "group", "metric", "value", "unit", "n"]
        )
        w.writeheader()
        # data: overall
        w.writerow(
            {
                "type": "data",
                "group": "ALL",
                "metric": "mean_rmse_z",
                "value": f"{data_summary['data_overall_rmse_z']:.6e}",
                "unit": "z",
                "n": sum(1 for r in data_rows if r["kind"] == "data"),
            }
        )
        w.writerow(
            {
                "type": "data",
                "group": "CONT",
                "metric": "mean_rmse_z",
                "value": f"{data_summary['data_cont_rmse_z']:.6e}",
                "unit": "z",
                "n": sum(
                    1
                    for r in data_rows
                    if (r["kind"] == "data") and (r["group"] != "actuator")
                ),
            }
        )
        # data: per sensor-group
        for g, val in sorted(data_summary["per_group"].items()):
            w.writerow(
                {
                    "type": "data",
                    "group": g,
                    "metric": "mean_rmse_z",
                    "value": f"{val:.6e}",
                    "unit": "z",
                    "n": sum(
                        1 for r in data_rows if r["kind"] == "data" and r["group"] == g
                    ),
                }
            )
        # physics groups
        w.writerow(
            {
                "type": "physics",
                "group": "tank",
                "metric": "mean_rms",
                "value": f"{phys_summary.get('tank_mean_rms_si', float('nan')):.6e}",
                "unit": "m3/s",
                "n": sum(1 for r in phys_rows if r["group"] == "tank"),
            }
        )
        w.writerow(
            {
                "type": "physics",
                "group": "split",
                "metric": "mean_rms",
                "value": f"{phys_summary.get('split_mean_rms_si', float('nan')):.6e}",
                "unit": "m3/s",
                "n": sum(1 for r in phys_rows if r["group"] == "split"),
            }
        )

    print(f"group summary (1-step)     -> {group_path}")


@torch.no_grad()
def save_parity_and_rmse(
    model,
    registry,
    normalizer,
    test_loader,
    *,
    target_indices=None,
    target_cols=None,
    run_name="run",
    plot_dir=Path("outputs/plots"),
    table_dir=Path("outputs/metrics"),
):
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if target_indices is None:
        target_indices = sorted(
            v.index
            for v in registry._vars.values()
            if getattr(v, "vtype", "continuous") == "continuous"
        )
    if target_cols is None:
        target_cols = [registry._by_index[i].name for i in target_indices]

    preds_list, true_list = [], []
    model.eval()

    for batch in test_loader:
        b = {k: v.to(device) for k, v in batch.items()}
        preds = model(b["x_norm"])  # (B,F) or (B,H,F)

        y = b["y_norm"]

        preds_phys = normalizer.inverse(preds.unsqueeze(1)).squeeze(1)
        tgt_phys = normalizer.inverse(y.unsqueeze(1)).squeeze(1)

        preds_list.append(preds_phys[:, target_indices].detach().cpu())
        true_list.append(tgt_phys[:, target_indices].detach().cpu())

    pred_all = torch.cat(preds_list, dim=0).numpy()
    true_all = torch.cat(true_list, dim=0).numpy()

    def _si_to_orig_factor(name: str) -> float:
        if name.startswith("LIT"):
            return 1_000.0
        if name.startswith("FIT"):
            return 3_600.0
        if name.startswith(("PIT", "DPIT")):
            return 1.0 / 1_000.0
        return 1.0

    factors = np.array(
        [_si_to_orig_factor(n) for n in target_cols], dtype=float
    ).reshape(1, -1)
    true_plot = true_all * factors
    pred_plot = pred_all * factors

    n_targets = len(
        [cont for cont in target_cols if infer_group_from_name(cont) != "actuator"]
    )
    n_cols = 5
    n_rows = math.ceil(n_targets / n_cols)
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 3.0))
    axes2 = np.array(axes2).reshape(n_rows, n_cols)

    plt_count = 0
    for i, name in enumerate(target_cols):
        if infer_group_from_name(name) == "actuator":
            continue

        ax2 = axes2[plt_count // n_cols, plt_count % n_cols]
        ax2.scatter(true_plot[:, i], pred_plot[:, i], s=6, alpha=0.6)

        mn, mx = true_plot[:, i].min(), true_plot[:, i].max()
        if mn == mx:
            eps = 1e-6
            mn -= eps
            mx += eps

        ax2.plot([mn, mx], [mn, mx], lw=0.8)
        ax2.set_xlim(mn, mx)
        ax2.set_title(name, fontsize=8)
        ax2.tick_params(axis="both", which="major", labelsize=6)
        ax2.set_xlabel("True (orig)", fontsize=6)
        ax2.set_ylabel("Pred (orig)", fontsize=6)
        plt_count += 1

    for j in range(plt_count, n_rows * n_cols):
        axes2.flat[j].axis("off")

    fig2.tight_layout()
    parity_path = plot_dir / f"{run_name}_parity.png"
    fig2.savefig(parity_path, dpi=300)
    plt.close(fig2)

    print(f"parity plot -> {parity_path}")


def save_loss_grid(
    trainer,
    *,
    run_name="run",
    out_dir=Path("outputs/losses"),
    n_cols: int = 2,
    figsize_single: tuple[int, int] = (5, 3),
):
    if not getattr(trainer, "history", None):
        raise RuntimeError("`trainer.history` is empty - run training first.")

    fig_dir = Path(out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sample = trainer.history[0]["dev_phys_metrics"]
    metrics = [m for m in sample.keys() if m != "total"]

    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_single[0], n_rows * figsize_single[1]),
        squeeze=False,
    )

    epochs = [rec["epoch"] for rec in trainer.history]

    for i, metric in enumerate(metrics):
        ax_left = axes[i // n_cols][i % n_cols]
        ax_right = None

        if metric == "data":
            # Dev MSE (H=1)
            dev_mse1 = [
                v["dev_phys_metrics"]["data"]["dev_mse"] for v in trainer.history
            ]
            # Train MSE (always tracked)
            train_M = [
                v["dev_phys_metrics"]["data"]["train_mse"] for v in trainer.history
            ]

            plotted_left = []

            if np.isfinite(dev_mse1).any():
                ax_left.plot(epochs, dev_mse1, label="dev MSE@1")
                plotted_left.append(np.array(dev_mse1, dtype=float))

            ax_left.set_ylabel("MSE (normalized space)")

            if np.isfinite(train_M).any():
                ax_right = ax_left.twinx()
                ax_right.plot(epochs, train_M, label="train MSE", color="red")
                ax_right.set_ylabel("train MSE")

                # Harmonize y-lims across left & right axes using whichever series exist
                all_series = []
                if plotted_left:
                    all_series.extend(plotted_left)
                all_series.append(np.array(train_M, dtype=float))

                flat = (
                    np.concatenate(
                        [s[np.isfinite(s)] for s in all_series if np.isfinite(s).any()]
                    )
                    if any(np.isfinite(s).any() for s in all_series)
                    else np.array([])
                )
                if flat.size > 0:
                    y_min, y_max = float(np.nanmin(flat)), float(np.nanmax(flat))
                    if y_min == y_max:
                        center = y_min
                        eps = 1e-6 if center == 0.0 else abs(center) * 0.1
                        y_min, y_max = center - eps, center + eps
                else:
                    y_min, y_max = 0.0, 1.0

                ax_left.set_ylim(y_min, y_max)
                ax_right.set_ylim(y_min, y_max)

        else:
            # Physics objectives: raw RMS (left) and scale (right, if present)
            r_vals = [v["dev_phys_metrics"][metric]["raw_rms"] for v in trainer.history]
            ax_left.plot(epochs, r_vals, label="raw RMS")
            ax_left.set_ylabel("raw RMS")

            scale_vals = [
                v["dev_phys_metrics"][metric]["scale"] for v in trainer.history
            ]
            if np.isfinite(scale_vals).any():
                ax_right = ax_left.twinx()
                ax_right.plot(epochs, scale_vals, label="scale", color="red")
                ax_right.set_ylabel("scale")

        ax_left.set_xlabel("epoch")
        ax_left.set_title(metric, fontsize=9)
        ax_left.tick_params(axis="both", labelsize=7)

        handles, labels = ax_left.get_legend_handles_labels()
        if ax_right is not None:
            h2, l2 = ax_right.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            ax_left.legend(handles, labels, loc="upper right", fontsize=7)

    # Hide any unused subplots
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")

    fig.tight_layout()
    path = fig_dir / f"{run_name}_loss_grid.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"loss grid    -> {path}")


def save_run_json(
    outdir,
    args,
    model,
    trainer,
    registry,
    objective_set=None,
    *,
    filename="run_config.json",
    include_model_repr=False,
    pretty=True,
):
    """
    Write JSON file with all run info.

    Parameters
    ----------
    outdir : str | pathlib.Path
    args   : argparse.Namespace
    model  : torch.nn.Module
    trainer: object expected to expose:
             - history: list[dict] (per-epoch logs)
             - loaders: dict like {"train": DataLoader, "dev": ..., "test": ...}
             - target_indices: list[int] (optional)
    registry: object with metadata:
             - _by_index: dict[int] -> obj with .name, .unit?, .group?, .vtype?
             - _vars: dict[name]   -> obj with .index, .unit?, .group?, .vtype?
    objective_set: object with:
             - objectives: list[obj with .name, .weight, .scale, .auto_scale?, .param_dict?]
             - scale_mode: str  ("online" / "frozen"), optional
    filename: output JSON filename
    include_model_repr: include model __repr__ (can be long)
    pretty: indent + sorted keys
    """

    # -------------------- helpers --------------------------------------------
    def _len_loader(x):
        try:
            return len(getattr(x, "dataset", x))
        except Exception:
            return None

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(getattr(x, "item", lambda: x)())
            except Exception:
                return None

    def _unwrap_chain(m):
        """Return [OuterClass, ..., BaseClass], and the base module object."""
        names = []
        seen = set()
        cur = m
        for _ in range(16):  # cap to avoid loops
            if cur is None or id(cur) in seen:
                break
            seen.add(id(cur))
            names.append(cur.__class__.__name__)
            nxt = None
            # common wrapper attributes
            for attr in ("base", "module", "model", "net", "backbone", "_model"):
                if hasattr(cur, attr):
                    nxt = getattr(cur, attr)
                    if nxt is not cur:
                        break
            if nxt is None:
                break
            cur = nxt
        base = cur
        return names, base

    def _model_info(m):
        try:
            n_params = sum(p.numel() for p in m.parameters())
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        except Exception:
            n_params = n_train = None
        chain, base = _unwrap_chain(m)
        info = {
            "class": chain[0]
            if chain
            else getattr(m, "__class__", type("X", (object,), {})).__name__,
            "class_chain": chain,  # e.g., ["NextStepForecaster", "TemporalCNN1d"]
            "base_class": base.__class__.__name__ if base is not None else None,
            "parameters_total": n_params,
            "parameters_trainable": n_train,
        }
        if include_model_repr:
            info["repr"] = str(m)
        return info

    def _collect_objectives(obj_set):
        if obj_set is None:
            return [], None
        out = []
        for obj in getattr(obj_set, "objectives", []) or []:
            row = {
                "name": getattr(obj, "name", "?"),
                "weight": _safe_float(getattr(obj, "weight", None)),
                "scale": _safe_float(getattr(obj, "scale", None)),
                "auto_scale": bool(getattr(obj, "auto_scale", False)),
            }
            pdict = getattr(obj, "param_dict", None)
            if isinstance(pdict, dict):
                row["learned_params"] = {k: _safe_float(v) for k, v in pdict.items()}
            out.append(row)
        return out, getattr(obj_set, "scale_mode", None)

    def _registry_dump(reg):
        items = []
        by_idx = getattr(reg, "_by_index", None)
        if isinstance(by_idx, dict):
            for i in sorted(by_idx.keys()):
                v = by_idx[i]
                items.append(
                    {
                        "index": int(i),
                        "name": getattr(v, "name", None),
                        "unit": getattr(v, "unit", None),
                        "group": getattr(v, "group", None),
                        "vtype": getattr(v, "vtype", None),
                    }
                )
        else:
            by_name = getattr(reg, "_vars", None) or {}
            for name, v in by_name.items():
                items.append(
                    {
                        "index": int(getattr(v, "index", -1)),
                        "name": name,
                        "unit": getattr(v, "unit", None),
                        "group": getattr(v, "group", None),
                        "vtype": getattr(v, "vtype", None),
                    }
                )
            items.sort(key=lambda r: r.get("index", 1 << 30))
        return items

    def _targets(trainer_obj, reg):
        t_idx = getattr(trainer_obj, "target_indices", None)
        if t_idx is None:
            all_meta = _registry_dump(reg)
            t_idx = [
                m["index"]
                for m in all_meta
                if m.get("vtype", "continuous") == "continuous"
            ]
        names = None
        if t_idx is not None and hasattr(reg, "_by_index"):
            try:
                names = [reg._by_index[i].name for i in t_idx]
            except Exception:
                names = None
        return list(map(int, t_idx)) if t_idx is not None else None, names

    def _splits(trainer_obj):
        sizes = {}
        for k, ld in (getattr(trainer_obj, "loaders", {}) or {}).items():
            sizes[k] = _len_loader(ld)
        return sizes

    def _infer_stride(args_obj, trainer_obj):
        """
        Detect temporal stride from args first (including --temporal_stride),
        then fall back to dataset attributes inside loaders.
        """
        arg_keys = (
            "temporal_stride",
            "stride",
            "sample_stride",
            "step_stride",
            "every_n",
            "downsample",
            "ds_stride",
            "subsample",
            "hop",
            "time_stride",
            "ts_stride",
        )
        for key in arg_keys:
            if hasattr(args_obj, key):
                try:
                    iv = int(getattr(args_obj, key))
                    if iv >= 1:
                        return iv
                except Exception:
                    pass
        # Try datasets in loaders
        for ld in (getattr(trainer_obj, "loaders", {}) or {}).values():
            ds = getattr(ld, "dataset", None)
            if ds is None:
                continue
            for key in arg_keys:
                if hasattr(ds, key):
                    try:
                        iv = int(getattr(ds, key))
                        if iv >= 1:
                            return iv
                    except Exception:
                        continue
        return 1

    def _effective_W(args_obj, trainer_obj):
        """
        Effective window when keeping wall-clock coverage:
          W_eff = ceil(W_arg / stride) if keep_seconds and stride>1 else W_arg
        """
        W_arg = getattr(args_obj, "window", getattr(args_obj, "S", None))
        W_arg = int(W_arg) if W_arg is not None else None

        stride = _infer_stride(args_obj, trainer_obj)
        keep = bool(getattr(args_obj, "keep_seconds", False))

        W_eff = W_arg
        if keep and stride > 1:
            if W_arg is not None:
                W_eff = max(1, math.ceil(W_arg / stride))

        return (
            {"arg": W_arg, "effective": W_eff},
            keep,
            stride,
        )

    # -------------------- build payload ---------------------------------------
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_info = _model_info(model)
    objectives, scale_mode = _collect_objectives(objective_set)
    variables = _registry_dump(registry)
    tgt_idx, tgt_names = _targets(trainer, registry)
    window_blk, keep_seconds, stride = _effective_W(args, trainer)

    data_block = {
        "source": getattr(args, "data", None),
        "row_start": getattr(args, "row_start", None),
        "rows": getattr(args, "rows", None),
        "splits": _splits(trainer),
        "window": window_blk,  # {"arg": W, "effective": W_eff}
        "keep_seconds": keep_seconds,
        "stride": stride,
    }

    args_dict = dict(vars(args)) if hasattr(args, "__dict__") else {}
    # extract training time
    timing = None
    try:
        timing = (getattr(trainer, "state", {}) or {}).get("timing")
    except Exception:
        timing = None

    payload = {
        "run": {
            "name": args_dict.get("name"),
            "seed": args_dict.get("seed"),
            "device": args_dict.get("device"),
            "time": _dt.datetime.now().isoformat(timespec="seconds"),
            "command": " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        },
        "args": args_dict,
        "data": data_block,
        "model": model_info,  # includes wrapper->base chain + base_class
        "pinn": {
            "w_phys": getattr(args, "w_phys", None),
            "scale_mode": scale_mode,
            "objectives": objectives,
        },
        "variables": variables,  # full index↔name map + meta
        "targets": {"indices": tgt_idx, "names": tgt_names},
    }

    # attach timing into the "run" section
    if timing:
        payload["run"]["train_start_utc"] = timing.get("train_start_utc")
        payload["run"]["train_end_utc"] = timing.get("train_end_utc")
        payload["run"]["train_wall_time_sec"] = float(
            timing.get("train_wall_time_sec", 0.0)
        )
        payload["run"]["train_wall_time_min"] = round(
            payload["run"]["train_wall_time_sec"] / 60.0, 3
        )

    # epoch counts for the cost decomposition (total = per-epoch x epochs):
    #   total_epochs = epochs actually run before early-stop,
    #   best_epoch   = epoch with the best dev metric (the saved checkpoint).
    _hist = getattr(trainer, "history", None) or []
    _state = getattr(trainer, "state", {}) or {}
    payload["run"]["total_epochs"] = len(_hist) if _hist else None
    payload["run"]["best_epoch"] = _state.get("epoch")

    # Physics-loss balancing scalar: loss = loss_data + (w_phys / phys_scale) * loss_phys.
    # Estimated from the untrained model (Trainer._estimate_phys_scale), so it shifts
    # when the output parameterisation changes -- recorded here so runs trained with
    # different --wrapper values can be compared on effective physics weight.
    payload["pinn"]["phys_scale"] = _state.get("phys_scale")

    # -------------------- write JSON ------------------------------------------
    fp = outdir / filename
    if pretty:
        fp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        fp.write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )

    # console hint incl. effective coverage
    W_a, W_e = data_block["window"]["arg"], data_block["window"]["effective"]
    print(
        f"[saved] {fp}  |  stride={stride} keep_seconds={keep_seconds}  "
        f"window {W_a}->{W_e}"
    )

    return payload
