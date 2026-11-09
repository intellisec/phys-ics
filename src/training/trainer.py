"""Training loop for the next-step forecaster: loss balancing, early stopping,
checkpointing, and per-epoch logging."""

from __future__ import annotations

import math

from training.metrics import compute_batch_metrics
import torch
import copy


class Trainer:
    """Next-step training loop with early stopping and physics-loss balancing.

    Loss = data MSE + w_phys * physics loss; the physics term is rescaled on clean
    batches so w_phys=1 weights data and physics equally (w_phys=0 = Data-Only).
    """

    def __init__(
        self,
        model,
        objective_set,
        optimizer,
        scheduler,
        loaders,
        registry,
        normalizer,
        device,
        print_every=50,
        *,
        target_indices=None,
        w_phys: float = 1.0,
        early_stop_patience: int = 12,
        min_epochs: int = 25,
        early_stop_delta: float = 0.0,
    ):
        self.model = model
        self.objective_set = objective_set
        self.opt = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.registry = registry
        self.normalizer = normalizer
        self.device = device
        self.print_every = print_every

        self.state = {}
        self.history = []

        self.target_indices = (
            list(target_indices) if target_indices is not None else None
        )
        self.w_phys = float(w_phys)

        # early stopping config
        self.early_stop_patience = int(early_stop_patience)
        self.min_epochs = int(min_epochs)
        self.early_stop_delta = float(early_stop_delta)
        # early stopping state
        self._es_best = math.inf
        self._es_best_epoch = 0
        self._es_bad_epochs = 0

        self._best_ckpt = None  # will hold the best weights in memory

    def _estimate_phys_scale(self, num_batches: int = 8) -> float:
        self.model.eval()
        self.objective_set.train(True)

        data_losses = []
        phys_losses = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loaders["test"], start=1):
                losses = compute_batch_metrics(
                    self.model,
                    batch,
                    self.objective_set,
                    self.registry,
                    self.normalizer,
                    self.device,
                    self.target_indices,
                )

                loss_data = losses["loss_data"].detach().float().cpu()
                loss_phys = losses["loss_phys"].detach().float().cpu()

                if not torch.isfinite(loss_data):
                    raise ValueError(
                        f"Non-finite loss_data during scale estimation: {loss_data.item()}"
                    )

                if not torch.isfinite(loss_phys):
                    raise ValueError(
                        f"Non-finite loss_phys during scale estimation: {loss_phys.item()}"
                    )

                data_losses.append(loss_data)
                phys_losses.append(loss_phys)

                if batch_idx >= num_batches:
                    break

        if len(data_losses) == 0:
            raise RuntimeError("Scale estimation used zero batches.")

        mean_data = torch.stack(data_losses).mean()
        mean_phys = torch.stack(phys_losses).mean()

        if mean_data.item() <= 0.0:
            raise ValueError(
                f"Mean data loss must be > 0 for scale estimation, got {mean_data.item()}"
            )

        scale = mean_phys / mean_data

        if not torch.isfinite(scale):
            raise ValueError(f"Estimated scale is non-finite: {scale.item()}")

        if scale.item() <= 0.0:
            raise ValueError(f"Estimated scale must be > 0, got {scale.item()}")

        return float(scale.item())

    def train(self, epochs: int):
        self.state["epoch"] = 0
        # reset early stopping stats
        self._es_best = math.inf
        self._es_best_epoch = 0
        self._es_bad_epochs = 0

        self.phys_scale = self._estimate_phys_scale(num_batches=16)
        # Recorded (not just printed) because this scalar divides the physics term
        # for the whole run: loss = loss_data + (w_phys / phys_scale) * loss_phys.
        # It is estimated from the UNTRAINED model, so a change of output
        # parameterisation (absolute vs delta head) moves it and therefore changes
        # what w_phys=1.0 actually weights. save_run_json writes it to
        # run_config.json under pinn.phys_scale.
        self.state["phys_scale"] = float(self.phys_scale)
        print(f"Estimated physics scale: {self.phys_scale:.6f}")

        for epoch in range(1, epochs + 1):
            self.state["epoch"] = epoch

            self.model.train()
            # ensure physics set can update scales online (it will no-op if scale_mode='frozen')
            self.objective_set.train(True)

            train_data_losses = []
            train_phys_losses = []
            for batch in self.loaders["train"]:
                losses = compute_batch_metrics(
                    self.model,
                    batch,
                    self.objective_set,
                    self.registry,
                    self.normalizer,
                    self.device,
                    self.target_indices,
                )

                loss = (
                    losses["loss_data"]
                    + (self.w_phys / self.phys_scale) * losses["loss_phys"]
                )
                train_data_losses.append(float(losses["loss_data"].detach().cpu()))
                train_phys_losses.append(float(losses["loss_phys"].detach().cpu()))

                has_params = any(p.requires_grad for p in self.model.parameters())

                if has_params:
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.opt.step()

            train_mse = torch.nanmean(
                torch.as_tensor(train_data_losses, dtype=torch.float32)
            ).item()
            train_phys = torch.nanmean(
                torch.as_tensor(train_phys_losses, dtype=torch.float32)
            ).item()

            # dev metrics
            self.model.eval()
            with torch.no_grad():
                dev_data = []
                dev_phys = []
                dev_phys_resid = {}  # name -> dict with running sums for physical residuals
                for batch in self.loaders["dev"]:
                    metrics_batch = compute_batch_metrics(
                        self.model,
                        batch,
                        self.objective_set,
                        self.registry,
                        self.normalizer,
                        self.device,
                        self.target_indices,
                    )
                    dev_data.append(metrics_batch.get("loss_data", float("nan")))
                    dev_phys.append(metrics_batch.get("loss_phys", float("nan")))

                    metrics_phys = metrics_batch.get("metrics_phys", {})
                    for name, d in metrics_phys.items():
                        resid_phys = dev_phys_resid.setdefault(
                            name,
                            {"loss": [], "raw_rms_sq": 0.0, "count": 0, "scale": []},
                        )
                        resid_phys["loss"].append(float(d["loss"]))
                        resid_phys["scale"].append(float(d["scale"]))

                        r = d["raw_resid"]  # tensor (B, 1) or (B, K)
                        resid_phys["raw_rms_sq"] += float(torch.sum(r**2))
                        resid_phys["count"] += r.numel()

                dev_data_term = torch.nanmean(
                    torch.as_tensor(dev_data, dtype=torch.float32)
                ).item()
                dev_phys_term = torch.nanmean(
                    torch.as_tensor(dev_phys, dtype=torch.float32)
                ).item()

                # gather physical residuals to provide loss grid plots
                dev_phys_metrics = {}
                for name, d in dev_phys_resid.items():
                    loss_tensor = torch.as_tensor(
                        d["loss"], dtype=torch.float32, device="cpu"
                    )
                    scale_tensor = torch.as_tensor(
                        d["scale"], dtype=torch.float32, device="cpu"
                    )
                    rms_sq_tensor = torch.as_tensor(
                        d["raw_rms_sq"], dtype=torch.float32, device="cpu"
                    )
                    count_tensor = torch.as_tensor(
                        d["count"], dtype=torch.float32, device="cpu"
                    )

                    dev_phys_metrics[name] = {
                        "loss": loss_tensor.mean().item(),  # float
                        "raw_rms": torch.sqrt(
                            rms_sq_tensor / count_tensor + 1e-12
                        ).item(),
                        "scale": scale_tensor.mean().item(),
                    }
                dev_phys_metrics["data"] = {
                    "dev_mse": dev_data_term,
                    "train_mse": train_mse,
                    "train_phys": train_phys,
                }

            # dev objective for LR scheduling
            if self.scheduler is not None:
                self.scheduler.step(dev_data_term)

            self.history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "train_phys": train_phys,
                    "dev_mse": dev_data_term,
                    "dev_phys": dev_phys_term,
                    "dev_phys_metrics": dev_phys_metrics,
                }
            )

            if epoch % self.print_every == 0 or epoch == 1:
                self.print_summary()

            # ----- early stopping -----
            es_crit = dev_data_term
            if es_crit < self._es_best - self.early_stop_delta:
                self._es_best = es_crit
                self._es_best_epoch = epoch
                self._es_bad_epochs = 0

                # deep-copy everything you need for a full resume
                self._best_ckpt = {
                    "epoch": epoch,
                    "model_state": copy.deepcopy(self.model.state_dict()),
                    "objective_state": copy.deepcopy(self.objective_set.state_dict()),
                    "optimizer_state": copy.deepcopy(self.opt.state_dict()),
                }
            else:
                self._es_bad_epochs += 1

            if (
                epoch >= self.min_epochs
                and self._es_bad_epochs >= self.early_stop_patience
            ):
                self.state["early_stopped"] = True
                self.state["early_stop_epoch"] = epoch
                self.state["early_stop_best_epoch"] = self._es_best_epoch
                self.state["early_stop_best"] = self._es_best
                print(
                    f"[early-stop] stopping at epoch {epoch} | best @ {self._es_best_epoch}: {self._es_best:.6f}"
                )
                break

        return self._best_ckpt

    def print_summary(self):
        """Pretty-print the latest training / validation metrics."""
        if not self.history:  # nothing to show yet
            return

        # Print a header only once (on the first epoch)
        if self.history[-1]["epoch"] == 1:
            header = (
                f"{'Epoch':>6} | {'Train MSE':>11} | {'Train Phys':>11} | {'Dev MSE':>11} |"
                f"{'Dev Phys':>11}"
            )
            print(header)
            print("-" * len(header))

        h = self.history[-1]  # latest metrics
        print(
            f"{h['epoch']:6d} | "
            f"{h['train_mse']:11.4e} | "
            f"{h['train_phys']:11.4e} | "
            f"{h['dev_mse']:11.4e} | "
            f"{h['dev_phys']:11.4e} | "
        )
