"""Wrap a backbone into a next-step state forecaster.

Feeds the backbone only the sensor channels (auxiliary actuator/mask inputs
dropped), predicts the next state, and writes it back into the full state vector
so the physics residual sees every variable.

Two output parameterisations, selected by ``residual``:

* ``residual=False`` (absolute) — the backbone predicts the next state directly.
* ``residual=True`` (delta) — the backbone predicts the *change* from the last
  observed step and the wrapper adds it back, ResNet-style. Every tank objective's
  level term is ``(pred_level - obs_last_level)/dt``, so under the delta head the
  backbone output *is* the constrained quantity instead of being recovered by
  cancelling two large near-equal numbers.

Both modes produce identical ``state_dict`` keys and shapes, so a checkpoint cannot
tell you which one trained it — the mode is recorded in ``meta["args"]["wrapper"]``
and must be passed back in at construction (see ``models.config.load_model_bundle``).
"""

import torch


class NextStepForecaster(torch.nn.Module):
    """Backbone wrapper that predicts the next state from the sensor channels."""

    def __init__(self, base: torch.nn.Module, *, target_indices, residual: bool = False):
        super().__init__()
        self.base = base
        # Plain attribute, not a buffer: keeping it out of the state_dict means
        # absolute and delta checkpoints stay interchangeable on disk.
        self.residual = bool(residual)
        self.register_buffer(
            "target_indices", torch.as_tensor(list(target_indices), dtype=torch.long)
        )

    def extra_repr(self) -> str:
        return f"residual={self.residual}"

    def forward(self, x_norm_full: torch.Tensor) -> torch.Tensor:
        # x_norm_full: (B, W, F_full). Contains aux masks too.
        x_in = x_norm_full[
            :, :, self.target_indices
        ]  # drop aux; keep only real features

        pred = self.base(x_in)  # expected (B, T) where T=len(target_indices)
        if pred.ndim != 2:
            raise ValueError(f"base must return (B, T), got {tuple(pred.shape)}")
        if pred.shape[1] != int(self.target_indices.numel()):
            raise ValueError(
                f"base output dim {pred.shape[1]} != len(target_indices) {int(self.target_indices.numel())}"
            )

        y_full = x_norm_full[:, -1, :].clone()
        if self.residual:
            # pred is already in target order, so it is added to the last observed
            # target values directly -- never index `pred` with target_indices.
            y_full[:, self.target_indices] = x_in[:, -1, :] + pred
        else:
            y_full[:, self.target_indices] = pred
        return y_full
