"""Windowing, batching, and block-mean downsampling for the SWaT series:
sliding input windows with next-step targets, wrapped in dataloaders."""

import torch
from torch.utils.data import Dataset, DataLoader


class ForecasterDataset(Dataset):
    """
    Returns dict with:
      x_norm:      (S, F_in)
      x_phys:      (S, F_in)
      y_norm:  (F_out)
      y_phys:  (F_out)
    """

    def __init__(self, x_norm, x_phys, y_norm, y_phys, slc):
        self.xn = x_norm[slc]
        self.xp = x_phys[slc]
        self.yn = y_norm[slc]
        self.yp = y_phys[slc]

    def __len__(self):
        return self.xn.shape[0]

    def __getitem__(self, i):
        out = {
            "x_norm": self.xn[i],
            "x_phys": self.xp[i],
            "y_norm": self.yn[i],
            "y_phys": self.yp[i],
        }
        return out


def make_windows(data_phys_full: torch.Tensor, window: int, stride: int = 1):
    # data_phys_full: (N, F_in) on device
    N, F_in = data_phys_full.shape
    num_seq = N - window
    if num_seq < 0:
        raise ValueError(f"Not enough data for window={window}, (N={N})")
    elif num_seq == 0:
        return torch.as_tensor(data_phys_full).unsqueeze(0)
    else:
        x_phys = torch.stack(
            [data_phys_full[i : i + window] for i in range(0, num_seq, stride)]
        )
        return x_phys  # (num_seq, S, F_in)


def make_targets(
    data_phys_full: torch.Tensor, target_indices, window: int, stride: int = 1
):
    num_seq = data_phys_full.shape[0] - window

    y_phys = torch.stack(
        [data_phys_full[i + window, target_indices] for i in range(0, num_seq, stride)]
    )
    return y_phys  # (num_seq, F_out)


def make_dataloaders(
    inputs_norm,
    inputs_phys,
    y_norm,
    y_phys,
    train_ratio=0.80,
    dev_ratio=0.10,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    train_sl: slice | None = None,
    dev_sl: slice | None = None,
    test_sl: slice | None = None,
):
    num_seq = inputs_norm.shape[0]

    if all(s is not None for s in (train_sl, dev_sl, test_sl)):
        # use provided explicit slices
        pass
    else:
        # fall back to ratio-based splits (unchanged behavior)
        n_train = int(train_ratio * num_seq)
        n_dev = int(dev_ratio * num_seq)
        train_sl = slice(0, n_train)
        dev_sl = slice(n_train, n_train + n_dev)
        test_sl = slice(n_train + n_dev, None)

    ds = {
        "train": ForecasterDataset(inputs_norm, inputs_phys, y_norm, y_phys, train_sl),
        "dev": ForecasterDataset(inputs_norm, inputs_phys, y_norm, y_phys, dev_sl),
        "test": ForecasterDataset(inputs_norm, inputs_phys, y_norm, y_phys, test_sl),
    }

    def _ldr(d, shuffle):
        return DataLoader(
            ds[d],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

    return _ldr("train", True), _ldr("dev", False), _ldr("test", False), ds


# downsampling helpers ────────────────────────────────────────────────────────
def _column_masks(registry, *, vtype: str) -> torch.BoolTensor:
    """
    Build a boolean mask of columns whose VariableMeta.vtype matches `vtype`.
    Returned tensor lives on CPU; move to device of your data if needed.
    """
    return torch.tensor(
        [meta.vtype == vtype for _, meta in sorted(registry._by_index.items())],
        dtype=torch.bool,
    )


def temporal_downsample(
    arr: torch.Tensor,
    temporal_stride: int = 1,
    registry=None,
    agg: str = "block_mean",
) -> torch.Tensor:
    """Downsample the time axis (dim=0) by 'temporal_stride'."""

    def safe_mode(t, dim):
        if t.device.type == "mps":  # M-series GPUs
            return torch.mode(t.cpu(), dim=dim).values.to(t.device)
        else:
            return torch.mode(t, dim=dim).values

    K = int(temporal_stride)
    if K <= 1:
        y = arr
    else:
        if arr.dim() < 1:
            raise ValueError("temporal_downsample expects arr with time on dim=0")

        T = arr.shape[0]
        T_trim = (T // K) * K
        if T_trim == 0:
            raise ValueError(f"temporal_stride={K} too large for length T={T}")

        x = arr[:T_trim].view(T_trim // K, K, *arr.shape[1:])  # [N, K, ...]
        if agg == "block_mean" and registry is not None and arr.dim() >= 2:
            cont_mask = _column_masks(registry, vtype="continuous").to(arr.device)
            cat_mask = ~cont_mask
            if cat_mask.any():
                out_cont = x[:, :, cont_mask].mean(dim=1)  # [N, C_cont]
                out_cat = safe_mode(x[:, :, cat_mask], dim=1)  # [N, C_cat]
                y = torch.empty_like(x[:, 0, :])
                y[:, cont_mask] = out_cont
                y[:, cat_mask] = out_cat
            else:
                y = x.mean(dim=1)
        elif agg == "block_mean":
            y = x.mean(dim=1)
        elif agg == "pick_last":
            y = x[:, -1, ...]
        elif agg == "pick_first":
            y = x[:, 0, ...]
        else:
            raise ValueError(
                f"Unknown agg='{agg}' (use 'block_mean'|'pick_last'|'pick_first')"
            )

    return y
