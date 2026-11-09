# Physics-Informed Defenses against Poisoning Attacks on Learning-based Anomaly Detection in Industrial Control Systems

Code artifact for our work on defending **process-based anomaly detection** in
industrial control systems against **training-time data poisoning**. The detector
is a **next-step forecaster**: it learns normal plant behavior and raises an alarm
when the prediction error exceeds a threshold. We add **physical process knowledge**
(tank mass balance) as a prior to make the detector robust to poisoned training data,
and compare three training variants:

- **Data-Only** — standard forecaster, no physics (`--w_phys 0`).
- **PINN** — adds a physics-residual loss as a soft constraint (`--w_phys 1`).
- **Filtering** — drops training windows whose physics residual is too large, then
  trains Data-Only (`--physics_filter q`).

> **Scope**: Experiments run on the SWaT testbed, scoped to the first process stage (PLC1: LIT101, FIT101, FIT201, MV101, P101, P102) and extended to the full six-stage plant.

---

## Repository layout

| Path | Contents |
| ---- | -------- |
| **src/data/** | SWaT → SI-unit conversion, windowing/downsampling, normalizer, mass-balance residual + bias fitting, poisoning/fault transforms, synthetic-attack generation |
| **src/models/** | MLP / CNN1d / LSTM / Transformer backbones, next-step forecaster wrapper, copy-last baseline, model factory |
| **src/physics/** | Physics objectives (residual → scale → loss) and the tank/flow equation factory |
| **src/training/** | Training loop, early stopping, data + physics loss balancing, batch metrics |
| **src/inference/** | Residual-based detectors (MSE and per-sensor), recall/FPR metrics |
| **src/scripts/** | CLI entry points: training, hyperparameter sweep, model selection, evaluation, invariant-rule filtering |
| **src/plotting/** | Builds the figures and exports their pgfplots CSVs |


### Code ↔ method map

| Method component | Where |
| ---------------- | ----- |
| Forecasters & input wrappers | `src/models/` |
| Mass-balance residual, static bias `b`, residual scales `s_k` | `src/data/data_utils.py`, `src/physics/objectives.py` |
| Physics (PINN) loss & multi-objective training | `src/physics/objectives.py`, `src/training/trainer.py` |
| Filtering defense + invariant-rule filter | `src/scripts/train.py`, `src/scripts/eval_invariant_rules.py` |
| Poisoning (inconsistent/consistent ramps) & synthetic attacks | `src/data/transforms.py`, `src/data/generate_synthetic_ramps.py` |
| Detectors & metrics (recall, FPR) | `src/inference/`, `src/scripts/evaluation.py` |
| Sweep → selection → evaluation → figures | `src/scripts/`, `src/plotting/` |

---

## Quick-start

### 0 · Install

```bash
pip install -e .                  # plus a PyTorch build for your platform,
pip install -e ".[torch-cu128]"   # e.g. this one for CUDA 12.8
```

The modules import each other as top-level packages (`from data...`,
`from training...`), which the editable install provides. Without installing,
prefix every command below with `PYTHONPATH=src`. All commands run from the
repository root, since configs refer to `data/processed/...` and `outputs/...`.



### 1 · Obtain the dataset

The SWaT dataset is available upon request from iTrust Singapore:
<http://itrust.sutd.edu.sg/>.

We use **SWaT.A1 & A2_Dec 2015**. The dataset is not included in
this repository. Place the files in a `data/` directory at the
repository root:

```text
data/
├── raw/
│   ├── SWaT_Dataset_Normal_v0.xlsx
│   └── SWaT_Dataset_Attack_v0.xlsx
└── processed/        # generated in step 2
```

### 2 · Prepare data

```bash
# Convert the raw SWaT Excel logs to fast Parquet in SI units (run once per sheet).
python -m src.data.convert_swat \
    --xlsx data/raw/SWaT_Dataset_Normal_v0.xlsx \
    --sheet Normal.csv \
    --out data/processed/normal_si.parquet
```

The converter rescales to SI units, drops helper columns, and adds the
`UF_REJECT_MASK` feature. Every later stage reads the Parquet files directly.

The synthetic test attacks (linear / staircase / sigmoid / noisy ramps,
physics-consistent and multistage attacks) are generated from the normal data
with `python -m src.data.generate_synthetic_ramps`; its module docstring lists
the invocations.

### 3 · Train a model

```bash
python -m src.scripts.train \
    --data data/processed/normal_si.parquet --outdir outputs/run \
    --window 50 --temporal_stride 10 --keep_seconds \
    --batch_size 256 --epochs 250 --device cuda --lr 5e-4 \
    --model mlp --w_phys 1.0 \
    --row_start 21600 --rows 345600 --test_rows 43200 \
    --save_artifacts --name "mlp_pinn"
```

`--w_phys 0` trains Data-Only; `--physics_filter 0.99` trains the Filtering variant;
`--full_swat` switches from PLC1 to the full plant. Poisoning is configured with
`--corrupt_type / --corrupt_sensor / --corrupt_param / --ramp_rate`.

`--wrapper delta` swaps the absolute output head for a residual one (the network
predicts the change from the last observed step); `--drop_cols A,B` removes columns
from both model input and prediction. Both are recorded in the checkpoint and must
match at inference — `load_model_bundle` reads them back automatically.

### 4 · Sweep, select, evaluate, plot

```bash
# Expand the config blocks into train.py runs (see CONFIGS in gridsearch.py).
python -m src.scripts.gridsearch --data data/processed/normal_si.parquet \
    --outbase outputs/v5-gridsearch --all_blocks

# Per experiment, keep the run with the lowest test_rmse on the held-out
# selection slice within each (arch, sweep value, variant, seed) group.
python -m src.scripts.select_best_models --batch-all

# Run the detectors and cache recall/FPR; the flags add the transfer, grid and
# appendix passes the figures need. Plots are rendered at the end.
python -m src.scripts.evaluation --device cuda --grid --cross-sensor \
    --cross-shape --ramp-rates --rate-sweep --full-swat-synthetic --appendix-table
```


`python -m src.plotting.plots` re-renders the figures from existing caches.

---

If you would like to cite our work, please use the reference as provided below:
```
@InProceedings{Grossmann2026Physics,
author    = {Simeon Johannes Grossmann and Alessandro Erba and Christian Wressnegger},
booktitle = {Proc. of the Joint Workshop on CPS \& IoT Security and Privacy ({CPSIoTSec})},
title     = {Physics-Informed Defenses against Poisoning Attacks on Learning-based Anomaly Detection in ICS},
year      = {2026},
month     = nov
}
```
