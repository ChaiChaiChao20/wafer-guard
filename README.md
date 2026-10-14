# WaferGuard — audited wafer anomaly detector

This repo repairs and replaces the supplied wafer detector with an auditable, causality-safe supervised autoencoder. It uses a strict chronological train/validation/test split, learns imputation and scaling from the training period only, treats the legacy PS-L route marker and firmware codes explicitly, aligns every tensor by batch, and chooses its alert threshold from validation labels only.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m wafer_guard.train --csv data/secom_telemetry_modified.csv --artifacts artifacts --epochs 80
```

Training writes `metrics.json`, `model_config.json`, `preprocessor.npz`, and `wafer_guard_state.pt` under `artifacts/`. The run is deterministic on a given PyTorch/device combination as far as PyTorch's deterministic support permits; the committed report records the development run and is not a performance guarantee.

Score new rows (CSV requires the 590 `sensor_N` columns; `label` and `timestamp` are optional):

```powershell
python -m wafer_guard.predict production_batch.csv --artifacts artifacts --output wafer_guard_scores.csv
```

Output contains the source fields, `wafer_guard_score`, `reconstruction_error`, and a validation-threshold review flag. Scores are not calibrated defect probabilities. Review the `AUDIT.md` limitations before operational use.

## Repaired pipeline

1. Sort rows by UTC timestamp and reserve the latest 15% as a natural-prevalence test block; the preceding 15% is validation; the earliest 70% is training.
2. Fit finite-value medians, means and scales using training rows only. Missingness indicators travel with the scaled values. New rows cannot change learned preprocessing state.
3. For `sensor_0 == 0`, mask the route sentinel and `sensor_1` firmware code out of continuous statistics. Keep one-hot indicators for legacy codes 100/200/300 and an unknown legacy code, plus a PS-L route flag.
4. Train a supervised autoencoder with weighted binary cross-entropy and an aligned per-row reconstruction objective. Both reconstructions and logits are shape-checked against their targets before the loss is calculated.
5. Select the threshold using validation F2 only. Test prevalence remains natural and test labels never influence preprocessing, training, checkpoint choice or threshold selection.

See [AUDIT.md](AUDIT.md) for the defect-by-defect audit, actual metrics and temporal-drift caveats. The source notebook required by the challenge is [wafer_guard_fixed.ipynb](wafer_guard_fixed.ipynb).
