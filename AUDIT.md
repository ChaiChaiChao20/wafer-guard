# Wafer detector audit and repair

## Scope

The supplied starter was inspected against the challenge's leakage, tensor-shape, imbalance and PS-L requirements. The replacement is a causal supervised autoencoder, not a claim that the supplied wafer sample can support production-grade defect prediction.

## Findings and repairs

| Starter defect | Why it invalidates the result | Repair |
|---|---|---|
| Imputer and scaler fit on the complete CSV before any split | Future validation/test rows influence the representation used for training and evaluation. | Chronologically split first; fit medians, means and scales on training rows only; use those frozen values for validation, test and inference. |
| SMOTE runs before a random train/test split | Synthetic points can be built using future/evaluation neighbors, and an artificially balanced test set cannot measure performance at the real defect rate. | Remove SMOTE; use training-period positive-class weighting. Keep validation/test at their observed class rates and report those rates. |
| Classifier logits have shape `[B]`, reconstruction has shape `[B,F]`, and the input is expanded to `[B,1,F]` | Broadcasting subtraction creates `[B,B,F]`, coupling rows and corrupting reconstruction loss/gradients. | Keep input and reconstruction as `[B,F]`, logits and labels as `[B]`; assert exact shapes before computing either loss. |
| `sensor_0` zeros and `sensor_1` legacy values 100/200/300 are treated as continuous | A route sentinel and firmware categories are given false numeric meaning and can distort imputation/scaling. | Exclude both fields for PS-L rows from continuous statistics; add explicit route, missingness and firmware-category indicators. |
| Random split ignores the ordered process chronology; fixed `0.5` cutoff is not selected from validation | Temporal drift is hidden and the threshold is not tuned to the stated operating trade-off. | Preserve time order; use the latest 15% as final test; choose the recall-weighted F2 threshold on validation only. |

## Reproducible run

Command used:

```powershell
python -m wafer_guard.train --csv data/secom_telemetry_modified.csv --artifacts artifacts --epochs 80
```

Seed: `1337`; device: CPU; early stopping selected epoch 3 from validation average precision. The dataset has 1,567 rows, 590 sensors, 104 defects (6.64% overall), and 146 rows with the PS-L sentinel. The test block remains at natural prevalence.

| Split | Rows | Defects | Prevalence | Time block |
|---|---:|---:|---:|---|
| Train | 1,096 | 78 | 7.12% | 2008-07-19 through 2008-09-26 |
| Validation | 235 | 17 | 7.23% | 2008-09-26 through 2008-10-05 |
| Test | 236 | 9 | 3.81% | 2008-10-05 through 2008-10-17 |

### Held-out chronological test

Validation-selected threshold: `0.0529508`.

| Metric | Result |
|---|---:|
| Average precision | 0.0842 |
| Test prevalence (no-skill AP reference) | 0.0381 |
| ROC AUC | 0.6163 |
| Precision | 0.0435 |
| Recall | 0.5556 |
| F1 | 0.0806 |
| F2 | 0.1656 |
| Confusion counts | TN 117 · FP 110 · FN 4 · TP 5 |

These metrics are from the actual run on the provided modified CSV and use all 236 chronological test rows. The model finds 5 of 9 defects at its validation-selected threshold but raises many false alarms. Test average precision is above the 3.81% prevalence reference, yet the operating point is not suitable for automated process rejection. Treat scores as a review queue and collect a larger, later-dated, independently labeled cohort before considering deployment. The validation block contains only 17 defects, so its threshold is itself noisy.

## Causality and sentinel checks

- Timestamp sorting uses the UTC timestamps and is stable for ties.
- Split indices are fixed before the preprocessor is fitted.
- The preprocessor has no access to labels or future population statistics.
- Checkpoint selection reads validation average precision; test labels are evaluated once after threshold selection.
- `sensor_0 == 0` rows do not contribute their `sensor_0` sentinel or `sensor_1` category to continuous means/scales; output includes explicit categories 100, 200, 300 and unknown.
- Each training batch asserts reconstruction shape equals input shape and logit shape equals label shape.
- No resampling occurs. Train, validation and test class prevalence is emitted into `artifacts/metrics.json`.

## Remaining limitations

The sample is small, labels are rare, the chronological test prevalence falls to 3.81%, and one validation-selected threshold is especially sensitive to the 17 validation defects. The learned score is not calibrated. No class-conditioned or sensor-group stability guarantee is made. Before production use, evaluate on multiple later, plant-separated cohorts, calibrate with an untouched set, include process cost curves and monitor drift. A generic benchmark result must not override those limits.
