from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import chronological_split, load_dataset
from .metrics import average_precision, choose_threshold, precision_recall_fbeta, roc_auc
from .model import WaferGuardSAE
from .preprocess import CausalPreprocessor


SEED = 1337


def _predict(model: WaferGuardSAE, x: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    reconstructions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            xb = torch.as_tensor(x[start : start + 256], dtype=torch.float32, device=device)
            reconstructed, logits = model(xb)
            reconstructions.append(reconstructed.detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(reconstructions), np.concatenate(probabilities)


def _evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (scores >= threshold).astype(np.int64)
    result: dict[str, float | int] = precision_recall_fbeta(y, predicted, beta=2.0)
    result.update({
        "average_precision": average_precision(y, scores),
        "roc_auc": roc_auc(y, scores),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "rows": int(len(y)),
        "defects": int(y.sum()),
    })
    return result


def run_pipeline(
    csv_path: str | Path,
    artifact_dir: str | Path,
    epochs: int = 80,
    patience: int = 12,
    batch_size: int = 64,
) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass

    data = load_dataset(csv_path)
    train_idx, val_idx, test_idx = chronological_split(data.timestamps)
    preprocessor = CausalPreprocessor().fit(data.features[train_idx])
    x_train = preprocessor.transform(data.features[train_idx])
    x_val = preprocessor.transform(data.features[val_idx])
    x_test = preprocessor.transform(data.features[test_idx])
    y_train, y_val, y_test = data.labels[train_idx], data.labels[val_idx], data.labels[test_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaferGuardSAE(x_train.shape[1]).to(device)
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    positive = max(1, int(y_train.sum()))
    negative = int(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_ap = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            reconstruction, logits = model(xb)
            if reconstruction.shape != xb.shape or logits.shape != yb.shape:
                raise RuntimeError(
                    f"Loss shape mismatch: x={tuple(xb.shape)}, "
                    f"reconstruction={tuple(reconstruction.shape)}, "
                    f"y={tuple(yb.shape)}, logits={tuple(logits.shape)}"
                )
            reconstruction_loss = torch.mean((reconstruction - xb) ** 2)
            classification_loss = bce(logits, yb)
            loss = 0.25 * reconstruction_loss + classification_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            totals += (loss.item(), reconstruction_loss.item(), classification_loss.item())
            batches += 1
        scheduler.step()

        _, val_scores = _predict(model, x_val, device)
        val_ap = average_precision(y_val, val_scores)
        history.append({
            "epoch": float(epoch),
            "loss": float(totals[0] / max(1, batches)),
            "reconstruction_loss": float(totals[1] / max(1, batches)),
            "classification_loss": float(totals[2] / max(1, batches)),
            "validation_average_precision": val_ap,
        })
        if val_ap > best_ap + 1e-8:
            best_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:02d} loss={history[-1]['loss']:.4f} "
                f"val_ap={val_ap:.4f} best={best_ap:.4f}@{best_epoch}"
            )
        if no_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint")
    model.load_state_dict(best_state)
    _, val_scores = _predict(model, x_val, device)
    threshold = choose_threshold(y_val, val_scores, beta=2.0)
    _, test_scores = _predict(model, x_test, device)
    test_metrics = _evaluate(y_test, test_scores, threshold)

    legacy_total = int(np.sum(np.isfinite(data.features[:, 0]) & (data.features[:, 0] == 0.0)))
    output = {
        "seed": SEED,
        "device": str(device),
        "model": "WaferGuardSAE",
        "best_epoch": best_epoch,
        "features": int(x_train.shape[1]),
        "training": {
            "rows": int(len(train_idx)),
            "defects": int(y_train.sum()),
            "prevalence": float(y_train.mean()),
            "start": data.timestamps[int(train_idx[0])].isoformat(),
            "end": data.timestamps[int(train_idx[-1])].isoformat(),
        },
        "validation": {
            "rows": int(len(val_idx)),
            "defects": int(y_val.sum()),
            "prevalence": float(y_val.mean()),
            "start": data.timestamps[int(val_idx[0])].isoformat(),
            "end": data.timestamps[int(val_idx[-1])].isoformat(),
            "threshold_selection": "maximum validation F2; ties prefer precision then higher threshold",
            "selected_threshold": float(threshold),
            "average_precision": average_precision(y_val, val_scores),
        },
        "test": {
            **test_metrics,
            "start": data.timestamps[int(test_idx[0])].isoformat(),
            "end": data.timestamps[int(test_idx[-1])].isoformat(),
        },
        "dataset": {
            "rows": int(len(data.labels)),
            "sensors": int(data.features.shape[1]),
            "defects": int(data.labels.sum()),
            "prevalence": float(data.labels.mean()),
            "legacy_rows": legacy_total,
        },
        "history": history,
    }

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), artifact_dir / "wafer_guard_state.pt")
    (artifact_dir / "model_config.json").write_text(
        json.dumps({"model": "WaferGuardSAE", "input_features": int(x_train.shape[1]), "hidden": 128, "latent": 24}, indent=2),
        encoding="utf-8",
    )
    preprocessor.save(str(artifact_dir / "preprocessor.npz"))
    (artifact_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate WaferGuard in chronological order")
    parser.add_argument("--csv", default="data/secom_telemetry_modified.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.csv, args.artifacts, epochs=args.epochs), indent=2))


if __name__ == "__main__":
    main()

