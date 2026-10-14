from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .model import WaferGuardSAE
from .preprocess import CausalPreprocessor


def predict_csv(input_path: str | Path, artifact_dir: str | Path, output_path: str | Path) -> int:
    artifact_dir = Path(artifact_dir)
    with (Path(input_path)).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = tuple(sorted((c for c in (reader.fieldnames or []) if c.startswith("sensor_")), key=lambda n: int(n.removeprefix("sensor_"))))
        rows = list(reader)
    if not names or any(not row for row in rows):
        raise ValueError("Input CSV must contain sensor_N columns and at least one row")
    x = np.array([[float(row[name]) if row.get(name, "").strip() else np.nan for name in names] for row in rows], dtype=np.float32)
    if x.shape[1] != 590:
        raise ValueError(f"Expected the 590 SECOM sensor columns, found {x.shape[1]}")
    preprocessor = CausalPreprocessor.load(str(artifact_dir / "preprocessor.npz"))
    model_cfg = json.loads((artifact_dir / "model_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaferGuardSAE(model_cfg["input_features"], model_cfg["hidden"], model_cfg["latent"]).to(device)
    model.load_state_dict(torch.load(artifact_dir / "wafer_guard_state.pt", map_location=device, weights_only=True))
    model.eval()
    prepared = torch.as_tensor(preprocessor.transform(x), dtype=torch.float32, device=device)
    with torch.no_grad():
        reconstruction, logits = model(prepared)
        scores = torch.sigmoid(logits).cpu().numpy()
        reconstruction_error = torch.mean((reconstruction-prepared)**2, dim=1).cpu().numpy()
    threshold = float(metrics["validation"]["selected_threshold"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(reader.fieldnames or []) + ["wafer_guard_score", "reconstruction_error", "review_flag"]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, score, error in zip(rows, scores, reconstruction_error, strict=True):
            writer.writerow({**row, "wafer_guard_score": f"{float(score):.7f}", "reconstruction_error": f"{float(error):.7f}", "review_flag": int(score >= threshold)})
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a CSV with the saved WaferGuard artifact")
    parser.add_argument("input_csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="wafer_guard_scores.csv")
    args = parser.parse_args()
    count = predict_csv(args.input_csv, args.artifacts, args.output)
    print(f"Scored {count} rows -> {args.output}")


if __name__ == "__main__":
    main()
