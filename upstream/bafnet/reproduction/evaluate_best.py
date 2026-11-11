"""额外评估两阶段训练中验证 accuracy 最佳的 checkpoint。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from model import create_model
from train import DEFAULT_RUN, compile_model, evaluate_records, evaluate_segments

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "artifacts" / "bafnet_5min.npz")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as data:
        test = data["split"] == 1
        features = data["features"][test]
        labels = data["labels"][test]
        records = data["record"][test]
        minutes = data["minute"][test]
    model = create_model()
    compile_model(model)
    model.load_weights(args.run / "best.weights.h5")
    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))
    run_metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    best_stage = ""
    best_epoch = 0
    best_val_accuracy = -1.0
    for stage in ("stage1", "stage2"):
        for epoch, accuracy in enumerate(history[stage]["val_outputs_accuracy"], start=1):
            if accuracy > best_val_accuracy:
                best_stage, best_epoch, best_val_accuracy = stage, epoch, accuracy
    scores_path = args.run / "best_test_scores.csv"
    segment_metrics = evaluate_segments(model, features, labels, records, minutes, scores_path)
    record_metrics = (
        evaluate_records(scores_path)
        if run_metrics.get("record_eval", "apnea_ecg") == "apnea_ecg" else None
    )
    result = {
        "checkpoint": "best val_outputs_accuracy",
        "selected_stage": best_stage,
        "selected_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "segment_test": segment_metrics,
        "record_test": record_metrics,
    }
    (args.run / "best_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("segment_test", segment_metrics)
    if record_metrics is not None:
        print("record_test", {k: v for k, v in record_metrics.items() if k != "predicted_events_per_hour"})


if __name__ == "__main__":
    main()
