"""汇总Fold 0原始DAEAC、mRMR-64、MI-128门控与RF概率融合结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = {
    "original_daeac": (
        ROOT / "runs" / "daeac_filtered_rhythm_amplitude_e3_epoch5fold_2class" / "fold_0"
    ),
    "mrmr64_concat": (
        ROOT / "runs"
        / "daeac_filtered_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold_2class"
        / "fold_0"
    ),
    "mi128_gated": (
        ROOT / "runs"
        / "daeac_filtered_rhythm_amplitude_handcrafted_mi128_gated_fold0_2class"
        / "fold_0"
    ),
    "mi128_rf_probability": (
        ROOT / "runs" / "daeac_mi128_rf_probability_fusion_fold0_2class" / "fold_0"
    ),
}
METRICS = (
    "accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity",
    "f1", "macro_f1", "mcc", "roc_auc", "average_precision",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "fold0_daeac_fusion_comparison.csv",
    )
    args = parser.parse_args()

    missing = [
        str(path / "metrics.json")
        for path in DEFAULT_RUNS.values()
        if not (path / "metrics.json").is_file()
    ]
    if missing:
        raise FileNotFoundError("以下实验尚未完成：\n" + "\n".join(missing))
    rows = []
    for experiment, run_dir in DEFAULT_RUNS.items():
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": experiment,
                **{metric: payload.get(metric) for metric in METRICS},
                "best_epoch": payload.get("best_epoch", payload.get("daeac_best_epoch")),
                "run_dir": str(run_dir),
            }
        )
    frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
