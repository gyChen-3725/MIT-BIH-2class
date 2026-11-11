"""校验五个测试折的预测文件，并计算整体折外（OOF）指标。

每个样本必须且只能在一个测试折中出现。先验证逐折覆盖，再验证全数据覆盖，
最后按 master 数据的行索引排序并计算总体指标。
"""

from __future__ import annotations

import csv
import json

import numpy as np

from mitbih_2class.config import ExperimentConfig
from mitbih_2class.data import load_split
from mitbih_2class.metrics import compute_metrics


def aggregate_oof(config: ExperimentConfig) -> dict[str, object]:
    """汇总完整五折的预测；缺折、重复或漏样本时拒绝生成 OOF 结果。

    该函数读取已经训练完成的逐折 CSV，不重新推理，也不重新挑选模型。
    """

    pooled: list[dict[str, str]] = []
    for fold in range(5):
        predictions_path = config.output_root / config.name / f"fold_{fold}" / "predictions.csv"
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Missing fold predictions: {predictions_path}")
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = set(load_split(config.split_dir / f"fold_{fold}.json")["test_indices"])
        observed = {int(row["dataset_index"]) for row in rows}
        # 集合相等检查漏样本；行数相等同时排除同一折中的重复预测。
        if observed != expected or len(rows) != len(expected):
            raise ValueError(
                f"Fold {fold} predictions do not exactly cover its test split: "
                f"rows={len(rows)}, unique={len(observed)}, expected={len(expected)}"
            )
        pooled.extend(rows)

    indices = [int(row["dataset_index"]) for row in pooled]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate dataset indices found across held-out folds")
    with np.load(config.data_path, allow_pickle=False) as archive:
        expected_total = len(archive["raw_label"])
    if set(indices) != set(range(expected_total)):
        missing = expected_total - len(set(indices))
        raise ValueError(f"OOF predictions do not cover the full dataset (delta={missing})")

    # 统一按原始行号输出，方便与 master 数据、外部标签表逐行核对。
    pooled.sort(key=lambda row: int(row["dataset_index"]))
    metrics = compute_metrics(
        [int(row["true_label"]) for row in pooled],
        [int(row["pred_label"]) for row in pooled],
        [float(row["prob_apnea"]) for row in pooled],
    )
    metrics.update({"experiment": config.name, "folds": 5, "split_unit": config.split_unit})
    output_dir = config.output_root / config.name / "oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pooled[0]))
        writer.writeheader()
        writer.writerows(pooled)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics

