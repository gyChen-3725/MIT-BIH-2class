"""固定使用MI-128-RF完成E3样本级五折测试并汇总OOF指标。

每折均使用 ``fusion3_expanded`` 特征组、互信息前128维和随机森林。由于整套配方
已经固定，不再使用验证集选择超参数；该折训练集与验证集合并后重新计算互信息排名、
拟合RF，最后只在测试集评价一次。脚本同时保存每折结果、OOF结果和运行日志。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "selection"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

from optimize_feature_dimension import fit_feature_ranking
from train_handcrafted_2class import (
    DEFAULT_FEATURES,
    DEFAULT_MASTER,
    DEFAULT_SPLITS,
    evaluate,
    load_design_matrix,
    load_fold,
    sha256_file,
)
from train_handcrafted_2class import build_model as build_table_model

FEATURE_GROUP = "fusion3_expanded"
SELECTOR = "mutual_info"
DIMENSION = 128
MODEL_NAME = "random_forest"
DEFAULT_OUTPUT = ROOT / "runs" / "mi128_rf_fixed_e3_epoch5fold_2class"


def build_logger(log_path: Path) -> logging.Logger:
    """同时写控制台和UTF-8日志，避免再次丢失完整运行过程。"""

    logger = logging.getLogger("mi128_rf_5fold")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_path = args.features.resolve()
    master_path = args.master.resolve()
    split_dir = args.split_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录已有内容，避免覆盖历史结果：{output}")
    output.mkdir(parents=True, exist_ok=True)
    logger = build_logger(output / "run.log")
    started = time.perf_counter()

    logger.info("读取347维扩展特征并核对MIT master行顺序")
    matrices, labels, metadata, names_by_group = load_design_matrix(
        feature_path, master_path
    )
    matrix = matrices[FEATURE_GROUP]
    feature_names = names_by_group[FEATURE_GROUP]
    if matrix.shape[1] < DIMENSION:
        raise ValueError(
            f"{FEATURE_GROUP}只有{matrix.shape[1]}维，无法选择{DIMENSION}维"
        )

    oof_probability = np.full(len(labels), np.nan, dtype=np.float64)
    oof_fold = np.full(len(labels), -1, dtype=np.int16)
    fold_metric_rows: list[dict[str, object]] = []

    for fold in range(5):
        fold_started = time.perf_counter()
        split = load_fold(split_dir / f"fold_{fold}.json", len(labels))
        development = np.concatenate((split["train"], split["val"]))
        test = split["test"]
        logger.info(
            "Fold %d开始：开发集=%d，测试集=%d，计算互信息排名",
            fold,
            len(development),
            len(test),
        )
        ranking = fit_feature_ranking(
            matrix[development], labels[development], SELECTOR, args.seed + fold
        )
        selected = ranking[:DIMENSION]
        selected_names = [feature_names[int(index)] for index in selected]

        logger.info("Fold %d拟合%d维随机森林", fold, DIMENSION)
        estimator = build_table_model(MODEL_NAME, args.seed + fold)
        estimator.fit(matrix[development][:, selected], labels[development])
        probability = estimator.predict_proba(matrix[test][:, selected])[:, 1]
        metrics = evaluate(labels[test], probability)
        predicted = (probability >= 0.5).astype(np.int64)

        oof_probability[test] = probability
        oof_fold[test] = fold
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics = {
            **metrics,
            "fold": fold,
            "feature_group": FEATURE_GROUP,
            "selector": SELECTOR,
            "dimension": DIMENSION,
            "model": MODEL_NAME,
            "seed": args.seed + fold,
            "ranking_fit_rows": "train+validation",
            "test_labels_used_for_selection": False,
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(fold_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (fold_dir / "selected_features.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "source_feature_group": FEATURE_GROUP,
                    "source_feature_count": matrix.shape[1],
                    "selector": SELECTOR,
                    "dimension": DIMENSION,
                    "ranking_fit_rows": "train+validation",
                    "selected_group_indices": selected.tolist(),
                    "selected_names": selected_names,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        fold_predictions = pd.DataFrame(
            {
                "dataset_index": test,
                "fold": fold,
                "subject_id": metadata["subject_id"][test],
                "record": metadata["record"][test],
                "true_label": labels[test],
                "pred_label": predicted,
                "prob_apnea": probability,
            }
        ).sort_values("dataset_index")
        fold_predictions.to_csv(
            fold_dir / "predictions.csv", index=False, encoding="utf-8-sig"
        )
        fold_metric_rows.append(
            {
                key: value
                for key, value in fold_metrics.items()
                if key != "confusion_matrix"
            }
        )
        logger.info(
            "Fold %d完成：accuracy=%.4f，balanced_accuracy=%.4f，"
            "macro_f1=%.4f，roc_auc=%.4f，耗时=%.1f秒",
            fold,
            float(metrics["accuracy"]),
            float(metrics["balanced_accuracy"]),
            float(metrics["macro_f1"]),
            float(metrics["roc_auc"]),
            fold_metrics["elapsed_seconds"],
        )

    if np.isnan(oof_probability).any() or np.any(oof_fold < 0):
        raise RuntimeError("五折测试预测没有完整覆盖全部样本")
    oof_predicted = (oof_probability >= 0.5).astype(np.int64)
    oof_metrics = evaluate(labels, oof_probability)
    oof_payload = {
        **oof_metrics,
        "experiment": "mi128_rf_fixed_e3_epoch5fold_2class",
        "folds": 5,
        "split_unit": "epoch",
        "feature_group": FEATURE_GROUP,
        "selector": SELECTOR,
        "dimension": DIMENSION,
        "model": MODEL_NAME,
        "elapsed_seconds": time.perf_counter() - started,
    }
    oof_dir = output / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    (oof_dir / "metrics.json").write_text(
        json.dumps(oof_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "dataset_index": np.arange(len(labels)),
            "fold": oof_fold,
            "subject_id": metadata["subject_id"],
            "record": metadata["record"],
            "true_label": labels,
            "pred_label": oof_predicted,
            "prob_apnea": oof_probability,
        }
    ).to_csv(oof_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metric_rows).to_csv(
        output / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "experiment": "mi128_rf_fixed_e3_epoch5fold_2class",
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "master_path": str(master_path),
        "split_dir": str(split_dir),
        "feature_group": FEATURE_GROUP,
        "selector": SELECTOR,
        "dimension": DIMENSION,
        "model": MODEL_NAME,
        "base_seed": args.seed,
        "sklearn": sklearn.__version__,
        "development_rows": "current fold train+validation",
        "test_labels_used_for_selection": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "五折完成：OOF accuracy=%.4f，balanced_accuracy=%.4f，"
        "macro_f1=%.4f，roc_auc=%.4f，总耗时=%.1f秒",
        float(oof_metrics["accuracy"]),
        float(oof_metrics["balanced_accuracy"]),
        float(oof_metrics["macro_f1"]),
        float(oof_metrics["roc_auc"]),
        oof_payload["elapsed_seconds"],
    )
    logger.info("结果目录：%s", output)


if __name__ == "__main__":
    main()
