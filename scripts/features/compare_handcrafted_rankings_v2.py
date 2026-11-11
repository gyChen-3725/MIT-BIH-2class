"""比较v2完整排序在不同保留维度下的五折OOF分类指标。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "features"))

from compare_handcrafted_selection import build_estimator, selection_stability

from mitbih_2class.config import load_config
from mitbih_2class.data import load_master, load_split, validate_split
from mitbih_2class.metrics import compute_metrics

DEFAULT_CONFIG = ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
DEFAULT_RANKING = ROOT / "protocols" / "handcrafted_ranking_e3_subject_aware_v2"
DEFAULT_OUTPUT = ROOT / "runs" / "handcrafted_selection_v2_dimension_sweep"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ranking-dir", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[64, 96, 128, 160, 192, 263]
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    if not sizes or sizes[0] < 1 or sizes[-1] > 263:
        raise ValueError("候选维度必须位于1到263")
    config = load_config(args.config)
    master = load_master(config)
    if master.handcrafted_signals is None or master.handcrafted_names is None:
        raise ValueError("配置没有加载完整手工特征")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"维度对比目录已有内容，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    predictions: dict[int, list[dict[str, object]]] = {size: [] for size in sizes}
    selections: dict[int, list[set[int]]] = {size: [] for size in sizes}
    fold_rows = []
    for fold in range(5):
        fold_config = config.with_overrides(fold=fold)
        split = load_split(fold_config.split_path)
        validate_split(master, split, split_unit=fold_config.split_unit)
        ranking_path = args.ranking_dir.resolve() / f"fold_{fold}.json"
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        indices = list(map(int, ranking["ranked_indices"]))
        names = [master.handcrafted_names[index] for index in indices]
        if names != list(map(str, ranking["ranked_names"])) or len(indices) != 263:
            raise ValueError(f"{ranking_path}: 排序与当前特征矩阵不一致")
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        for size in sizes:
            selected = np.asarray(indices[:size], dtype=np.int64)
            selections[size].append(set(selected.tolist()))
            matrix = master.handcrafted_signals[:, selected]
            estimator = build_estimator(args.seed + fold)
            estimator.fit(matrix[train_indices], master.labels[train_indices])
            probability = estimator.predict_proba(matrix[test_indices])[:, 1]
            predicted = (probability >= 0.5).astype(np.int64)
            metrics = compute_metrics(master.labels[test_indices], predicted, probability)
            fold_rows.append({
                "feature_count": size,
                "fold": fold,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            })
            predictions[size].extend(
                {
                    "dataset_index": int(dataset_index),
                    "fold": fold,
                    "true_label": int(label),
                    "pred_label": int(prediction),
                    "prob_apnea": float(score),
                }
                for dataset_index, label, prediction, score in zip(
                    test_indices, master.labels[test_indices], predicted, probability, strict=True
                )
            )
            print(f"fold {fold} k={size}: accuracy={metrics['accuracy']:.4f}", flush=True)

    with (output / "fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    summary = {}
    for size, rows in predictions.items():
        rows.sort(key=lambda row: int(row["dataset_index"]))
        if [int(row["dataset_index"]) for row in rows] != list(range(len(master))):
            raise ValueError(f"k={size}: OOF预测未完整覆盖数据集")
        metrics = compute_metrics(
            [int(row["true_label"]) for row in rows],
            [int(row["pred_label"]) for row in rows],
            [float(row["prob_apnea"]) for row in rows],
        )
        summary[str(size)] = {
            "metrics": metrics,
            "selection_stability": selection_stability(selections[size]),
        }
    result = {
        "ranking": "subject-aware AUC + within-subject Spearman v2",
        "protocol": "E3 epoch-level frozen five-fold",
        "model": "HistGradientBoostingClassifier",
        "candidate_sizes": sizes,
        "results": summary,
        "warning": "Dimension sweep is exploratory; do not repeatedly tune on these test folds.",
    }
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# v2手工特征维度对比", "",
        "| 维度 | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | ROC AUC | 折间Jaccard |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        row = summary[str(size)]["metrics"]
        stability = summary[str(size)]["selection_stability"]
        lines.append(
            f"| {size} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['sensitivity']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | "
            f"{row['roc_auc']:.4f} | {stability['pairwise_jaccard_mean']:.3f} |"
        )
    lines.extend([
        "", "所有排序只读取对应外层折的训练索引。候选维度使用测试折比较，",
        "因此本表用于探索合理容量；最终结论需要新的受试者级或独立测试。",
    ])
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
