"""用相同五折和HistGradientBoosting比较全部263维与按折筛选64维特征。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import load_config
from mitbih_2class.data import apply_handcrafted_selection, load_master, load_split, validate_split
from mitbih_2class.metrics import compute_metrics

DEFAULT_CONFIG = ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
DEFAULT_SELECTION = ROOT / "protocols" / "handcrafted_selection_e3_auc_corr_k64"
DEFAULT_OUTPUT = ROOT / "runs" / "handcrafted_selection_comparison_e3_k64"


def build_estimator(seed: int):
    """保持与既有全部手工特征表格基线相同的固定参数。"""

    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
    )


def selection_stability(selections: list[set[int]]) -> dict[str, object]:
    """统计五折选择集合的Jaccard重叠，不读取任何标签或预测。"""

    pairwise = []
    for left in range(len(selections)):
        for right in range(left + 1, len(selections)):
            union = selections[left] | selections[right]
            pairwise.append(len(selections[left] & selections[right]) / len(union))
    counts: dict[int, int] = {}
    for selected in selections:
        for index in selected:
            counts[index] = counts.get(index, 0) + 1
    return {
        "pairwise_jaccard_mean": float(np.mean(pairwise)),
        "pairwise_jaccard_min": float(np.min(pairwise)),
        "selected_in_all_folds": sum(value == len(selections) for value in counts.values()),
        "selected_in_at_least_four_folds": sum(value >= 4 for value in counts.values()),
    }


def write_report(path: Path, metrics: dict[str, dict[str, object]], stability: dict[str, object]) -> None:
    """生成可直接阅读的筛选前后五折OOF对比。"""

    lines = [
        "# 手工特征选择前后对比", "",
        "两组均使用同一E3冻结五折、同一HistGradientBoosting参数和0.5阈值。",
        "`selected64` 的列在每折只由该折训练集决定；验证与测试标签不参与筛选。", "",
        "| 特征组 | 维度 | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, dimension in (("all263", 263), ("selected64", 64)):
        row = metrics[group]
        lines.append(
            f"| `{group}` | {dimension} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['sensitivity']:.4f} | "
            f"{row['specificity']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |"
        )
    lines.extend([
        "", "## 五折选择稳定性", "",
        f"- 折间Jaccard均值：{stability['pairwise_jaccard_mean']:.3f}",
        f"- 折间Jaccard最小值：{stability['pairwise_jaccard_min']:.3f}",
        f"- 五折均入选：{stability['selected_in_all_folds']}项",
        f"- 至少四折入选：{stability['selected_in_at_least_four_folds']}项", "",
        "该结果用于判断降维是否保留预测信息。最终仍需比较DAEAC融合版的完整263维和筛选64维。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    master = load_master(config)
    if master.handcrafted_signals is None or master.handcrafted_names is None:
        raise ValueError("配置没有加载完整手工特征")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"对比目录已有内容，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, list[dict[str, object]]] = {"all263": [], "selected64": []}
    fold_rows: list[dict[str, object]] = []
    selection_sets: list[set[int]] = []
    for fold in range(5):
        fold_config = config.with_overrides(fold=fold)
        split = load_split(fold_config.split_path)
        validate_split(master, split, split_unit=fold_config.split_unit)
        selection_path = args.selection_dir.resolve() / f"fold_{fold}.json"
        selected_matrix, _ = apply_handcrafted_selection(
            master.handcrafted_signals, master.handcrafted_names, selection_path
        )
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_sets.append(set(map(int, payload["selected_indices"])))
        matrices = {"all263": master.handcrafted_signals, "selected64": selected_matrix}
        for group, matrix in matrices.items():
            estimator = build_estimator(args.seed + fold)
            train_indices = np.asarray(split["train_indices"], dtype=np.int64)
            test_indices = np.asarray(split["test_indices"], dtype=np.int64)
            estimator.fit(matrix[train_indices], master.labels[train_indices])
            probability = estimator.predict_proba(matrix[test_indices])[:, 1]
            predicted = (probability >= 0.5).astype(np.int64)
            metrics = compute_metrics(master.labels[test_indices], predicted, probability)
            fold_rows.append({
                "group": group,
                "fold": fold,
                "feature_count": matrix.shape[1],
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            })
            predictions[group].extend(
                {
                    "dataset_index": int(dataset_index),
                    "fold": fold,
                    "true_label": int(label),
                    "prob_apnea": float(score),
                    "pred_label": int(prediction),
                }
                for dataset_index, label, score, prediction in zip(
                    test_indices, master.labels[test_indices], probability, predicted, strict=True
                )
            )
            print(f"{group} fold {fold}: accuracy={metrics['accuracy']:.4f}", flush=True)

    fieldnames = list(fold_rows[0])
    with (output / "fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)

    oof_metrics: dict[str, dict[str, object]] = {}
    for group, rows in predictions.items():
        rows.sort(key=lambda row: int(row["dataset_index"]))
        if [int(row["dataset_index"]) for row in rows] != list(range(len(master))):
            raise ValueError(f"{group}: OOF预测未完整覆盖数据集")
        with (output / f"oof_{group}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        oof_metrics[group] = compute_metrics(
            [int(row["true_label"]) for row in rows],
            [int(row["pred_label"]) for row in rows],
            [float(row["prob_apnea"]) for row in rows],
        )
    stability = selection_stability(selection_sets)
    result = {
        "protocol": "E3 epoch-level frozen five-fold",
        "model": "HistGradientBoostingClassifier",
        "sklearn_version": sklearn_version,
        "metrics": oof_metrics,
        "selection_stability": stability,
    }
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output / "comparison.md", oof_metrics, stability)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
