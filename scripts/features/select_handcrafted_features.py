"""仅用每折训练集筛选低冗余手工特征，生成可供 DAEAC 融合读取的清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import load_config
from mitbih_2class.data import load_master, load_split, validate_split

DEFAULT_CONFIG = ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
DEFAULT_OUTPUT = ROOT / "protocols" / "handcrafted_selection_e3_auc_corr_k64"


def sha256_file(path: Path) -> str:
    """计算输入文件摘要，便于确认服务器和本机使用同一份特征。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_features(
    values: np.ndarray,
    labels: np.ndarray,
    names: tuple[str, ...],
    *,
    max_features: int = 64,
    correlation_threshold: float = 0.90,
    maximum_missing_fraction: float = 0.20,
) -> dict[str, object]:
    """按训练集AUC排序，相关性过高时仅保留更靠前的代表特征。"""

    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("特征矩阵与名称数量不一致")
    if len(values) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("训练标签必须包含完整二分类")
    if max_features < 1 or not 0 < correlation_threshold < 1:
        raise ValueError("max_features 和 correlation_threshold 无效")

    finite = np.isfinite(values)
    missing_fraction = 1.0 - finite.mean(axis=0)
    medians = np.zeros(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        observed = values[finite[:, column], column]
        if len(observed):
            medians[column] = np.median(observed)
    filled = np.where(finite, values, medians)
    standard_deviation = filled.std(axis=0)
    eligible = np.flatnonzero(
        (missing_fraction <= maximum_missing_fraction) & (standard_deviation > 1e-8)
    )
    if len(eligible) < max_features:
        raise ValueError(f"质量筛选后仅剩 {len(eligible)} 项，少于目标 {max_features} 项")

    relevance = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column in eligible:
        auc = roc_auc_score(labels, filled[:, column])
        relevance[column] = abs(float(auc) - 0.5) * 2.0

    # Spearman 相关等价于秩变换后的 Pearson 相关；只对合格列计算。
    ranked = np.column_stack([rankdata(filled[:, column]) for column in eligible])
    correlation = np.corrcoef(ranked, rowvar=False)
    local_index = {int(column): position for position, column in enumerate(eligible)}
    ordered = sorted(eligible.tolist(), key=lambda index: (-relevance[index], names[index]))
    selected: list[int] = []
    redundancy: list[dict[str, object]] = []
    for candidate in ordered:
        candidate_position = local_index[candidate]
        conflicts = [
            (kept, abs(float(correlation[candidate_position, local_index[kept]])))
            for kept in selected
            if abs(float(correlation[candidate_position, local_index[kept]]))
            >= correlation_threshold
        ]
        if conflicts:
            representative, rho = max(conflicts, key=lambda item: item[1])
            redundancy.append({
                "excluded_index": candidate,
                "excluded_name": names[candidate],
                "representative_index": representative,
                "representative_name": names[representative],
                "absolute_spearman": rho,
            })
            continue
        selected.append(candidate)
        if len(selected) == max_features:
            break
    if len(selected) < max_features:
        raise ValueError(
            f"相关性筛选后只能保留 {len(selected)} 项；请降低数量或提高相关阈值"
        )

    return {
        "source_feature_count": values.shape[1],
        "selected_indices": selected,
        "selected_names": [names[index] for index in selected],
        "selected_relevance": [float(relevance[index]) for index in selected],
        "quality_excluded_indices": [
            int(index) for index in range(values.shape[1]) if index not in set(eligible.tolist())
        ],
        "redundancy_exclusions_encountered": redundancy,
        "criteria": {
            "relevance": "2 * abs(train ROC AUC - 0.5)",
            "maximum_missing_fraction": maximum_missing_fraction,
            "minimum_standard_deviation": 1e-8,
            "absolute_spearman_threshold": correlation_threshold,
            "max_features": max_features,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    parser.add_argument("--maximum-missing-fraction", type=float, default=0.20)
    args = parser.parse_args()

    config = load_config(args.config)
    master = load_master(config)
    if master.handcrafted_signals is None or master.handcrafted_names is None:
        raise ValueError("配置没有加载手工特征融合输入")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"筛选目录已有内容，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        fold_config = config.with_overrides(fold=fold)
        split = load_split(fold_config.split_path)
        validate_split(master, split, split_unit=fold_config.split_unit)
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        result = select_features(
            master.handcrafted_signals[train_indices],
            master.labels[train_indices],
            master.handcrafted_names,
            max_features=args.max_features,
            correlation_threshold=args.correlation_threshold,
            maximum_missing_fraction=args.maximum_missing_fraction,
        )
        result.update({
            "fold": fold,
            "selection_rows": "train_indices only",
            "train_n": len(train_indices),
            "train_positive_n": int(master.labels[train_indices].sum()),
            "split_path": str(fold_config.split_path),
            "split_sha256": sha256_file(fold_config.split_path),
            "feature_path": str(config.handcrafted_path),
            "feature_sha256": sha256_file(config.handcrafted_path),
        })
        (output / f"fold_{fold}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"fold {fold}: selected {len(result['selected_indices'])} features")

    manifest = {
        "method": "train-only quality filter + univariate AUC ranking + Spearman pruning",
        "folds": 5,
        "max_features": args.max_features,
        "correlation_threshold": args.correlation_threshold,
        "maximum_missing_fraction": args.maximum_missing_fraction,
        "test_or_validation_labels_used": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
