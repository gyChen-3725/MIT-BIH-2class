"""以受试者内统计生成完整手工特征排序，不使用验证集或测试集。"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import load_config
from mitbih_2class.data import load_master, load_split, validate_split

DEFAULT_CONFIG = ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
DEFAULT_OUTPUT = ROOT / "protocols" / "handcrafted_ranking_e3_subject_aware_v2"


def subject_aware_relevance(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    minimum_per_class: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按受试者分别计算AUC，再用方向一致性修正跨人中位效应。"""

    feature_count = values.shape[1]
    score = np.zeros(feature_count, dtype=np.float64)
    median_auc = np.full(feature_count, 0.5, dtype=np.float64)
    consistency = np.zeros(feature_count, dtype=np.float64)
    subject_count = np.zeros(feature_count, dtype=np.int64)
    for column in range(feature_count):
        aucs = []
        for subject in np.unique(subjects):
            selected = (subjects == subject) & np.isfinite(values[:, column])
            target = labels[selected]
            positive = int(target.sum())
            negative = len(target) - positive
            if min(positive, negative) < minimum_per_class:
                continue
            aucs.append(float(roc_auc_score(target, values[selected, column])))
        if not aucs:
            continue
        auc_array = np.asarray(aucs)
        median = float(np.median(auc_array))
        direction = np.sign(median - 0.5)
        agreement = float(np.mean(np.sign(auc_array - 0.5) == direction))
        median_auc[column] = median
        consistency[column] = agreement
        subject_count[column] = len(aucs)
        score[column] = 2.0 * abs(median - 0.5) * agreement
    return score, median_auc, consistency, subject_count


def within_subject_absolute_spearman(
    values: np.ndarray,
    subjects: np.ndarray,
    eligible: np.ndarray,
    *,
    minimum_rows: int = 30,
) -> np.ndarray:
    """逐受试者计算秩相关后取绝对值中位数，降低个体幅度差异影响。"""

    matrices = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        if int(selected.sum()) < minimum_rows:
            continue
        local = values[selected][:, eligible]
        ranked = np.column_stack([rankdata(local[:, column]) for column in range(local.shape[1])])
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(ranked, rowvar=False)
        correlation[~np.isfinite(correlation)] = np.nan
        matrices.append(np.abs(correlation))
    if not matrices:
        raise ValueError("没有受试者满足相关性计算的最少窗口数")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmedian(np.stack(matrices), axis=0)
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 1.0)
    return result


def rank_features_v2(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    names: tuple[str, ...],
    *,
    correlation_threshold: float = 0.95,
    maximum_missing_fraction: float = 0.20,
) -> dict[str, object]:
    """完整扫描所有字段：先排非冗余代表，再附加相关簇中的次级特征。"""

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
    relevance, median_auc, consistency, subject_count = subject_aware_relevance(
        values, labels, subjects
    )
    correlation = within_subject_absolute_spearman(filled, subjects, eligible)
    local_index = {int(column): position for position, column in enumerate(eligible)}
    ordered = sorted(eligible.tolist(), key=lambda index: (-relevance[index], names[index]))

    representatives: list[int] = []
    deferred: list[int] = []
    redundancy = []
    for candidate in ordered:
        position = local_index[candidate]
        conflicts = [
            (kept, float(correlation[position, local_index[kept]]))
            for kept in representatives
            if correlation[position, local_index[kept]] >= correlation_threshold
        ]
        if not conflicts:
            representatives.append(candidate)
            continue
        representative, rho = max(conflicts, key=lambda item: item[1])
        deferred.append(candidate)
        redundancy.append({
            "feature": names[candidate],
            "representative": names[representative],
            "median_absolute_within_subject_spearman": rho,
        })

    quality_excluded = [index for index in range(values.shape[1]) if index not in set(eligible)]
    ranking = representatives + deferred + quality_excluded
    if len(ranking) != values.shape[1] or len(set(ranking)) != values.shape[1]:
        raise RuntimeError("v2排序没有完整覆盖原始特征")
    return {
        "source_feature_count": values.shape[1],
        "ranked_indices": ranking,
        "ranked_names": [names[index] for index in ranking],
        "nonredundant_representative_count": len(representatives),
        "quality_excluded_indices": quality_excluded,
        "redundancy": redundancy,
        "statistics": {
            names[index]: {
                "subject_aware_score": float(relevance[index]),
                "median_within_subject_auc": float(median_auc[index]),
                "direction_consistency": float(consistency[index]),
                "eligible_subject_count": int(subject_count[index]),
                "missing_fraction": float(missing_fraction[index]),
            }
            for index in range(values.shape[1])
        },
        "criteria": {
            "relevance": "2 * abs(median within-subject AUC - 0.5) * direction consistency",
            "correlation": "median absolute within-subject Spearman",
            "correlation_threshold": correlation_threshold,
            "maximum_missing_fraction": maximum_missing_fraction,
            "ranking": "all nonredundant representatives first, then correlated members",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--maximum-missing-fraction", type=float, default=0.20)
    args = parser.parse_args()

    config = load_config(args.config)
    master = load_master(config)
    if master.handcrafted_signals is None or master.handcrafted_names is None:
        raise ValueError("配置没有加载完整手工特征")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"v2排序目录已有内容，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)
    representatives = []
    for fold in range(5):
        fold_config = config.with_overrides(fold=fold)
        split = load_split(fold_config.split_path)
        validate_split(master, split, split_unit=fold_config.split_unit)
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        result = rank_features_v2(
            master.handcrafted_signals[train_indices],
            master.labels[train_indices],
            master.subjects[train_indices],
            master.handcrafted_names,
            correlation_threshold=args.correlation_threshold,
            maximum_missing_fraction=args.maximum_missing_fraction,
        )
        result.update({"fold": fold, "selection_rows": "train_indices only", "train_n": len(train_indices)})
        representatives.append(result["nonredundant_representative_count"])
        (output / f"fold_{fold}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"fold {fold}: {result['nonredundant_representative_count']} representatives")
    manifest = {
        "version": 2,
        "folds": 5,
        "uses_train_indices_only": True,
        "uses_validation_or_test_labels": False,
        "correlation_threshold": args.correlation_threshold,
        "maximum_missing_fraction": args.maximum_missing_fraction,
        "nonredundant_representative_counts": representatives,
        "candidate_sizes": [64, 96, 128, 160, 192, 263],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
