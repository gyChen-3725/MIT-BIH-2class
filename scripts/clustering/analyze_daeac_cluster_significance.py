"""检验已保存的 DAEAC 三组聚类与二分类标签的关联，不运行训练。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from cluster_daeac_features import COMBINATIONS
from plot_daeac_clusters import load_existing_clusters
from scipy.stats import chi2, mannwhitneyu, wilcoxon


def contingency(cluster_ids: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """构造簇×类别列联表；列 0 为 Normal，列 1 为 Apnea。"""

    return np.bincount(cluster_ids * 2 + labels, minlength=n_clusters * 2).reshape(
        n_clusters, 2
    )


def pearson_chi_square(table: np.ndarray) -> tuple[float, float]:
    """返回 Pearson χ² 和最小期望频数，供检查渐近近似的适用性。"""

    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    return float(np.sum((table - expected) ** 2 / expected)), float(expected.min())


def within_subject_permutation_p(
    assignments: dict[str, np.ndarray],
    labels: np.ndarray,
    subjects: np.ndarray,
    n_clusters: int,
    observed: dict[str, float],
    repetitions: int,
    seed: int,
    mode: str = "shuffle",
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """在每位受试者内部重排标签；循环平移还保留标签序列的成段结构。"""

    if mode not in {"shuffle", "circular_shift"}:
        raise ValueError(f"未知的受试者内置换方式：{mode}")

    groups = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    expected = {}
    for name in COMBINATIONS:
        table = contingency(assignments[name], labels, n_clusters)
        expected[name] = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    hits = {name: 0 for name in COMBINATIONS}
    null_statistics = {name: np.empty(repetitions, dtype=np.float64) for name in COMBINATIONS}
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        shuffled = labels.copy()
        for group in groups:
            if mode == "shuffle":
                shuffled[group] = shuffled[group][rng.permutation(len(group))]
            else:
                shuffled[group] = np.roll(shuffled[group], rng.integers(len(group)))
        for name in COMBINATIONS:
            table = contingency(assignments[name], shuffled, n_clusters)
            statistic = float(np.sum((table - expected[name]) ** 2 / expected[name]))
            null_statistics[name][repetition] = statistic
            hits[name] += statistic >= observed[name] - 1e-12
    p_values = {name: (hits[name] + 1) / (repetitions + 1) for name in COMBINATIONS}
    quantiles = {
        name: np.quantile(null_statistics[name], [0.05, 0.5, 0.95]).tolist()
        for name in COMBINATIONS
    }
    return p_values, quantiles


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """对本次三项聚类和两项特征检验进行 Holm 多重比较校正。"""

    adjusted = {}
    previous = 0.0
    total = len(p_values)
    for rank, (name, p_value) in enumerate(sorted(p_values.items(), key=lambda item: item[1])):
        previous = max(previous, min(1.0, (total - rank) * p_value))
        adjusted[name] = previous
    return adjusted


def feature_group_difference(values: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> dict:
    """同时报告样本级秩检验与受试者内差值检验，避免只看大样本 P 值。"""

    normal = values[labels == 0]
    apnea = values[labels == 1]
    rank_test = mannwhitneyu(apnea, normal, alternative="two-sided", method="asymptotic")
    subject_deltas = {}
    for subject in np.unique(subjects):
        subset = subjects == subject
        if np.any(subset & (labels == 0)) and np.any(subset & (labels == 1)):
            delta = values[subset & (labels == 1)].mean() - values[subset & (labels == 0)].mean()
            subject_deltas[str(subject)] = float(delta)
    paired = np.asarray(list(subject_deltas.values()), dtype=float)
    subject_test = wilcoxon(paired, alternative="two-sided", method="auto")
    return {
        "normal_mean": float(normal.mean()),
        "apnea_mean": float(apnea.mean()),
        "mean_difference_apnea_minus_normal": float(apnea.mean() - normal.mean()),
        "mann_whitney_p_epoch_level": float(rank_test.pvalue),
        "rank_biserial_apnea_vs_normal": float(2 * rank_test.statistic / (len(apnea) * len(normal)) - 1),
        "subjects_with_both_labels": len(subject_deltas),
        "subject_delta_median": float(np.median(paired)),
        "wilcoxon_p_subject_level": float(subject_test.pvalue),
        "subject_deltas": subject_deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.permutations < 99:
        raise ValueError("置换次数至少为 99")

    output, summary, inputs, assignments = load_existing_clusters(args.config, args.fold)
    with (output / "assignments.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["dataset_index"]) for row in rows] != assignments["dataset_indices"].tolist():
        raise ValueError("CSV 与 NPZ 样本顺序不一致")
    if np.any(np.diff(assignments["dataset_indices"]) <= 0):
        raise ValueError("循环平移检验要求训练样本按 master 行号保持时间顺序")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    subjects = np.asarray([row["subject_id"] for row in rows])
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("本脚本只分析 Normal/Apnea 二分类")

    observed = {}
    clusters = {}
    for name in COMBINATIONS:
        table = contingency(assignments[name], labels, summary["n_clusters"])
        statistic, minimum_expected = pearson_chi_square(table)
        observed[name] = statistic
        clusters[name] = {
            "table_normal_apnea": table.tolist(),
            "chi_square": statistic,
            "degrees_of_freedom": summary["n_clusters"] - 1,
            "chi_square_asymptotic_p": float(chi2.sf(statistic, summary["n_clusters"] - 1)),
            "minimum_expected_count": minimum_expected,
            "cramers_v": float(np.sqrt(statistic / len(labels))),
        }
    permutation_p, null_quantiles = within_subject_permutation_p(
        assignments, labels, subjects, summary["n_clusters"], observed,
        args.permutations, args.seed,
    )
    for name in COMBINATIONS:
        clusters[name]["within_subject_permutation_p"] = permutation_p[name]
        clusters[name]["permutation_null_chi_square_quantiles_05_50_95"] = null_quantiles[name]
    # 连续 30 秒 epoch 的标签有时间相关性；循环平移保留每人的标签段结构。
    circular_p, circular_null_quantiles = within_subject_permutation_p(
        assignments, labels, subjects, summary["n_clusters"], observed,
        args.permutations, args.seed + 1, mode="circular_shift",
    )
    for name in COMBINATIONS:
        clusters[name]["within_subject_circular_shift_p"] = circular_p[name]
        clusters[name]["circular_null_chi_square_quantiles_05_50_95"] = circular_null_quantiles[name]

    # 每个 30 秒 epoch 先归约成一个均值，避免把 256 个插值点当成独立样本。
    scaler = summary["scaler"]
    rri_mean = inputs["rri"].mean(axis=1) * scaler["std"][0] + scaler["mean"][0]
    amplitude_mean = inputs["amplitude"].mean(axis=1) * scaler["std"][1] + scaler["mean"][1]
    features = {
        "rri_mean_seconds_after_clip": feature_group_difference(rri_mean, labels, subjects),
        "amplitude_mean_after_asinh": feature_group_difference(amplitude_mean, labels, subjects),
    }
    primary_p = {
        **{name: clusters[name]["within_subject_circular_shift_p"] for name in COMBINATIONS},
        **{name: item["wilcoxon_p_subject_level"] for name, item in features.items()},
    }
    corrected = holm_adjust(primary_p)
    for name in COMBINATIONS:
        clusters[name]["holm_adjusted_circular_shift_p"] = corrected[name]
    for name, item in features.items():
        item["holm_adjusted_wilcoxon_p"] = corrected[name]

    result = {
        "fold": args.fold,
        "n": len(labels),
        "normal_n": int(np.sum(labels == 0)),
        "apnea_n": int(np.sum(labels == 1)),
        "subjects_n": len(np.unique(subjects)),
        "permutations": args.permutations,
        "seed": args.seed,
        "cluster_test": "Pearson chi-square on 5x2 tables; within-subject label shuffling and circular shifts",
        "feature_test": "per-epoch processed-feature mean; Mann-Whitney U across epochs; Wilcoxon of within-subject mean differences",
        "multiple_testing": "Holm across three cluster circular-shift P values and two feature Wilcoxon P values",
        "limitations": [
            "Within-subject label shuffling breaks temporal correlation. Circular shifts preserve runs in the retained train sequence but train/validation gaps remain.",
            "Cluster P values do not measure predictive accuracy or compare clustering combinations directly.",
            "The paper did not specify the statistical test used for its RRI/RPA P values; these are new tests on MIT-BIH data.",
        ],
        "clusters": clusters,
        "features": features,
        "training_started": False,
    }
    target = output / "significance.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n={len(labels)}, Normal={result['normal_n']}, Apnea={result['apnea_n']}")
    for name in COMBINATIONS:
        row = clusters[name]
        print(
            f"{name}: chi2={row['chi_square']:.3f}, V={row['cramers_v']:.3f}, "
            f"p_shuffle={row['within_subject_permutation_p']:.6g}, "
            f"p_shift={row['within_subject_circular_shift_p']:.6g}, "
            f"p_holm={row['holm_adjusted_circular_shift_p']:.6g}"
        )
    for name, row in features.items():
        print(
            f"{name}: delta={row['subject_delta_median']:.4g}, "
            f"p_wilcoxon={row['wilcoxon_p_subject_level']:.6g}, "
            f"p_holm={row['holm_adjusted_wilcoxon_p']:.6g}"
        )
    print(f"saved: {target}")


if __name__ == "__main__":
    main()
