"""按受试者评估手工特征对三分类任务的单变量区分能力，不拟合模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "features" / "expanded347" / "features_30s_expanded.npz"
TASKS = {
    "Apnea_vs_Normal": ("class", "Normal", "Apnea"),
    "Hypopnea_vs_Normal": ("class", "Normal", "Hypopnea"),
    "Any_event_vs_Normal": ("class", "Normal", "Event"),
    "Central_vs_Obstructive": ("subtype", "Obstructive", "Central"),
}

REDUNDANCY_PAIRS = (
    ("hrv.sdnn_seconds", "hrv.rmssd_seconds"),
    ("hrv.sdnn_seconds", "fm_rr.resp_peak_to_peak"),
    ("hrv.sdnn_seconds", "am_qr.envelope_cv"),
    ("hrv.sdnn_seconds", "am_r.log_baseline_q10_ratio"),
    ("am_qr.envelope_cv", "am_qr.envelope_modulation_depth"),
    ("am_qr.envelope_cv", "am_r.envelope_cv"),
    ("am_qr.envelope_cv", "sm_angle.envelope_cv"),
    ("am_r.log_baseline_q10_ratio", "am_r.log_baseline_min10_mean_ratio"),
    ("am_r.log_baseline_q10_ratio", "am_qr.log_baseline_q10_ratio"),
    ("am_r.log_baseline_q10_ratio", "am_qr.envelope_cv"),
)


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """秩统计量实现 AUC；保留原始方向，低于 0.5 也可能有判别价值。"""

    y = np.asarray(y_true, dtype=bool)
    x = np.asarray(scores, dtype=np.float64)
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    if n_positive == 0 or n_negative == 0 or not np.all(np.isfinite(x)):
        return np.nan
    ranks = rankdata(x, method="average")
    return float(
        (np.sum(ranks[y]) - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def load_feature_columns(
    path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """展平特征轴，但保留每个方法与特征名以供独立消融。"""

    with np.load(path, allow_pickle=False) as archive:
        labels = archive["class_label"].astype(str)
        subjects = archive["subject_id"].astype(str)
        stages = archive["sleep_stage"].astype(str)
        annotations = archive["raw_annotation"].astype(str)
        methods = archive["method_names"].astype(str)
        feature_names = archive["feature_names"].astype(str)
        hrv_names = archive["hrv_feature_names"].astype(str)
        cross_names = archive["cross_feature_names"].astype(str)
        method_features = archive["method_features"]
        hrv_features = archive["hrv_features"]
        cross_features = archive["cross_features"]
        optional_groups = {}
        for prefix in ("advanced_hrv", "ecg", "qrs"):
            names_key = f"{prefix}_feature_names"
            values_key = f"{prefix}_features"
            if names_key in archive.files and values_key in archive.files:
                optional_groups[prefix] = (
                    archive[names_key].astype(str), archive[values_key]
                )
    if method_features.shape != (len(labels), len(methods), len(feature_names)):
        raise ValueError("单路 EDR 特征维度与元数据不一致")
    columns = {
        f"{method}.{name}": method_features[:, method_index, feature_index]
        for method_index, method in enumerate(methods)
        for feature_index, name in enumerate(feature_names)
    }
    columns.update({f"hrv.{name}": hrv_features[:, index] for index, name in enumerate(hrv_names)})
    columns.update({f"cross.{name}": cross_features[:, index] for index, name in enumerate(cross_names)})
    for prefix, (names, values) in optional_groups.items():
        if values.shape != (len(labels), len(names)):
            raise ValueError(f"{prefix}扩展特征维度与元数据不一致")
        columns.update(
            {f"{prefix}.{name}": values[:, index] for index, name in enumerate(names)}
        )
    return columns, labels, subjects, stages, annotations


def apnea_subtype(annotation: str) -> str:
    """混合事件不参与中枢与阻塞对比，避免把并存标签误当作单一亚型。"""

    tokens = set(annotation.split()[1:])
    central = bool(tokens & {"CA", "CAA"})
    obstructive = bool(tokens & {"OA", "X"})
    hypopnea = bool(tokens & {"H", "HA"})
    if central + obstructive + hypopnea != 1:
        return "Other"
    if central:
        return "Central"
    if obstructive:
        return "Obstructive"
    return "Other"


def screen(
    columns: dict[str, np.ndarray], labels: np.ndarray, subjects: np.ndarray,
    subtypes: np.ndarray,
    minimum_per_class: int = 5,
    minimum_subjects: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每位受试者内算 AUC，再跨人汇总，避免多窗口受试者占过大权重。"""

    summaries = []
    subject_rows = []
    for task, (source, negative_name, positive_name) in TASKS.items():
        source_labels = subtypes if source == "subtype" else labels
        positive = (
            labels != "Normal" if positive_name == "Event" else source_labels == positive_name
        )
        eligible = (source_labels == negative_name) | positive
        for name, values in columns.items():
            valid = eligible & np.isfinite(values)
            aucs = []
            for subject in np.unique(subjects):
                selected = valid & (subjects == subject)
                target = positive[selected]
                n_positive = int(target.sum())
                n_negative = len(target) - n_positive
                if min(n_positive, n_negative) < minimum_per_class:
                    continue
                auc = binary_auc(target, values[selected])
                aucs.append(auc)
                subject_rows.append({
                    "task": task, "feature": name, "subject_id": subject,
                    "auc": auc, "positive_n": n_positive, "negative_n": n_negative,
                })
            if len(aucs) < minimum_subjects:
                continue
            median = float(np.median(aucs))
            direction = np.sign(median - 0.5)
            consistency = float(np.mean(np.sign(np.asarray(aucs) - 0.5) == direction))
            summaries.append({
                "task": task,
                "feature": name,
                "family": name.split(".")[0],
                "subject_n": len(aucs),
                "valid_epoch_n": int(valid.sum()),
                "available_fraction": float(valid.sum() / eligible.sum()),
                "median_auc": median,
                "q25_auc": float(np.quantile(aucs, 0.25)),
                "q75_auc": float(np.quantile(aucs, 0.75)),
                "direction_consistency": consistency,
                "absolute_distance_from_half": abs(median - 0.5),
            })
    summary = pd.DataFrame(summaries).sort_values(
        ["task", "absolute_distance_from_half"], ascending=[True, False]
    )
    return summary, pd.DataFrame(subject_rows)


def write_report(
    summary: pd.DataFrame, destination: Path, n_epochs: int, n_subjects: int,
    scope: str, filename: str,
) -> None:
    lines = [
        f"# EDR 手工特征的单变量筛选：{scope}", "",
        f"分析 {n_epochs:,} 个 30 秒窗口、{n_subjects} 位受试者。",
        "每位受试者内分别计算两组标签的 AUC，再取跨受试者中位数。",
        "AUC 保留特征原方向；低于 0.5 表示数值降低更偏向任务名中的后一类。",
        "此处没有拟合分类器，排序属于探索性特征筛选，不是留出测试集性能。", "",
    ]
    for task in TASKS:
        selected = summary[summary["task"] == task].head(12)
        lines.extend([
            f"## {task}", "",
            "| 特征 | 受试者数 | AUC 中位数 [Q1,Q3] | 方向一致比例 | 有效窗口比例 |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in selected.itertuples():
            lines.append(
                f"| `{row.feature}` | {row.subject_n} | "
                f"{row.median_auc:.3f} [{row.q25_auc:.3f}, {row.q75_auc:.3f}] | "
                f"{row.direction_consistency:.2f} | {row.available_fraction:.2f} |"
            )
        lines.append("")
    lines.extend([
        "## 使用边界", "",
        "- 同一数据既用于筛选又用于汇总，不能据此宣称泛化性能。",
        "- 特征之间可能高度相关；高排名不说明组合后有增益。",
        "- 质量特征可能反映记录条件，应单独消融以防受试者或设备偏差。",
        "- 120 秒基线只用当前窗口之前的 ECG 派生信号；连续滤波为离线零相位处理。",
        "- 真实鼻气流、胸腹努力和标签均未用于手工特征提取。",
    ])
    (destination / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def redundancy_n2(
    columns: dict[str, np.ndarray], subjects: np.ndarray, stages: np.ndarray,
) -> pd.DataFrame:
    """逐受试者计算 N2 期 Spearman 相关，避免跨人幅度尺度产生虚假冗余。"""

    rows = []
    for first, second in REDUNDANCY_PAIRS:
        left = columns[first]
        right = columns[second]
        correlations = []
        for subject in np.unique(subjects):
            selected = (
                (subjects == subject) & (stages == "2")
                & np.isfinite(left) & np.isfinite(right)
            )
            if selected.sum() < 30:
                continue
            value = spearmanr(left[selected], right[selected]).statistic
            if np.isfinite(value):
                correlations.append(float(value))
        rows.append({
            "first": first, "second": second, "subject_n": len(correlations),
            "median_spearman": float(np.median(correlations)),
            "q25_spearman": float(np.quantile(correlations, 0.25)),
            "q75_spearman": float(np.quantile(correlations, 0.75)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--minimum-per-class", type=int, default=5)
    parser.add_argument("--minimum-subjects", type=int, default=5)
    args = parser.parse_args()
    source = args.features.resolve()
    columns, labels, subjects, stages, annotations = load_feature_columns(source)
    subtypes = np.asarray([apnea_subtype(value) for value in annotations])
    destination = source.parent
    scopes = {
        "all": np.ones(len(labels), dtype=bool),
        "sleep": np.isin(stages, ["1", "2", "3", "4", "R"]),
        "n2": stages == "2",
    }
    for scope, selected in scopes.items():
        scoped_columns = {name: values[selected] for name, values in columns.items()}
        scoped_labels = labels[selected]
        scoped_subjects = subjects[selected]
        scoped_subtypes = subtypes[selected]
        summary, subject_auc = screen(
            scoped_columns, scoped_labels, scoped_subjects, scoped_subtypes,
            args.minimum_per_class, args.minimum_subjects,
        )
        suffix = "" if scope == "all" else f"_{scope}"
        summary.to_csv(
            destination / f"univariate_auc{suffix}.csv", index=False, encoding="utf-8-sig"
        )
        subject_auc.to_csv(
            destination / f"subject_auc{suffix}.csv", index=False, encoding="utf-8-sig"
        )
        report_name = "筛选报告.md" if scope == "all" else f"筛选报告_{scope}.md"
        write_report(
            summary, destination, len(scoped_labels), len(np.unique(scoped_subjects)),
            scope, report_name,
        )
        print(f"{scope}: {len(scoped_labels)} epochs, {len(summary)} feature-task rows")
    metadata = {
        "feature_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "minimum_per_class_per_subject": args.minimum_per_class,
        "minimum_subjects": args.minimum_subjects,
        "tasks": TASKS,
        "scopes": {name: int(mask.sum()) for name, mask in scopes.items()},
        "subject_weighting": "median of within-subject AUC values",
        "classifier_fitted": False,
    }
    (destination / "screening_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    redundancy_n2(columns, subjects, stages).to_csv(
        destination / "redundancy_n2.csv", index=False, encoding="utf-8-sig"
    )
    print(f"screened {len(columns)} features across {len(TASKS)} tasks: {destination}")


if __name__ == "__main__":
    main()
