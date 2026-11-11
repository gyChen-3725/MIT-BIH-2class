"""在冻结五折内比较特征选择方法、维度和机器学习模型，并仅用验证集选优。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

from train_handcrafted_2class import (
    DEFAULT_MASTER,
    DEFAULT_SPLITS,
    build_model,
    evaluate,
    load_design_matrix,
    load_fold,
    sha256_file,
)

DEFAULT_FEATURES = (
    ROOT / "features" / "expanded347" / "features_30s_expanded.npz"
)
SELECTORS = ("anova", "mutual_info", "l1", "random_forest", "mrmr")
MODELS = ("logistic", "random_forest", "hist_gb")
DEFAULT_DIMENSIONS = (16, 32, 48, 64, 96, 128, 192, 263)
DEFAULT_FEATURE_GROUPS = (
    "hrv_all",
    "ecg",
    "qrs",
    "ecg_qrs",
    "non_edr_with_hrv",
    "edr3",
    "fusion3",
    "edr_all",
    "all",
    "fusion3_expanded",
    "expanded_all",
)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")


def _filled_training_matrix(values: np.ndarray) -> np.ndarray:
    """所有选择器共用仅在当前训练行拟合的中位数填补。"""

    return SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(values)


def _finite_score_order(scores: np.ndarray) -> np.ndarray:
    """把非有限得分放到末尾，并用原始列号稳定打破同分。"""

    values = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    return np.lexsort((np.arange(len(values)), -values)).astype(np.int64)


def _mrmr_order(values: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """用互信息衡量相关性，并逐步惩罚与已选列的最大Pearson冗余。"""

    relevance = mutual_info_classif(values, labels, random_state=seed)
    if np.ptp(relevance) > 1e-12:
        relevance = (relevance - relevance.min()) / np.ptp(relevance)
    # 常量列的标准差为0，直接调用np.corrcoef会产生除零警告。这里仅对
    # 非常量列计算中心化余弦相似度，它与Pearson相关系数等价。
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    valid = norms > np.finfo(np.float64).eps
    correlation = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    normalized = centered[:, valid] / norms[valid]
    correlation[np.ix_(valid, valid)] = np.clip(
        np.abs(normalized.T @ normalized), 0.0, 1.0
    )
    np.fill_diagonal(correlation, 0.0)
    remaining = np.ones(values.shape[1], dtype=bool)
    maximum_redundancy = np.zeros(values.shape[1], dtype=np.float64)
    order = np.empty(values.shape[1], dtype=np.int64)
    for position in range(values.shape[1]):
        scores = relevance - 0.5 * maximum_redundancy
        scores[~remaining] = -np.inf
        selected = int(np.argmax(scores))
        order[position] = selected
        remaining[selected] = False
        maximum_redundancy = np.maximum(maximum_redundancy, correlation[selected])
    return order


def fit_feature_ranking(
    values: np.ndarray,
    labels: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    """只用传入训练样本拟合一个覆盖全部列的稳定排名。"""

    if method not in SELECTORS:
        raise ValueError(f"未知特征选择方法：{method}")
    filled = _filled_training_matrix(values)
    if method == "anova":
        scores = f_classif(filled, labels)[0]
        return _finite_score_order(scores)
    if method == "mutual_info":
        scores = mutual_info_classif(filled, labels, random_state=seed)
        return _finite_score_order(scores)
    if method == "l1":
        scaled = StandardScaler().fit_transform(filled)
        estimator = LinearSVC(
            penalty="l1",
            dual=False,
            C=0.1,
            class_weight="balanced",
            # 347维特征中存在强相关列，适当放宽收敛阈值并增加上限，
            # 避免liblinear在已经稳定的排名附近过早发出警告。
            tol=1e-3,
            max_iter=20000,
            random_state=seed,
        ).fit(scaled, labels)
        return _finite_score_order(np.abs(estimator.coef_[0]))
    if method == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ).fit(filled, labels)
        return _finite_score_order(estimator.feature_importances_)
    return _mrmr_order(filled, labels, seed)


def candidate_key(
    row: dict[str, object], selection_metric: str
) -> tuple[float, float, float, float, int, str, str]:
    """验证指标优先，同分时依次偏向宏F1、平衡准确率、AUC和更少维度。"""

    metrics = row["validation_metrics"]
    assert isinstance(metrics, dict)
    return (
        float(metrics[selection_metric]),
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["roc_auc"]),
        -int(row["dimension"]),
        str(row["selector"]),
        str(row["model"]),
    )


def flatten_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    """把混淆矩阵序列化，便于验证网格写入单个CSV。"""

    return {
        f"{prefix}_{name}": json.dumps(value) if name == "confusion_matrix" else value
        for name, value in metrics.items()
    }


def write_report(
    path: Path,
    fold_best: pd.DataFrame,
    oof_metrics: dict[str, object],
    selection_metric: str,
    feature_group_count: int,
    top_groups: int,
    selectors: list[str],
    dimensions: list[int],
    models: list[str],
) -> None:
    model_names = "、".join(f"`{name}`" for name in models)
    selector_names = "、".join(f"`{name}`" for name in selectors)
    dimension_names = "、".join(str(value) for value in dimensions)
    lines = [
        "# 扩展手工特征：特征选择与机器学习组合搜索",
        "",
        f"每折使用验证集 `{selection_metric}` 选择方法、维度和模型；测试集只在选定后评价一次。",
        "候选比较阶段的缺失值填补、特征排名和分类器只使用当前折训练集拟合。",
        "确定特征组合、选择器、维度和模型后，在训练集＋验证集上重新拟合填补、特征排名和最终模型，",
        "测试集始终不参与拟合或方案选择，只在最后评价一次。",
        "",
        "## 每折的数据职责",
        "",
        "```text",
        "训练集：拟合候选方案的填补、特征排名和分类器",
        "   ↓",
        f"验证集：以 {selection_metric} 比较特征组合＋选择器＋维度＋模型",
        "   ↓  固定最佳方案，不再修改超参数",
        "训练集＋验证集：重新拟合该方案的填补、特征排名和最终分类器",
        "   ↓",
        "测试集：只评价一次，生成该折最终预测",
        "```",
        "",
        "重新拟合特征排名可能改变具体入选列，但选择器类型和目标维度已经由验证集固定。",
        "因此测试标签从未参与特征选择；若看完测试结果再更换方案，就会形成测试泄漏。",
        "",
        "## 两阶段候选搜索",
        "",
        (
            f"第一阶段不做列级选择。每折对{feature_group_count}个完整特征组分别训练 "
            f"{len(models)} 个分类器（{model_names}），共 "
            f"{feature_group_count * len(models)} 个候选。每组取最高验证 `{selection_metric}` "
            f"作为初筛成绩，前{top_groups}组进入第二阶段。"
        ),
        "",
        (
            f"第二阶段在晋级组内比较选择器 {selector_names}、候选维度 {dimension_names} "
            "和全部分类器。候选维度只有小于该组完整维度时才运行；完整维度方案 "
            "已经在第一阶段评价。"
        ),
        "",
        "`random_forest`可能表示随机森林重要性选择器，也可能表示随机森林分类器，二者角色独立；",
        "HGB只作为分类器。最终冠军从第一阶段完整维度候选和第二阶段降维候选的并集中产生，",
        "因此 `selector=none` 的完整特征组也可能胜出。",
        "",
        "## 每折选择",
        "",
        "| 折 | 特征组合 | 选择器 | 维度 | 模型 | 验证准确率 | 测试准确率 | 测试macro F1 |",
        "|---:|---|---|---:|---|---:|---:|---:|",
    ]
    for row in fold_best.itertuples(index=False):
        lines.append(
            f"| {row.fold} | `{row.feature_group}` | `{row.selector}` | "
            f"{row.dimension} | `{row.model}` | "
            f"{row.val_accuracy:.4f} | {row.test_accuracy:.4f} | {row.test_macro_f1:.4f} |"
        )
    lines.extend([
        "",
        "## 自适应五折OOF",
        "",
        f"- accuracy：{float(oof_metrics['accuracy']):.4f}",
        f"- balanced accuracy：{float(oof_metrics['balanced_accuracy']):.4f}",
        f"- macro F1：{float(oof_metrics['macro_f1']):.4f}",
        f"- ROC AUC：{float(oof_metrics['roc_auc']):.4f}",
        "",
        "不同折可能选择不同维度；这代表完整的自适应选择流程。不能读取测试结果后再挑一个全局维度，",
        "否则会把测试集重新用于调参。`validation_grid.csv`可用于观察维度趋势，但不替代独立验证。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selectors", nargs="+", choices=SELECTORS, default=list(SELECTORS))
    parser.add_argument(
        "--feature-groups", nargs="+", default=list(DEFAULT_FEATURE_GROUPS),
        help="先按完整特征族组合筛查，再对验证集靠前组合做列级选择",
    )
    parser.add_argument(
        "--top-groups", type=int, default=3,
        help="每折进入列级特征选择的特征族组合数量",
    )
    parser.add_argument("--dimensions", nargs="+", type=int, default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--selection-metric", choices=METRICS, default="accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只核对数据与组合数量，不拟合选择器或模型",
    )
    args = parser.parse_args()

    feature_path = args.features.resolve()
    master_path = args.master.resolve()
    split_dir = args.split_dir.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else feature_path.parent / "selection_ml_grid_e3_v1"
    )
    if not args.plan_only and output.exists():
        # 中断发生在首折完成前时，旧版本只会留下一个空selected_features目录。
        # 这种占位目录不算实验产物，允许原命令安全重启。
        meaningful_entries = [
            path
            for path in output.iterdir()
            if not (
                path.name == "selected_features"
                and path.is_dir()
                and not any(path.iterdir())
            )
        ]
        if meaningful_entries:
            raise FileExistsError(f"组合搜索目录已有内容，避免覆盖：{output}")
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("折号不能重复")
    matrices, labels, metadata, names_by_group = load_design_matrix(feature_path, master_path)
    unavailable_groups = sorted(set(args.feature_groups) - set(matrices))
    if unavailable_groups:
        raise ValueError(f"特征文件不包含组合：{unavailable_groups}")
    if not 1 <= args.top_groups <= len(args.feature_groups):
        raise ValueError("top-groups必须位于1和特征组合数量之间")
    group_matrices = {name: matrices[name] for name in args.feature_groups}
    group_names = {name: tuple(names_by_group[name]) for name in args.feature_groups}
    dimensions = sorted({value for value in args.dimensions if value >= 1})
    if not dimensions:
        raise ValueError("至少提供一个正整数候选维度")
    folds = {
        fold: load_fold(split_dir / f"fold_{fold}.json", len(labels))
        for fold in args.folds
    }
    screening_fits_per_fold = len(group_matrices) * len(args.models)
    selection_fit_counts = sorted(
        (
            len(args.selectors)
            * sum(dimension < matrix.shape[1] for dimension in dimensions)
            * len(args.models)
            for matrix in group_matrices.values()
        ),
        reverse=True,
    )
    validation_fits_upper_bound = screening_fits_per_fold + sum(
        selection_fit_counts[: args.top_groups]
    )
    if args.plan_only:
        print(
            "feature_groups="
            f"{[(name, group_matrices[name].shape[1]) for name in args.feature_groups]}"
        )
        print(f"selectors={args.selectors}")
        print(f"dimensions={dimensions} plus each group full dimension")
        print(f"models={args.models}")
        print(f"folds={args.folds}")
        print(f"top_groups_per_fold={args.top_groups}")
        print(f"screening_fits_per_fold={screening_fits_per_fold}")
        print(f"validation_fits_per_fold_upper_bound={validation_fits_upper_bound}")
        print(
            "total_model_fits_upper_bound="
            f"{validation_fits_upper_bound * len(args.folds) + len(args.folds)}"
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    selected_dir = output / "selected_features"
    selected_dir.mkdir(exist_ok=True)

    validation_rows: list[dict[str, object]] = []
    fold_best_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    selected_name_counter: Counter[str] = Counter()
    experiment_started = time.perf_counter()

    for fold in args.folds:
        print(
            f"fold {fold}: 开始特征族初筛，共{len(group_matrices)}组，"
            f"每组{len(args.models)}个模型",
            flush=True,
        )
        indices = folds[fold]
        train_indices = indices["train"]
        validation_indices = indices["val"]
        candidates: list[dict[str, object]] = []
        group_best_candidates: list[dict[str, object]] = []
        # 第一阶段使用每个特征族组合的完整维度，先筛掉明显不合适的组合。
        for feature_group, matrix in group_matrices.items():
            group_started = time.perf_counter()
            group_candidates = []
            for model_name in args.models:
                estimator = build_model(model_name, args.seed + fold)
                estimator.fit(matrix[train_indices], labels[train_indices])
                probability = estimator.predict_proba(matrix[validation_indices])[:, 1]
                validation = evaluate(labels[validation_indices], probability)
                candidate = {
                    "fold": fold,
                    "stage": "group_screen",
                    "feature_group": feature_group,
                    "selector": "none",
                    "dimension": matrix.shape[1],
                    "model": model_name,
                    "validation_metrics": validation,
                }
                candidates.append(candidate)
                group_candidates.append(candidate)
                validation_rows.append({
                    "fold": fold,
                    "stage": "group_screen",
                    "feature_group": feature_group,
                    "selector": "none",
                    "dimension": matrix.shape[1],
                    "model": model_name,
                    **flatten_metrics("val", validation),
                })
            group_best_candidates.append(
                max(
                    group_candidates,
                    key=lambda row: candidate_key(row, args.selection_metric),
                )
            )
            print(
                f"fold {fold}: 初筛完成 {feature_group} ({matrix.shape[1]}维)，"
                f"耗时{time.perf_counter() - group_started:.1f}秒",
                flush=True,
            )

        ranked_groups = sorted(
            group_best_candidates,
            key=lambda row: candidate_key(row, args.selection_metric),
            reverse=True,
        )
        refined_groups = [
            str(row["feature_group"]) for row in ranked_groups[: args.top_groups]
        ]
        print(f"fold {fold}: 进入列级选择的组合={refined_groups}", flush=True)
        # 第二阶段只对验证表现靠前的特征组合做列级选择和维度搜索。
        for feature_group in refined_groups:
            matrix = group_matrices[feature_group]
            rankings = {}
            for selector in args.selectors:
                ranking_started = time.perf_counter()
                rankings[selector] = fit_feature_ranking(
                    matrix[train_indices],
                    labels[train_indices],
                    selector,
                    args.seed + fold,
                )
                print(
                    f"fold {fold}: {feature_group}/{selector} 排名完成，"
                    f"耗时{time.perf_counter() - ranking_started:.1f}秒",
                    flush=True,
                )
            for selector in args.selectors:
                for dimension in dimensions:
                    if dimension >= matrix.shape[1]:
                        continue
                    selected = rankings[selector][:dimension]
                    dimension_started = time.perf_counter()
                    for model_name in args.models:
                        if model_name == "random_forest":
                            print(
                                f"fold {fold}: 正在拟合 {feature_group}/{selector}/"
                                f"{dimension}/random_forest",
                                flush=True,
                            )
                        estimator = build_model(model_name, args.seed + fold)
                        estimator.fit(
                            matrix[train_indices][:, selected], labels[train_indices]
                        )
                        probability = estimator.predict_proba(
                            matrix[validation_indices][:, selected]
                        )[:, 1]
                        validation = evaluate(labels[validation_indices], probability)
                        candidate = {
                            "fold": fold,
                            "stage": "within_group_selection",
                            "feature_group": feature_group,
                            "selector": selector,
                            "dimension": dimension,
                            "model": model_name,
                            "validation_metrics": validation,
                        }
                        candidates.append(candidate)
                        validation_rows.append({
                            "fold": fold,
                            "stage": "within_group_selection",
                            "feature_group": feature_group,
                            "selector": selector,
                            "dimension": dimension,
                            "model": model_name,
                            **flatten_metrics("val", validation),
                        })
                    print(
                        f"fold {fold}: {feature_group}/{selector}/{dimension}维的"
                        f"{len(args.models)}个模型完成，"
                        f"耗时{time.perf_counter() - dimension_started:.1f}秒",
                        flush=True,
                    )

        best = max(candidates, key=lambda row: candidate_key(row, args.selection_metric))
        combined = np.concatenate((train_indices, validation_indices))
        feature_group = str(best["feature_group"])
        matrix = group_matrices[feature_group]
        feature_names = group_names[feature_group]
        selector = str(best["selector"])
        dimension = int(best["dimension"])
        if selector == "none":
            selected = np.arange(matrix.shape[1])
        else:
            # 验证集已经完成超参数选择，最终排名可在训练＋验证上重新拟合。
            selected = fit_feature_ranking(
                matrix[combined], labels[combined], selector, args.seed + fold
            )[:dimension]
        model_name = str(best["model"])
        final_estimator = build_model(model_name, args.seed + fold)
        final_estimator.fit(matrix[combined][:, selected], labels[combined])
        test_probability = final_estimator.predict_proba(
            matrix[indices["test"]][:, selected]
        )[:, 1]
        test = evaluate(labels[indices["test"]], test_probability)
        validation = best["validation_metrics"]
        assert isinstance(validation, dict)
        chosen_names = [feature_names[index] for index in selected]
        selected_name_counter.update(chosen_names)
        selection_payload = {
            "fold": fold,
            "feature_group": feature_group,
            "source_feature_count": matrix.shape[1],
            "selector": selector,
            "dimension": dimension,
            "model": model_name,
            "selection_metric": args.selection_metric,
            "selected_indices": selected.tolist(),
            "selected_names": chosen_names,
            "ranking_fit_rows": "train+validation after validation-based combination choice",
            "test_rows_used_for_selection": False,
        }
        (selected_dir / f"fold_{fold}.json").write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_best_rows.append({
            "fold": fold,
            "feature_group": feature_group,
            "selector": selector,
            "dimension": dimension,
            "model": model_name,
            **flatten_metrics("val", validation),
            **flatten_metrics("test", test),
        })
        prediction_parts.append(pd.DataFrame({
            "dataset_index": indices["test"],
            "fold": fold,
            "subject_id": metadata["subject_id"][indices["test"]],
            "label": labels[indices["test"]],
            "probability": test_probability,
            "feature_group": feature_group,
            "selector": selector,
            "dimension": dimension,
            "model": model_name,
        }))
        print(
            f"fold {fold}: {feature_group}/{selector}/{dimension}/{model_name} "
            f"val_{args.selection_metric}={float(validation[args.selection_metric]):.4f} "
            f"test_accuracy={float(test['accuracy']):.4f}，"
            f"累计耗时{(time.perf_counter() - experiment_started) / 60:.1f}分钟",
            flush=True,
        )

    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(output / "validation_grid.csv", index=False)
    dimension_summary = (
        validation_frame.groupby(
            ["stage", "feature_group", "selector", "dimension", "model"],
            as_index=False,
        )
        .agg(
            fold_n=("fold", "count"),
            mean_val_accuracy=("val_accuracy", "mean"),
            std_val_accuracy=("val_accuracy", "std"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            mean_val_balanced_accuracy=("val_balanced_accuracy", "mean"),
            mean_val_roc_auc=("val_roc_auc", "mean"),
        )
        .sort_values(
            [f"mean_val_{args.selection_metric}", "mean_val_macro_f1", "dimension"],
            ascending=[False, False, True],
        )
    )
    dimension_summary.to_csv(output / "validation_dimension_summary.csv", index=False)
    fold_best = pd.DataFrame(fold_best_rows).sort_values("fold")
    fold_best.to_csv(output / "fold_best.csv", index=False)
    oof = pd.concat(prediction_parts, ignore_index=True).sort_values("dataset_index")
    if len(oof["dataset_index"].unique()) != len(oof):
        raise ValueError("所选测试折之间存在重复样本")
    oof.to_csv(output / "oof_predictions.csv", index=False)
    oof_metrics = evaluate(oof["label"].to_numpy(), oof["probability"].to_numpy())
    (output / "oof_metrics.json").write_text(
        json.dumps(oof_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"feature": name, "selected_fold_count": count}
        for name, count in selected_name_counter.most_common()
    ]).to_csv(output / "selected_feature_frequency.csv", index=False)
    manifest = {
        "feature_file": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "master_file": str(master_path),
        "split_dir": str(split_dir),
        "feature_groups": {
            name: group_matrices[name].shape[1] for name in args.feature_groups
        },
        "top_groups_per_fold": args.top_groups,
        "selectors": args.selectors,
        "dimensions": dimensions,
        "models": args.models,
        "folds": args.folds,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "test_policy": "one test evaluation per fold after validation-based combination choice",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output / "组合搜索报告.md",
        fold_best,
        oof_metrics,
        args.selection_metric,
        len(args.feature_groups),
        args.top_groups,
        list(args.selectors),
        dimensions,
        list(args.models),
    )
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
