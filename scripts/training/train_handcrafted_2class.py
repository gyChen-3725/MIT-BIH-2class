"""使用冻结 E3 五折比较常见表格模型的 MIT-BIH 二分类指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ROOT / "features" / "expanded347" / "features_30s_expanded.npz"
DEFAULT_SPLITS = ROOT / "protocols" / "e3_full_epoch5fold_20260913"
DEFAULT_MASTER = (
    ROOT.parent / "dataset" / "mitbih_master" / "mitbih_master_30s.npz"
)
LEGACY_FEATURE_GROUPS = ("hrv", "edr3", "fusion3", "all")
FEATURE_GROUPS = (
    *LEGACY_FEATURE_GROUPS,
    "edr_all",
    "advanced_hrv",
    "hrv_all",
    "ecg",
    "qrs",
    "ecg_qrs",
    "hrv_ecg",
    "hrv_qrs",
    "non_edr",
    "non_edr_with_hrv",
    "all_plus_advanced_hrv",
    "all_plus_ecg",
    "all_plus_qrs",
    "fusion3_expanded",
    "expanded_all",
)
MODELS = ("logistic", "random_forest", "hist_gb")
QUALITY_FEATURES = {
    "coverage_fraction", "longest_uncovered_seconds", "baseline_coverage_fraction"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_design_matrix(
    feature_path: Path, master_path: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, list[str]]]:
    """按 master 行号验证 EDR 对齐，再构建预先固定的特征组。"""

    with np.load(feature_path, allow_pickle=False) as stored:
        methods = stored["method_names"].astype(str)
        feature_names = stored["feature_names"].astype(str)
        hrv_names = stored["hrv_feature_names"].astype(str)
        cross_names = stored["cross_feature_names"].astype(str)
        values = stored["method_features"]
        hrv = stored["hrv_features"]
        cross = stored["cross_features"]
        expanded = None
        expanded_keys = {
            "advanced_hrv_feature_names", "advanced_hrv_features",
            "ecg_feature_names", "ecg_features",
            "qrs_feature_names", "qrs_features",
        }
        if expanded_keys.issubset(stored.files):
            expanded = {
                "advanced_hrv_names": stored["advanced_hrv_feature_names"].astype(str),
                "advanced_hrv": stored["advanced_hrv_features"],
                "ecg_names": stored["ecg_feature_names"].astype(str),
                "ecg": stored["ecg_features"],
                "qrs_names": stored["qrs_feature_names"].astype(str),
                "qrs": stored["qrs_features"],
            }
        metadata = {
            "record": stored["record"].astype(str),
            "subject_id": stored["subject_id"].astype(str),
            "start_sample": stored["start_sample"].astype(np.int64),
            "class_label": stored["class_label"].astype(str),
        }
    with np.load(master_path, allow_pickle=False) as master:
        if not np.array_equal(metadata["record"], master["record"].astype(str)):
            raise ValueError("EDR 与 MIT master 的记录顺序不一致")
        if not np.array_equal(metadata["start_sample"], master["start_sample"]):
            raise ValueError("EDR 与 MIT master 的 epoch 起点不一致")
        binary_master = (master["raw_label"].astype(str) != "Normal").astype(np.int64)
    labels = (metadata["class_label"] != "Normal").astype(np.int64)
    if not np.array_equal(labels, binary_master):
        raise ValueError("EDR 与 MIT master 的二分类标签不一致")

    # 质量字段单独保留，不让它们在首轮模型中充当记录条件的捷径。
    selected_methods = {"am_qr", "am_r", "sm_angle"}
    method_columns = [
        (method, name, values[:, method_index, feature_index])
        for method_index, method in enumerate(methods)
        for feature_index, name in enumerate(feature_names)
        if name not in QUALITY_FEATURES
    ]
    edr3 = [
        (f"{method}.{name}", values)
        for method, name, values in method_columns if method in selected_methods
    ]
    all_edr = [(f"{method}.{name}", values) for method, name, values in method_columns]
    hrv_columns = [(f"hrv.{name}", hrv[:, index]) for index, name in enumerate(hrv_names)]
    cross_columns = [
        (f"cross.{name}", cross[:, index]) for index, name in enumerate(cross_names)
    ]
    # 各组输入按固定顺序拼接；所有填补与缩放都在每折训练样本上拟合。
    definitions = {
        "hrv": hrv_columns,
        "edr3": edr3 + cross_columns,
        "fusion3": hrv_columns + edr3 + cross_columns,
        "edr_all": all_edr + cross_columns,
        "all": hrv_columns + all_edr + cross_columns,
    }
    if expanded is not None:
        advanced_columns = [
            (f"advanced_hrv.{name}", expanded["advanced_hrv"][:, index])
            for index, name in enumerate(expanded["advanced_hrv_names"])
        ]
        ecg_columns = [
            (f"ecg.{name}", expanded["ecg"][:, index])
            for index, name in enumerate(expanded["ecg_names"])
        ]
        qrs_columns = [
            (f"qrs.{name}", expanded["qrs"][:, index])
            for index, name in enumerate(expanded["qrs_names"])
        ]
        non_edr = advanced_columns + ecg_columns + qrs_columns
        legacy_all = hrv_columns + all_edr + cross_columns
        definitions.update({
            "advanced_hrv": advanced_columns,
            "hrv_all": hrv_columns + advanced_columns,
            "ecg": ecg_columns,
            "qrs": qrs_columns,
            "ecg_qrs": ecg_columns + qrs_columns,
            "hrv_ecg": hrv_columns + advanced_columns + ecg_columns,
            "hrv_qrs": hrv_columns + advanced_columns + qrs_columns,
            "non_edr": non_edr,
            "non_edr_with_hrv": hrv_columns + non_edr,
            "all_plus_advanced_hrv": legacy_all + advanced_columns,
            "all_plus_ecg": legacy_all + ecg_columns,
            "all_plus_qrs": legacy_all + qrs_columns,
            "fusion3_expanded": hrv_columns + edr3 + cross_columns + non_edr,
            "expanded_all": legacy_all + non_edr,
        })
    matrices = {
        name: np.column_stack([column for _, column in columns]).astype(np.float32)
        for name, columns in definitions.items()
    }
    names = {name: [key for key, _ in columns] for name, columns in definitions.items()}
    return matrices, labels, metadata, names


def load_fold(path: Path, n_samples: int) -> dict[str, np.ndarray]:
    """检查每折索引互斥且完整，防止训练和测试样本重叠。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_unit") != "epoch":
        raise ValueError(f"{path}: 只接受冻结样本级 E3 划分")
    arrays = {
        role: np.asarray(payload[f"{role}_indices"], dtype=np.int64)
        for role in ("train", "val", "test")
    }
    combined = np.concatenate(list(arrays.values()))
    if (
        len(combined) != n_samples
        or len(np.unique(combined)) != n_samples
        or combined.min() != 0
        or combined.max() != n_samples - 1
    ):
        raise ValueError(f"{path}: 训练/验证/测试索引必须互斥并覆盖全部 epoch")
    return arrays


def build_model(name: str, seed: int):
    """固定首轮算法与参数；测试标签不参与超参数选择。"""

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "logistic":
        return make_pipeline(
            imputer, StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1500, random_state=seed),
        )
    if name == "random_forest":
        return make_pipeline(
            imputer,
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, max_features="sqrt",
                n_jobs=-1, random_state=seed,
            ),
        )
    if name == "hist_gb":
        return make_pipeline(
            imputer,
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=25, l2_regularization=1.0,
                early_stopping=False, random_state=seed,
            ),
        )
    raise ValueError(f"未知模型：{name}")


def evaluate(labels: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    """与现有二分类实验统一使用 0.5 阈值，报告正类 Apnea 的各项指标。"""

    predicted = (probability >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(matrix[0, 0] / matrix[0].sum()),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--groups", nargs="+", choices=FEATURE_GROUPS, default=list(LEGACY_FEATURE_GROUPS)
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_path = args.features.resolve()
    master_path = args.master.resolve()
    split_dir = args.split_dir.resolve()
    output = (
        args.output_dir.resolve() if args.output_dir is not None
        else feature_path.parent / "ml_2class_e3_5fold_v1"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"实验目录已有内容，避免覆盖：{output}")
    matrices, labels, metadata, feature_names = load_design_matrix(feature_path, master_path)
    unavailable = sorted(set(args.groups) - set(matrices))
    if unavailable:
        raise ValueError(
            f"特征文件不包含请求的扩展特征组：{unavailable}；请使用扩展347维文件"
        )
    folds = {fold: load_fold(split_dir / f"fold_{fold}.json", len(labels)) for fold in args.folds}
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("折号不能重复")
    output.mkdir(parents=True, exist_ok=True)
    (output / "oof").mkdir(exist_ok=True)
    manifest = {
        "feature_file": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "master_file": str(master_path),
        "split_dir": str(split_dir),
        "split_sha256": {
            str(fold): sha256_file(split_dir / f"fold_{fold}.json") for fold in args.folds
        },
        "split_unit": "epoch",
        "models": args.models,
        "groups": {name: feature_names[name] for name in args.groups},
        "folds": args.folds,
        "seed": args.seed,
        "decision_threshold": 0.5,
        "sklearn_version": sklearn.__version__,
        "note": "EDR uses previous 120 s and offline zero-phase filtering; epoch-level folds are for protocol comparison, not subject-independent generalization.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fold_rows = []
    oof_rows = []
    # 多数类基线只用每折训练标签的阳性比例，不读取验证或测试标签作预测。
    baseline_parts = []
    for fold in args.folds:
        indices = folds[fold]
        train_prevalence = float(np.mean(labels[indices["train"]]))
        validation_prob = np.full(len(indices["val"]), train_prevalence)
        test_prob = np.full(len(indices["test"]), train_prevalence)
        validation = evaluate(labels[indices["val"]], validation_prob)
        test = evaluate(labels[indices["test"]], test_prob)
        fold_rows.append({
            "group": "none", "model": "majority", "fold": fold,
            **{f"val_{key}": value for key, value in validation.items() if key != "confusion_matrix"},
            **{f"test_{key}": value for key, value in test.items() if key != "confusion_matrix"},
        })
        baseline_parts.append(pd.DataFrame({
            "dataset_index": indices["test"], "fold": fold,
            "subject_id": metadata["subject_id"][indices["test"]],
            "label": labels[indices["test"]], "probability": test_prob,
        }))
    baseline_oof = pd.concat(baseline_parts, ignore_index=True).sort_values("dataset_index")
    baseline_oof.to_csv(output / "oof" / "none_majority.csv", index=False)
    baseline_metrics = evaluate(
        baseline_oof["label"].to_numpy(), baseline_oof["probability"].to_numpy()
    )
    oof_rows.append({
        "group": "none", "model": "majority", "feature_count": 0,
        **{key: value for key, value in baseline_metrics.items() if key != "confusion_matrix"},
        "confusion_matrix": json.dumps(baseline_metrics["confusion_matrix"]),
    })
    for group_name in args.groups:
        matrix = matrices[group_name]
        for model_name in args.models:
            predictions = []
            for fold in args.folds:
                indices = folds[fold]
                estimator = build_model(model_name, args.seed + fold)
                estimator.fit(matrix[indices["train"]], labels[indices["train"]])
                validation_prob = estimator.predict_proba(matrix[indices["val"]])[:, 1]
                test_prob = estimator.predict_proba(matrix[indices["test"]])[:, 1]
                validation = evaluate(labels[indices["val"]], validation_prob)
                test = evaluate(labels[indices["test"]], test_prob)
                fold_rows.append({
                    "group": group_name, "model": model_name, "fold": fold,
                    **{f"val_{key}": value for key, value in validation.items() if key != "confusion_matrix"},
                    **{f"test_{key}": value for key, value in test.items() if key != "confusion_matrix"},
                })
                predictions.append(pd.DataFrame({
                    "dataset_index": indices["test"],
                    "fold": fold,
                    "subject_id": metadata["subject_id"][indices["test"]],
                    "label": labels[indices["test"]],
                    "probability": test_prob,
                }))
                print(
                    f"{group_name}/{model_name} fold {fold}: "
                    f"val_acc={validation['accuracy']:.4f} test_acc={test['accuracy']:.4f}",
                    flush=True,
                )
            oof = pd.concat(predictions, ignore_index=True).sort_values("dataset_index")
            if len(oof) != len(oof["dataset_index"].unique()):
                raise ValueError("测试折之间存在重复样本")
            oof.to_csv(output / "oof" / f"{group_name}_{model_name}.csv", index=False)
            metrics = evaluate(oof["label"].to_numpy(), oof["probability"].to_numpy())
            oof_rows.append({
                "group": group_name, "model": model_name, "feature_count": len(feature_names[group_name]),
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            })
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    result = pd.DataFrame(oof_rows).sort_values("accuracy", ascending=False)
    result.to_csv(output / "oof_metrics.csv", index=False)
    print(f"saved: {output / 'oof_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
