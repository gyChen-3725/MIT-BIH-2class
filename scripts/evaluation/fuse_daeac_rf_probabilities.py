"""在验证集选择权重，将 DAEAC 与机器学习最优 RF 做概率级融合。

本脚本不会训练 DAEAC。它读取已经完成的最佳 checkpoint；RF 在训练集上拟合后
产生验证概率用于选权重，再在训练＋验证集上重新拟合并只评估一次测试集。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "selection"))

from optimize_feature_dimension import fit_feature_ranking
from train_handcrafted_2class import build_model as build_table_model
from train_handcrafted_2class import (
    evaluate,
    load_design_matrix,
    load_fold,
)

from mitbih_2class.config import load_config
from mitbih_2class.data import build_dataloaders, load_master, load_split
from mitbih_2class.models import build_model as build_daeac

# 概率融合以原始DAEAC为深度学习分量，和MI-128门控实验保持相互独立。
DEFAULT_CONFIG = (
    ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_e3_epoch5fold.yaml"
)
DEFAULT_SEARCH = ROOT / "features" / "expanded347" / "selection_ml_grid_e3_v1"
DEFAULT_FEATURES = ROOT / "features" / "expanded347" / "features_30s_expanded.npz"
DEFAULT_OUTPUT = (
    ROOT / "runs" / "daeac_mi128_rf_probability_fusion_fold0_2class" / "fold_0"
)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_probabilities(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> pd.DataFrame:
    """按 master 行号返回概率，供RF结果严格对齐。"""

    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            aux = batch.get("aux_features")
            handcrafted = batch.get("handcrafted_features")
            if aux is not None:
                aux = aux.to(device, non_blocking=True)
            if handcrafted is not None:
                handcrafted = handcrafted.to(device, non_blocking=True)
            logits = model(
                features, aux_features=aux, handcrafted_features=handcrafted
            )
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            for index, label, probability in zip(
                batch["dataset_index"], batch["label"], probabilities, strict=True
            ):
                rows.append(
                    {
                        "dataset_index": int(index),
                        "label": int(label),
                        "daeac_probability": float(probability),
                    }
                )
    return pd.DataFrame(rows).sort_values("dataset_index").reset_index(drop=True)


def fit_rf_probabilities(
    feature_path: Path,
    master_path: Path,
    split_dir: Path,
    search_dir: Path,
    fold: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """复现折内最优RF：训练行预测验证，训练＋验证行预测测试。"""

    best_frame = pd.read_csv(search_dir / "fold_best.csv")
    matches = best_frame.loc[best_frame["fold"] == fold]
    if len(matches) != 1:
        raise ValueError(f"fold_best.csv 中 Fold {fold} 应恰有一行")
    best = matches.iloc[0]
    feature_group = str(best["feature_group"])
    selector = str(best["selector"])
    dimension = int(best["dimension"])
    model_name = str(best["model"])
    if model_name != "random_forest":
        raise ValueError(f"当前概率融合要求机器学习最优模型为RF，实际为{model_name}")

    matrices, labels, _metadata, _names = load_design_matrix(feature_path, master_path)
    matrix = matrices[feature_group]
    indices = load_fold(split_dir / f"fold_{fold}.json", len(labels))
    train = indices["train"]
    validation = indices["val"]
    test = indices["test"]

    train_ranking = fit_feature_ranking(
        matrix[train], labels[train], selector, seed + fold
    )
    train_columns = train_ranking[:dimension]
    validation_model = build_table_model(model_name, seed + fold)
    validation_model.fit(matrix[train][:, train_columns], labels[train])
    validation_probability = validation_model.predict_proba(
        matrix[validation][:, train_columns]
    )[:, 1]

    combined = np.concatenate((train, validation))
    final_ranking = fit_feature_ranking(
        matrix[combined], labels[combined], selector, seed + fold
    )
    final_columns = final_ranking[:dimension]
    final_model = build_table_model(model_name, seed + fold)
    final_model.fit(matrix[combined][:, final_columns], labels[combined])
    test_probability = final_model.predict_proba(matrix[test][:, final_columns])[:, 1]

    recipe = {
        "feature_group": feature_group,
        "selector": selector,
        "dimension": dimension,
        "model": model_name,
        "seed": seed + fold,
        "validation_fit_rows": "train only",
        "test_fit_rows": "train+validation",
        "test_labels_used": False,
    }
    validation_frame = pd.DataFrame(
        {
            "dataset_index": validation,
            "label": labels[validation],
            "rf_probability": validation_probability,
        }
    ).sort_values("dataset_index").reset_index(drop=True)
    test_frame = pd.DataFrame(
        {
            "dataset_index": test,
            "label": labels[test],
            "rf_probability": test_probability,
        }
    ).sort_values("dataset_index").reset_index(drop=True)
    return validation_frame, test_frame, recipe


def align_probabilities(daeac: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """一对一连接并核对标签，防止两条链路的样本顺序错位。"""

    merged = daeac.merge(rf, on="dataset_index", suffixes=("_daeac", "_rf"))
    if len(merged) != len(daeac) or len(merged) != len(rf):
        raise ValueError("DAEAC与RF没有覆盖完全相同的样本")
    if not np.array_equal(merged["label_daeac"], merged["label_rf"]):
        raise ValueError("DAEAC与RF标签不一致")
    return merged.rename(columns={"label_daeac": "label"}).drop(columns="label_rf")


def choose_daeac_weight(
    validation: pd.DataFrame, step: float
) -> tuple[float, pd.DataFrame]:
    """以验证accuracy为主选择DAEAC权重，并保存完整搜索轨迹。"""

    if not 0.0 < step <= 1.0:
        raise ValueError("weight-step 必须位于 (0, 1] 区间")
    weights = np.unique(np.append(np.arange(0.0, 1.0 + step / 2.0, step), 1.0))
    rows: list[dict[str, object]] = []
    labels = validation["label"].to_numpy(dtype=np.int64)
    for weight in weights:
        probability = (
            weight * validation["daeac_probability"].to_numpy()
            + (1.0 - weight) * validation["rf_probability"].to_numpy()
        )
        metrics = evaluate(labels, probability)
        rows.append({"daeac_weight": float(weight), **metrics})
    frame = pd.DataFrame(rows)
    # 同分时依次看宏F1、平衡准确率、AUC；仍同分则偏向简单的0.5等权融合。
    frame["distance_to_equal_weight"] = (frame["daeac_weight"] - 0.5).abs()
    best = frame.sort_values(
        ["accuracy", "macro_f1", "balanced_accuracy", "roc_auc", "distance_to_equal_weight"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return float(best["daeac_weight"]), frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--daeac-run-dir", type=Path)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-step", type=float, default=0.05)
    args = parser.parse_args()

    if args.fold not in range(5):
        raise ValueError("fold 必须位于0..4")
    config = load_config(args.config, {"fold": args.fold, "device": args.device})
    run_dir = (
        args.daeac_run_dir.resolve() if args.daeac_run_dir is not None else config.run_dir
    )
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"请先完成DAEAC训练，缺少checkpoint：{checkpoint_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA，但当前环境不可用")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    master = load_master(config)
    split = load_split(config.split_path)
    _train_loader, validation_loader, test_loader = build_dataloaders(
        config, master, split
    )
    daeac = build_daeac(config).to(device)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    daeac.load_state_dict(checkpoint["model_state"])
    daeac_validation = infer_probabilities(daeac, validation_loader, device)
    daeac_test = infer_probabilities(daeac, test_loader, device)

    rf_validation, rf_test, recipe = fit_rf_probabilities(
        args.features.resolve(),
        config.data_path,
        config.split_dir,
        args.search_dir.resolve(),
        args.fold,
        args.seed,
    )
    validation = align_probabilities(daeac_validation, rf_validation)
    test = align_probabilities(daeac_test, rf_test)
    daeac_weight, weight_search = choose_daeac_weight(validation, args.weight_step)
    test["fused_probability"] = (
        daeac_weight * test["daeac_probability"]
        + (1.0 - daeac_weight) * test["rf_probability"]
    )
    test["pred_label"] = (test["fused_probability"] >= 0.5).astype(np.int64)
    fused_metrics = evaluate(
        test["label"].to_numpy(dtype=np.int64), test["fused_probability"].to_numpy()
    )
    daeac_metrics = evaluate(
        test["label"].to_numpy(dtype=np.int64), test["daeac_probability"].to_numpy()
    )
    rf_metrics = evaluate(
        test["label"].to_numpy(dtype=np.int64), test["rf_probability"].to_numpy()
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    weight_search.to_csv(output / "validation_weight_search.csv", index=False)
    test.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        **fused_metrics,
        "fold": args.fold,
        "method": "validation-selected probability fusion",
        "daeac_weight": daeac_weight,
        "rf_weight": 1.0 - daeac_weight,
        "weight_selection_metric": "validation accuracy",
        "weight_step": args.weight_step,
        "daeac_run_dir": str(run_dir),
        "daeac_best_epoch": int(checkpoint["epoch"]),
        "rf_recipe": recipe,
        "component_test_metrics": {"daeac": daeac_metrics, "random_forest": rf_metrics},
        "test_labels_used_for_weight_selection": False,
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"completed: weight(DAEAC)={daeac_weight:.2f}, "
        f"test_accuracy={float(fused_metrics['accuracy']):.4f}, output={output}"
    )


if __name__ == "__main__":
    main()
