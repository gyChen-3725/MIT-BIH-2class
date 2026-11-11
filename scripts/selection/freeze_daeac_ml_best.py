"""把机器学习网格的折内最优配方冻结为 DAEAC 可读取的特征掩码。

默认只处理 Fold 0。配方（特征组、选择器、维度和表格分类器）来自验证集搜索，
但具体特征排名仅使用当前折训练行重新拟合，避免把验证标签直接写入 DAEAC 输入列。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "selection"))

from optimize_feature_dimension import fit_feature_ranking
from train_handcrafted_2class import (
    DEFAULT_MASTER,
    DEFAULT_SPLITS,
    load_design_matrix,
    load_fold,
    sha256_file,
)

DEFAULT_FEATURES = ROOT / "features" / "expanded347" / "features_30s_expanded.npz"
DEFAULT_SEARCH = ROOT / "features" / "expanded347" / "selection_ml_grid_e3_v1"
DEFAULT_OUTPUT = ROOT / "protocols" / "handcrafted_selection_expanded347_ml_best_fold0"


def map_to_full_indices(
    selected_names: list[str], full_names: list[str]
) -> list[int]:
    """按名称把特征组内列映射到完整347维矩阵，防止依赖组内列号。"""

    if len(set(full_names)) != len(full_names):
        raise ValueError("完整347维特征名称必须唯一")
    full_index = {name: index for index, name in enumerate(full_names)}
    missing = [name for name in selected_names if name not in full_index]
    if missing:
        raise ValueError(f"最优配方中的特征不在完整输入中：{missing[:3]}")
    return [full_index[name] for name in selected_names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    folds = args.fold or [0]
    if any(fold not in range(5) for fold in folds) or len(set(folds)) != len(folds):
        raise ValueError("--fold 必须是不重复的0..4整数")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"冻结掩码目录已有内容，避免覆盖：{output}")

    feature_path = args.features.resolve()
    master_path = args.master.resolve()
    split_dir = args.split_dir.resolve()
    search_dir = args.search_dir.resolve()
    fold_best = pd.read_csv(search_dir / "fold_best.csv").set_index("fold")
    matrices, labels, _metadata, names_by_group = load_design_matrix(
        feature_path, master_path
    )
    full_names = list(names_by_group["expanded_all"])
    if len(full_names) != 347:
        raise ValueError(f"完整扩展输入应为347维，实际为{len(full_names)}维")

    output.mkdir(parents=True, exist_ok=True)
    choices: list[dict[str, object]] = []
    for fold in folds:
        if fold not in fold_best.index:
            raise ValueError(f"fold_best.csv 缺少 Fold {fold}")
        best = fold_best.loc[fold]
        feature_group = str(best["feature_group"])
        selector = str(best["selector"])
        dimension = int(best["dimension"])
        matrix = matrices[feature_group]
        group_names = list(names_by_group[feature_group])
        split = load_fold(split_dir / f"fold_{fold}.json", len(labels))
        train_indices = split["train"]
        if selector == "none":
            ranking = np.arange(matrix.shape[1], dtype=np.int64)
        else:
            ranking = fit_feature_ranking(
                matrix[train_indices], labels[train_indices], selector, args.seed + fold
            )
        selected_names = [group_names[int(index)] for index in ranking[:dimension]]
        selected_indices = map_to_full_indices(selected_names, full_names)
        payload = {
            "fold": fold,
            "feature_group": feature_group,
            "source_group_feature_count": matrix.shape[1],
            "selector": selector,
            "dimension": dimension,
            "validation_model": str(best["model"]),
            "val_accuracy": float(best["val_accuracy"]),
            "val_macro_f1": float(best["val_macro_f1"]),
            "val_balanced_accuracy": float(best["val_balanced_accuracy"]),
            "val_roc_auc": float(best["val_roc_auc"]),
            "selection_recipe_rows": "validation rows choose the ML-best recipe",
            "ranking_fit_rows": "train_indices only",
            "ranking_fit_n": len(train_indices),
            "source_feature_count": len(full_names),
            "selected_indices": selected_indices,
            "selected_names": selected_names,
            "test_labels_used": False,
        }
        (output / f"fold_{fold}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        choices.append(payload)
        print(
            f"fold {fold}: {feature_group}/{selector}/{dimension}/"
            f"{best['model']} -> {len(selected_indices)} frozen columns"
        )

    manifest = {
        "protocol": "E3 fold-specific ML-best recipe for DAEAC fusion",
        "source_search_dir": str(search_dir),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "master_path": str(master_path),
        "split_dir": str(split_dir),
        "seed": args.seed,
        "ranking_fit_rows": "current fold train_indices only",
        "test_labels_used": False,
        "fold_choices": [
            {
                key: choice[key]
                for key in (
                    "fold", "feature_group", "selector", "dimension",
                    "validation_model", "val_accuracy",
                )
            }
            for choice in choices
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme_lines = [
        "# DAEAC使用机器学习折内最优配方的冻结掩码",
        "",
        "配方来自既有验证网格；具体列只使用当前折训练行重新排名。",
        "因此它复用的是机器学习最优的特征组、选择器和维度，而不是直接复用",
        "机器学习最终阶段在训练＋验证行上重排后的列。测试标签完全不参与。",
        "",
        "| 折 | 特征组 | 选择器 | 维度 | 表格模型 | 验证accuracy |",
        "|---:|---|---|---:|---|---:|",
    ]
    for choice in choices:
        readme_lines.append(
            f"| {choice['fold']} | `{choice['feature_group']}` | "
            f"`{choice['selector']}` | {choice['dimension']} | "
            f"`{choice['validation_model']}` | {choice['val_accuracy']:.4f} |"
        )
    (output / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
