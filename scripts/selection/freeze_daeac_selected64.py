"""从已完成的347维组合搜索中冻结DAEAC每折64维手工特征掩码。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))

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
DEFAULT_OUTPUT = ROOT / "protocols" / "handcrafted_selection_expanded347_e3_grid_k64"
SELECTION_DIMENSION = 64


def choose_validation_candidate(
    validation_grid: pd.DataFrame,
    fold: int,
    dimension: int = SELECTION_DIMENSION,
) -> pd.Series:
    """按原组合搜索的同分规则选择当前折、目标维度下的最佳验证候选。"""

    candidates = validation_grid.loc[
        (validation_grid["fold"] == fold)
        & (validation_grid["stage"] == "within_group_selection")
        & (validation_grid["dimension"] == dimension)
    ].copy()
    if candidates.empty:
        raise ValueError(f"fold {fold}: 没有{dimension}维的第二阶段候选")
    # 原搜索以accuracy为主；同分时依次比较macro F1、balanced accuracy、AUC，
    # 最后使用选择器和模型名称稳定打破完全同分。
    candidates = candidates.sort_values(
        [
            "val_accuracy",
            "val_macro_f1",
            "val_balanced_accuracy",
            "val_roc_auc",
            "selector",
            "model",
        ],
        ascending=[False, False, False, False, False, False],
    )
    return candidates.iloc[0]


def map_group_selection_to_full_indices(
    group_names: tuple[str, ...],
    ranking: np.ndarray,
    full_names: tuple[str, ...],
    dimension: int,
) -> tuple[list[int], list[str]]:
    """把组内排名转换为347维冻结矩阵的全局列号。"""

    if len(set(full_names)) != len(full_names):
        raise ValueError("完整347维特征名称必须唯一")
    if len(ranking) != len(group_names) or len(np.unique(ranking)) != len(ranking):
        raise ValueError("组内特征排名必须是完整排列")
    if not 1 <= dimension <= len(group_names):
        raise ValueError("目标维度超出候选特征组范围")
    full_index = {name: index for index, name in enumerate(full_names)}
    selected_names = [group_names[int(index)] for index in ranking[:dimension]]
    missing = [name for name in selected_names if name not in full_index]
    if missing:
        raise ValueError(f"组内特征未出现在完整347维输入中：{missing[:3]}")
    selected_indices = [full_index[name] for name in selected_names]
    return selected_indices, selected_names


def write_readme(path: Path, choices: list[dict[str, object]]) -> None:
    """记录五折选择配方及使用边界，避免把掩码误认为全局固定特征。"""

    lines = [
        "# DAEAC扩展手工特征：折内64维冻结掩码",
        "",
        "本目录供 `daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml` 使用。",
        "每折先从既有组合搜索的64维候选中，用验证accuracy确定特征组和选择器；随后只使用",
        "该折训练行重新拟合特征排名并冻结64列。测试标签不参与配方选择或列排名。",
        "",
        "| 折 | 特征组 | 选择器 | 表格分类器 | 验证accuracy | 64维排名拟合数据 |",
        "|---:|---|---|---|---:|---|",
    ]
    for choice in choices:
        lines.append(
            f"| {choice['fold']} | `{choice['feature_group']}` | `{choice['selector']}` | "
            f"`{choice['validation_model']}` | {choice['val_accuracy']:.4f} | 仅当前折训练集 |"
        )
    lines.extend(
        [
            "",
            "五折都向DAEAC提供64维输入，但具体特征名称允许不同。每折的中位数填补、均值和标准差",
            "仍由DAEAC数据加载器仅在该折训练行拟合。该目录不是一个可直接部署的全局64维列表；",
            "部署前需要在最终开发集上重新冻结一套单独掩码。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--search-dir", type=Path, default=DEFAULT_SEARCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=SELECTION_DIMENSION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_path = args.features.resolve()
    master_path = args.master.resolve()
    split_dir = args.split_dir.resolve()
    search_dir = args.search_dir.resolve()
    output = args.output_dir.resolve()
    if args.dimension < 1:
        raise ValueError("dimension必须为正整数")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"冻结掩码目录已有内容，避免覆盖：{output}")

    validation_grid = pd.read_csv(search_dir / "validation_grid.csv")
    matrices, labels, _metadata, names_by_group = load_design_matrix(
        feature_path, master_path
    )
    full_names = tuple(names_by_group["expanded_all"])
    if len(full_names) != 347:
        raise ValueError(f"完整扩展输入应为347维，实际为{len(full_names)}维")

    output.mkdir(parents=True, exist_ok=True)
    choices: list[dict[str, object]] = []
    for fold in range(5):
        candidate = choose_validation_candidate(validation_grid, fold, args.dimension)
        feature_group = str(candidate["feature_group"])
        selector = str(candidate["selector"])
        matrix = matrices[feature_group]
        group_names = tuple(names_by_group[feature_group])
        split = load_fold(split_dir / f"fold_{fold}.json", len(labels))
        train_indices = split["train"]
        ranking = fit_feature_ranking(
            matrix[train_indices], labels[train_indices], selector, args.seed + fold
        )
        selected_indices, selected_names = map_group_selection_to_full_indices(
            group_names, ranking, full_names, args.dimension
        )
        choice = {
            "fold": fold,
            "feature_group": feature_group,
            "source_group_feature_count": matrix.shape[1],
            "selector": selector,
            "dimension": args.dimension,
            "validation_model": str(candidate["model"]),
            "val_accuracy": float(candidate["val_accuracy"]),
            "val_macro_f1": float(candidate["val_macro_f1"]),
            "val_balanced_accuracy": float(candidate["val_balanced_accuracy"]),
            "val_roc_auc": float(candidate["val_roc_auc"]),
            "selection_recipe_rows": "validation rows choose group/selector/model",
            "ranking_fit_rows": "train_indices only",
            "ranking_fit_n": len(train_indices),
            "source_feature_count": len(full_names),
            "selected_indices": selected_indices,
            "selected_names": selected_names,
            "test_labels_used": False,
        }
        (output / f"fold_{fold}.json").write_text(
            json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        choices.append(choice)
        print(
            f"fold {fold}: {feature_group}/{selector}/{args.dimension} -> "
            f"{len(selected_indices)} frozen columns",
            flush=True,
        )

    manifest = {
        "protocol": "E3 epoch-level frozen five-fold",
        "source_search_dir": str(search_dir),
        "source_validation_grid": str(search_dir / "validation_grid.csv"),
        "feature_path": str(feature_path),
        "feature_sha256": sha256_file(feature_path),
        "master_path": str(master_path),
        "split_dir": str(split_dir),
        "dimension": args.dimension,
        "seed": args.seed,
        "recipe_selection_metric": "validation accuracy",
        "recipe_tie_breakers": ["macro_f1", "balanced_accuracy", "roc_auc"],
        "ranking_fit_rows": "current fold train_indices only",
        "validation_labels_used_for": "feature_group and selector recipe choice only",
        "test_labels_used": False,
        "fold_choices": [
            {
                key: choice[key]
                for key in (
                    "fold",
                    "feature_group",
                    "selector",
                    "validation_model",
                    "val_accuracy",
                )
            }
            for choice in choices
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(output / "README.md", choices)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
