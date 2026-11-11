"""对263维手工特征执行两层聚类：先全局K=2，再细分训练集中最混合的父簇。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cluster_handcrafted_features import (
    association_tests,
    cluster_summary,
    plot_feature_profiles,
    plot_subjects,
    search_k,
)

from mitbih_2class.config import load_config
from mitbih_2class.data import load_master, load_split, validate_split

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "models"
    / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "handcrafted_hierarchical_refine_mixed_e3"
    / "fold_0"
    / "clusters"
)


def select_most_mixed_parent(
    parent_ids: np.ndarray,
    labels: np.ndarray,
    n_parent_clusters: int,
) -> tuple[int, list[dict[str, float | int]]]:
    """仅用训练标签选择阳性比例最接近0.5的父簇，并返回选择依据。"""

    rows: list[dict[str, float | int]] = []
    for cluster in range(n_parent_clusters):
        selected = parent_ids == cluster
        n = int(selected.sum())
        positive_fraction = float(labels[selected].mean()) if n else float("nan")
        rows.append({
            "cluster": cluster,
            "n": n,
            "apnea_fraction": positive_fraction,
            "distance_to_half": abs(positive_fraction - 0.5),
        })
    # 距离相同时优先细分样本更多的簇，避免在很小的局部样本上继续切分。
    chosen = min(rows, key=lambda row: (float(row["distance_to_half"]), -int(row["n"])))
    return int(chosen["cluster"]), rows


def compose_leaf_ids(
    parent_ids: np.ndarray,
    refined_parent: int,
    local_ids: np.ndarray,
    local_k: int,
) -> tuple[np.ndarray, list[str]]:
    """把未细分父簇与局部子簇编码为连续叶节点编号。"""

    if len(parent_ids) != len(local_ids):
        raise ValueError("parent_ids与local_ids长度必须一致")
    leaf_ids = np.zeros(len(parent_ids), dtype=np.int16)
    refined = parent_ids == refined_parent
    if np.any((local_ids[refined] < 0) | (local_ids[refined] >= local_k)):
        raise ValueError("待细分父簇存在非法局部簇编号")
    # 叶节点0表示保留的另一个父簇；1..local_k表示混合父簇的局部子簇。
    leaf_ids[refined] = local_ids[refined].astype(np.int16) + 1
    other_parent = next(cluster for cluster in np.unique(parent_ids) if cluster != refined_parent)
    names = [f"P{int(other_parent)}"] + [
        f"P{refined_parent}/S{cluster}" for cluster in range(local_k)
    ]
    return leaf_ids, names


def plot_hierarchy(
    path: Path,
    global_scores: np.ndarray,
    leaf_ids: np.ndarray,
    test_indices: np.ndarray,
    test_rows: list[dict[str, object]],
    leaf_names: list[str],
    explained: np.ndarray,
) -> None:
    """同时展示全局PCA位置和独立测试折的叶节点类别构成。"""

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        global_scores[:, 0],
        global_scores[:, 1],
        c=leaf_ids,
        cmap="tab10",
        s=8,
        alpha=0.45,
        rasterized=True,
    )
    axes[0].set_title("Hierarchical leaf clusters in global PCA space")
    axes[0].set_xlabel(f"Global PC1 ({explained[0]:.1%})")
    axes[0].set_ylabel(f"Global PC2 ({explained[1]:.1%})")
    figure.colorbar(scatter, ax=axes[0], label="Leaf cluster")

    x = np.arange(len(test_rows))
    normal = np.asarray([int(row["normal"]) for row in test_rows])
    apnea = np.asarray([int(row["apnea"]) for row in test_rows])
    total = normal + apnea
    axes[1].bar(x, normal / total, label="Normal")
    axes[1].bar(x, apnea / total, bottom=normal / total, label="Apnea")
    axes[1].set_xticks(
        x,
        [f"{leaf_names[index]}\n(n={int(total[index])})" for index in x],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction within held-out test leaf")
    axes[1].set_title(f"Held-out test composition (n={len(test_indices)})")
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_local_stage(
    path: Path,
    local_scores: np.ndarray,
    local_ids: np.ndarray,
    local_test_mask: np.ndarray,
    local_test_rows: list[dict[str, object]],
    explained: np.ndarray,
) -> None:
    """单独展示被细分父簇的局部PCA和测试集子簇构成。"""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        local_scores[:, 0],
        local_scores[:, 1],
        c=local_ids,
        cmap="tab10",
        s=9,
        alpha=0.5,
        rasterized=True,
    )
    axes[0].set_title("Refined parent cluster in local PCA space")
    axes[0].set_xlabel(f"Local PC1 ({explained[0]:.1%})")
    axes[0].set_ylabel(f"Local PC2 ({explained[1]:.1%})")
    figure.colorbar(scatter, ax=axes[0], label="Local cluster")

    x = np.arange(len(local_test_rows))
    normal = np.asarray([int(row["normal"]) for row in local_test_rows])
    apnea = np.asarray([int(row["apnea"]) for row in local_test_rows])
    total = normal + apnea
    axes[1].bar(x, normal / total, label="Normal")
    axes[1].bar(x, apnea / total, bottom=normal / total, label="Apnea")
    axes[1].set_xticks(x, [f"S{i}\n(n={int(total[i])})" for i in x])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction within held-out test subcluster")
    axes[1].set_title(f"Refined-parent test composition (n={int(local_test_mask.sum())})")
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--parent-cluster", type=int, choices=(0, 1))
    parser.add_argument("--local-min-k", type=int, default=2)
    parser.add_argument("--local-max-k", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.local_min_k < 2 or args.local_max_k < args.local_min_k:
        raise ValueError("局部k范围必须满足2 <= local-min-k <= local-max-k")

    config = load_config(args.config, {"fold": args.fold})
    master = load_master(config)
    if master.handcrafted_signals is None or master.handcrafted_names is None:
        raise ValueError("配置没有加载完整263维手工特征")
    split = load_split(config.split_path)
    validate_split(master, split, split_unit=config.split_unit)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT
        / "runs"
        / "handcrafted_hierarchical_refine_mixed_e3"
        / f"fold_{args.fold}"
        / "clusters"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"聚类目录已有内容，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    val_indices = np.asarray(split["val_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)

    # 第一层的缺失填补、缩放、PCA和KMeans均只在训练折拟合。
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    global_scaler = StandardScaler()
    train_filled = imputer.fit_transform(master.handcrafted_signals[train_indices])
    all_filled = imputer.transform(master.handcrafted_signals)
    train_standardized = global_scaler.fit_transform(train_filled)
    all_standardized = global_scaler.transform(all_filled)
    global_pca = PCA(n_components=0.95, svd_solver="full")
    train_global_scores = global_pca.fit_transform(train_standardized)
    all_global_scores = global_pca.transform(all_standardized)
    parent_model = KMeans(n_clusters=2, n_init=30, random_state=args.seed).fit(
        train_global_scores
    )
    train_parent_ids = parent_model.labels_.astype(np.int16)
    all_parent_ids = parent_model.predict(all_global_scores).astype(np.int16)

    auto_parent, parent_train_rows = select_most_mixed_parent(
        train_parent_ids, master.labels[train_indices], n_parent_clusters=2
    )
    refined_parent = auto_parent if args.parent_cluster is None else args.parent_cluster
    parent_selection = (
        "training-label fraction closest to 0.5"
        if args.parent_cluster is None
        else "explicit command-line parent cluster"
    )

    # 第二层重新在父簇内部估计尺度和PCA，避免全局主变化掩盖局部结构。
    local_train_mask = train_parent_ids == refined_parent
    local_all_mask = all_parent_ids == refined_parent
    local_scaler = StandardScaler()
    local_train_standardized = local_scaler.fit_transform(train_filled[local_train_mask])
    local_all_standardized = local_scaler.transform(all_filled[local_all_mask])
    local_pca = PCA(n_components=0.95, svd_solver="full")
    local_train_scores = local_pca.fit_transform(local_train_standardized)
    local_all_scores = local_pca.transform(local_all_standardized)
    local_k, local_search_rows = search_k(
        local_train_scores,
        range(args.local_min_k, args.local_max_k + 1),
        seed=args.seed,
    )
    local_model = KMeans(n_clusters=local_k, n_init=30, random_state=args.seed).fit(
        local_train_scores
    )
    all_local_ids = np.full(len(master), -1, dtype=np.int16)
    all_local_ids[local_all_mask] = local_model.predict(local_all_scores).astype(np.int16)
    leaf_ids, leaf_names = compose_leaf_ids(
        all_parent_ids, refined_parent, all_local_ids, local_k
    )

    roles = np.full(len(master), "", dtype=object)
    roles[train_indices], roles[val_indices], roles[test_indices] = "train", "val", "test"
    leaf_count = local_k + 1
    test_rows = cluster_summary(
        leaf_ids[test_indices],
        master.labels[test_indices],
        master.subjects[test_indices],
        leaf_count,
    )
    all_rows = cluster_summary(leaf_ids, master.labels, master.subjects, leaf_count)
    overall_tests = association_tests(
        leaf_ids[test_indices],
        master.labels[test_indices],
        master.subjects[test_indices],
        master.records[test_indices],
        leaf_count,
        permutations=args.permutations,
        seed=args.seed,
    )

    test_parent_mask = all_parent_ids[test_indices] == refined_parent
    refined_test_indices = test_indices[test_parent_mask]
    local_test_rows = cluster_summary(
        all_local_ids[refined_test_indices],
        master.labels[refined_test_indices],
        master.subjects[refined_test_indices],
        local_k,
    )
    local_tests = association_tests(
        all_local_ids[refined_test_indices],
        master.labels[refined_test_indices],
        master.subjects[refined_test_indices],
        master.records[refined_test_indices],
        local_k,
        permutations=args.permutations,
        seed=args.seed + 1,
    )

    with (output / "local_k_search.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(local_search_rows[0]))
        writer.writeheader()
        writer.writerows(local_search_rows)
    with (output / "assignments.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fields = [
            "dataset_index",
            "split",
            "subject_id",
            "record",
            "label",
            "parent_cluster",
            "local_cluster",
            "leaf_cluster",
            "leaf_name",
            "global_pc1",
            "global_pc2",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(master)):
            writer.writerow({
                "dataset_index": index,
                "split": roles[index],
                "subject_id": master.subjects[index],
                "record": master.records[index],
                "label": int(master.labels[index]),
                "parent_cluster": int(all_parent_ids[index]),
                "local_cluster": int(all_local_ids[index]),
                "leaf_cluster": int(leaf_ids[index]),
                "leaf_name": leaf_names[int(leaf_ids[index])],
                "global_pc1": float(all_global_scores[index, 0]),
                "global_pc2": float(all_global_scores[index, 1]),
            })

    plot_hierarchy(
        output / "hierarchical_overview.png",
        all_global_scores,
        leaf_ids,
        test_indices,
        test_rows,
        leaf_names,
        global_pca.explained_variance_ratio_,
    )
    plot_local_stage(
        output / "local_stage_overview.png",
        local_all_scores,
        all_local_ids[local_all_mask],
        test_parent_mask,
        local_test_rows,
        local_pca.explained_variance_ratio_,
    )
    profile_features = plot_feature_profiles(
        output / "leaf_feature_profiles.png",
        all_standardized,
        leaf_ids,
        master.handcrafted_names,
        leaf_count,
    )
    plot_subjects(
        output / "leaf_subjects.png", leaf_ids, master.subjects, leaf_count
    )

    summary = {
        "fold": args.fold,
        "fit_rows": "train_indices only for both stages",
        "train_n": len(train_indices),
        "val_n": len(val_indices),
        "test_n": len(test_indices),
        "source_feature_count": master.handcrafted_signals.shape[1],
        "global_pca_components": int(global_pca.n_components_),
        "global_pca_explained_variance": float(
            global_pca.explained_variance_ratio_.sum()
        ),
        "parent_k": 2,
        "refined_parent": refined_parent,
        "parent_selection": parent_selection,
        "parent_train_summary": parent_train_rows,
        "local_train_n": int(local_train_mask.sum()),
        "local_pca_components": int(local_pca.n_components_),
        "local_pca_explained_variance": float(local_pca.explained_variance_ratio_.sum()),
        "local_k": local_k,
        "leaf_names": leaf_names,
        "all_leaf_summary": all_rows,
        "held_out_test_leaf_summary": test_rows,
        "held_out_test_overall_association": overall_tests,
        "held_out_test_refined_parent_summary": local_test_rows,
        "held_out_test_refined_parent_association": local_tests,
        "profile_features": profile_features,
        "warning": (
            "The parent selected for refinement uses training labels only. Local K selection "
            "is unsupervised; test labels are used only for final descriptive evaluation."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"refined parent={refined_parent}, local k={local_k}, "
        f"leaf clusters={leaf_count}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

