"""可视化 DAEAC 两路辅助特征的三组已有聚类，不重新拟合 K-means。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

# 兼容没有图形桌面的训练服务器。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cluster_daeac_features import COMBINATIONS, DEFAULT_CONFIG, prepare_cluster_inputs

from mitbih_2class.config import load_config
from mitbih_2class.data import load_split

DISPLAY_NAMES = {"rri": "RRI", "amplitude": "R-peak amplitude", "both": "RRI + amplitude"}
COLORS = plt.get_cmap("tab10").colors[:5]


def load_existing_clusters(config_path: Path, fold: int) -> tuple[Path, dict, dict, dict]:
    """核对聚类文件与冻结划分和输入数据相同，避免画错实验结果。"""

    config = load_config(config_path, {"fold": fold})
    output = config.run_dir / "clusters"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    split = load_split(config.split_path)
    with np.load(output / "assignments.npz") as archive:
        assignments = {key: archive[key] for key in ("dataset_indices", *COMBINATIONS)}
    original_train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    excluded_indices = np.asarray(summary.get("excluded_indices", []), dtype=np.int64)
    train_indices = original_train_indices[
        ~np.isin(original_train_indices, excluded_indices)
    ]
    if summary["fold"] != fold or not np.array_equal(assignments["dataset_indices"], train_indices):
        raise ValueError("已保存的簇编号与当前折训练索引不一致")
    if hashlib.sha256(config.split_path.read_bytes()).hexdigest() != summary["split_sha256"]:
        raise ValueError("冻结划分文件与聚类时不一致")

    # 只读两路辅助特征，不加载 7500 点 ECG；重现聚类时的行顺序与缩放。
    with np.load(config.data_path, allow_pickle=False) as archive:
        train_aux = np.stack(
            (archive["rri_256"][train_indices], archive["r_amp_256"][train_indices]), axis=1
        ).astype(np.float32)
    if hashlib.sha256(train_aux.tobytes()).hexdigest() != summary["train_aux_sha256"]:
        raise ValueError("master 的训练特征与聚类时不一致")
    inputs, scaler = prepare_cluster_inputs(train_aux)
    for key in ("mean", "std"):
        if not np.allclose(scaler[key], summary["scaler"][key]):
            raise ValueError(f"预处理尺度 {key} 与聚类时不一致")
    for name in COMBINATIONS:
        counts = np.bincount(assignments[name], minlength=summary["n_clusters"])
        expected = [row["n"] for row in summary["combinations"][name]["clusters"]]
        if counts.tolist() != expected:
            raise ValueError(f"{name} 的簇编号与汇总文件不一致")
    return output, summary, inputs, assignments


def plot_pca_and_labels(output: Path, summary: dict, inputs: dict, assignments: dict) -> Path:
    """左列显示二维 PCA 投影，右列显示各簇的 Normal/Apnea 构成。"""

    figure, axes = plt.subplots(3, 2, figsize=(15, 13), gridspec_kw={"width_ratios": [1.4, 1]})
    for row, name in enumerate(COMBINATIONS):
        ids = assignments[name]
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        points = pca.fit_transform(inputs[name])
        with np.load(output / f"{name}_centers.npz") as archive:
            centers = pca.transform(archive["centers"])

        ax = axes[row, 0]
        for cluster_id in range(summary["n_clusters"]):
            mask = ids == cluster_id
            ax.scatter(
                points[mask, 0], points[mask, 1], s=5, alpha=0.35,
                color=COLORS[cluster_id], rasterized=True,
                label=f"C{cluster_id} (n={int(mask.sum())})",
            )
            ax.scatter(
                centers[cluster_id, 0], centers[cluster_id, 1],
                s=95, marker="X", color=COLORS[cluster_id],
                edgecolors="black", linewidths=0.8, zorder=5,
            )
        ax.set_title(f"{DISPLAY_NAMES[name]}: cluster projection")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8, markerscale=2)

        ax = axes[row, 1]
        clusters = summary["combinations"][name]["clusters"]
        y = np.arange(len(clusters))
        normal_share = np.array([item["normal"] / item["n"] for item in clusters])
        apnea_share = 1 - normal_share
        ax.barh(y, normal_share, color="#6b8fb3", label="Normal")
        ax.barh(y, apnea_share, left=normal_share, color="#e89b62", label="Apnea")
        ax.set_yticks(y, [f"C{i} (n={item['n']})" for i, item in enumerate(clusters)])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.16)
        ax.set_xticks(np.linspace(0, 1, 6), [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
        for i, fraction in enumerate(apnea_share):
            ax.text(1.015, i, f"{fraction:.1%}", va="center", fontsize=8)
        ax.set_xlabel("Within-cluster label fraction")
        ax.set_title(f"{DISPLAY_NAMES[name]}: label mix")
        ax.grid(axis="x", alpha=0.15)
    figure.suptitle(
        f"DAEAC auxiliary clustering | fold {summary['fold']} train | "
        f"n={summary['train_n']} | excluded={summary.get('excluded_n', 0)} | k={summary['n_clusters']}",
        fontsize=15,
    )
    figure.text(
        0.5, 0.01,
        "Bars: blue Normal, orange Apnea. K-means used 256/512 dimensions; 2D PCA overlap does not measure full-space separation.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.965))
    target = output / "cluster_overview.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def plot_pca_zoom(output: Path, summary: dict, inputs: dict, assignments: dict) -> Path:
    """放大中心区域，避免极少数远离主群的样本压缩其余簇的显示。"""

    figure, axes = plt.subplots(3, 1, figsize=(10, 12))
    for ax, name in zip(axes, COMBINATIONS, strict=True):
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        points = pca.fit_transform(inputs[name])
        bounds = np.quantile(points, [0.005, 0.995], axis=0)
        in_view = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
        for cluster_id in range(summary["n_clusters"]):
            mask = (assignments[name] == cluster_id) & in_view
            ax.scatter(
                points[mask, 0], points[mask, 1], s=7, alpha=0.45,
                color=COLORS[cluster_id], rasterized=True,
                label=f"C{cluster_id} (shown {int(mask.sum())})",
            )
        ax.set_xlim(bounds[:, 0])
        ax.set_ylim(bounds[:, 1])
        ax.set_title(f"{DISPLAY_NAMES[name]}: central projection ({int((~in_view).sum())} outside view)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8, markerscale=1.8)
    figure.suptitle(
        f"PCA close-up | n={summary['train_n']} | excluded={summary.get('excluded_n', 0)} "
        "| axes display 0.5th–99.5th percentile",
        fontsize=15,
    )
    figure.text(0.5, 0.01, "Clusters were fitted in the full feature space; this zoom changes only the display.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.025, 1, 0.965))
    target = output / "cluster_pca_zoom.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def plot_subjects(output: Path, summary: dict, assignments: dict) -> Path:
    """显示每簇受试者来源占比，检查聚类是否主要按受试者分组。"""

    with (output / "assignments.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["dataset_index"]) for row in rows] != assignments["dataset_indices"].tolist():
        raise ValueError("CSV 与 NPZ 中的样本行号不一致")
    subjects = np.array([row["subject_id"] for row in rows])
    unique_subjects = sorted(set(subjects))
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, name in zip(axes, COMBINATIONS, strict=True):
        matrix = np.zeros((summary["n_clusters"], len(unique_subjects)), dtype=float)
        for cluster_id in range(summary["n_clusters"]):
            counts = Counter(subjects[assignments[name] == cluster_id])
            total = sum(counts.values())
            matrix[cluster_id] = [counts.get(subject, 0) / total for subject in unique_subjects]
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
        cluster_rows = summary["combinations"][name]["clusters"]
        ax.set_yticks(range(len(cluster_rows)), [f"C{i} (n={item['n']})" for i, item in enumerate(cluster_rows)])
        ax.set_title(f"{DISPLAY_NAMES[name]}: subject share within each cluster")
    axes[-1].set_xticks(range(len(unique_subjects)), unique_subjects, rotation=45, ha="right")
    figure.colorbar(image, ax=axes, fraction=0.015, pad=0.015, label="Fraction of cluster")
    figure.suptitle(
        f"Cluster composition by subject | fold {summary['fold']} train | "
        f"excluded={summary.get('excluded_n', 0)}",
        fontsize=15,
    )
    figure.subplots_adjust(left=0.1, right=0.9, bottom=0.09, top=0.92, hspace=0.32)
    target = output / "cluster_subjects.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    output, summary, inputs, assignments = load_existing_clusters(args.config, args.fold)
    print(plot_pca_and_labels(output, summary, inputs, assignments))
    print(plot_pca_zoom(output, summary, inputs, assignments))
    print(plot_subjects(output, summary, assignments))


if __name__ == "__main__":
    main()
