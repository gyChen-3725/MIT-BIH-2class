"""在训练折上拟合PCA-KMeans，并在独立测试折解释手工特征聚类。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import load_config
from mitbih_2class.data import load_master, load_split, validate_split

DEFAULT_CONFIG = ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "handcrafted_all_pca_kmeans_e3" / "fold_0" / "clusters"


def contingency(cluster_ids: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """构造簇×二分类列联表，行顺序固定为0到k-1。"""

    return np.bincount(
        cluster_ids.astype(np.int64) * 2 + labels.astype(np.int64),
        minlength=n_clusters * 2,
    ).reshape(n_clusters, 2)


def choose_k(rows: list[dict[str, float | int]]) -> int:
    """对四项内部指标做等权秩汇总，分数相同时选择较小的k。"""

    silhouette = np.asarray([float(row["silhouette"]) for row in rows])
    calinski = np.asarray([float(row["calinski_harabasz"]) for row in rows])
    davies = np.asarray([float(row["davies_bouldin"]) for row in rows])
    stability = np.asarray([float(row["stability_ari"]) for row in rows])
    ranks = np.column_stack((
        rankdata(-silhouette, method="average"),
        rankdata(-calinski, method="average"),
        rankdata(davies, method="average"),
        rankdata(-stability, method="average"),
    ))
    for row, score in zip(rows, ranks.mean(axis=1), strict=True):
        row["composite_rank"] = float(score)
    return int(min(rows, key=lambda row: (float(row["composite_rank"]), int(row["k"]))) ["k"])


def search_k(
    train: np.ndarray,
    candidates: range,
    *,
    seed: int,
    silhouette_sample: int = 2500,
) -> tuple[int, list[dict[str, float | int]]]:
    """只使用训练特征比较候选k，并评估五个随机种子的分配稳定性。"""

    rows: list[dict[str, float | int]] = []
    sample_size = min(silhouette_sample, len(train))
    for k in candidates:
        reference = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(train)
        alternatives = [
            KMeans(n_clusters=k, n_init=10, random_state=seed + offset).fit_predict(train)
            for offset in range(1, 6)
        ]
        stability = float(np.mean([
            adjusted_rand_score(reference.labels_, labels) for labels in alternatives
        ]))
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(
                train,
                reference.labels_,
                sample_size=sample_size,
                random_state=seed,
            )),
            "calinski_harabasz": float(calinski_harabasz_score(train, reference.labels_)),
            "davies_bouldin": float(davies_bouldin_score(train, reference.labels_)),
            "stability_ari": stability,
            "minimum_cluster_n": int(np.bincount(reference.labels_, minlength=k).min()),
        })
    return choose_k(rows), rows


def association_tests(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    records: np.ndarray,
    n_clusters: int,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """在测试折计算标签关联，并用受试者置换和记录内循环移位构造零分布。"""

    table = contingency(cluster_ids, labels, n_clusters)
    observed, asymptotic_p, _, _ = chi2_contingency(table, correction=False)
    n = int(table.sum())
    cramer_v = float(np.sqrt(observed / (n * min(n_clusters - 1, 1)))) if n else 0.0
    rng = np.random.default_rng(seed)
    permutation_null = np.empty(permutations, dtype=np.float64)
    circular_null = np.empty(permutations, dtype=np.float64)
    subject_groups = [np.flatnonzero(subjects == subject) for subject in np.unique(subjects)]
    record_groups = [np.flatnonzero(records == record) for record in np.unique(records)]
    for iteration in range(permutations):
        shuffled = labels.copy()
        for group in subject_groups:
            shuffled[group] = rng.permutation(shuffled[group])
        permutation_null[iteration] = chi2_contingency(
            contingency(cluster_ids, shuffled, n_clusters), correction=False
        )[0]
        shifted = labels.copy()
        for group in record_groups:
            if len(group) > 1:
                shifted[group] = np.roll(shifted[group], int(rng.integers(0, len(group))))
        circular_null[iteration] = chi2_contingency(
            contingency(cluster_ids, shifted, n_clusters), correction=False
        )[0]
    return {
        "test_contingency_normal_apnea": table.tolist(),
        "pearson_chi_square": float(observed),
        "degrees_of_freedom": n_clusters - 1,
        "asymptotic_p": float(asymptotic_p),
        "cramer_v": cramer_v,
        "within_subject_permutation_p": float(
            (1 + np.count_nonzero(permutation_null >= observed)) / (permutations + 1)
        ),
        "within_record_circular_shift_p": float(
            (1 + np.count_nonzero(circular_null >= observed)) / (permutations + 1)
        ),
        "permutations": permutations,
    }


def cluster_summary(
    ids: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    n_clusters: int,
) -> list[dict[str, object]]:
    """汇总各簇样本量、阳性比例和受试者覆盖。"""

    rows = []
    for cluster in range(n_clusters):
        selected = ids == cluster
        rows.append({
            "cluster": cluster,
            "n": int(selected.sum()),
            "normal": int(np.count_nonzero(labels[selected] == 0)),
            "apnea": int(np.count_nonzero(labels[selected] == 1)),
            "apnea_fraction": float(labels[selected].mean()),
            "subject_n": len(np.unique(subjects[selected])),
        })
    return rows


def plot_overview(
    path: Path,
    scores: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    test_indices: np.ndarray,
    rows: list[dict[str, object]],
    explained: np.ndarray,
) -> None:
    """显示二维PCA投影以及独立测试折各簇标签构成。"""

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    scatter = axes[0].scatter(
        scores[:, 0], scores[:, 1], c=ids, cmap="tab10", s=8, alpha=0.45, rasterized=True
    )
    axes[0].set_title("All epochs assigned by train-fitted KMeans")
    axes[0].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({explained[1]:.1%})")
    figure.colorbar(scatter, ax=axes[0], label="Cluster")

    x = np.arange(len(rows))
    normal = np.asarray([row["normal"] for row in rows])
    apnea = np.asarray([row["apnea"] for row in rows])
    total = normal + apnea
    axes[1].bar(x, normal / total, label="Normal")
    axes[1].bar(x, apnea / total, bottom=normal / total, label="Apnea")
    axes[1].set_xticks(x, [f"C{i}\n(n={int(total[i])})" for i in x])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction within held-out test cluster")
    axes[1].set_title(f"Held-out test composition (n={len(test_indices)})")
    axes[1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_feature_profiles(
    path: Path,
    standardized: np.ndarray,
    ids: np.ndarray,
    names: tuple[str, ...],
    n_clusters: int,
    top_n: int = 30,
) -> list[str]:
    """选取簇间均值方差最大的字段，绘制标准化均值热图。"""

    profiles = np.vstack([standardized[ids == cluster].mean(axis=0) for cluster in range(n_clusters)])
    importance = profiles.var(axis=0)
    selected = np.argsort(importance)[-top_n:][::-1]
    figure, axis = plt.subplots(figsize=(12, max(7, top_n * 0.25)), constrained_layout=True)
    image = axis.imshow(profiles[:, selected].T, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axis.set_xticks(range(n_clusters), [f"C{i}" for i in range(n_clusters)])
    axis.set_yticks(range(len(selected)), [names[index] for index in selected], fontsize=8)
    axis.set_title("Cluster profiles: mean train-standardized feature value")
    figure.colorbar(image, ax=axis, label="Standardized mean")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [names[index] for index in selected]


def plot_subjects(
    path: Path,
    ids: np.ndarray,
    subjects: np.ndarray,
    n_clusters: int,
) -> None:
    """检查簇是否被单一受试者主导。"""

    unique = np.unique(subjects)
    matrix = np.zeros((n_clusters, len(unique)), dtype=np.float64)
    for cluster in range(n_clusters):
        selected = ids == cluster
        total = max(1, int(selected.sum()))
        matrix[cluster] = [np.count_nonzero(selected & (subjects == subject)) / total for subject in unique]
    figure, axis = plt.subplots(figsize=(12, max(3, n_clusters * 0.7)), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, matrix.max()))
    axis.set_xticks(range(len(unique)), unique, rotation=45, ha="right")
    axis.set_yticks(range(n_clusters), [f"C{i}" for i in range(n_clusters)])
    axis.set_title("Subject fraction within each cluster")
    figure.colorbar(image, ax=axis, label="Fraction of cluster")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config, {"fold": args.fold})
    master = load_master(config)
    if master.handcrafted_signals is None or master.handcrafted_names is None:
        raise ValueError("配置没有加载完整263维手工特征")
    split = load_split(config.split_path)
    validate_split(master, split, split_unit=config.split_unit)
    output = (
        args.output_dir.resolve() if args.output_dir is not None
        else ROOT / "runs" / "handcrafted_all_pca_kmeans_e3" / f"fold_{args.fold}" / "clusters"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"聚类目录已有内容，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    val_indices = np.asarray(split["val_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_filled = imputer.fit_transform(master.handcrafted_signals[train_indices])
    train_standardized = scaler.fit_transform(train_filled)
    all_standardized = scaler.transform(imputer.transform(master.handcrafted_signals))
    pca = PCA(n_components=0.95, svd_solver="full")
    train_scores = pca.fit_transform(train_standardized)
    all_scores = pca.transform(all_standardized)

    chosen_k, search_rows = search_k(
        train_scores, range(args.min_k, args.max_k + 1), seed=args.seed
    )
    model = KMeans(n_clusters=chosen_k, n_init=30, random_state=args.seed).fit(train_scores)
    ids = model.predict(all_scores).astype(np.int16)
    roles = np.full(len(master), "", dtype=object)
    roles[train_indices], roles[val_indices], roles[test_indices] = "train", "val", "test"

    test_rows = cluster_summary(
        ids[test_indices], master.labels[test_indices], master.subjects[test_indices], chosen_k
    )
    all_rows = cluster_summary(ids, master.labels, master.subjects, chosen_k)
    tests = association_tests(
        ids[test_indices],
        master.labels[test_indices],
        master.subjects[test_indices],
        master.records[test_indices],
        chosen_k,
        permutations=args.permutations,
        seed=args.seed,
    )

    with (output / "k_search.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(search_rows[0]))
        writer.writeheader()
        writer.writerows(search_rows)
    with (output / "assignments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["dataset_index", "split", "subject_id", "record", "label", "cluster", "pc1", "pc2"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(master)):
            writer.writerow({
                "dataset_index": index,
                "split": roles[index],
                "subject_id": master.subjects[index],
                "record": master.records[index],
                "label": int(master.labels[index]),
                "cluster": int(ids[index]),
                "pc1": float(all_scores[index, 0]),
                "pc2": float(all_scores[index, 1]),
            })

    plot_overview(
        output / "cluster_overview.png",
        all_scores,
        ids,
        master.labels,
        test_indices,
        test_rows,
        pca.explained_variance_ratio_,
    )
    profile_features = plot_feature_profiles(
        output / "cluster_feature_profiles.png",
        all_standardized,
        ids,
        master.handcrafted_names,
        chosen_k,
    )
    plot_subjects(output / "cluster_subjects.png", ids, master.subjects, chosen_k)

    summary = {
        "fold": args.fold,
        "fit_rows": "train_indices only",
        "train_n": len(train_indices),
        "val_n": len(val_indices),
        "test_n": len(test_indices),
        "source_feature_count": master.handcrafted_signals.shape[1],
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "chosen_k": chosen_k,
        "k_selection": "equal-weight rank of silhouette, CH, DB, and five-seed ARI",
        "all_cluster_summary": all_rows,
        "held_out_test_cluster_summary": test_rows,
        "held_out_test_label_association": tests,
        "profile_features": profile_features,
        "warning": (
            "Clusters are descriptive feature phenotypes. Association does not establish "
            "clinical subtypes or predictive generalization."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"chosen k={chosen_k}, PCA components={pca.n_components_}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
