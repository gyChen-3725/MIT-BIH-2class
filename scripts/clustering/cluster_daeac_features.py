"""只用冻结训练折的 RRI、R 峰幅度及两者组合生成 K-means 聚类。

BAFNet 作者用单路 RRI 做五簇聚类。本脚本把这个思路用于 DAEAC 的
30 秒两路辅助特征，分别输出三组簇编号；不筛困难样本、不更新模型权重。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import load_config
from mitbih_2class.data import load_master, load_split, prepare_aux_features, validate_split

DEFAULT_CONFIG = ROOT / "configs" / "clustering" / "daeac_aspp2_feature_clustering_e3.yaml"
COMBINATIONS = ("rri", "amplitude", "both")


def prepare_cluster_inputs(raw_aux: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """沿用 DAEAC 辅助分支预处理，并按训练集尺度标准化两路特征。

    输入只能是本折训练样本；均值和标准差也只从这些样本计算。
    每路用一个全局均值和标准差，不逐 epoch 去均值，保留平均 RRI/幅度信息。
    两路拼接时各占 256 维，标准化后不会因原始量纲不同而偏向一侧。
    """

    if raw_aux.ndim != 3 or raw_aux.shape[1:] != (2, 256):
        raise ValueError(f"应提供 [训练样本数, 2, 256]，实际为 {raw_aux.shape}")
    if len(raw_aux) < 2:
        raise ValueError("聚类至少需要两个训练样本")
    processed = prepare_aux_features(torch.from_numpy(raw_aux).float()).numpy()
    mean = processed.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = processed.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    scaled = (processed - mean[None, :, None]) / std[None, :, None]
    inputs = {
        "rri": np.ascontiguousarray(scaled[:, 0, :]),
        "amplitude": np.ascontiguousarray(scaled[:, 1, :]),
        "both": np.ascontiguousarray(scaled.reshape(len(scaled), 512)),
    }
    scaler = {"mean": mean.tolist(), "std": std.tolist()}
    return inputs, scaler


def cluster_counts(
    cluster_ids: np.ndarray, labels: np.ndarray, subjects: np.ndarray, n_clusters: int
) -> list[dict[str, object]]:
    """汇总每簇训练样本、类别与受试者分布，便于检查聚类偏向。"""

    rows: list[dict[str, object]] = []
    for cluster in range(n_clusters):
        mask = cluster_ids == cluster
        counts = np.bincount(labels[mask], minlength=2)
        subject_counts = Counter(subjects[mask].tolist())
        total = int(mask.sum())
        rows.append(
            {
                "cluster": cluster,
                "n": total,
                "normal": int(counts[0]),
                "apnea": int(counts[1]),
                "apnea_fraction": float(counts[1] / total) if total else None,
                "subject_count": len(subject_counts),
                "top_subjects": subject_counts.most_common(3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-index-file", type=Path,
        help="仅从聚类输入中排除所列 master 行号；不会修改 master 或冻结划分",
    )
    args = parser.parse_args()

    config = load_config(args.config, {"fold": args.fold})
    if config.model != "daeac" or config.model_params.get("use_third_aspp", False):
        raise ValueError("聚类配置必须使用关闭 aspp3 的 DAEAC 模型")
    if config.signal_key != "ecg_filtered_rri_r_amp_256":
        raise ValueError("聚类配置必须同时读取滤波 ECG、RRI 和 R 峰幅度")
    master = load_master(config)
    split = load_split(config.split_path)
    validate_split(master, split, split_unit=config.split_unit)
    original_train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    excluded_indices = np.empty(0, dtype=np.int64)
    exclusion_metadata: dict[str, object] = {}
    if args.exclude_index_file is not None:
        exclusion_file = args.exclude_index_file.resolve()
        exclusion = json.loads(exclusion_file.read_text(encoding="utf-8"))
        declared = np.asarray(exclusion["dataset_indices"], dtype=np.int64)
        if declared.ndim != 1 or len(np.unique(declared)) != len(declared):
            raise ValueError("异常点行号必须是一维且无重复")
        if len(declared) and (declared.min() < 0 or declared.max() >= len(master)):
            raise ValueError("异常点行号超出 master 范围")
        # 文件记录全局 master 行号；本折仅排除落在训练集的样本。
        excluded_indices = original_train_indices[np.isin(original_train_indices, declared)]
        expected_fractions = exclusion.get("rri_invalid_fraction", {})
        actual_fractions = {}
        for index in excluded_indices:
            rri = master.aux_signals[index, 0] if master.aux_signals is not None else None
            if rri is None:
                raise ValueError("master 未加载 RRI 特征")
            fraction = float(np.mean(~np.isfinite(rri) | (rri < 0) | (rri > 3)))
            actual_fractions[str(index)] = fraction
            if str(index) in expected_fractions and not np.isclose(
                fraction, expected_fractions[str(index)], atol=1 / 256
            ):
                raise ValueError(f"样本 {index} 的异常 RRI 比例与审计记录不一致")
        exclusion_metadata = {
            "exclusion_file": str(exclusion_file),
            "exclusion_sha256": hashlib.sha256(exclusion_file.read_bytes()).hexdigest(),
            "excluded_rri_invalid_fraction": actual_fractions,
        }
    train_indices = original_train_indices[~np.isin(original_train_indices, excluded_indices)]
    if not 2 <= args.clusters <= len(train_indices):
        raise ValueError("聚类数必须在 2 与训练样本数之间")
    if master.aux_signals is None:
        raise ValueError("master 未加载 RRI 和 R 峰幅度")

    # 仅取训练索引；验证和测试样本不进入标准化、K-means 拟合或簇统计。
    train_aux = np.ascontiguousarray(master.aux_signals[train_indices])
    inputs, scaler = prepare_cluster_inputs(train_aux)
    labels = master.labels[train_indices]
    subjects = master.subjects[train_indices]
    records = master.records[train_indices]
    output = config.run_dir / "clusters"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"聚类产物已存在，避免覆盖：{output}")
    output.mkdir(parents=True, exist_ok=True)

    assignments: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, object]] = {}
    for combination in COMBINATIONS:
        # 复用 BAFNet 的五簇 K-means、10 次初始化；三组输入用同一随机种子。
        model = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed)
        cluster_ids = model.fit_predict(inputs[combination]).astype(np.int16)
        assignments[combination] = cluster_ids
        summaries[combination] = {
            "input_shape": list(inputs[combination].shape),
            "inertia": float(model.inertia_),
            "clusters": cluster_counts(cluster_ids, labels, subjects, args.clusters),
        }
        # 保存中心可复核簇分配；三种输入维数不同，不直接比较 inertia 大小。
        np.savez_compressed(output / f"{combination}_centers.npz", centers=model.cluster_centers_)
        print(f"{combination}: {[row['n'] for row in summaries[combination]['clusters']]}")

    with (output / "assignments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("dataset_index", "subject_id", "record", "label", "rri", "amplitude", "both")
        )
        for position, index in enumerate(train_indices):
            writer.writerow(
                (
                    int(index), subjects[position], records[position], int(labels[position]),
                    *(int(assignments[name][position]) for name in COMBINATIONS),
                )
            )
    np.savez_compressed(
        output / "assignments.npz",
        dataset_indices=train_indices,
        rri=assignments["rri"],
        amplitude=assignments["amplitude"],
        both=assignments["both"],
    )
    metadata = {
        "config": str(config.config_path),
        "data": str(config.data_path),
        "split": str(config.split_path),
        "split_sha256": hashlib.sha256(config.split_path.read_bytes()).hexdigest(),
        "train_aux_sha256": hashlib.sha256(train_aux.tobytes()).hexdigest(),
        "fold": config.fold,
        "split_unit": config.split_unit,
        "train_n": len(train_indices),
        "train_n_before_exclusion": len(original_train_indices),
        "excluded_n": len(excluded_indices),
        "excluded_indices": excluded_indices.tolist(),
        **exclusion_metadata,
        "n_clusters": args.clusters,
        "n_init": 10,
        "seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "preprocessing": "RRI clip [0,3] seconds; amplitude asinh; train-only channel scaling",
        "scaler": scaler,
        "combinations": summaries,
        "training_started": False,
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
