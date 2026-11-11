"""读取 MIT-BIH master 数据，并按配置检查受试者级或样本级五折划分。

这里仅把原始标签映射为二分类并按索引取样，不重新生成或修改 master 数据。
训练、验证、测试索引均来自外部冻结的划分 JSON。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mitbih_2class.config import ExperimentConfig

QUALITY_FEATURES = {
    "coverage_fraction", "longest_uncovered_seconds", "baseline_coverage_fraction"
}


def binary_label(raw_label: str) -> int:
    """把 Normal 映射为 0，其余已标注呼吸事件映射为 1（Apnea）。

当前函数假定 master 数据的原始标签已经过上游校验；若接入新数据源，
应先检查标签全集，避免未知字符串被自动归入阳性类。
    """

    return 0 if str(raw_label) == "Normal" else 1


def prepare_aux_features(raw: torch.Tensor) -> torch.Tensor:
    """把 master 的 RRI 和 R 峰幅度变成 DAEAC 辅助分支实际接收的数值。

    接受 ``[2, 256]`` 单样本或 ``[N, 2, 256]`` 批量输入。聚类和训练共用
    此函数，避免对同一份 master 特征采用不同预处理；不修改调用者的张量。
    """

    if raw.shape[-2:] != (2, 256):
        raise ValueError(f"辅助特征应以 [2, 256] 结尾，实际为 {tuple(raw.shape)}")
    aux = torch.nan_to_num(raw.clone(), nan=0.0, posinf=0.0, neginf=0.0)
    # RRI 单位为秒；master 的三次插值可能出现负值和极端过冲。
    aux[..., 0, :] = aux[..., 0, :].clamp(0.0, 3.0)
    # 幅度保留绝对水平，用 asinh 压缩极端峰值。
    aux[..., 1, :] = torch.asinh(aux[..., 1, :])
    return aux


@dataclass(frozen=True)
class MasterArrays:
    """保持 ECG、可选辅助特征、标签与来源信息处于同一行索引空间。"""

    signals: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    records: np.ndarray
    aux_signals: np.ndarray | None = None
    handcrafted_signals: np.ndarray | None = None
    handcrafted_names: tuple[str, ...] | None = None

    def __len__(self) -> int:
        """返回 master 数据中的 epoch 数量。"""

        return len(self.labels)


@dataclass(frozen=True)
class FeatureNormalizer:
    """仅由当前折训练样本拟合的逐列填补及标准化参数。"""

    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> FeatureNormalizer:
        """缺失列填零，常数列缩放因子置一；不读取验证或测试样本。"""

        median = np.zeros(values.shape[1], dtype=np.float32)
        for column in range(values.shape[1]):
            finite = values[np.isfinite(values[:, column]), column]
            if len(finite):
                median[column] = np.median(finite)
        filled = np.where(np.isfinite(values), values, median)
        mean = filled.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = filled.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(median, mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """复用训练折参数处理任一集合，保留原始特征的跨样本差异。"""

        filled = np.where(np.isfinite(values), values, self.median)
        return ((filled - self.mean) / self.scale).astype(np.float32)


def load_handcrafted_all(
    path: Path, master: np.lib.npyio.NpzFile
) -> tuple[np.ndarray, tuple[str, ...]]:
    """读取旧263维或扩展手工特征矩阵，并逐行核对MIT master。"""

    with np.load(path, allow_pickle=False) as archive:
        required = {"record", "start_sample", "class_label"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"Handcrafted dataset is missing keys: {sorted(missing)}")
        for key in ("record", "start_sample"):
            if not np.array_equal(archive[key].astype(str), master[key].astype(str)):
                raise ValueError(f"手工特征与 MIT master 的 {key} 不一致")
        expected_labels = (master["raw_label"].astype(str) != "Normal")
        actual_labels = (archive["class_label"].astype(str) != "Normal")
        if not np.array_equal(actual_labels, expected_labels):
            raise ValueError("手工特征与 MIT master 的二分类标签不一致")
        if {"all_features", "all_feature_names"}.issubset(archive.files):
            # 扩展文件显式冻结列顺序，避免下游重新拼接时遗漏新的特征族。
            matrix = archive["all_features"].astype(np.float32)
            names = archive["all_feature_names"].astype(str).tolist()
            if matrix.shape != (len(expected_labels), len(names)):
                raise ValueError("all_features 的样本数或特征名称数量不一致")
        else:
            legacy_required = {
                "method_names", "feature_names", "method_features",
                "hrv_feature_names", "hrv_features",
                "cross_feature_names", "cross_features",
            }
            legacy_missing = legacy_required - set(archive.files)
            if legacy_missing:
                raise KeyError(
                    f"Handcrafted dataset is missing keys: {sorted(legacy_missing)}"
                )
            methods = archive["method_names"].astype(str)
            feature_names = archive["feature_names"].astype(str)
            hrv_names = archive["hrv_feature_names"].astype(str)
            cross_names = archive["cross_feature_names"].astype(str)
            values = archive["method_features"]
            hrv = archive["hrv_features"]
            cross = archive["cross_features"]
            if values.shape[:2] != (len(expected_labels), len(methods)):
                raise ValueError("method_features 的样本或方法数量不一致")
            if values.shape[2] != len(feature_names):
                raise ValueError("method_features 的特征名称数量不一致")
            # 与 EDR 表格基线的 all 组保持同一顺序及同一排除规则。
            columns = [hrv]
            names = [f"hrv.{name}" for name in hrv_names]
            columns.extend(
                values[:, method_index, feature_index, None]
                for method_index in range(len(methods))
                for feature_index, name in enumerate(feature_names)
                if name not in QUALITY_FEATURES
            )
            names.extend(
                f"{method}.{name}"
                for method in methods
                for name in feature_names
                if name not in QUALITY_FEATURES
            )
            columns.append(cross)
            names.extend(f"cross.{name}" for name in cross_names)
            matrix = np.concatenate(columns, axis=1).astype(np.float32)
    if matrix.shape[1] != len(names):
        raise ValueError("手工特征矩阵与名称数量不一致")
    return matrix, tuple(names)


def apply_handcrafted_selection(
    matrix: np.ndarray,
    names: tuple[str, ...],
    selection_path: Path,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """按当前折已冻结的索引选列，并核对名称以防特征顺序漂移。"""

    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    indices = np.asarray(payload.get("selected_indices", []), dtype=np.int64)
    selected_names = tuple(map(str, payload.get("selected_names", [])))
    if (
        len(indices) == 0 or len(indices) != len(np.unique(indices))
        or indices.min() < 0 or indices.max() >= matrix.shape[1]
    ):
        raise ValueError(f"{selection_path}: selected_indices 无效")
    actual_names = tuple(names[index] for index in indices)
    if actual_names != selected_names:
        raise ValueError(f"{selection_path}: 特征名称与当前矩阵顺序不一致")
    if int(payload.get("source_feature_count", -1)) != matrix.shape[1]:
        raise ValueError(f"{selection_path}: 原始特征维度不一致")
    return matrix[:, indices], actual_names


def load_master(config: ExperimentConfig) -> MasterArrays:
    """读取配置指定的信号及标签，检查所有数组的行数与输入长度。

读取后使用 float32 信号和 int64 标签，以匹配 PyTorch 的卷积和交叉熵输入。
    """

    with np.load(config.data_path, allow_pickle=False) as archive:
        # 三模态实验直接读取 master 已保存的同一行特征，不重新检测 R 峰。
        fusion = config.signal_key == "ecg_filtered_rri_r_amp_256_handcrafted_all"
        with_aux = config.signal_key in {
            "ecg_filtered_rri_r_amp_256", "ecg_filtered_rri_r_amp_256_handcrafted_all"
        }
        handcrafted_only = config.signal_key == "handcrafted_all"
        if handcrafted_only:
            signal_keys = ()
        elif config.signal_key == "rri_r_amp_256":
            signal_keys = ("rri_256", "r_amp_256")
        elif with_aux:
            signal_keys = ("ecg_filtered",)
        else:
            signal_keys = (config.signal_key,)
        required = {*signal_keys, "raw_label", "subject_id", "record"}
        if handcrafted_only or fusion:
            required.add("start_sample")
        if with_aux:
            required.update(("rri_256", "r_amp_256"))
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"Dataset is missing keys: {sorted(missing)}")
        handcrafted_signals = None
        handcrafted_names = None
        if handcrafted_only:
            signals, handcrafted_names = load_handcrafted_all(config.handcrafted_path, archive)
        elif fusion:
            signals = archive["ecg_filtered"].astype(np.float32, copy=False)
            handcrafted_signals, handcrafted_names = load_handcrafted_all(
                config.handcrafted_path, archive
            )
            if config.handcrafted_selection_dir is not None:
                handcrafted_signals, handcrafted_names = apply_handcrafted_selection(
                    handcrafted_signals,
                    handcrafted_names,
                    config.handcrafted_selection_dir / f"fold_{config.fold}.json",
                )
        elif len(signal_keys) == 2:
            signals = np.stack([archive[key] for key in signal_keys], axis=1).astype(np.float32)
        else:
            signals = archive[signal_keys[0]].astype(np.float32, copy=False)
        aux_signals = (
            np.stack((archive["rri_256"], archive["r_amp_256"]), axis=1).astype(np.float32)
            if with_aux else None
        )
        raw_labels = archive["raw_label"].astype(str)
        subjects = archive["subject_id"].astype(str)
        records = archive["record"].astype(str)

    lengths = {len(signals), len(raw_labels), len(subjects), len(records)}
    if len(lengths) != 1:
        raise ValueError("Master dataset arrays have inconsistent lengths")
    expected_shape = (
        (len(raw_labels), 2, config.expected_input_length)
        if len(signal_keys) == 2
        else (len(raw_labels), config.expected_input_length)
    )
    if signals.shape != expected_shape:
        raise ValueError(
            f"Expected {config.signal_key} shape {expected_shape}, "
            f"got {signals.shape}"
        )
    if aux_signals is not None and aux_signals.shape != (len(raw_labels), 2, 256):
        raise ValueError(f"Expected auxiliary shape {(len(raw_labels), 2, 256)}, got {aux_signals.shape}")
    if handcrafted_signals is not None and handcrafted_signals.shape != (
        len(raw_labels), int(config.model_params["handcrafted_dim"])
    ):
        raise ValueError(
            "Expected handcrafted shape "
            f"{(len(raw_labels), int(config.model_params['handcrafted_dim']))}, "
            f"got {handcrafted_signals.shape}"
        )
    labels = np.asarray([binary_label(value) for value in raw_labels], dtype=np.int64)
    return MasterArrays(
        signals, labels, subjects, records, aux_signals,
        handcrafted_signals, handcrafted_names,
    )


def load_split(path: Path) -> dict[str, Any]:
    """读取一折划分，确认索引和受试者名单这两组关键字段齐全。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "train_indices",
        "val_indices",
        "test_indices",
        "train_subjects",
        "val_subjects",
        "test_subjects",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Split is missing keys: {sorted(missing)}")
    return payload


def validate_split(
    master: MasterArrays, split: dict[str, Any], split_unit: str = "subject"
) -> None:
    """检查一折内的索引完整性，并按协议检查受试者隔离。

    subject 协议要求训练、验证和测试受试者互斥；epoch 协议允许受试者重叠，
    但三组样本索引仍须互斥且覆盖全部 master 行。两种协议都用索引反查
    实际受试者，防止 JSON 元数据与波形数据错位。
    """

    declared_unit = split.get("split_unit")
    if declared_unit is not None and declared_unit != split_unit:
        raise ValueError(f"划分文件声明 {declared_unit}，配置要求 {split_unit}")
    if declared_unit is None and split_unit != "subject":
        raise ValueError("样本级划分文件必须显式声明 split_unit: epoch")
    role_subjects = {
        role: set(map(str, split[f"{role}_subjects"])) for role in ("train", "val", "test")
    }
    if split_unit == "subject" and (
        role_subjects["train"] & role_subjects["val"]
        or role_subjects["train"] & role_subjects["test"]
        or role_subjects["val"] & role_subjects["test"]
    ):
        raise ValueError("Subject overlap found between train, validation, and test")

    # 逐个集合核对索引边界与实际受试者，避免只看 JSON 的声明而漏检错位索引。
    role_indices: dict[str, np.ndarray] = {}
    for role in ("train", "val", "test"):
        indices = np.asarray(split[f"{role}_indices"], dtype=np.int64)
        if len(indices) == 0:
            raise ValueError(f"{role} split is empty")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{role} split contains duplicate indices")
        if indices.min() < 0 or indices.max() >= len(master):
            raise IndexError(f"{role} split contains out-of-range indices")
        actual_subjects = set(master.subjects[indices])
        if actual_subjects != role_subjects[role]:
            raise ValueError(
                f"{role} subject metadata mismatch: JSON={sorted(role_subjects[role])}, "
                f"data={sorted(actual_subjects)}"
            )
        role_indices[role] = indices

    index_sets = {role: set(values.tolist()) for role, values in role_indices.items()}
    if (
        index_sets["train"] & index_sets["val"]
        or index_sets["train"] & index_sets["test"]
        or index_sets["val"] & index_sets["test"]
    ):
        raise ValueError("Sample index overlap found between train, validation, and test")
    if set.union(*index_sets.values()) != set(range(len(master))):
        raise ValueError("Train, validation, and test must cover every master row exactly once")


class EpochDataset(Dataset):
    """按冻结索引惰性获取单个 30 秒 epoch，返回模型输入与来源元数据。"""

    def __init__(
        self, master: MasterArrays, indices: list[int], normalization: str = "per_epoch_zscore",
        feature_normalizer: FeatureNormalizer | None = None,
    ) -> None:
        self.master = master
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalization = normalization
        self.feature_normalizer = feature_normalizer

    def __len__(self) -> int:
        """返回该集合包含的 epoch 数，不是 master 数据总行数。"""

        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """返回 ECG、可选辅助特征及可追溯到 master 的行索引。"""

        index = int(self.indices[item])
        values = self.master.signals[index].copy()
        if self.normalization == "train_feature_zscore":
            if self.feature_normalizer is None:
                raise ValueError("手工特征输入需要训练折的 FeatureNormalizer")
            values = self.feature_normalizer.transform(values)
        signal = torch.from_numpy(values).float()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        if self.normalization != "train_feature_zscore":
            signal = self._normalize(signal)
        result = {
            "features": signal,
            "label": int(self.master.labels[index]),
            "dataset_index": index,
            "subject_id": str(self.master.subjects[index]),
            "record": str(self.master.records[index]),
        }
        if self.master.aux_signals is not None:
            # 保留 RRI 的秒单位和幅度的绝对水平；逐 epoch z-score 会抹掉均值信息。
            aux = torch.from_numpy(self.master.aux_signals[index].copy()).float()
            result["aux_features"] = prepare_aux_features(aux)
        if self.master.handcrafted_signals is not None:
            if self.feature_normalizer is None:
                raise ValueError("融合手工特征需要训练折的 FeatureNormalizer")
            handcrafted = self.feature_normalizer.transform(
                self.master.handcrafted_signals[index].copy()
            )
            result["handcrafted_features"] = torch.from_numpy(handcrafted).float()
        return result

    def _normalize(self, signal: torch.Tensor) -> torch.Tensor:
        """按通道处理单个 epoch，供 ECG 和辅助特征复用。"""

        # 非有限值先置零，再对当前 epoch 的每一路特征独立缩放。
        finite = torch.isfinite(signal)
        if not bool(finite.all()):
            signal = torch.where(finite, signal, torch.zeros_like(signal))
        if self.normalization == "per_epoch_zscore":
            signal = (signal - signal.mean(dim=-1, keepdim=True)) / signal.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(1e-6)
        elif self.normalization == "per_channel_minmax":
            low = signal.amin(dim=-1, keepdim=True)
            high = signal.amax(dim=-1, keepdim=True)
            signal = (signal - low) / (high - low).clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        return signal


def class_counts(master: MasterArrays, indices: list[int]) -> tuple[int, int]:
    """统计指定索引中的阴性/阳性数量；训练权重只能传入训练集索引。"""

    counts = Counter(master.labels[np.asarray(indices, dtype=np.int64)].tolist())
    return counts.get(0, 0), counts.get(1, 0)


def build_dataloaders(
    config: ExperimentConfig, master: MasterArrays, split: dict[str, Any]
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """先验证划分，再创建 train、val、test 三个 DataLoader。

只打乱训练集；固定随机种子使每折的训练样本顺序可重复。
    """

    validate_split(master, split, split_unit=config.split_unit)
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    if master.handcrafted_signals is not None:
        feature_normalizer = FeatureNormalizer.fit(master.handcrafted_signals[train_indices])
    elif config.normalization == "train_feature_zscore":
        feature_normalizer = FeatureNormalizer.fit(master.signals[train_indices])
    else:
        feature_normalizer = None
    generator = torch.Generator().manual_seed(config.seed + config.fold)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "persistent_workers": config.num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(
            EpochDataset(master, split["train_indices"], config.normalization, feature_normalizer),
            shuffle=True,
            generator=generator,
            **common,
        ),
        DataLoader(
            EpochDataset(master, split["val_indices"], config.normalization, feature_normalizer),
            shuffle=False,
            **common,
        ),
        DataLoader(
            EpochDataset(master, split["test_indices"], config.normalization, feature_normalizer),
            shuffle=False,
            **common,
        ),
    )

