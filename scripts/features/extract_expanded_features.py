"""在旧263维EDR特征之外，提取HRV、原始ECG和QRS形态特征，生成347维数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.features.expanded import (
    ADVANCED_HRV_FEATURE_NAMES,
    ECG_FEATURE_NAMES,
    QRS_FEATURE_NAMES,
    epoch_advanced_hrv_features,
    epoch_ecg_features,
    epoch_qrs_morphology_features,
    expanded_feature_names,
    lowpass_filter,
)


def find_edr_output() -> Path:
    """兼容本机Projects布局和服务器CGY目录下的同级仓库布局。"""

    candidates = (
        ROOT.parent / "EDR" / "outputs" / "slpdb_paper_aligned_v1",
        ROOT.parent.parent / "EDR" / "outputs" / "slpdb_paper_aligned_v1",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


DEFAULT_EDR_RUN = find_edr_output()
DEFAULT_LEGACY = DEFAULT_EDR_RUN / "handcrafted_v1" / "features_30s.npz"
DEFAULT_DATASET = ROOT.parent / "dataset" / "mit-bih-polysomnographic-database-1.0.0"
DEFAULT_OUTPUT = ROOT / "features" / "expanded347"
QUALITY_FEATURES = {
    "coverage_fraction",
    "longest_uncovered_seconds",
    "baseline_coverage_fraction",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_ecg(dataset_dir: Path, record_name: str) -> tuple[np.ndarray, float]:
    """只读取SLPDB的ECG通道，避免为本实验加载真实呼吸参考。"""

    base = dataset_dir / record_name
    header = wfdb.rdheader(str(base))
    ecg_names = [name for name in header.sig_name if name.strip().lower() == "ecg"]
    if len(ecg_names) != 1:
        raise RuntimeError(f"{record_name}: ECG通道数量异常 {ecg_names}")
    record = wfdb.rdrecord(str(base), channel_names=ecg_names, physical=True)
    if record.p_signal is None:
        raise RuntimeError(f"{record_name}: 无法读取物理单位ECG")
    return np.asarray(record.p_signal[:, 0], dtype=np.float64), float(record.fs)


def legacy_model_matrix(
    method_features: np.ndarray,
    methods: np.ndarray,
    feature_names: np.ndarray,
    hrv_features: np.ndarray,
    hrv_names: np.ndarray,
    cross_features: np.ndarray,
    cross_names: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """严格复用旧模型的263维顺序，并继续排除每路3项质量字段。"""

    columns: list[np.ndarray] = [hrv_features]
    names = [f"hrv.{name}" for name in hrv_names]
    columns.extend(
        method_features[:, method_index, feature_index, None]
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
    columns.append(cross_features)
    names.extend(f"cross.{name}" for name in cross_names)
    matrix = np.concatenate(columns, axis=1).astype(np.float32)
    if matrix.shape[1] != 263 or len(names) != 263:
        raise ValueError(f"旧模型特征应为263维，实际为{matrix.shape[1]}")
    return matrix, tuple(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-features", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_EDR_RUN / "records")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    legacy_path = args.legacy_features.resolve()
    record_dir = args.record_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    destination = args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"扩展特征目录已有内容，避免覆盖：{destination}")
    if not legacy_path.is_file() or not record_dir.is_dir() or not dataset_dir.is_dir():
        raise FileNotFoundError("旧特征、记录产物或SLPDB原始数据目录不存在")

    with np.load(legacy_path, allow_pickle=False) as archive:
        legacy = {key: archive[key] for key in archive.files}
    required = {
        "record",
        "subject_id",
        "start_sample",
        "class_label",
        "raw_annotation",
        "sleep_stage",
        "method_names",
        "feature_names",
        "method_features",
        "hrv_feature_names",
        "hrv_features",
        "cross_feature_names",
        "cross_features",
    }
    missing = required - set(legacy)
    if missing:
        raise KeyError(f"旧特征文件缺少字段：{sorted(missing)}")
    records = legacy["record"].astype(str)
    starts = legacy["start_sample"].astype(np.int64)
    n_epochs = len(records)
    legacy_matrix, legacy_names = legacy_model_matrix(
        legacy["method_features"],
        legacy["method_names"].astype(str),
        legacy["feature_names"].astype(str),
        legacy["hrv_features"],
        legacy["hrv_feature_names"].astype(str),
        legacy["cross_features"],
        legacy["cross_feature_names"].astype(str),
    )
    advanced_hrv = np.full(
        (n_epochs, len(ADVANCED_HRV_FEATURE_NAMES)), np.nan, dtype=np.float32
    )
    ecg_features = np.full((n_epochs, len(ECG_FEATURE_NAMES)), np.nan, dtype=np.float32)
    qrs_features = np.full((n_epochs, len(QRS_FEATURE_NAMES)), np.nan, dtype=np.float32)
    source_record_hashes: dict[str, str] = {}

    for record_name in dict.fromkeys(records):
        indices = np.flatnonzero(records == record_name)
        ecg, fs = load_raw_ecg(dataset_dir, record_name)
        stored_path = record_dir / f"{record_name}.npz"
        with np.load(stored_path, allow_pickle=False) as stored:
            beat_times = stored["beat_times"].astype(np.float64)
            q_samples = stored["q_samples"].astype(np.int64)
            r_samples = stored["r_samples"].astype(np.int64)
            s_samples = stored["s_samples"].astype(np.int64)
            normal_beat = stored["normal_beat"].astype(bool)
            polarity = float(stored["polarity"])
            stored_fs = float(stored["fs_ecg"])
        if not np.isclose(stored_fs, fs) or not np.allclose(
            r_samples.astype(np.float64) / fs,
            beat_times,
            rtol=0.0,
            # beat_times在旧record产物中保存为float32；允许小于半个ECG采样周期的舍入。
            atol=0.5 / fs,
        ):
            raise ValueError(f"{record_name}: 保存的心搏时间轴与原始记录不一致")
        filtered = lowpass_filter(ecg, fs, 35.0, order=4)
        source_record_hashes[record_name] = sha256_file(stored_path)
        epoch_length = round(30 * fs)
        for epoch_index in indices:
            start = int(starts[epoch_index])
            stop = start + epoch_length
            if start < 0 or stop > len(ecg):
                raise ValueError(f"{record_name}: epoch {epoch_index}越过ECG边界")
            start_seconds = start / fs
            advanced = epoch_advanced_hrv_features(
                beat_times, normal_beat, start_seconds, start_seconds + 30.0
            )
            advanced_hrv[epoch_index] = [
                advanced[name] for name in ADVANCED_HRV_FEATURE_NAMES
            ]
            raw = epoch_ecg_features(ecg[start:stop], fs)
            ecg_features[epoch_index] = [raw[name] for name in ECG_FEATURE_NAMES]
            morphology = epoch_qrs_morphology_features(
                filtered,
                q_samples,
                r_samples,
                s_samples,
                normal_beat,
                polarity,
                start,
                stop,
                fs,
            )
            qrs_features[epoch_index] = [
                morphology[name] for name in QRS_FEATURE_NAMES
            ]
        print(f"{record_name}: {len(indices)} expanded epochs", flush=True)

    all_names = expanded_feature_names(legacy_names)
    all_features = np.concatenate(
        (legacy_matrix, advanced_hrv, ecg_features, qrs_features), axis=1
    ).astype(np.float32)
    if all_features.shape != (n_epochs, 347) or len(all_names) != 347:
        raise RuntimeError(f"扩展矩阵应为{(n_epochs, 347)}，实际为{all_features.shape}")

    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "features_30s_expanded.npz"
    np.savez_compressed(
        output_path,
        **legacy,
        advanced_hrv_feature_names=np.asarray(ADVANCED_HRV_FEATURE_NAMES),
        advanced_hrv_features=advanced_hrv,
        ecg_feature_names=np.asarray(ECG_FEATURE_NAMES),
        ecg_features=ecg_features,
        qrs_feature_names=np.asarray(QRS_FEATURE_NAMES),
        qrs_features=qrs_features,
        all_feature_names=np.asarray(all_names),
        all_features=all_features,
    )
    manifest = {
        "version": "expanded_v2",
        "source_legacy_file": str(legacy_path),
        "source_legacy_sha256": sha256_file(legacy_path),
        "source_record_sha256": source_record_hashes,
        "dataset_dir": str(dataset_dir),
        "n_epochs": n_epochs,
        "legacy_feature_count": 263,
        "advanced_hrv_feature_count": len(ADVANCED_HRV_FEATURE_NAMES),
        "ecg_feature_count": len(ECG_FEATURE_NAMES),
        "qrs_feature_count": len(QRS_FEATURE_NAMES),
        "all_feature_count": len(all_names),
        "feature_order": list(all_names),
        "window_seconds": 30,
        "raw_ecg_bandpass_hz": [0.5, 35.0],
        "uses_reference_respiration": False,
        "uses_labels_during_extraction": False,
        "short_window_limitations": (
            "30-second windows do not support conventional LF/HF estimation; "
            "frequency-domain HRV was intentionally omitted."
        ),
        "missing_fraction": {
            "advanced_hrv": float(np.mean(~np.isfinite(advanced_hrv))),
            "ecg": float(np.mean(~np.isfinite(ecg_features))),
            "qrs": float(np.mean(~np.isfinite(qrs_features))),
            "all": float(np.mean(~np.isfinite(all_features))),
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output_path}")
    print(f"shape: {all_features.shape}")


if __name__ == "__main__":
    main()
