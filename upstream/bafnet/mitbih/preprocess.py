"""从 MIT-BIH master 构造 BAFNet 所需的五分钟 RRI/R 波幅值输入。"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import biosppy.signals.tools as signal_tools
import numpy as np
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.interpolate import splev, splrep
from scipy.signal import medfilt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_MASTER = PROJECT_ROOT.parent / "dataset" / "mitbih_master" / "mitbih_master_30s.npz"
DEFAULT_SPLIT = PROJECT_ROOT.parent / "dataset" / "mitbih_master" / "splits" / "split_dev_subjectwise.json"
DEFAULT_OUTPUT = HERE / "artifacts" / "mitbih_5min.npz"
SAMPLING_RATE = 250
EPOCH_SAMPLES = 30 * SAMPLING_RATE
WINDOW_SAMPLES = 5 * 60 * SAMPLING_RATE
GRID = np.arange(0, 300, step=1 / 3, dtype=np.float64)


def minmax(values: np.ndarray) -> np.ndarray | None:
    """每个五分钟窗口、每条输入路分别缩放到 0～1。"""

    span = float(values.max() - values.min())
    if not np.isfinite(span) or span <= 0:
        return None
    return (values - values.min()) / span


def process_record(task: tuple[str, str, np.ndarray, np.ndarray]) -> tuple:
    """只在同一条记录内取窗口，标签来自中心的 30 秒 epoch。"""

    record, subject, ecg_epochs, labels = task
    if ecg_epochs.ndim != 2 or ecg_epochs.shape[1] != EPOCH_SAMPLES:
        raise ValueError(f"{record}: ECG 必须是每 epoch {EPOCH_SAMPLES} 点")
    if len(ecg_epochs) != len(labels):
        raise ValueError(f"{record}: ECG 与标签数量不一致")
    ecg = np.asarray(ecg_epochs, dtype=np.float64).reshape(-1)
    features = []
    targets = []
    centers = []
    skipped = {"candidate": 0, "peak_count": 0, "heart_rate": 0, "interpolation": 0}

    for epoch in range(len(labels)):
        # 30 秒 epoch 的中点对齐五分钟窗口；每侧各取 150 秒。
        midpoint = epoch * EPOCH_SAMPLES + EPOCH_SAMPLES // 2
        start = midpoint - WINDOW_SAMPLES // 2
        stop = midpoint + WINDOW_SAMPLES // 2
        if start < 0 or stop > len(ecg):
            continue
        skipped["candidate"] += 1
        signal, _, _ = signal_tools.filter_signal(
            ecg[start:stop], ftype="FIR", band="bandpass", order=30,
            frequency=[3, 45], sampling_rate=SAMPLING_RATE,
        )
        try:
            peaks, = hamilton_segmenter(signal, sampling_rate=SAMPLING_RATE)
            peaks, = correct_rpeaks(
                signal, rpeaks=peaks, sampling_rate=SAMPLING_RATE, tol=0.1
            )
        except (ValueError, IndexError):
            skipped["peak_count"] += 1
            continue
        if not 40 <= len(peaks) / 5 <= 200:
            skipped["peak_count"] += 1
            continue

        rri = medfilt(np.diff(peaks) / SAMPLING_RATE, kernel_size=3)
        if not np.all((60 / rri >= 20) & (60 / rri <= 300)):
            skipped["heart_rate"] += 1
            continue
        amplitude = signal[peaks]
        rri_scaled = minmax(rri)
        amplitude_scaled = minmax(amplitude)
        if rri_scaled is None or amplitude_scaled is None:
            skipped["interpolation"] += 1
            continue
        try:
            rri_curve = splev(GRID, splrep(peaks[1:] / SAMPLING_RATE, rri_scaled, k=3), ext=1)
            amplitude_curve = splev(
                GRID, splrep(peaks / SAMPLING_RATE, amplitude_scaled, k=3), ext=1
            )
        except (ValueError, IndexError):
            skipped["interpolation"] += 1
            continue
        pair = np.column_stack((rri_curve, amplitude_curve)).astype(np.float32)
        if not np.isfinite(pair).all():
            skipped["interpolation"] += 1
            continue
        features.append(pair)
        targets.append(int(labels[epoch] != "Normal"))
        centers.append(epoch)

    arrays = (
        np.stack(features) if features else np.empty((0, 900, 2), dtype=np.float32),
        np.asarray(targets, dtype=np.int8),
        np.asarray(centers, dtype=np.int32),
    )
    return record, subject, *arrays, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--subject-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--records", nargs="*", help="仅用于小范围预处理检查")
    args = parser.parse_args()

    split_spec = json.loads(args.subject_split.read_text(encoding="utf-8"))
    test_subjects = set(split_spec["test_subjects"])
    if not test_subjects:
        raise ValueError("独立测试被试列表为空")
    with np.load(args.master, allow_pickle=False) as data:
        records = data["record"].astype(str)
        subjects = data["subject_id"].astype(str)
        epoch_numbers = data["epoch_in_record"]
        labels = data["raw_label"].astype(str)
        ecg = data["ecg_raw"]
    if set(labels) - {"Normal", "H", "HA", "OA", "X", "CA", "CAA"}:
        raise ValueError("MIT-BIH 出现未定义的原始标签")
    if len({len(records), len(subjects), len(epoch_numbers), len(labels), len(ecg)}) != 1:
        raise ValueError("master 数组行数不一致")

    names = list(dict.fromkeys(records))
    if args.records:
        if set(args.records) - set(names):
            raise ValueError("--records 包含 master 中不存在的记录")
        names = args.records
    tasks = []
    for name in names:
        indices = np.flatnonzero(records == name)
        order = np.argsort(epoch_numbers[indices])
        indices = indices[order]
        if not np.array_equal(epoch_numbers[indices], np.arange(len(indices))):
            raise ValueError(f"{name}: epoch 索引不连续，不能拼接五分钟波形")
        record_subjects = set(subjects[indices])
        if len(record_subjects) != 1:
            raise ValueError(f"{name}: 一条记录对应了多个被试")
        tasks.append((name, subjects[indices[0]], ecg[indices], labels[indices]))

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_record, task): task[0] for task in tasks}
        for future in as_completed(futures):
            record, subject, features, targets, centers, skipped = future.result()
            results[record] = (subject, features, targets, centers, skipped)
            print(f"{record}: {len(targets)} windows; skipped={skipped}", flush=True)

    features = np.concatenate([results[name][1] for name in names])
    labels = np.concatenate([results[name][2] for name in names])
    center_epochs = np.concatenate([results[name][3] for name in names])
    record_names = np.concatenate([
        np.full(len(results[name][2]), name, dtype="U16") for name in names
    ])
    subject_names = np.concatenate([
        np.full(len(results[name][2]), results[name][0], dtype="U16") for name in names
    ])
    split = np.isin(subject_names, list(test_subjects)).astype(np.int8)
    if set(np.unique(split)) != {0, 1} or set(np.unique(labels)) != {0, 1}:
        raise ValueError("预处理后训练池/测试集或二分类标签不完整")
    if set(subject_names[split == 0]) & set(subject_names[split == 1]):
        raise ValueError("训练池和测试集出现被试重叠")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=features, labels=labels, record=record_names,
        subject=subject_names, minute=center_epochs / 2,
        center_epoch=center_epochs, split=split,
    )
    manifest = {
        "dataset": "MIT-BIH PSG master V3",
        "source_master": str(args.master.resolve()),
        "source_subject_split": str(args.subject_split.resolve()),
        "test_subjects": sorted(test_subjects),
        "train_pool_subjects": sorted(set(subject_names[split == 0])),
        "shape": list(features.shape),
        "train_pool_rows": int((split == 0).sum()),
        "test_rows": int((split == 1).sum()),
        "class_counts": {"normal": int((labels == 0).sum()), "apnea": int((labels == 1).sum())},
        "per_record_skips": {name: results[name][4] for name in names},
        "label_rule": "Normal=0; H/HA/OA/X/CA/CAA=1; label from center 30-second epoch",
        "window_rule": "300 seconds centered on target 30-second epoch, within one recording",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.output}: {len(features)} rows", flush=True)


if __name__ == "__main__":
    main()
