"""从 Apnea-ECG master 原始 ECG 重建 BAFNet 的五分钟双路输入。"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import biosppy.signals.tools as st
import numpy as np
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.interpolate import splev, splrep
from scipy.signal import medfilt

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_MASTER = PROJECT_ROOT.parent / "dataset" / "apnea_ecg_master"
DEFAULT_ANSWERS = HERE.parent / "source" / "dataset" / "event-2-answers"
DEFAULT_OUTPUT = HERE / "artifacts" / "bafnet_5min.npz"
TRAIN_RECORDS = tuple(
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)
TEST_RECORDS = tuple(f"x{i:02d}" for i in range(1, 36))


def read_answers(path: Path) -> dict[str, list[str]]:
    """按作者源码的块格式读取 x 记录盲测答案。"""

    answers = {
        block[:3]: list("".join(block.split()[2::2]))
        for block in path.read_text(encoding="utf-8").split("\n\n")
        if block.strip()
    }
    if set(answers) != set(TEST_RECORDS):
        raise ValueError("event-2-answers 的 x 记录不完整")
    if any(set(labels) - {"A", "N"} for labels in answers.values()):
        raise ValueError("event-2-answers 出现未知标签")
    return answers


def _minmax(values: np.ndarray) -> np.ndarray | None:
    """保持作者的逐窗口逐路 0～1 缩放；常数序列无法缩放时丢弃。"""

    span = float(values.max() - values.min())
    if not np.isfinite(span) or span <= 0:
        return None
    return (values - values.min()) / span


def process_record(args: tuple[str, list[str], Path]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """复用原始 3～45 Hz FIR、Hamilton R 峰、校正与生理阈值处理。"""

    name, labels, master_root = args
    with np.load(master_root / "records" / f"{name}.npz", allow_pickle=False) as archive:
        ecg_epochs = np.asarray(archive["ecg"], dtype=np.float64)
        if name in TRAIN_RECORDS:
            master_labels = archive["raw_label"].astype(str)
            if len(master_labels) < len(labels) or any(
                ("N" if value == "Normal" else "A") != labels[i]
                for i, value in enumerate(master_labels[: len(labels)])
            ):
                raise ValueError(f"{name}: master 标签与原始 apn 不一致")
    if ecg_epochs.ndim != 2 or ecg_epochs.shape[1] != 6000:
        raise ValueError(f"{name}: 期望 100 Hz、每分钟 6000 点 ECG")
    ecg = ecg_epochs.reshape(-1)
    grid = np.arange(0, 300, step=1 / 3, dtype=np.float64)
    if len(grid) != 900:
        raise RuntimeError("插值时间轴应为 900 点")
    features: list[np.ndarray] = []
    targets: list[int] = []
    minutes: list[int] = []
    counts = {"candidate": 0, "peak_count": 0, "heart_rate": 0, "interpolation": 0}

    for minute, label in enumerate(labels):
        if minute < 2 or minute + 3 > len(ecg_epochs):
            continue
        counts["candidate"] += 1
        signal = ecg[(minute - 2) * 6000 : (minute + 3) * 6000]
        signal, _, _ = st.filter_signal(
            signal, ftype="FIR", band="bandpass", order=30,
            frequency=[3, 45], sampling_rate=100,
        )
        peaks, = hamilton_segmenter(signal, sampling_rate=100)
        peaks, = correct_rpeaks(signal, rpeaks=peaks, sampling_rate=100, tol=0.1)
        if len(peaks) / 5 < 40 or len(peaks) / 5 > 200:
            counts["peak_count"] += 1
            continue

        rri = medfilt(np.diff(peaks) / 100.0, kernel_size=3)
        if not np.all((60 / rri >= 20) & (60 / rri <= 300)):
            counts["heart_rate"] += 1
            continue
        amp = signal[peaks]
        rri_scaled = _minmax(rri)
        amp_scaled = _minmax(amp)
        if rri_scaled is None or amp_scaled is None:
            counts["interpolation"] += 1
            continue
        rri_curve = splev(grid, splrep(peaks[1:] / 100.0, rri_scaled, k=3), ext=1)
        amp_curve = splev(grid, splrep(peaks / 100.0, amp_scaled, k=3), ext=1)
        pair = np.column_stack((rri_curve, amp_curve)).astype(np.float32)
        if not np.isfinite(pair).all():
            counts["interpolation"] += 1
            continue
        features.append(pair)
        targets.append(0 if label == "N" else 1)
        minutes.append(minute)

    values = np.stack(features) if features else np.empty((0, 900, 2), np.float32)
    return name, values, np.asarray(targets, np.int8), np.asarray(minutes, np.int32), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--records", nargs="*", help="仅用于小范围数据链路检查")
    args = parser.parse_args()
    answers = read_answers(args.answers)
    records = tuple(args.records) if args.records else TRAIN_RECORDS + TEST_RECORDS
    if not records or len(set(records)) != len(records) or set(records) - set(TRAIN_RECORDS + TEST_RECORDS):
        raise ValueError("--records 包含未知或重复记录")

    tasks = []
    for name in records:
        if name in answers:
            labels = answers[name]
        else:
            with np.load(args.master_root / "records" / f"{name}.npz", allow_pickle=False) as archive:
                labels = ["N" if value == "Normal" else "A" for value in archive["raw_label"].astype(str)]
        tasks.append((name, labels, args.master_root))
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_record, task): task[0] for task in tasks}
        for future in as_completed(futures):
            name, features, labels, minutes, counts = future.result()
            result[name] = (features, labels, minutes, counts)
            print(f"{name}: {len(labels)} windows; skipped={counts}", flush=True)

    ordered = [result[name] for name in records]
    features = np.concatenate([item[0] for item in ordered])
    labels = np.concatenate([item[1] for item in ordered])
    minutes = np.concatenate([item[2] for item in ordered])
    groups = np.concatenate([
        np.full(len(result[name][1]), name, dtype="U3") for name in records
    ])
    split = np.asarray([0 if name in TRAIN_RECORDS else 1 for name in groups], np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, features=features, labels=labels, minute=minutes,
        record=groups, split=split,
    )
    manifest = {
        "source_master": str(args.master_root.resolve()),
        "source_answers": str(args.answers.resolve()),
        "source_commit": "ac6ea317319f9fd9bbe17f4a45e0e184e748ce2b",
        "shape": list(features.shape),
        "train_rows": int((split == 0).sum()),
        "test_rows": int((split == 1).sum()),
        "class_counts": {"normal": int((labels == 0).sum()), "apnea": int((labels == 1).sum())},
        "records": list(records),
        "per_record_skips": {name: result[name][3] for name in records},
        "note": "5-minute ECG is reconstructed by concatenating master 60-second raw epochs",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved {args.output}: {len(features)} rows", flush=True)


if __name__ == "__main__":
    main()
