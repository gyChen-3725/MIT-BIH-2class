"""核对 master 重建的 x 盲测窗口与作者随仓库提供的预测 CSV。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_CSV = (
    HERE.parent / "source" / "utils" / "code_for_calculating_per-recording"
    / "output" / "BAFNet.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "artifacts" / "bafnet_5min.npz")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    args = parser.parse_args()
    source: dict[str, list[int]] = defaultdict(list)
    with args.source_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source[row["subject"]].append(int(float(row["y_true"])))

    current: dict[str, list[int]] = defaultdict(list)
    with np.load(args.data, allow_pickle=False) as data:
        for record, label, split in zip(data["record"], data["labels"], data["split"]):
            if split == 1:
                current[str(record)].append(int(label))
    if set(current) != set(source):
        raise ValueError(f"测试记录不同：当前 {len(current)}，作者 {len(source)}")
    differences = [
        record for record in sorted(source) if source[record] != current[record]
    ]
    if differences:
        raise ValueError(f"以下记录的窗口数或标签顺序不同：{differences}")
    print(f"OK: {len(source)} records, {sum(map(len, source.values()))} windows, labels match")


if __name__ == "__main__":
    main()
