"""训练前检查 master 数据形状、划分协议与测试集覆盖。

默认审计五折完整覆盖；传入 ``--fold`` 时只核验指定单折。这一步只读取数据和
划分，不训练模型、不写入实验结果。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import default_config_path, load_config
from mitbih_2class.data import class_counts, load_master, load_split, validate_split


def main() -> None:
    """逐折审计索引和类别分布，并确认所有样本恰好测试一次。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    config = load_config(args.config)
    master = load_master(config)
    print(
        f"dataset={config.data_path} signal={master.signals.shape} "
        f"aux={None if master.aux_signals is None else master.aux_signals.shape} "
        "handcrafted="
        f"{None if master.handcrafted_signals is None else master.handcrafted_signals.shape} "
        f"total={len(master)}"
    )
    all_test_indices: set[int] = set()
    folds = range(5) if args.fold is None else (args.fold,)
    for fold in folds:
        # 折内手工特征掩码可能不同；逐折重新加载可同时核对五份索引和名称，
        # 而不是只验证配置默认的fold 0。
        fold_config = config.with_overrides(fold=fold)
        fold_master = (
            load_master(fold_config)
            if config.handcrafted_selection_dir is not None else master
        )
        split = load_split(fold_config.split_path)
        validate_split(fold_master, split, split_unit=fold_config.split_unit)
        test_indices = set(split["test_indices"])
        if all_test_indices & test_indices:
            raise ValueError(f"Fold {fold} test set overlaps an earlier test fold")
        all_test_indices.update(test_indices)
        # 显示每折阴性/阳性数量，便于发现类别缺失或分布明显异常。
        counts = {
            role: class_counts(fold_master, split[f"{role}_indices"])
            for role in ("train", "val", "test")
        }
        handcrafted_dimension = (
            None
            if fold_master.handcrafted_signals is None
            else fold_master.handcrafted_signals.shape[1]
        )
        print(
            f"fold={fold} handcrafted_dim={handcrafted_dimension} "
            f"binary_counts={counts}"
        )
    if args.fold is None:
        if all_test_indices != set(range(len(master))):
            raise ValueError("Five test folds do not cover the master dataset exactly once")
        print(f"OK: {config.split_unit}-level splits and complete five-fold test coverage")
    else:
        print(f"OK: {config.split_unit}-level Fold {args.fold} split audit")


if __name__ == "__main__":
    main()

