"""按 fold 0～4 顺序运行五折，每折使用配置指定的划分文件。

各折独立构建模型和优化器，并写入各自的结果目录。运行结束后，
可另行执行 aggregate_oof.py 检查完整覆盖并计算总体指标。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import default_config_path, load_config
from mitbih_2class.training import train_fold


def main() -> None:
    """读取同一份实验配置，逐折覆盖 fold 参数并启动训练。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    for fold in range(5):
        config = load_config(args.config, {"fold": fold, "device": args.device})
        train_fold(config, debug=args.debug)


if __name__ == "__main__":
    main()

