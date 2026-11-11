"""汇总五折测试预测并输出折外指标；未完成五折时会直接报错。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.aggregate import aggregate_oof
from mitbih_2class.config import default_config_path, load_config


def main() -> None:
    """读取实验配置并调用统一的 OOF 校验与汇总函数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(aggregate_oof(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

