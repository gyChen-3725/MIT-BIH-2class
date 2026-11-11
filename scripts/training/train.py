"""单折训练入口；参数和训练流程由 mitbih_2class.training 统一维护。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 无需先安装包即可从项目根目录或其他工作目录直接运行此脚本。
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.training import cli_main  # noqa: I001


if __name__ == "__main__":
    cli_main()

