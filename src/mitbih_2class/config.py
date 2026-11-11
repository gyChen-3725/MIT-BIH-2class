"""读取并校验实验配置，统一解析本机和服务器上的路径。

YAML 中的相对路径以项目根目录为基准；运行时先转成绝对路径，
这样从不同工作目录调用脚本时仍会读取同一份数据和划分。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


def find_repository_root(start: Path | None = None) -> Path:
    """从给定文件或目录向上寻找 pyproject.toml 所在的项目根目录。"""

    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not find pyproject.toml from {current}")


def _resolve(value: str | Path, root: Path) -> Path:
    """保留绝对路径，并把相对路径解析为相对于项目根目录的路径。"""

    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    """一次实验的完整参数；冻结对象可避免训练中意外修改配置。

    数据路径指向仓库外共享的 MIT-BIH master 数据；fold 对应已有划分文件。
    labels 的顺序同时决定模型输出列和评价指标的正负类约定。
    model 对应 models/ 下的模块名，model_params 保存该论文模型专有的构造参数。
    """

    name: str
    data_path: Path
    split_dir: Path
    split_unit: str
    output_root: Path
    signal_key: str
    normalization: str
    expected_input_length: int
    model: str
    model_params: dict[str, Any]
    labels: tuple[str, str]
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    class_weight: str
    seed: int
    fold: int
    num_workers: int
    device: str
    repository_root: Path
    config_path: Path
    checkpoint_metric: str = "val_macro_f1"
    handcrafted_path: Path | None = None
    handcrafted_selection_dir: Path | None = None

    @property
    def run_dir(self) -> Path:
        """当前折的产物目录，包含 checkpoint、预测和配置快照。"""

        return self.output_root / self.name / f"fold_{self.fold}"

    @property
    def split_path(self) -> Path:
        """当前折的 train/val/test 划分文件；划分单位由 split_unit 指定。"""

        return self.split_dir / f"fold_{self.fold}.json"

    def with_overrides(self, **values: Any) -> ExperimentConfig:
        """生成经校验的新配置，供命令行覆盖 fold、设备或 debug 参数。"""

        unknown = set(values) - {item.name for item in fields(self)}
        if unknown:
            raise KeyError(f"Unknown override(s): {sorted(unknown)}")
        config = replace(self, **values)
        config.validate()
        return config

    def validate(self, require_files: bool = True) -> None:
        """在读取大数据和开始训练前拦截无效参数与缺失文件。

        require_files=False 可用于只检查配置结构的场景；正常训练应检查数据和
        当前折划分都存在。模型名必须是 models/ 下的单个模块名。
        """

        if not self.name or any(char in self.name for char in '<>:"/\\|?*'):
            raise ValueError(f"Invalid experiment name: {self.name!r}")
        if self.fold not in range(5):
            raise ValueError("fold must be in 0..4")
        if self.split_unit not in {"subject", "epoch"}:
            raise ValueError("split_unit 必须是 subject 或 epoch")
        if self.labels != ("Normal", "Apnea"):
            raise ValueError("Binary label order is fixed as [Normal, Apnea]")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.model):
            raise ValueError(f"model 必须是小写 Python 模块名: {self.model!r}")
        if not isinstance(self.model_params, dict) or not all(
            isinstance(key, str) for key in self.model_params
        ):
            raise ValueError("model_params 必须是字符串键组成的 YAML 映射")
        if self.signal_key not in {
            "ecg", "ecg_raw", "ecg_filtered", "rri_256", "r_amp_256",
            "rri_r_amp_256", "ecg_filtered_rri_r_amp_256", "handcrafted_all",
            "ecg_filtered_rri_r_amp_256_handcrafted_all",
        }:
            raise ValueError(f"Unsupported signal_key: {self.signal_key}")
        fusion = self.signal_key == "ecg_filtered_rri_r_amp_256_handcrafted_all"
        uses_aux = self.signal_key in {
            "ecg_filtered_rri_r_amp_256", "ecg_filtered_rri_r_amp_256_handcrafted_all"
        }
        if uses_aux != bool(self.model_params.get("use_rhythm_amplitude", False)):
            raise ValueError("含 RRI/幅度的 signal_key 与 use_rhythm_amplitude 必须同时启用")
        if uses_aux and self.model != "daeac":
            raise ValueError("三模态输入目前只由 daeac 模型支持")
        handcrafted_only = self.signal_key == "handcrafted_all"
        uses_handcrafted = handcrafted_only or fusion
        if handcrafted_only:
            if self.model != "handcrafted_mlp" or self.handcrafted_path is None:
                raise ValueError("handcrafted_all 需要 handcrafted_mlp 和 handcrafted_path")
            if self.model_params.get("input_dim") != self.expected_input_length:
                raise ValueError("input_dim 必须等于 expected_input_length")
        if fusion:
            if self.model != "daeac" or self.handcrafted_path is None:
                raise ValueError("融合输入需要 daeac 模型和 handcrafted_path")
            if not self.model_params.get("use_handcrafted", False):
                raise ValueError("融合输入需要 DAEAC use_handcrafted=true")
            if int(self.model_params.get("handcrafted_dim", 0)) <= 0:
                raise ValueError("融合输入需要正整数 handcrafted_dim")
        elif self.model_params.get("use_handcrafted", False):
            raise ValueError("DAEAC 手工特征分支需要对应的融合 signal_key")
        if handcrafted_only != (self.normalization == "train_feature_zscore"):
            raise ValueError("handcrafted_all 必须使用 train_feature_zscore，其他输入不能使用")
        if self.normalization not in {
            "per_epoch_zscore", "per_channel_minmax", "train_feature_zscore"
        }:
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        if self.class_weight not in {"none", "balanced"}:
            raise ValueError("class_weight must be 'none' or 'balanced'")
        if self.checkpoint_metric not in {"val_macro_f1", "val_accuracy"}:
            raise ValueError("checkpoint_metric 必须是 val_macro_f1 或 val_accuracy")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        positive = {
            "expected_input_length": self.expected_input_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"Values must be positive: {positive}")
        if self.weight_decay < 0 or self.early_stopping_patience < 0 or self.num_workers < 0:
            raise ValueError("weight_decay, patience, and num_workers must be non-negative")
        if require_files:
            if not self.data_path.is_file():
                raise FileNotFoundError(f"Missing dataset: {self.data_path}")
            if not self.split_path.is_file():
                raise FileNotFoundError(f"Missing split: {self.split_path}")
            if uses_handcrafted and not self.handcrafted_path.is_file():
                raise FileNotFoundError(f"Missing handcrafted features: {self.handcrafted_path}")
            if self.handcrafted_selection_dir is not None:
                selection_path = self.handcrafted_selection_dir / f"fold_{self.fold}.json"
                if not selection_path.is_file():
                    raise FileNotFoundError(f"Missing handcrafted selection: {selection_path}")

    def serializable(self) -> dict[str, Any]:
        """将 Path 和标签元组转换为可写入 JSON 的基本类型。"""

        payload = asdict(self)
        for key in ("data_path", "split_dir", "output_root", "repository_root", "config_path"):
            payload[key] = str(payload[key])
        payload["handcrafted_path"] = (
            str(self.handcrafted_path) if self.handcrafted_path is not None else None
        )
        payload["handcrafted_selection_dir"] = (
            str(self.handcrafted_selection_dir)
            if self.handcrafted_selection_dir is not None else None
        )
        payload["labels"] = list(self.labels)
        return payload

    def write_snapshot(self) -> None:
        """在每折目录保存最终生效的配置，便于复核运行参数。"""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(self.serializable(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None, require_files: bool = True
) -> ExperimentConfig:
    """加载 YAML，应用非空命令行覆盖，并严格核对配置字段。

    未知字段和缺失字段直接报错，避免参数拼写错误被静默忽略。
    """

    config_path = Path(path).expanduser().resolve()
    root = find_repository_root(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")
    values = dict(raw)
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    values["repository_root"] = root
    values["config_path"] = config_path
    for key in ("data_path", "split_dir", "output_root"):
        values[key] = _resolve(values[key], root)
    values["handcrafted_path"] = (
        _resolve(values["handcrafted_path"], root)
        if values.get("handcrafted_path") is not None else None
    )
    values["handcrafted_selection_dir"] = (
        _resolve(values["handcrafted_selection_dir"], root)
        if values.get("handcrafted_selection_dir") is not None else None
    )
    values["labels"] = tuple(values["labels"])
    # 旧配置未声明选择指标时沿用原来的 macro F1，保证历史实验含义不变。
    values.setdefault("checkpoint_metric", "val_macro_f1")

    expected = {item.name for item in fields(ExperimentConfig)}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown:
        raise KeyError(f"Unknown configuration key(s): {sorted(unknown)}")
    if missing:
        raise KeyError(f"Missing configuration key(s): {sorted(missing)}")
    config = ExperimentConfig(**values)
    config.validate(require_files=require_files)
    return config


def default_config_path() -> Path:
    """返回项目内置的链路验证配置；正式复现应提供单独的论文配置。"""

    return find_repository_root() / "configs" / "models" / "smoke_cnn.yaml"

