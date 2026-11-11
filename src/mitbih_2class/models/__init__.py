"""按配置名加载独立的论文模型模块。

每个模型位于 ``models/<模型名>.py`` 或 ``models/<模型名>/__init__.py``，
并提供 ``build_model(config)`` 工厂函数。训练循环只依赖返回的 PyTorch 模型，
因此增加论文实现时无需修改训练代码或此处的模型名单。
"""

from __future__ import annotations

from importlib import import_module

from torch import nn

from mitbih_2class.config import ExperimentConfig
from mitbih_2class.models.smoke_cnn import SmokeCNN


def build_model(config: ExperimentConfig) -> nn.Module:
    """导入配置指定的模型模块并构造网络。

    ``config.model`` 已限制为单个 Python 模块名，例如 ``paper_2026``；
    因此不会把任意文件路径当作可执行代码加载。模型模块内部依赖缺失时，
    保留原始导入错误，以便定位缺少的依赖库。
    """

    module_name = f"{__name__}.{config.model}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ValueError(
                f"模型 {config.model!r} 不存在；请在 src/mitbih_2class/models/ 下添加同名模块"
            ) from exc
        raise

    factory = getattr(module, "build_model", None)
    if not callable(factory):
        raise TypeError(f"模型模块 {module_name} 必须提供 build_model(config) 函数")
    model = factory(config)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{module_name}.build_model() 必须返回 torch.nn.Module")
    return model


__all__ = ["SmokeCNN", "build_model"]
