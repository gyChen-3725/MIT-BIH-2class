"""使用全部手工特征进行二分类的轻量 MLP。

输入是已经提取完成并按训练折标准化的 263 维统计特征。模型只包含一层
非线性隐藏层和分类头，不把特征转换为伪时间序列，也不使用卷积、ASPP、
SE 或其他时序特征提取模块。
"""

from __future__ import annotations

import torch
from torch import nn

from mitbih_2class.config import ExperimentConfig


class HandcraftedMLP(nn.Module):
    """263 维特征到二分类 logits 的简单全连接分类器。"""

    def __init__(
        self,
        input_dim: int = 263,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, num_classes) < 1:
            raise ValueError("input_dim、hidden_dim 和 num_classes 必须为正整数")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间")
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """接受 ``[batch, 1, 263]``，返回 ``[batch, 2]`` logits。"""

        if features.ndim != 3 or features.shape[1:] != (1, self.input_dim):
            raise ValueError(
                f"HandcraftedMLP 需要 [batch, 1, {self.input_dim}] 输入，"
                f"实际为 {tuple(features.shape)}"
            )
        return self.classifier(features.squeeze(1))


def build_model(config: ExperimentConfig) -> nn.Module:
    """从配置构建手工特征分类器，类别顺序由实验标签固定。"""

    return HandcraftedMLP(num_classes=len(config.labels), **config.model_params)
