"""用于检查数据、训练与评估链路的小型 CNN；不对应目标论文。"""

from __future__ import annotations

import torch
from torch import nn

from mitbih_2class.config import ExperimentConfig


class SmokeCNN(nn.Module):
    """输入形状为 (batch, 1, 时间点数)，输出形状为 (batch, 2) 的 logits。

    自适应平均池化将不同时间长度的卷积特征压缩为固定长度；网络结构和
    超参数仅用于验证程序能否运行，不能当作论文模型或正式实验结果。
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # 两层卷积提取局部波形特征；步幅和池化用于缩短链路测试耗时。
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """将每条信号映射为两个类别的未归一化分数。"""

        return self.classifier(self.features(features).flatten(1))


def build_model(config: ExperimentConfig) -> nn.Module:
    """使用全局标签顺序确定输出维度。"""

    return SmokeCNN(num_classes=len(config.labels))
