"""BAFNet 双路 RRI/R 峰幅值网络的 MIT-BIH 30 秒二分类适配。

结构参照 ``upstream/bafnet/source/BAFNet.ipynb``：两路卷积编码、瓶颈自注意力、
分别对两路特征做交叉注意力、通道融合、分类头和两路重建头。原模型的输入是
Apnea-ECG 数据集带前后文的 900 点序列；这里使用 master 已有的 256 点、
单个 30 秒 epoch 特征。重建输出按输入长度对齐，以适配新的时间轴长度。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mitbih_2class.config import ExperimentConfig


class ScaledDotProductAttention(nn.Module):
    """按上游实现用通道维平方根缩放 query/key 相似度。"""

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        # PyTorch 卷积输出为 [B, C, T]；注意力按时间点计算，因此临时转为 [B, T, C]。
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        return torch.matmul(torch.softmax(scores, dim=-1), value).transpose(1, 2)


class ReconstructionHead(nn.Module):
    """沿用上游的 5 倍、3 倍上采样及三层转置卷积重建一路输入。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(32, 24, 11, padding=5)
        self.conv2 = nn.ConvTranspose1d(24, 16, 11, stride=2, padding=5, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(16, 1, 11, padding=5)

    def forward(self, features: torch.Tensor, output_length: int) -> torch.Tensor:
        features = F.interpolate(features, scale_factor=5, mode="nearest")
        features = self.conv1(features)
        features = F.interpolate(features, scale_factor=3, mode="nearest")
        features = self.conv2(features)
        features = self.conv3(features)
        # 900 点原输入恰好可由池化比例还原；256 点会因向上取整产生 270 点。
        if features.shape[-1] != output_length:
            features = F.interpolate(features, size=output_length, mode="linear")
        return features


class BAFNet(nn.Module):
    """RRI 与 R 峰幅值各占一路，分类列固定为 [Normal, Apnea]。

    ``forward`` 仅给出分类 logits，满足共用推理接口；``forward_with_aux``
    额外返回重建损失，供训练、验证和测试计算总损失。
    """

    def __init__(self, num_classes: int = 2, reconstruction_weight: float = 1.0) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError("BAFNet 当前适配固定输出 Normal/Apnea 两类")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight 不能为负数")
        self.reconstruction_weight = reconstruction_weight

        # 两路 FCN 的通道数、11 点卷积核、步长和池化比例与上游保持一致。
        self.rri_stem = self._stem()
        self.amp_stem = self._stem()
        self.rri_stage = nn.Sequential(
            nn.Conv1d(24, 32, 11, padding=5), nn.ReLU(), nn.MaxPool1d(5, 5, 2)
        )
        self.amp_stage = nn.Sequential(
            nn.Conv1d(24, 32, 11, padding=5), nn.ReLU(), nn.MaxPool1d(5, 5, 2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(48, 32, 11, padding=5), nn.ReLU(), nn.MaxPool1d(5, 5, 2)
        )
        self.attention = ScaledDotProductAttention()
        self.fusion = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
        self.classifier = nn.Linear(64, num_classes)
        self.rri_decoder = ReconstructionHead()
        self.amp_decoder = ReconstructionHead()
        self.apply(self._initialize)
        # 新输入长度需要对重建结果插值；先让末层输出接近零，避免初始 MSE
        # 远大于分类交叉熵而淹没分类梯度。末层仍可从第一步开始学习。
        nn.init.normal_(self.rri_decoder.conv3.weight, std=1e-3)
        nn.init.normal_(self.amp_decoder.conv3.weight, std=1e-3)

    @staticmethod
    def _stem() -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(1, 16, 11, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, 24, 11, stride=2, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(3, 3, 1),
        )

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError(f"BAFNet 期望 [batch, 2, time]，收到 {tuple(features.shape)}")
        rri = self.rri_stem(features[:, 0:1])
        amp = self.amp_stem(features[:, 1:2])
        query = self.bottleneck(torch.cat((rri, amp), dim=1))
        query = self.attention(query, query, query)
        rri = self.rri_stage(rri)
        amp = self.amp_stage(amp)
        rri = self.attention(query, rri, rri)
        amp = self.attention(query, amp, amp)
        joined = torch.cat((rri, amp), dim=1)
        weights = self.fusion(joined.mean(dim=-1)).unsqueeze(-1)
        logits = self.classifier((joined * weights).mean(dim=-1))
        return logits, rri, amp

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self._encode(features)
        return logits

    def forward_with_aux(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回分类 logits 与两路同名特征的 MSE 重建损失。

        上游 notebook 在 fit 参数中将两路重建目标互换；这里按重建头名称对应
        原输入，避免让 RRI 头拟合幅值、幅值头拟合 RRI。
        """

        logits, rri, amp = self._encode(features)
        rri_reconstruction = self.rri_decoder(rri, features.shape[-1])
        amp_reconstruction = self.amp_decoder(amp, features.shape[-1])
        auxiliary_loss = self.reconstruction_weight * (
            F.mse_loss(rri_reconstruction, features[:, 0:1])
            + F.mse_loss(amp_reconstruction, features[:, 1:2])
        )
        return logits, auxiliary_loss


def build_model(config: ExperimentConfig) -> nn.Module:
    """从实验配置构造 BAFNet，并保留独立的重建权重超参数。"""

    return BAFNet(num_classes=len(config.labels), **config.model_params)
