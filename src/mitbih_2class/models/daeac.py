"""DAEAC 的 MACNN-SE 分类主干，适配单导联 30 秒 ECG 二分类。

依据 ``upstream/daeac/source/src/model/ACNN.py`` 中的 ``MACNN_SE`` 和
三分类参考仓库 ``src/trignet_repro/models/daeac.py``。原网络处理心拍级
``[batch, lead, 3, length]``，三行分别包含 ECG 和两个 RR 特征。当前数据集
输入为 ``[batch, 1, 7500]`` 的 ECG 波形，因此首层二维卷积改成时间轴上的
一维卷积；其后高度已为 1 的卷积、SE、残差与池化按一维形式实现。可选的
RRI/幅度分支直接读取 master 中每个 epoch 已保存的两路 256 点特征，并在
分类头前融合。该融合是睡眠呼吸暂停任务的适配，不等同于论文的心拍输入。
可选的手工特征分支用一层 MLP 将已标准化的统计特征映射为紧凑向量，随后
与 ECG、RRI/幅度嵌入在分类头前拼接；它不把手工特征作为伪序列重复卷积。

论文结构有两个 ASPP-SE-残差块，之后还有一个独立的末端 ASPP-SE。
``use_third_aspp`` 在两个重复块后再插入一个 ASPP-SE-残差块；开启后
共有四个 ASPP。该额外块属于本项目的容量实验，默认关闭以兼容旧模型。

这里只接入监督分类主干。上游的域适配训练步骤和相关损失不属于这个模型模块。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from mitbih_2class.config import ExperimentConfig


def _activation(name: str, channels: int) -> nn.Module:
    """使用与参考适配实现相同的激活函数选项。"""

    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "prelu":
        return nn.PReLU(init=0.05)
    if name == "cprelu":
        return nn.PReLU(num_parameters=channels, init=0.05)
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"不支持的 DAEAC 激活函数: {name}")


class ConvBlock1D(nn.Module):
    """上游 ASPP 中卷积、批归一化与激活的单时间轴版本。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        use_bn: bool,
        use_activation: bool,
        activation: str,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = _activation(activation, out_channels)
        self.use_bn = use_bn
        self.use_activation = use_activation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.conv(features)
        if self.use_bn:
            features = self.bn(features)
        if self.use_activation:
            features = self.activation(features)
        return features


class ASPP1D(nn.Module):
    """并联多个膨胀卷积分支，沿通道维拼接多尺度特征。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Sequence[int],
        *,
        use_bn: bool,
        use_activation: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvBlock1D(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    dilation=int(dilation),
                    use_bn=use_bn,
                    use_activation=use_activation,
                    activation=activation,
                )
                for dilation in dilations
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(features) for branch in self.branches], dim=1)


class SqueezeExcitation1D(nn.Module):
    """保持上游 SELayer 的通道压缩比例与无偏置线性层。"""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = channels // reduction
        if hidden < 1:
            raise ValueError("SE 压缩后通道数不能为零")
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.pool(features).squeeze(-1)
        weights = self.fc(weights).unsqueeze(-1)
        return features * weights


class ResidualBlock1D(nn.Module):
    """预激活残差块；stride=2 时用平均池化同步下采样捷径。"""

    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        first_padding = (kernel_size - stride + 1) // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, stride=stride, padding=first_padding
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.pool = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features if self.stride == 1 else self.pool(features)
        output = self.conv1(self.relu1(self.bn1(features)))
        output = self.conv2(self.relu2(self.bn2(output)))
        return output + residual


def _initialize_like_source(module: nn.Module) -> None:
    """复用参考适配版的卷积、归一化和线性层初始化规则。"""

    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight, 1.0, 0.01)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DAEACMACNNSE(nn.Module):
    """单导联 MACNN-SE 主干和可配置类别数的线性分类头。

    默认宽度 16 与三分类参考仓库的参数量匹配设置一致。类别数由当前任务的
    ``labels`` 决定；二分类输出列顺序固定为 ``[Normal, Apnea]``。
    默认使用三个 ASPP；开启 ``use_third_aspp`` 后，在末端 ASPP 前多一组
    ASPP、SE 和残差层，共四个 ASPP。原有三路输入实验保持默认结构。
    """

    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 16,
        dilations: Sequence[int] = (1, 6, 12, 18),
        activation: str = "relu",
        final_activation: str = "relu",
        use_rhythm_amplitude: bool = False,
        use_third_aspp: bool = False,
        use_handcrafted: bool = False,
        handcrafted_dim: int = 263,
        handcrafted_hidden_dim: int = 128,
        handcrafted_dropout: float = 0.2,
        use_handcrafted_gate: bool = False,
        handcrafted_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 至少为 2")
        if base_channels < 1:
            raise ValueError("base_channels 必须为正整数")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ValueError("dilations 必须包含正整数")
        if use_handcrafted and min(handcrafted_dim, handcrafted_hidden_dim) < 1:
            raise ValueError("手工特征维度和隐藏层维度必须为正整数")
        if use_handcrafted_gate and not use_handcrafted:
            raise ValueError("手工特征门控只能在 use_handcrafted=true 时启用")
        if not 0.0 <= handcrafted_dropout < 1.0:
            raise ValueError("handcrafted_dropout 必须位于 [0, 1) 区间")
        if use_handcrafted_gate and not 0.0 < handcrafted_gate_init < 1.0:
            raise ValueError("handcrafted_gate_init 必须位于 (0, 1) 区间")
        scale_count = len(dilations)
        stage1_channels = scale_count * base_channels
        stage2_channels = scale_count * stage1_channels
        final_channels = scale_count * stage2_channels
        self.base_channels = base_channels
        self.use_rhythm_amplitude = use_rhythm_amplitude
        self.use_third_aspp = use_third_aspp
        self.use_handcrafted = use_handcrafted
        self.handcrafted_dim = handcrafted_dim
        self.use_handcrafted_gate = use_handcrafted_gate
        # 两路辅助序列保持原有 256 点长度；不把它们插值成 7500 点伪波形。
        self.rhythm_amplitude_branch = (
            nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(4),
                nn.Flatten(),
            ) if use_rhythm_amplitude else None
        )
        # 手工特征已经具有明确统计含义，只做一次非线性压缩与正则化。
        self.handcrafted_branch = (
            nn.Sequential(
                nn.Linear(handcrafted_dim, handcrafted_hidden_dim),
                nn.ReLU(),
                nn.Dropout(handcrafted_dropout),
            ) if use_handcrafted else None
        )
        # 门控只缩放手工特征嵌入，不改变其维度。用较小初值可使模型从稳定的
        # ECG 主干附近开始优化，再由反向传播决定是否逐步放大手工特征贡献。
        gate_logit = torch.logit(torch.tensor(float(handcrafted_gate_init)))
        self.handcrafted_gate_logit = (
            nn.Parameter(gate_logit) if use_handcrafted and use_handcrafted_gate else None
        )
        self.feature_dim = (
            final_channels
            + (128 if use_rhythm_amplitude else 0)
            + (handcrafted_hidden_dim if use_handcrafted else 0)
        )

        # 源首层跨 ECG/RR 三行卷积；当前单路 ECG 使用相同宽度的时间卷积。
        self.stem = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        # 第一组：四个不同膨胀率的卷积分支拼接后，经 SE 筛选通道。
        # 残差块 stride=1，因此时间长度仍为 7500。
        self.aspp1 = ASPP1D(
            base_channels,
            base_channels,
            dilations,
            use_bn=True,
            use_activation=True,
            activation=activation,
        )
        self.se1 = SqueezeExcitation1D(stage1_channels, reduction=4)
        self.residual1 = ResidualBlock1D(stage1_channels, kernel_size=3, stride=1)

        # 第二组输出 256 通道；残差块 stride=2，把时间长度减为 3750。
        self.aspp2 = ASPP1D(
            stage1_channels,
            stage1_channels,
            dilations,
            use_bn=True,
            use_activation=True,
            activation=activation,
        )
        self.se2 = SqueezeExcitation1D(stage2_channels, reduction=8)
        self.residual2 = ResidualBlock1D(stage2_channels, kernel_size=3, stride=2)

        if use_third_aspp:
            # 新增第三组保留 256 通道：每个 ASPP 分支输出 64 通道，
            # 四路拼接后仍为 256。这样增加一层多尺度提取，同时控制显存；
            # 若继续按四倍扩张通道，末端 ASPP 会大幅增加参数和激活内存。
            # 新残差块再次下采样，7500 点 ECG 在此处约剩 1875 点。
            third_branch_channels = stage2_channels // scale_count
            self.aspp3 = ASPP1D(
                stage2_channels,
                third_branch_channels,
                dilations,
                use_bn=True,
                use_activation=True,
                activation=activation,
            )
            self.se3 = SqueezeExcitation1D(stage2_channels, reduction=8)
            self.residual3 = ResidualBlock1D(stage2_channels, kernel_size=3, stride=2)
        else:
            # 不启用时不创建参数，旧配置与旧 checkpoint 的结构保持一致。
            self.aspp3 = None
            self.se3 = None
            self.residual3 = None

        # 最后一个 ASPP 不属于上面的重复块；它把 256 通道扩为 1024，
        # 随后做 SE 与全局平均池化，得到供分类头使用的 ECG 向量。
        self.final_bn = nn.BatchNorm1d(stage2_channels)
        self.final_relu = nn.ReLU()

        self.final_aspp = ASPP1D(
            stage2_channels,
            stage2_channels,
            dilations,
            use_bn=True,
            use_activation=True,
            activation=final_activation,
        )
        self.final_se = SqueezeExcitation1D(final_channels, reduction=16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.apply(_initialize_like_source)

    def extract_features(
        self,
        features: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        handcrafted_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """提取 ECG 与可选节律/幅度向量，返回分类头之前的融合特征。

        默认 ECG 路径是两组 ASPP-SE-残差块加末端 ASPP-SE；启用第三组时，
        在第二组与末端 ASPP 之间多提取一次多尺度信息。末端全局池化得到
        1024 维 ECG 向量；RRI/幅度分支和手工特征分支各自产生 128 维向量，
        手工分支的输出宽度由 handcrafted_hidden_dim 决定。
        """

        if features.ndim != 3 or features.shape[1] != 1:
            raise ValueError(
                "DAEACMACNNSE 需要 [batch, 1, length] 输入，"
                f"实际为 {tuple(features.shape)}"
            )
        features = self.stem(features)
        features = self.residual1(self.se1(self.aspp1(features)))
        features = self.residual2(self.se2(self.aspp2(features)))
        if self.use_third_aspp:
            # 只在新实验启用；第三组的 SE 及残差层同样参加端到端反向传播。
            features = self.residual3(self.se3(self.aspp3(features)))
        features = self.final_relu(self.final_bn(features))
        features = self.final_se(self.final_aspp(features))
        ecg_features = self.pool(features).squeeze(-1)
        embeddings = [ecg_features]
        if self.rhythm_amplitude_branch is None:
            if aux_features is not None:
                raise ValueError("当前模型未启用 RRI/幅度分支")
        else:
            if (
                aux_features is None or aux_features.ndim != 3
                or aux_features.shape != (len(features), 2, 256)
            ):
                raise ValueError("启用 RRI/幅度分支时需要 [batch, 2, 256] 的 aux_features")
            embeddings.append(self.rhythm_amplitude_branch(aux_features))
        if self.handcrafted_branch is None:
            if handcrafted_features is not None:
                raise ValueError("当前模型未启用手工特征分支")
        else:
            if (
                handcrafted_features is None or handcrafted_features.ndim != 2
                or handcrafted_features.shape != (len(features), self.handcrafted_dim)
            ):
                raise ValueError(
                    "启用手工特征分支时需要 "
                    f"[batch, {self.handcrafted_dim}] 的 handcrafted_features"
                )
            handcrafted_embedding = self.handcrafted_branch(handcrafted_features)
            if self.handcrafted_gate_logit is not None:
                handcrafted_embedding = (
                    handcrafted_embedding * self.handcrafted_gate_logit.sigmoid()
                )
            embeddings.append(handcrafted_embedding)
        return torch.cat(embeddings, dim=1) if len(embeddings) > 1 else ecg_features

    def forward(
        self,
        features: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        handcrafted_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """输出两个未经 softmax 的 logits，供交叉熵损失计算。"""

        return self.classifier(
            self.extract_features(features, aux_features, handcrafted_features)
        )


def build_model(config: ExperimentConfig) -> nn.Module:
    """从实验配置构建 DAEAC 主干，类别数始终由标签顺序决定。"""

    return DAEACMACNNSE(num_classes=len(config.labels), **config.model_params)
