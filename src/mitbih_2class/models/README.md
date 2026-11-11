# 模型加载目录

每篇论文的模型实现放在这里，文件名或子目录名就是实验配置中的 `model`。例如：

```text
models/
├── __init__.py            # 按配置名加载模型，不维护论文名单
├── smoke_cnn.py           # 链路验证模型
├── daeac.py               # DAEAC MACNN-SE 的二分类适配
├── bafnet.py              # BAFNet 双路 RRI/幅值网络的二分类适配
├── paper_a.py             # 简单论文网络
└── paper_b/               # 需要拆分层和模块的论文网络
    ├── __init__.py        # 提供 build_model(config)
    └── layers.py
```

每个模型模块必须提供 `build_model(config)`，返回 `torch.nn.Module`。模型输入默认是
`(batch, 1, expected_input_length)`；BAFNet 使用两路输入 `(batch, 2, 256)`。
输出必须是 `(batch, 2)` 的 logits，列顺序固定为
`[Normal, Apnea]`。例如：

```python
from torch import nn


def build_model(config) -> nn.Module:
    return PaperNet(num_classes=len(config.labels), **config.model_params)
```

然后新增 `configs/models/paper_a.yaml`，其中 `model: paper_a`。论文网络的通道数、层数等
构造参数写入 `model_params`；数据、优化器等公共参数仍用现有 YAML 字段。

这里存放网络源码；训练产生的最佳权重自动保存在
`runs/<实验名>/fold_<折号>/best.pt`。不同论文应使用不同实验名，避免覆盖彼此产物。
