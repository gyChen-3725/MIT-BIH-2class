# BAFNet 双路网络的 MIT-BIH 二分类适配

## 代码与输入

- 原仓库快照：[`upstream/bafnet/SOURCE.md`](../../upstream/bafnet/SOURCE.md)
- 当前 PyTorch 实现：[`src/mitbih_2class/models/bafnet.py`](../../src/mitbih_2class/models/bafnet.py)
- 实验配置：[`configs/models/bafnet_e3_epoch5fold.yaml`](../../configs/models/bafnet_e3_epoch5fold.yaml)

原 notebook 使用 Apnea-ECG 的 RRI、R 峰幅值，两路各 900 点，覆盖目标分钟和
前后各两分钟。当前数据源是 MIT-BIH master 中同一行的 `rri_256` 与 `r_amp_256`：
每路是单个 30 秒 epoch 内的插值特征，没有额外上下文。每个 epoch、每一路分别
做 0～1 归一化；常数序列保持为零。两路合并后形状为 `[batch, 2, 256]`。
master 的 R 峰检测与插值实现位于共享数据脚本
`../dataset/build_mitbih_master_dataset.py`；原 notebook 使用不同的 R 峰检测、
生理范围筛选和三次样条插值步骤，这些步骤没有在当前仓库重新执行。
这属于**结构适配实验**，不能当作原论文在 Apnea-ECG 数据集上的复现结果。

## 保留的网络结构与训练差异

保留两路 11 点卷积、瓶颈自注意力、从瓶颈分别到两路的交叉注意力、
通道融合、二分类头及两路转置卷积重建头。输入从 900 点改为 256 点后，
重建输出插值到 256 点，与同名输入特征计算 MSE。原 notebook 调用 `fit` 时
把 RRI 和幅值重建目标互换；本实现按重建头名称对应同名输入。训练总损失是
带类别权重的交叉熵加两路 MSE，权重在配置中固定为 1.0。
当前适配网络共有 72,868 个可训练参数。

原 notebook 的 K-means 困难样本筛选和 Apnea-ECG 固定训练/测试划分没有用于
这次 MIT-BIH 实验。训练优化器、早停和 checkpoint 选择沿用本仓库的公共训练器，
分别是 AdamW、验证集 macro F1 最佳模型和 20 epoch 早停耐心。

## 划分与结果边界

使用与 DAEAC 样本级 E3 实验相同的五折索引，10,197 个 epoch 各在一个测试折
出现一次；同一受试者可跨训练、验证和测试集合。该索引相同仅便于按相同样本
比较，不能把双路 RRI/幅值与 DAEAC 的原始 ECG 输入视为相同实验条件。

```powershell
& 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe' -B scripts\data\verify_data.py --config configs\models\bafnet_e3_epoch5fold.yaml
& 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe' -u -B scripts\training\train_5fold.py --config configs\models\bafnet_e3_epoch5fold.yaml
& 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe' -B scripts\evaluation\aggregate_oof.py --config configs\models\bafnet_e3_epoch5fold.yaml
& 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe' -B scripts\evaluation\plot_loss.py --config configs\models\bafnet_e3_epoch5fold.yaml
```

## 本次五折结果

2026-10-27 在本机完成五折训练。逐折测试预测合并后恰好覆盖 10,197 个样本，
结果保存在 [`runs/bafnet_rri_amp_30s_e3_epoch5fold_2class/oof/`](../../runs/bafnet_rri_amp_30s_e3_epoch5fold_2class/oof/metrics.json)。

| 折 | 测试 accuracy | 测试 macro F1 | 最佳 epoch |
|---:|---:|---:|---:|
| 0 | 72.55% | 68.86% | 14 |
| 1 | 72.65% | 69.00% | 22 |
| 2 | 72.19% | 69.22% | 11 |
| 3 | 75.58% | 71.79% | 20 |
| 4 | 70.67% | 66.99% | 9 |

OOF accuracy 为 **72.73%**，macro F1 为 **69.17%**，敏感度为 **59.16%**，
特异度为 **79.33%**，ROC AUC 为 **0.7585**。混淆矩阵按
`[Normal, Apnea]` 的真实行与预测列排列，为 `[[5440, 1417], [1364, 1976]]`。
五折 [loss 曲线](../../runs/bafnet_rri_amp_30s_e3_epoch5fold_2class/oof/loss_curves.png)
显示最佳 checkpoint 后训练目标继续下降、验证目标上升；这是该适配配置的
过拟合迹象，不能据此推断原 BAFNet 在其原始数据与五分钟输入上的效果。
