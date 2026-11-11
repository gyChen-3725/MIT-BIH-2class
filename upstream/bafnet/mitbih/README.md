# BAFNet 两阶段流程在 MIT-BIH PSG 上的迁移实验

这里沿用相邻 [`../reproduction/`](../reproduction/README.md) 的 Keras BAFNet
网络、三头损失、困难样本选择、100＋5 epoch 双阶段训练与验证集 checkpoint。
这是更换数据集的迁移实验，不是原论文在 Apnea-ECG 上的指标复现。

## 数据和输入

`preprocess.py` 从共享 `dataset/mitbih_master/mitbih_master_30s.npz` 读取
250 Hz 原始 ECG。每条记录内以目标 30 秒 epoch 的中点为中心，取前后各
150 秒形成五分钟窗口；窗口不能跨记录边界。目标 epoch 的原始标签映射为
`Normal=0`，`H/HA/OA/X/CA/CAA=1`。边界不足五分钟或不满足原作者 R 峰、
心率、插值条件的窗口丢弃。

每个窗口沿用原作者的 3～45 Hz FIR、Hamilton R 峰检测与校正、RRI 中值
滤波、逐路 0～1 缩放、三次样条插值，输出 900×2 的 RRI 和 R 波幅值特征。
采样率使用 MIT-BIH 的 250 Hz；Apnea-ECG 原数据为 100 Hz。

## 划分与训练

共享的 `split_dev_subjectwise.json` 冻结了独立测试被试 P14、P16、P61。
其他被试组成训练池；从中按二分类标签分层随机划出 70% 训练、30% 验证，
随机种子 42。测试被试与训练池严格隔离。训练和验证的五分钟窗口可能共享
同一条记录中的 ECG，因为该步骤遵循作者的样本级分层验证方式；验证指标
不能当作独立被试泛化结果。

本次预处理产生 9,966 个有效窗口：训练池 7,868 个，独立测试 2,098 个。
种子 42 的分层划分将训练池分为 5,507 个训练窗口和 2,361 个验证窗口。
测试集只含 P14、P16、P61；训练、验证、测试的样本索引互斥。
由于窗口每 30 秒移动一次，相邻五分钟窗口共享 90% 原始波形。
本次 2,361 个验证窗口中，全部都与某个训练窗口共享原始 ECG；
其中 2,154 个验证窗口的中心与训练窗口相距不超过 30 秒。
因此验证准确率不能解释为独立被试准确率。

训练复用 `../reproduction/train.py`：第一阶段全部训练样本 100 epoch；
第二阶段按 RRI 做五簇 K-means，重新选取簇内样本，再连续训练 5 epoch。
初始学习率 0.001，Adam，batch size 128，He 初始化与 L2 正则。
测试集只在各阶段训练结束后评估。MIT-BIH 缺少 Apnea-ECG 作者所用的
每小时事件率参考值，因此不应用其逐记录 >5 次/小时评价规则。

## 命令

```powershell
cd E:\envs\Projects\Sleep-apnea-detection\MIT-BIH-2class
$Python = 'C:\Users\cgy_3\miniconda3\envs\tf218-cpu\python.exe'
& $Python -u -B upstream\bafnet\mitbih\preprocess.py --workers 4
& $Python -u -B upstream\bafnet\reproduction\train.py --data upstream\bafnet\mitbih\artifacts\mitbih_5min.npz --output upstream\bafnet\mitbih\artifacts\paper_100plus5_seed42 --dataset-name MIT-BIH-PSG --record-eval none --stage1-epochs 100 --stage2-epochs 5
& $Python -B upstream\bafnet\reproduction\evaluate_best.py --data upstream\bafnet\mitbih\artifacts\mitbih_5min.npz --run upstream\bafnet\mitbih\artifacts\paper_100plus5_seed42
& $Python -B upstream\bafnet\reproduction\plot_history.py --run upstream\bafnet\mitbih\artifacts\paper_100plus5_seed42
```

输出存放在本目录的 `artifacts/`。它与先前 30 秒、256 点、PyTorch
五折结构适配实验相互独立，不能直接将两者的准确率归因于训练逻辑差别。

## 本次 100＋5 结果

训练历史包含第一阶段 100 epoch、第二阶段 5 epoch。独立测试集为 P14、P16、
P61 的 2,098 个五分钟窗口，以下指标均按窗口计算。混淆矩阵的行是真实
`[Normal, Apnea]`，列是预测 `[Normal, Apnea]`。

| 权重 | 准确率 | Apnea F1 | Macro F1 | 敏感度 | 特异度 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| 第一阶段最终权重 | 72.12% | 46.87% | 63.98% | 35.49% | 91.54% | 0.7711 |
| 第二阶段最终权重 | **73.31%** | **54.17%** | **67.67%** | **45.53%** | 88.04% | **0.7720** |
| 验证 accuracy 最佳权重 | 72.12% | 46.87% | 63.98% | 35.49% | 91.54% | 0.7711 |

第二阶段最终权重的混淆矩阵为 `[[1207, 164], [396, 331]]`。
最佳验证 checkpoint 来自第一阶段第 99 epoch，验证准确率 85.51%；
其独立测试准确率只有 72.12%。这说明样本级验证分数明显高于独立被试测试分数，
报告泛化性能时应以独立测试集为准。第二阶段相对第一阶段提高了 Apnea 敏感度，
同时降低了特异度。
按测试被试拆分，第二阶段最终权重在 P14、P16、P61 上的准确率分别为
81.25%、71.93%、66.76%；整体准确率为按窗口数加权的 73.31%。

[最终权重与两阶段指标](artifacts/paper_100plus5_seed42/metrics.json)、
[验证最佳权重指标](artifacts/paper_100plus5_seed42/best_metrics.json)、
[训练曲线](artifacts/paper_100plus5_seed42/training_curves.png)。
