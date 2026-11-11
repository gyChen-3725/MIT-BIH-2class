# DAEAC与机器学习最优MI-128特征的Fold 0融合对照

## 实验问题

已有机器学习网格在Fold 0选择出的最佳配方为：

```text
fusion3_expanded + mutual_info + 128维 + random_forest
```

此前DAEAC手工特征实验人为固定为64维，因此使用的是64维约束下最优的
`mRMR + 64维 + random_forest`，并非上述全局最优配方。本组实验拆成两个互相
独立的融合方向，和原始DAEAC、mRMR-64直接拼接进行同折比较。

## 四个比较对象

| 名称 | 深度模型输入 | 手工特征使用方式 |
|---|---|---|
| 原始DAEAC | ECG + RRI + R峰幅度 | 不使用 |
| mRMR-64直接拼接 | 原始三路输入 | 64维经MLP后直接拼接 |
| MI-128门控融合 | 原始三路输入 | 128维经MLP后乘可学习标量门控，再拼接 |
| MI-128-RF概率融合 | 原始DAEAC | 独立RF概率与DAEAC概率加权 |

门控参数使用Sigmoid约束到0至1，初值为0.05。小初值让模型从接近原始DAEAC的
状态开始，避免随机初始化的手工分支在训练初期明显扰动ECG表示。模型仍可通过
反向传播逐步放大该分支。

概率融合不使用门控DAEAC，而使用原始DAEAC checkpoint，从而与门控融合构成
两条独立实验臂。RF在训练集拟合并产生验证概率；融合权重仅由验证accuracy选择。
确定权重后，RF在训练＋验证集重新拟合，测试集只评价一次，测试标签不参与调权。

## MI-128冻结原则

配置配方来自已有验证网格，但DAEAC使用的128个具体列只在Fold 0训练行上重新计算
互信息排名。这避免直接把验证标签写入DAEAC输入列。冻结文件位于：

```text
protocols/handcrafted_selection_expanded347_ml_best_fold0/fold_0.json
```

机器学习最终测试模型仍遵循原流程：在训练＋验证行上重新计算互信息排名，然后拟合
RF。因此门控融合与RF概率融合复用相同的“特征组＋选择器＋维度”配方，但各自的
最终列排名严格匹配其训练边界。

## 执行顺序

项目约束要求训练由用户显式启动。先训练MI-128门控模型：

```powershell
python -u -B scripts\train.py `
  --config configs\models\daeac_ecg_rhythm_amplitude_handcrafted_mi128_gated_fold0.yaml `
  --fold 0 `
  --device cuda
```

原始DAEAC Fold 0已有checkpoint后，可以独立执行概率融合。该步骤不会重新训练
DAEAC，但会按冻结配方拟合RF：

```powershell
python -u -B scripts\evaluation\fuse_daeac_rf_probabilities.py `
  --fold 0 `
  --device cuda
```

两项结果都生成后，汇总四种方法：

```powershell
python -B scripts\evaluation\compare_fold0_fusions.py
```

比较表保存为：

```text
runs/fold0_daeac_fusion_comparison.csv
```

本实验只比较同一个Fold 0，不能替代五折总体结论。只有门控或概率融合在Fold 0显示
稳定收益后，才值得扩展到其余四折。
