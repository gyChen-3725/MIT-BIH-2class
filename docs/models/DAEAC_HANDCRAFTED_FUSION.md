# DAEAC 与手工特征融合实验

本实验使用三类信息完成同一个二分类任务：

- 滤波 ECG `[B, 1, 7500]` 进入 DAEAC ASPP-SE 主干，得到 1024 维表示；
- RRI 与 R峰幅度 `[B, 2, 256]` 进入现有辅助卷积分支，得到 128 维表示；
- 全部手工特征 `[B, 263]` 经过 `Linear(263,128) + ReLU + Dropout`，得到
  128 维表示。

三路向量在分类头前拼接为 `[B, 1280]`，最后由 `Linear(1280,2)` 输出。
手工特征不会被转换成伪时间序列，也不会经过 DAEAC 的 ASPP 或卷积。

每折仅使用训练索引拟合手工特征的中位数填补、均值和标准差。ECG 仍按
单个 epoch 做 z-score；RRI 与幅度沿用现有预处理。配置的其他训练参数与
`daeac_ecg_rhythm_amplitude_e3_epoch5fold.yaml` 一致，以便把指标变化主要
归因于新增手工特征分支。

在项目根目录运行一折：

```powershell
python scripts/training/train.py --config configs/models/daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml --fold 0 --device cuda
```

该命令由用户自行执行；完成一折后先与原 DAEAC 的 fold 0 做同折比较。

## 冗余筛选版本

`select_handcrafted_features.py` 对每折分别处理，并且只读取该折训练索引：

1. 删除缺失率超过20%或近常数的特征；
2. 按 `2 × |训练集 ROC AUC - 0.5|` 从高到低排序；
3. 若候选项与已保留项的绝对 Spearman 相关达到0.90，则删除候选项；
4. 每折固定保留64项，验证集和测试集标签不参与筛选。

先由用户生成五折筛选清单：

```powershell
python scripts/features/select_handcrafted_features.py
```

再运行筛选后的 fold 0 融合实验：

```powershell
python scripts/training/train.py --config configs/models/daeac_ecg_rhythm_amplitude_handcrafted_selected64_e3_epoch5fold.yaml --fold 0 --device cuda
```

每折可能选择不同的特征名称，但输入维数均固定为64。筛选清单记录原始索引、
名称、相关性排除关系、数据摘要和划分摘要，训练加载时会再次核对名称顺序。

在启动DAEAC前，可先用与既有表格基线相同的HistGradientBoosting完成五折
筛选前后对比：

```powershell
python scripts/features/compare_handcrafted_selection.py
```

结果写入 `runs/handcrafted_selection_comparison_e3_k64/`，包括逐折指标、完整
OOF指标、五折选择集合的Jaccard稳定性，以及可直接阅读的 `comparison.md`。

## v2优化筛选

v2不再以混合全部训练窗口的AUC排序，而是逐受试者计算AUC后汇总；相关性也
逐受试者计算Spearman后取绝对值中位数。它完整排序全部263维，并把相关阈值
提高到0.95。先生成五折训练集排序：

```powershell
python scripts/features/rank_handcrafted_features_v2.py
```

再比较64、96、128、160、192和263维：

```powershell
python scripts/features/compare_handcrafted_rankings_v2.py
```

结果保存到 `runs/handcrafted_selection_v2_dimension_sweep/`。维度对比读取外层
测试折，因此只用于探索容量；选定方案后的最终结论仍需新的独立划分确认。
