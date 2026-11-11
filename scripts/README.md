# 脚本分区

脚本按职责拆分，所有命令仍从仓库根目录执行。

| 目录 | 职责 | 主要入口 |
|---|---|---|
| `data/` | 数据核验与冻结划分生成 | `verify_data.py`、`prepare_e3_epoch_splits.py` |
| `features/` | 347维扩展特征提取、字典生成、单变量筛查及旧特征排序 | `extract_expanded_features.py`、`build_expanded_feature_dictionary.py`、`screen_handcrafted_features.py` |
| `selection/` | 特征族组合、选择器、维度与机器学习模型搜索，以及DAEAC折内64维掩码冻结 | `optimize_feature_dimension.py`、`freeze_daeac_selected64.py` |
| `clustering/` | DAEAC 辅助序列、263维全局聚类与层次化二次聚类 | `cluster_daeac_features.py`、`cluster_handcrafted_features.py`、`cluster_handcrafted_hierarchical.py` |
| `training/` | 深度模型训练及冻结五折表格基线 | `train.py`、`train_5fold.py`、`train_handcrafted_2class.py` |
| `evaluation/` | OOF 汇总、训练曲线和特征选择HTML仪表板 | `aggregate_oof.py`、`plot_loss.py`、`visualize_feature_selection.py` |

旧263维EDR特征仍由相邻 `EDR` 仓库生成；当前实验新增的84维HRV、原始ECG和
QRS形态特征由本仓库提取，并与旧263维按行对齐组成347维冻结输入。产物与依赖边界见
[`features/README.md`](../features/README.md)。

示例：

```powershell
python -B scripts/data/verify_data.py --config configs/models/daeac_e3_epoch5fold.yaml
python -B scripts/training/train.py --config configs/models/daeac_e3_epoch5fold.yaml --fold 0 --debug
python -B scripts/clustering/cluster_handcrafted_features.py --fold 0
python -B scripts/evaluation/plot_loss.py --config configs/models/daeac_e3_epoch5fold.yaml
```

