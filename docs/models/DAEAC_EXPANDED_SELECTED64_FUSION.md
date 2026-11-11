# DAEAC与折内64维扩展手工特征融合

## 实验目的

在现有DAEAC三路输入基础上增加一条轻量手工特征分支，检验扩展HRV、原始ECG统计、
QRS形态和EDR特征能否补充深度ECG表示。模型仍以30秒滤波ECG为主输入，不使用真实
呼吸通道。

## 四路输入与融合位置

```text
滤波ECG [B,1,7500] ── DAEAC ASPP-SE主干 ── 1024维 ┐
RRI/幅度 [B,2,256] ── 辅助卷积分支 ───────── 128维 ├─ 拼接 ─ Linear(1216,2)
手工特征 [B,64] ───── Linear+ReLU+Dropout ───── 64维 ┘
```

手工特征分支仅包含 `Linear(64,64) + ReLU + Dropout(0.2)`。它不把统计特征变成
伪时间序列，也不重复使用卷积或ASPP提取一次。拼接发生在DAEAC全局池化之后、最终
二分类头之前。`use_third_aspp=false`，因此本实验使用此前表现更稳定的主干结构。

## 每折64维如何确定

来源为347维组合搜索
`features/expanded347/selection_ml_grid_e3_v1/validation_grid.csv`：

1. 在每折已经进入第二阶段的候选中，只比较64维方案；
2. 按验证accuracy选择“特征组＋选择器”，同分时依次比较macro F1、balanced
   accuracy和ROC AUC；
3. 确定配方后，只用该折训练行重新拟合对应选择器的完整特征排名；
4. 取前64列并映射回冻结347维矩阵的全局列号；
5. 测试标签既不参与配方选择，也不参与列排名。

| 折 | 特征组 | 选择器 | 64维候选的表格分类器 | 验证accuracy |
|---:|---|---|---|---:|
| 0 | `fusion3_expanded` | `mrmr` | `random_forest` | 0.8490 |
| 1 | `expanded_all` | `l1` | `random_forest` | 0.8313 |
| 2 | `expanded_all` | `l1` | `hist_gb` | 0.8352 |
| 3 | `non_edr_with_hrv` | `random_forest` | `random_forest` | 0.8342 |
| 4 | `fusion3_expanded` | `l1` | `random_forest` | 0.8520 |

表格分类器只用于从既有验证网格中确定配方，不会被加载到DAEAC。五折输入宽度统一为
64，但具体列允许不同。每折的缺失值中位数、均值和标准差仍只由该折训练行拟合。

验证集既参与64维配方选择，也会在后续DAEAC训练中用于checkpoint和早停；二者都属于
当前折的开发阶段，测试集仍保持隔离。由于同一验证集比较过较多候选，本实验应视为探索性
模型选择，最终部署结论仍需新的独立数据或外层嵌套验证确认。

冻结掩码位于
[`protocols/handcrafted_selection_expanded347_e3_grid_k64/`](../../protocols/handcrafted_selection_expanded347_e3_grid_k64/README.md)，
模型配置为
[`daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml`](../../configs/models/daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml)。

## 运行边界

先做只读数据核验：

```powershell
python -B scripts\data\verify_data.py `
  --config configs\models\daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml
```

正式训练必须由用户显式启动。五折结果应与不含手工分支的
`daeac_filtered_rhythm_amplitude_e3_epoch5fold_2class` 在相同划分、主干和训练参数下比较。

这五份掩码只服务当前冻结五折评价，不构成一个全局部署特征集。部署模型需要在最终
开发数据上重新确定单独的64维掩码。
