# DAEAC扩展手工特征：折内64维冻结掩码

本目录供 `daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml` 使用。
每折先从既有组合搜索的64维候选中，用验证accuracy确定特征组和选择器；随后只使用
该折训练行重新拟合特征排名并冻结64列。测试标签不参与配方选择或列排名。

| 折 | 特征组 | 选择器 | 表格分类器 | 验证accuracy | 64维排名拟合数据 |
|---:|---|---|---|---:|---|
| 0 | `fusion3_expanded` | `mrmr` | `random_forest` | 0.8490 | 仅当前折训练集 |
| 1 | `expanded_all` | `l1` | `random_forest` | 0.8313 | 仅当前折训练集 |
| 2 | `expanded_all` | `l1` | `hist_gb` | 0.8352 | 仅当前折训练集 |
| 3 | `non_edr_with_hrv` | `random_forest` | `random_forest` | 0.8342 | 仅当前折训练集 |
| 4 | `fusion3_expanded` | `l1` | `random_forest` | 0.8520 | 仅当前折训练集 |

五折都向DAEAC提供64维输入，但具体特征名称允许不同。每折的中位数填补、均值和标准差
仍由DAEAC数据加载器仅在该折训练行拟合。该目录不是一个可直接部署的全局64维列表；
部署前需要在最终开发集上重新冻结一套单独掩码。
