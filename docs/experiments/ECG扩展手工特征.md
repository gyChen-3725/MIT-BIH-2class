# 347维ECG扩展手工特征

## 目的

旧模型输入263维主要由EDR统计量构成。扩展版保持旧263列的名称和顺序不变，
追加84项不局限于EDR的特征，总计347维：

| 特征族 | 新增维度 | 主要内容 |
|---|---:|---|
| 扩展HRV | 24 | NN分布、心率趋势、加减速比例、Poincaré SD1/SD2 |
| 原始ECG | 34 | 0.5–35 Hz统计量、功率谱、Hjorth、幅值熵、排列熵和基线漂移 |
| QRS形态 | 26 | 时限、振幅、面积、能量和斜率的中位数、IQR或MAD |

特征不读取分类标签，不使用真实鼻气流或胸腹呼吸信号。30秒窗口不足以可靠估计
常规LF/HF，因此没有为了增加维度而加入频域HRV。原始ECG使用离线零相位滤波，
当前结果不能直接解释为实时可用特征。

## 产物

- 特征文件：[`features/expanded347/features_30s_expanded.npz`](../../features/expanded347/features_30s_expanded.npz)
- 清单：[`features/expanded347/manifest.json`](../../features/expanded347/manifest.json)
- 完整字典：[`扩展手工特征字典_347维.md`](../features/扩展手工特征字典_347维.md)
- 机器可读字典：[`expanded_feature_dictionary_347.csv`](../features/expanded_feature_dictionary_347.csv)

最终矩阵形状为 `10197 × 347`，字段名无重复。全矩阵有限值比例为99.54%；
新增扩展HRV、原始ECG和QRS形态的有限值比例分别为99.77%、100%和99.78%。
缺失值保留为NaN，后续只能使用训练折统计量填补。

## 复现命令

```powershell
& C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe -B scripts\features\extract_expanded_features.py
& C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe -B scripts\features\build_expanded_feature_dictionary.py
```

脚本发现输出目录已有内容时会停止，避免覆盖当前冻结产物。

本仓库已支持直接读取 `all_features/all_feature_names`，并准备了
`configs/models/daeac_ecg_rhythm_amplitude_handcrafted_expanded347_e3_epoch5fold.yaml`。
该配置尚未启动训练。
