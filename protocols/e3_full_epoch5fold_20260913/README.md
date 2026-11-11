# E3 全量样本级五折

本目录保存 DAEAC 二分类实验使用的冻结样本索引。生成规则与三分类仓库 E3 完全一致：
从 `mitbih_master_30s.npz` 的原始七类标签映射出 Normal/Hypopnea/Apnea，执行
`StratifiedKFold(n_splits=5, shuffle=True, random_state=20260913)`；第 `i` 折测试，
下一折验证，其余三折训练。生成后已与三分类仓库的
`data/upstream_full_3class/mitpsg/split_indices.npz` 逐项比对，索引完全相同。

这是 **epoch/样本级** 协议：同一受试者可同时出现在训练、验证和测试中；同一行
epoch 只属于一个角色，五次测试合起来恰好覆盖全部 10,197 行。划分及来源哈希见
[`manifest.json`](manifest.json)。

重新生成命令：

```powershell
python -B scripts/data/prepare_e3_epoch_splits.py
```

`configs/models/daeac_e3_epoch5fold.yaml` 使用这些索引，但仍保留本项目的原始 ECG、
逐 epoch 标准化和二分类训练超参数。它与三分类 E3 对齐的是**折成员**，
不是三分类 E3 的滤波波形、全局标准化或完整训练超参数。
