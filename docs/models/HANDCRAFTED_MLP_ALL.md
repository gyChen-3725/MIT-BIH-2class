# 全部手工特征的 MLP 二分类实验

输入为 EDR 仓库生成的 263 维 `all` 特征：8 维 HRV、10 路 EDR 各 25 维
生理特征，以及 5 维三路关系特征。模型不读取 ECG 波形，也不读取 master
中的 256 点 RRI/幅度序列。

网络结构为 `263 → 128 → ReLU → Dropout(0.2) → 2`。它只完成非线性分类，
没有卷积、ASPP、SE，也不会把统计特征转换为伪时间序列。每折仅用训练集
拟合缺失值填补、逐列均值和标准差，并保存为 `feature_normalizer.npz`。

实验沿用 E3 冻结样本级五折，并按验证集准确率选择 checkpoint。E3 允许
同一受试者及相邻 epoch 分布在不同集合，因此结果用于与现有 E3 实验进行
协议内比较，不能解释为受试者独立泛化能力。

在 `MIT-BIH-2class` 目录运行一折：

```powershell
python scripts/training/train.py --config configs/models/handcrafted_mlp_all_e3_epoch5fold.yaml --fold 0 --device cuda
```

完成一折并确认结果后，由用户自行运行其余四折及汇总：

```powershell
1..4 | ForEach-Object { python scripts/training/train.py --config configs/models/handcrafted_mlp_all_e3_epoch5fold.yaml --fold $_ --device cuda }
python scripts/evaluation/aggregate_oof.py --config configs/models/handcrafted_mlp_all_e3_epoch5fold.yaml
```

配置中的 `handcrafted_path` 是相对于本仓库根目录的路径。迁移到服务器时，
需要改为服务器上实际的 EDR 特征文件位置。
