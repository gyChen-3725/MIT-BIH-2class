# MI-128-RF固定配方五折实验

## 目的

此前`MI-128-RF`只在Fold 0成为机器学习搜索的最终获胜方案，因此只有Fold 0测试
指标。其余四折虽然存在该组合的验证指标，但没有使用训练＋验证集重拟合后评价测试集。

本实验在五折中统一固定以下配方：

```text
特征组：fusion3_expanded
选择器：mutual_info
维度：128
分类器：random_forest
随机种子：42 + fold
```

## 每折数据边界

由于特征组、维度、选择器和分类器均已提前固定，不再使用验证集选择超参数。每折执行：

1. 合并当前折训练集和验证集；
2. 仅在合并后的开发集上计算互信息排名；
3. 选择前128维并拟合随机森林；
4. 测试集只预测和评价一次；
5. 五个测试折拼接为完整OOF预测。

测试标签不参与特征排名、模型拟合或参数选择。

## 启动命令

根据仓库约束，正式运行由用户手动启动：

```powershell
python -u -B scripts\training\train_mi128_rf_5fold.py
```

随机森林由scikit-learn在CPU运行，不使用CUDA。脚本会同时将进度写入控制台与：

```text
runs/mi128_rf_fixed_e3_epoch5fold_2class/run.log
```

## 输出

```text
runs/mi128_rf_fixed_e3_epoch5fold_2class/
├── run.log
├── manifest.json
├── fold_metrics.csv
├── fold_0/ ... fold_4/
│   ├── metrics.json
│   ├── predictions.csv
│   └── selected_features.json
└── oof/
    ├── metrics.json
    └── predictions.csv
```

最终五折结论以`oof/metrics.json`为准，不能用五折accuracy的简单平均代替完整OOF
指标计算。
