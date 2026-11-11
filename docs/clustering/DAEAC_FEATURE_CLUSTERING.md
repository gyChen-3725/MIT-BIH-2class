# DAEAC 两路辅助特征的三组聚类

## 范围

本轮模型回到两组 `ASPP → SE → 残差块`，再接原有末端 ASPP，
总计三个 ASPP。配置在
[`configs/clustering/daeac_aspp2_feature_clustering_e3.yaml`](../../configs/clustering/daeac_aspp2_feature_clustering_e3.yaml)，
其中 `use_third_aspp: false`。以前的 ASPP3 代码和实验配置保留，
以便复核历史结果。本轮**只做特征聚类，没有训练、筛困难样本或测试模型**。

BAFNet 作者在五分钟 RRI 序列上做五簇 K-means，然后按簇内类别分布
选择第二阶段训练样本。这里借用五簇和 10 次初始化的做法，但输入是
MIT-BIH PSG master 已保存的 **30 秒、256 点** `rri_256` 与 `r_amp_256`，
因此属于本任务的新实验，不是 BAFNet 论文的原始输入。

## 方法与数据边界

使用 E3 冻结划分的 fold 0，**仅取 6,117 个训练索引**。
验证集和测试集不参与标准化、K-means 拟合或簇类别统计。
RRI 限制到 0～3 秒，幅度使用 `asinh`；这与 DAEAC 辅助分支的读取处理一致。
随后每路使用训练样本的全局均值和标准差缩放，保留每个 epoch 的平均水平。
联合组合拼接两条 256 点序列，共 512 维；先对两路分别缩放，避免单位差异
让某一路主导欧氏距离。

| 组合 | 输入维度 | 五簇样本数 | 各簇 Apnea 比例 |
|---|---:|---|---|
| 仅 RRI | 256 | 1926、2191、1025、969、6 | 28.1%、34.6%、40.2%、30.2%、0% |
| 仅 R 峰幅度 | 256 | 585、1461、325、227、3519 | 44.3%、29.0%、25.2%、25.1%、33.6% |
| RRI＋幅度 | 512 | 2755、732、1189、6、1435 | 28.3%、39.5%、41.2%、0%、31.0% |

这些簇编号只在各自组合内有意义，不能把不同组合的“簇 0”当成同一群。
RRI 和联合组合均产生一个 6 样本的小簇；幅度组合的部分簇明显集中于
少数受试者。例如幅度簇 0 的 585 个样本中有 406 个来自 P04。
这些现象需要在困难样本选择前检查，不能直接把簇大小或簇编号解释为难度。
三组输入维度不同，K-means inertia 也不能直接作为组合优劣的统一分数。

## 产物与复现

输出目录：
[`runs/daeac_aspp2_feature_clustering_e3_2class/fold_0/clusters/`](../../runs/daeac_aspp2_feature_clustering_e3_2class/fold_0/clusters/)。

- `assignments.csv`：每个训练样本的 master 行号、标签、受试者、record 和三组簇编号。
- `assignments.npz`：供后续程序按 master 行号读取簇编号。
- `rri_centers.npz`、`amplitude_centers.npz`、`both_centers.npz`：标准化空间的簇中心。
- `summary.json`：类别和受试者分布、预处理、随机种子、输入摘要及划分哈希。
- `cluster_overview.png`：三种特征组合的二维 PCA、簇中心与类别构成。
- `cluster_pca_zoom.png`：按各轴 0.5%～99.5% 分位数放大主体样本；标题列出未显示的离群点数。
- `cluster_subjects.png`：各簇内部的受试者占比，检查被试偏向。

绘图读取已保存的簇编号和中心，不重新执行 K-means。聚类始终在原始的
256/512 维特征空间进行；二维投影只用于查看，重叠不等于高维空间无法区分。
放大图隐藏了离群点，只用于观察主体样本的结构，完整范围以总览图为准。
从图上看，RRI 的主要簇大致沿第一主成分排列；幅度的第一主成分解释
约 81.9% 的方差，联合特征在前两主成分上呈现更复杂的分布。
但各簇 Apnea 比例主要在 25%～44% 之间，聚类没有自然形成清晰的
Normal/Apnea 分界。6 样本离群簇和部分簇的受试者集中现象，意味着后续
困难样本选择还需要单独论证，不能仅凭这三张图决定使用哪种组合。

重新生成时运行：

```powershell
python -B scripts/clustering/cluster_daeac_features.py --config configs/clustering/daeac_aspp2_feature_clustering_e3.yaml --fold 0
python -B scripts/clustering/plot_daeac_clusters.py --config configs/clustering/daeac_aspp2_feature_clustering_e3.yaml --fold 0
```

聚类脚本发现目录已有产物时会停止，避免覆盖先前聚类；绘图脚本可重复运行。
`runs/` 已被 Git 忽略；
在另一台机器上需要重新运行该命令或单独复制产物。

## 排除原始聚类中的 6 个异常点

原始 fold 0 的 RRI 簇 4 与联合特征簇 3 包含完全相同的 master 行号：
`1406、1407、1413、1414、1420、5092`，其中 5 个来自 P03，
1 个来自 P37。这些样本的原始 RRI 中，**44.9%～99.2%** 的 256 个值
低于 0 秒、超过 3 秒或不是有限数。按当前 DAEAC 预处理，它们的大量值
会被限制到边界。这为把它们当作聚类异常点提供了数据质量依据。

具体行号、来源和每行异常比例写入
[`protocols/e3_full_epoch5fold_20260913/excluded_rri_outliers.json`](../../protocols/e3_full_epoch5fold_20260913/excluded_rri_outliers.json)。
复核时应保留上面的原始聚类产物；新实验使用独立配置
[`configs/clustering/daeac_aspp2_feature_clustering_exclude6_e3.yaml`](../../configs/clustering/daeac_aspp2_feature_clustering_exclude6_e3.yaml)，
只在聚类输入中排除这 6 个训练行，不改 master 或冻结五折划分，
也没有启动模型训练或评估。新结果保存到
[`runs/daeac_aspp2_feature_clustering_exclude6_e3_2class/fold_0/clusters/`](../../runs/daeac_aspp2_feature_clustering_exclude6_e3_2class/fold_0/clusters/)，
包含同样的三张可视化图。

| 组合 | 排除后五簇样本数，合计 6111 |
|---|---|
| 仅 RRI | 1892、959、9、2173、1078 |
| 仅 R 峰幅度 | 1443、586、227、3535、320 |
| RRI＋幅度 | 1289、2673、1338、611、200 |

重跑后 RRI 仍有一个 9 样本小簇，但其中只有 1 个样本的原始 RRI 大量
越界；其余 8 个的 RRI 没有超出 0～3 秒。**小簇本身不等于数据异常**。
整个训练折还有其他 RRI 越界样本，因此本实验只回答“排除这 6 个点后
聚类如何变化”，不代表已经完成全数据质量清理。以后若把排除规则用于
模型训练或五折评估，必须在各折训练、验证、测试中统一应用，并用相同
的评价样本比较结果。

已有原始结果时，只运行以下命令生成独立的排除版聚类和图片：

```powershell
python -B scripts/clustering/cluster_daeac_features.py --config configs/clustering/daeac_aspp2_feature_clustering_exclude6_e3.yaml --fold 0 --exclude-index-file protocols/e3_full_epoch5fold_20260913/excluded_rri_outliers.json
python -B scripts/clustering/plot_daeac_clusters.py --config configs/clustering/daeac_aspp2_feature_clustering_exclude6_e3.yaml --fold 0
```

## 聚类与二分类标签的统计关联

以下检验只用排除 6 点后的 fold 0 训练集：6111 个 epoch，其中 Normal 4107、
Apnea 2004，来自 16 位受试者。这里沿用项目的二分类映射：
原始标签为 Normal 记 0，其余已标注事件均记 1。每种组合用 5×2 列联表计算 Pearson χ² 和
Cramér's V。常规 χ² 检验假定每行样本独立；本数据同一受试者有许多相邻 epoch，
所以还做了 4999 次受试者内标签打乱，以及更保守的受试者内标签**循环平移**。
循环平移保留每位受试者标签序列的成段结构，但训练集抽样留下的间隙仍在。
下表中 Holm 校正涵盖三项聚类检验和后面的两项特征检验。

| 聚类特征 | χ² | 常规 χ² P | Cramér's V | 受试者内打乱 P | 保留标签段的循环平移 P | 循环平移 Holm P |
|---|---:|---:|---:|---:|---:|---:|
| RRI | 52.29 | 1.20×10⁻¹⁰ | 0.092 | 1.000 | 0.976 | 0.976 |
| R 峰幅度 | 57.61 | 9.24×10⁻¹² | 0.097 | 0.0002 | 0.113 | 0.452 |
| RRI＋幅度 | 142.07 | 1.02×10⁻²⁹ | 0.152 | 0.243 | 0.458 | 0.915 |

常规 χ² P 都很小，但在受试者和时间结构得到部分控制后，三组聚类
均未达到 0.05。尤其 RRI 的 9 样本簇有一个期望格数仅 2.95，
常规 χ² 渐近 P 值还受小格数影响。Cramér's V 也都较小，
不能把小 P 值解读成聚类可直接区分 Normal/Apnea。

为接近论文对 RRI 与 RPA 的讨论，还把每个 epoch 的 256 点归约成一个均值：
RRI 使用模型输入前的 0～3 秒截断，幅度使用 `asinh` 变换。
在 14 位同时有两类标签的受试者内，逐人计算 Apnea 减 Normal 的均值差，
再对 14 个差值做双侧 Wilcoxon 配对检验：

| 逐 epoch 特征均值 | 逐人差值的中位数 | Wilcoxon P | Holm P |
|---|---:|---:|---:|
| RRI | +0.0141 秒 | 0.0107 | 0.0537 |
| `asinh` 后的幅度 | +0.0236 | 0.241 | 0.724 |

RRI 的未校正 P 小于 0.05，但在本次五项探索性检验统一校正后为 0.0537。
论文未披露其 P 值使用的统计检验，故这里的数字不能直接与论文的
`1.58×10⁻¹⁶` 和 `0.284` 对比。上述关联性检验也不是预测性能或
三种聚类组合的优劣排名。完整列联表、受试者差值与随机化方法记录在
[`runs/daeac_aspp2_feature_clustering_exclude6_e3_2class/fold_0/clusters/significance.json`](../../runs/daeac_aspp2_feature_clustering_exclude6_e3_2class/fold_0/clusters/significance.json)。

复算命令（只分析现有聚类，不训练）：

```powershell
python -B scripts/clustering/analyze_daeac_cluster_significance.py --config configs/clustering/daeac_aspp2_feature_clustering_exclude6_e3.yaml --fold 0 --permutations 4999 --seed 42
```
