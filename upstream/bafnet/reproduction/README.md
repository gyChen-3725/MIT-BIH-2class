# BAFNet 在 Apnea-ECG 上的原流程复现

这里独立实现 `source/BAFNet.ipynb` 和 `source/Preprocessing.py` 的数据流、
网络、两阶段训练及逐记录评价。原始仓库保留在相邻的 `../source/`，不修改作者文件。
输入按要求来自共享的 `dataset/apnea_ecg_master/records/*.npz`。

## 数据与划分

`preprocess.py` 将每个 master 记录的 60 秒原始 ECG 重新连接，对目标分钟
及前后各两分钟形成 5 分钟窗口。各窗口执行作者的 3～45 Hz FIR 滤波、
Hamilton R 峰检测与校正、心率阈值、RRI 中值滤波、逐路 min-max 缩放和
三次样条插值，最终得到 `[900, 2]` 的 RRI/幅值特征。

- `a01–a20`、`b01–b05`、`c01–c10` 为训练池；其中 30% 按标签分层抽作验证集，种子 42。
- `x01–x35` 为独立测试记录。master 将这些记录标为 `Unknown`，所以复用
  [`../source/dataset/event-2-answers`](../source/dataset/event-2-answers)
  填充测试标签，不把 `Unknown` 当作 Normal。
- master ECG 为 float32 且只保留完整 60 秒片段；作者直接读 WFDB 全长波形。
  因此即使处理步骤相同，边界与小数精度也可能导致少量样本和数值差异。
- 本次重建产生 16,690 条 a/b/c 训练池窗口，比 notebook 写死的 16,709 少 19 条。
  x 盲测窗口为 16,945 条；`audit_source.py` 已验证 35 条 x 记录的窗口数和标签
  顺序与作者附带的 `BAFNet.csv` 完全一致。

## 网络与训练

`model.py` 按作者 notebook 的 900 点输入、两路卷积、瓶颈自注意力、
交叉注意力、通道融合、分类头和两路转置卷积重建头构造 72,868 参数网络。
`train.py` 使用 batch 128、Adam（初始学习率 0.001）、分类 BCE + 两路 MSE；
第一阶段按论文设置最多训练 100 epoch，每个 epoch 按 notebook 设为 92 steps。
第二阶段先对训练集 RRI 用 K-means 分五簇，按作者的簇内
困难样本选择函数组建新训练集，再训练 5 epoch。两阶段都用验证分类 accuracy
保存最佳 checkpoint；测试指标按作者 notebook 在当前 epoch 的权重上计算。
测试时先执行 `model.evaluate` 记录三头损失，再由分类头输出逐分钟预测。
卷积层和转置卷积层使用 He 正态初始化与 L2 正则。作者 notebook 的学习率
回调在 epoch 索引 71、81、91（对应第 72、82、92 个 epoch）各乘以 0.1；
第二阶段继续使用第一阶段末尾的学习率。
最终还按作者 `evaluate.py` 的每记录阳性分钟比例和 >5 次/小时规则计算指标。

| 论文原文中的设置 | 当前实现 | 与作者 notebook 的关系 |
|---|---|---|
| 训练/验证按标签分层 70%/30% | `test_size=0.3, stratify=labels[train_pool]` | 一致 |
| 第一阶段最多 100 epoch | 默认 `--stage1-epochs 100`，跑满 100 epoch | notebook 的 `fit` 写成 1 epoch |
| 精炼阶段最多 5 epoch | 默认 `--stage2-epochs 5`，跑满 5 epoch | 一致 |
| 初始学习率 0.001，Adam | `Adam(learning_rate=0.001)` | 与默认 Adam 学习率等价 |
| 前 70 epoch 学习率不变，之后每 10 epoch 乘 0.1 | 使用作者原版 `LearningRateScheduler` 条件 | notebook 零起始索引使实际下降发生在第 72、82、92 个 epoch 开始时 |
| batch size 128 | 默认 `--batch-size 128` | 一致 |
| Conv1D、Conv1DTranspose 使用 `he_normal` 和 L2 | 所有这两类层均设置 He 初始化与 L2(0.001) | 一致 |

论文只给出最大 epoch 数，未说明提前停止条件；作者 notebook 也没有
`EarlyStopping`。当前脚本会跑满指定 epoch，另保存验证准确率最高的权重。

为使重复运行一致，网络、K-means 与随机抽样统一设置种子 42。作者 notebook
没有固定全部这些随机源。保留了 notebook 的两路重建目标互换和困难样本列表
边遍历边删除的行为，以及按簇首次出现顺序组装第二阶段训练集；这些属于原代码
的训练语义，并非我们新改的模型结构。

## 运行

使用装有 TensorFlow、BioSPPy、SciPy 和 scikit-learn 的本机环境：

```powershell
cd E:\envs\Projects\Sleep-apnea-detection\MIT-BIH-2class
$Python = 'C:\Users\cgy_3\miniconda3\envs\tf218-cpu\python.exe'
& $Python -u -B upstream\bafnet\reproduction\preprocess.py --workers 8
& $Python -B upstream\bafnet\reproduction\audit_source.py
& $Python -u -B upstream\bafnet\reproduction\train.py
& $Python -B upstream\bafnet\reproduction\evaluate_best.py --run upstream\bafnet\reproduction\artifacts\paper_100plus5_seed42
& $Python -B upstream\bafnet\reproduction\plot_history.py --run upstream\bafnet\reproduction\artifacts\paper_100plus5_seed42
```

论文正文给出第一阶段最多 100 epoch、第二阶段 5 epoch；开源 notebook
的第一阶段 `model.fit` 却写成 `epochs=1`，与正文和学习率回调不一致。
本目录以论文的 100＋5 epoch 作为默认正式协议。此前的 1＋5 和 80＋5
运行分别是 notebook 字面设置与过渡诊断，两者都不是论文正式设置。
如需重现已经保存的 80＋5 诊断，可运行：

```powershell
& $Python -u -B upstream\bafnet\reproduction\train.py --stage1-epochs 80 --stage2-epochs 5 --output upstream\bafnet\reproduction\artifacts\extended_80plus5_seed42
& $Python -B upstream\bafnet\reproduction\evaluate_best.py --run upstream\bafnet\reproduction\artifacts\extended_80plus5_seed42
& $Python -B upstream\bafnet\reproduction\plot_history.py --run upstream\bafnet\reproduction\artifacts\extended_80plus5_seed42
```

预处理输出为 `artifacts/bafnet_5min.npz` 及其 manifest。训练输出位于
`artifacts/paper_100plus5_seed42/`，包含两阶段测试分数、最佳与最终权重、
历史、逐分钟和逐记录指标。`--smoke` 仅用少量样本验证运行链路，不可用作正式结果。
`evaluate_best.py` 另行评估已保存的最佳验证 checkpoint；作者 notebook 在
训练后的正式测试使用最后一个 epoch 权重，因此两个结果必须分开报告。

## 实际结果与解释

以下是**旧运行结果**，目前还没有 100＋5 epoch 的正式训练结果。
两组都在相同的 16,945 条 x 盲测窗口、35 条 x 记录上评估。下表的逐分钟
分类使用固定 0.5 阈值；逐记录分类使用作者的 >5 次/小时规则。

| 运行 | 逐分钟 accuracy | 逐分钟 F1 | 逐分钟 AUC | 逐记录 accuracy |
|---|---:|---:|---:|---:|
| notebook 字面 1＋5 epoch，最终权重 | 44.05% | 56.41% | 0.8353 | 65.71% |
| 延长 80＋5 epoch，最终权重 | 88.46% | 85.20% | 0.9522 | 97.14% |
| 延长 80＋5 epoch，验证最佳权重 | **89.18%** | **85.64%** | **0.9557** | **100%** |
| 作者仓库附带的 `BAFNet.csv` | 91.11% | 88.09% | 0.9677 | 100% |

字面运行的第一阶段仅预测 Normal；第二阶段最终权重则将多数 Normal 预测为
Apnea，说明 1＋5 epoch 与仓库附带的高分预测文件不对应。延长组第一阶段
验证 accuracy 的最佳值出现在第 42 epoch（91.83%），第二阶段最高为 91.73%；
因此延长组的最佳 checkpoint 来自第一阶段。两组运行均未达到论文正文所述的
第一阶段 100 epoch，不能把这些结果写成论文协议的复现指标。

详细结果：[字面运行](artifacts/original_flow_seed42/metrics.json)、
[延长运行](artifacts/extended_80plus5_seed42/metrics.json)、
[延长运行最佳权重](artifacts/extended_80plus5_seed42/best_metrics.json)、
[训练曲线](artifacts/extended_80plus5_seed42/training_curves.png)。

上述已有运行是在修正困难样本的簇遍历顺序、补充 `model.evaluate` 之前完成的。
修正会改变第二阶段的样本排列，因此表中指标仅代表当时保存的权重；要得到
当前脚本对应的结果，需重新执行训练命令并写入新的输出目录。
