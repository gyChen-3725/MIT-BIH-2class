# DAEAC MACNN-SE 二分类适配

## 来源与模型范围

- 作者代码快照：[`upstream/daeac/source/`](../../upstream/daeac/SOURCE.md)
- 三分类参数参考：`TriGNet-reproduction-3class/src/trignet_repro/models/daeac.py`
- 本项目模型：[`src/mitbih_2class/models/daeac.py`](../../src/mitbih_2class/models/daeac.py)
- 受试者级配置：[`configs/models/daeac.yaml`](../../configs/models/daeac.yaml)
- E3 样本级配置：[`configs/models/daeac_e3_epoch5fold.yaml`](../../configs/models/daeac_e3_epoch5fold.yaml)
- E3 三路输入配置：[`configs/models/daeac_ecg_rhythm_amplitude_e3_epoch5fold.yaml`](../../configs/models/daeac_ecg_rhythm_amplitude_e3_epoch5fold.yaml)
- E3 三路输入加第三组 ASPP 配置：[`configs/models/daeac_ecg_rhythm_amplitude_aspp3_e3_epoch5fold.yaml`](../../configs/models/daeac_ecg_rhythm_amplitude_aspp3_e3_epoch5fold.yaml)
- E3 第三组 ASPP、按验证准确率选权重配置：[`configs/models/daeac_ecg_rhythm_amplitude_aspp3_accuracy_patience50_e3_epoch5fold.yaml`](../../configs/models/daeac_ecg_rhythm_amplitude_aspp3_accuracy_patience50_e3_epoch5fold.yaml)
- 回到两组重复 ASPP 后的特征聚类：[`DAEAC_FEATURE_CLUSTERING.md`](../clustering/DAEAC_FEATURE_CLUSTERING.md)

模型复用上游 `MACNN_SE` 的 ASPP、SE、残差和全局池化分类主干。当前任务以
`[batch, 1, 7500]` 的单导联 ECG 为输入，首层改为一维时间卷积，输出头改为
`[Normal, Apnea]` 两类。上游心拍级 RR 特征、教师模型及域适配训练步骤没有进入
当前共用训练循环；因此本配置是网络主干的二分类适配，不代表原论文的完整训练协议。

## ECG、节律与幅度融合

三路输入配置直接从 `mitbih_master_30s.npz` 的同一行读取：滤波 ECG
`ecg_filtered`（7500 点）、RR 间期 `rri_256`（256 点）和 R 峰幅度
`r_amp_256`（256 点）。主数据集已经完成 R 峰检测与逐 epoch 插值，
ECG 也已经完成 0.5～30 Hz 带通滤波，这里不重复处理。ECG 按每段
z-score 输入 MACNN-SE 主干；另外两路
在原生 256 点上经过轻量卷积分支，最后与 ECG 的全局特征拼接后分类。

master 用三次插值生成稠密特征，偶尔会使 RRI 超出物理范围。因此读取时
将 RRI 限制在 0～3 秒，并对峰幅度取 `asinh` 以压缩极值；两路不做逐段
z-score，以保留平均心率和幅度水平。全零特征保持为零。这个融合结构与
预处理属于本项目的 30 秒呼吸暂停适配，不是原论文的三行心拍输入或域适配。
三路模型参数量为 **1,417,714**，单路基线仍为 **1,414,530**。

运行前可只检查数据与冻结折成员（不会启动训练）：

```powershell
& 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe' -B scripts\data\verify_data.py --config configs\models\daeac_ecg_rhythm_amplitude_e3_epoch5fold.yaml
```

需要正式比较时，新旧配置使用同一 E3 折划分、随机种子与训练超参数，
分别汇总 OOF 指标。样本级五折允许同一受试者出现在训练和测试集合，
因此也不能把该结果当作被试独立泛化能力。

## 新增第三组 ASPP 的容量实验

论文图中带“×2”的部分是两组 `ASPP → SE → 残差块`，其后还有一个
独立的末端 `ASPP → SE → GAP`，所以旧模型共有三个 ASPP。
新配置令 `use_third_aspp: true`，在末端 ASPP 之前再插入第三组
`ASPP → SE → 残差块`；新模型共有**四个** ASPP。每组仍有膨胀率
`1、6、12、18` 的四个并行卷积分支。

对于 7500 点 ECG，新块接收第二组输出的 256 通道、3750 点信号。
它的四个分支各输出 64 通道，拼接后仍为 256 通道；SE 重新加权，
残差块将时间长度减到约 1875 点。末端 ASPP 仍输出 1024 通道，
全局平均池化后与 128 维 RRI/幅度向量拼接，分类头仍接收 1152 维。
第三组保持通道数不变，以控制较长 ECG 波形的显存占用。

这组新增结构是本项目的容量实验，不属于论文原结构。新模型参数量
**2,026,226**，旧三路模型为 **1,417,714**。旧配置默认关闭
`use_third_aspp`，对应旧实验的结构和 checkpoint；新结构使用独立实验名，
便于在同一 E3 折划分上比较。目前新增结构有 fold 0 的结果，
尚未完成五折汇总。

### 准确率优先的 checkpoint 实验

`checkpoint_metric: val_accuracy` 令训练循环使用**验证集准确率**保存
`best.pt`，并根据同一指标计数早停轮数；同分时保留较早的 checkpoint。
未指定该字段的旧配置仍按 `val_macro_f1` 选择。验证 loss、准确率和
macro F1 均保存在 `history.json`，测试集只在恢复所选权重后评估一次。
准确率配置与第三组 ASPP 的 patience50 配置只差实验名和 checkpoint 指标，
因此可在同一 fold 0 比较两种选择规则。运行将从头训练；旧运行只保留
macro F1 最佳轮次权重，无法事后取回准确率最佳轮次权重。

## 参数量

| 构造方式 | 宽度 | 输出类别 | 参数量 |
|---|---:|---:|---:|
| 三分类参考仓库 | 16 | 3 | 1,415,555 |
| 当前二分类模型 | 16 | 2 | 1,414,530 |

两者只相差分类头的一个输出单元：`1024` 个权重和 `1` 个偏置，合计 `1,025` 个参数。
`tests/test_daeac.py` 会校验当前参数量、30 秒输入的特征维数和二分类输出形状。

## 运行入口

当前与三分类仓库 E3 的**折成员**对齐时，先校验划分：

```powershell
python -B scripts/data/verify_data.py --config configs/models/daeac_e3_epoch5fold.yaml
```

再做单折 CPU 链路检查：

```powershell
python -B scripts/training/train.py --config configs/models/daeac_e3_epoch5fold.yaml --fold 0 --debug
```

正式五折与汇总分别使用 `scripts/training/train_5fold.py` 和
`scripts/evaluation/aggregate_oof.py`，均指定
`--config configs/models/daeac_e3_epoch5fold.yaml`。这套配置只对齐 E3 的样本索引：
输入仍是本项目的原始 ECG 和逐 epoch 标准化，训练仍使用本项目的二分类超参数。
`configs/models/daeac.yaml` 则保留为另一套受试者独立划分，结果不得混合汇总。
