# MIT-BIH 二分类模型复现

这是从 `TriGNet-reproduction-3class` 提炼出的多论文复现骨架。它保留可复用主链路：

```text
YAML 配置 -> MIT-BIH master 数据 -> 配置指定的五折划分
          -> 模型 -> 训练/早停 -> 逐折预测与指标 -> 五折 OOF 汇总
```

当前 `SmokeCNN` 只是用来验证数据与训练链路的占位网络，不能作为待复现论文的实现或基线结论。
每篇论文使用独立模型模块、实验 YAML 和复现记录；训练流程共用。
论文作者的开源代码按论文分别存放在 [`upstream/`](upstream/README.md)，便于核对原始实现。
现已接入 [DAEAC MACNN-SE 二分类适配](docs/models/daeac.md)，配置为
`configs/models/daeac.yaml`。另有 [BAFNet 双路 RRI/幅值适配](docs/models/bafnet.md)，
使用独立配置和结果目录。脚本、配置与文档的分区索引分别见
[`scripts/`](scripts/README.md)、[`configs/`](configs/README.md) 和 [`docs/`](docs/README.md)。
BAFNet 基于 Apnea-ECG 的原流程复现单独放在
[`upstream/bafnet/reproduction/`](upstream/bafnet/reproduction/README.md)，使用五分钟输入及作者的两阶段训练，
与 MIT-BIH 30 秒五折实验分开记录。
将同一套五分钟输入和两阶段训练流程迁移到 MIT-BIH 的单次独立被试测试实验，
见 [`upstream/bafnet/mitbih/`](upstream/bafnet/mitbih/README.md)。

本仓库同时维护当前二分类实验使用的手工特征链路：冻结特征位于
[`features/`](features/README.md)，扩展提取、筛查、特征组合与机器学习入口分别位于
`scripts/features/`、`scripts/selection/` 和 `scripts/training/`。相邻 `EDR` 仓库只作为
旧263维EDR特征和心搏定位中间数据的上游来源，不再承载当前347维实验入口。

## 本机与实验室电脑同步

代码、配置和实验输出通过 GitHub 与 Git LFS 在两台电脑之间同步。开始工作前必须拉取，
结束工作后必须提交并推送；两台电脑不得同时写入同一个实验目录。首次配置、实验命名、
日常命令和冲突处理规则见
[`docs/guides/GIT_SYNC_GUIDE.md`](docs/guides/GIT_SYNC_GUIDE.md)。

## 数据与任务约定

- 数据：`../dataset/mitbih_master/mitbih_master_30s.npz`
- 划分：受试者级配置读取 `../dataset/mitbih_master/splits/`；与三分类 E3 对齐的
  样本级配置读取 [`protocols/e3_full_epoch5fold_20260913/`](protocols/e3_full_epoch5fold_20260913/README.md)
- ECG 模型默认使用 30 秒、250 Hz 的 `ecg_raw`，形状为 `1 x 7500`；BAFNet 使用
  `rri_256` 和 `r_amp_256` 两路，形状为 `2 x 256`
- 二分类：`Normal -> 0`，`H/HA/X/OA/CA/CAA -> 1`
- 划分单位由配置中的 `split_unit` 决定；样本级五折允许受试者重叠，但样本索引互斥
- 缩放方式由 `normalization` 指定：ECG 使用逐 epoch z-score，BAFNet 使用逐路
  0～1 归一化；两者都不使用全数据统计量

## 快速验证

```powershell
$Python = 'C:\Users\cgy_3\miniconda3\envs\pytorch-env\python.exe'

& $Python -B scripts\data\verify_data.py --config configs\models\smoke_cnn.yaml
& $Python -B scripts\training\train.py --config configs\models\smoke_cnn.yaml --fold 0 --debug
```

完整五折与汇总：

```powershell
& $Python -B scripts\training\train_5fold.py --config configs\models\smoke_cnn.yaml
& $Python -B scripts\evaluation\aggregate_oof.py --config configs\models\smoke_cnn.yaml
```

DAEAC 与三分类仓库 E3 折成员对齐的二分类五折：

```powershell
& $Python -B scripts\data\verify_data.py --config configs\models\daeac_e3_epoch5fold.yaml
& $Python -B scripts\training\train_5fold.py --config configs\models\daeac_e3_epoch5fold.yaml
& $Python -B scripts\evaluation\aggregate_oof.py --config configs\models\daeac_e3_epoch5fold.yaml
& $Python -B scripts\evaluation\plot_loss.py --config configs\models\daeac_e3_epoch5fold.yaml
```

每折写入 `runs/<experiment>/fold_<n>/`：配置快照、最佳 checkpoint、训练历史、
`metrics.json` 和 `predictions.csv`。OOF 汇总写入 `runs/<experiment>/oof/`。
绘图命令读取各折 `history.json` 和 `metrics.json`，生成各折 `loss_curve.png` 与
`oof/loss_curves.png`；绘图环境需安装可选依赖 `matplotlib`。

## 接入待复现模型

1. 将论文作者代码放在 `upstream/<paper_slug>/`，并记录代码来源、版本和许可证。
2. 在 [`src/mitbih_2class/models/`](src/mitbih_2class/models/README.md) 下为论文建立独立模块，例如 `paper_slug.py` 或 `paper_slug/__init__.py`，提供 `build_model(config)`。
3. 新增 `configs/models/<paper_slug>.yaml`，设置 `model: paper_slug`，将网络专有参数放在 `model_params` 中。
4. 复制 `docs/guides/REPRODUCTION_CHECKLIST.md` 为 `docs/models/<paper_slug>.md`，记录论文数据、划分、预处理和指标定义。
5. 先运行 `--debug`，再执行五折；不要把 debug 或 `SmokeCNN` 结果写进论文对比表。

模型模块由配置名称自动加载，新增论文无需编辑训练循环或维护模型白名单。不同论文使用
不同的 `name`，各自的训练权重仍保存到 `runs/<name>/fold_<n>/best.pt`。

如果论文使用 RRI、R 峰幅值或滤波 ECG，只需把 `signal_key` 改为 master NPZ 中相应键，
并按论文输入形状调整模型；无需复制数据集或训练框架。

