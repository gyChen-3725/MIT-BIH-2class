# BAFNet 原始代码记录

- 仓库名称：*Bottleneck Attention Based Fusion Network for Sleep Apnea Detection*（BAFNet）
- 公开代码：https://github.com/XianhuiChen/Bottleneck-Attention-Based-Fusion-Network-for-Sleep-Apnea-Detection
- 固定提交：`ac6ea317319f9fd9bbe17f4a45e0e184e748ce2b`，本地副本的工作树在整理前无修改
- 本地导入日期：2026-10-27
- 原始代码：[`source/`](source/)，保留原仓库文件与 Git 元数据；整理时只移动目录
- 许可证：本地仓库未发现 `LICENSE` 或 `COPYING`，不可据公开地址推定许可

原代码的网络在 `source/BAFNet.ipynb`，数据制备在 `source/Preprocessing.py`。
它以 Apnea-ECG 的 RRI 与 R 峰幅值为两路输入，窗口包含目标分钟及前后各两分钟，
插值后每路 900 点。当前 MIT-BIH 适配实现见
[`src/mitbih_2class/models/bafnet.py`](../../src/mitbih_2class/models/bafnet.py)，
输入和训练差异见 [`docs/bafnet.md`](../../docs/bafnet.md)。原始文件未在适配过程中修改。

按原始 Apnea-ECG 流程运行的独立代码放在 [`reproduction/`](reproduction/README.md)，
与前述 MIT-BIH 30 秒结构适配实验分开保存。
