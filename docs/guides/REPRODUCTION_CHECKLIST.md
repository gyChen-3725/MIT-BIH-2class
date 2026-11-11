# 新模型复现清单

每篇论文复制本清单为 `docs/models/<paper_slug>.md`，并配套建立
`src/mitbih_2class/models/<paper_slug>.py` 与 `configs/models/<paper_slug>.yaml`。
在实现目标论文前先冻结下列信息，避免“模型复现”与“实验协议变化”混在一起。

## 论文身份

- 论文标题 / DOI：待填写
- 官方代码地址与固定 commit：待填写
- 作者代码本地目录：`upstream/<paper_slug>/`（待填写）
- 许可证：待核对
- 复现范围：结构复现 / 官方代码复跑 / 同协议比较（待选择）
- 模型模块名与实验配置：待填写

## 输入与标签

- 输入信号：默认 `ecg_raw`
- 采样率与窗口：默认 250 Hz、30 s
- 归一化：默认逐 epoch z-score
- 标签：`Normal=0`，任意 apnea/hypopnea 事件 `=1`
- 上下文窗口：默认单 epoch；若使用上下文，必须防止相邻窗口跨集合重叠

## 划分与防泄漏

- 默认使用已有 subject-wise 五折 JSON
- train / val / test 受试者必须互斥
- 类别权重只能由训练集计算
- 阈值、早停和超参数只能依据验证集选择
- 测试集每折只用于最终评估

## 需要与论文对齐的项目

- [ ] 网络层、激活、padding、参数量
- [ ] 损失函数与类别不平衡策略
- [ ] 优化器、学习率策略、batch size、epoch、早停
- [ ] 数据增强及其应用范围
- [ ] 决策阈值
- [ ] 指标口径（尤其 apnea 是 positive class）
- [ ] 随机种子和重复次数
- [ ] 官方实现差异与必要修改

