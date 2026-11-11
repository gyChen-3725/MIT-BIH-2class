# 论文开源代码

这个目录存放论文作者公开的原始代码。每篇论文使用一个独立子目录，目录名使用
小写英文和下划线；同一标识可用于 `configs/` 和 `src/mitbih_2class/models/`。

```text
upstream/
├── README.md
└── daeac/
    ├── SOURCE.md             # 来源、版本、许可证和本地整理记录
    └── source/               # 作者仓库的原始文件
```

新增论文时，在对应的 `SOURCE.md` 中至少记录论文标题、代码仓库地址、固定 commit
或发布版本、许可证，以及获取日期。没有 Git 元数据或许可证时如实写明“未知”，
不要从文件夹名推定提交版本，也不要推定许可范围。作者文件放入 `source/` 并尽量
保持原貌；必要的适配实现在 `src/mitbih_2class/models/<paper_slug>.py` 或同名子包中，
并在该论文的 `docs/<paper_slug>.md` 说明差异。

这里存放源码。MIT-BIH 数据继续使用共享数据目录；本项目训练产生的权重和预测保存在
`runs/<实验名>/`。下载的原始数据、大型预训练权重和运行缓存不要直接混入作者代码目录。

当前已归档的作者仓库：[`daeac/SOURCE.md`](daeac/SOURCE.md)、
[`bafnet/SOURCE.md`](bafnet/SOURCE.md)。
