# DAEAC 原始代码记录

- 论文：*Inter-patient ECG Arrhythmia Heartbeat Classification Based on Unsupervised Domain Adaptation*
- 公开代码仓库：https://github.com/chenmingTHU/DAEAC
- 本地源码位置：[`source/`](source/)
- 本地导入日期：2026-10-26
- 原始放入路径：`upstream/paper_a_DEAC/DAEAC-master/`
- 本地整理：将外层目录命名为 `daeac`，将作者代码目录命名为 `source`；未修改作者源码内容。
- 固定 commit / 发布版本：未知。该本地副本不含 `.git` 元数据，不能据 `DAEAC-master` 推断具体提交。
- 许可证：本地副本未见 `LICENSE` 或 `COPYING` 文件；使用权限需另行确认。

校验记录：移动后，排除 `__pycache__` 和 `.pyc` 的 43 个源码与配置文件，按相对路径排序，
将每行写成 `<相对路径> <文件 SHA-256>` 并用换行符连接，再对整段 UTF-8 文本计算 SHA-256，
结果为 `61E451418FE99577775CF8958AB19029444F12D82C9ED46623BA4B306F8E2547`。
原目录中的 Python 缓存随快照保留，但已被项目的 `.gitignore` 排除。

本项目中的适配实现位于 `src/mitbih_2class/models/daeac.py`，结构与参数差异记录在
`docs/daeac.md`。作者源码继续保存在 `source/`，不在其中直接修改。
