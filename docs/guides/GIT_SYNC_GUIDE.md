# 本机与实验室电脑 Git 同步指南

本仓库使用 GitHub 在本机和实验室电脑之间同步代码、配置、文档及实验结果。
大型 `*.npz`、`*.pt`、`*.h5` 文件由 Git LFS 管理；`runs/` 和 BAFNet 的
`artifacts/` 都会进入版本控制。Python、pytest 和 Ruff 缓存不参与同步。

## 一、必须遵守的规则

1. **开始工作前先拉取，结束工作后立即提交并推送。** 不要在一台电脑存在未推送提交时切换到另一台电脑继续工作。
2. **同一个实验目录在同一时间只能由一台电脑写入。** 禁止两台电脑同时运行会写入同一 `runs/<experiment>/` 的任务。
3. **每次实验使用唯一名称。** YAML 中的 `name` 决定输出目录，建议加入机器、日期和版本，例如：
   - 本机：`daeac_e3_local_20261111_v1`
   - 实验室：`daeac_e3_lab_20261111_v1`
4. **不要覆盖已有实验目录。** 重跑实验时使用新的 `name`；确实需要续跑时，确认另一台电脑没有修改该目录。
5. **代码、配置和对应结果一起提交。** 这样每份结果都能追溯到产生它的代码与参数。
6. **不要强制推送 `main`。** 禁止使用 `git push --force` 或 `git push -f`。
7. **遇到冲突先停止。** 尤其不要尝试手工合并模型权重、NPZ 或 H5 文件。

## 二、实验室电脑实际路径与首次配置

以下是实验室电脑当前的实际目录，用来说明数据和项目之间的关系：

```text
D:\CGY\
├─ MIT-BIH-2class\
│  ├─ configs\
│  ├─ features\
│  ├─ runs\
│  └─ ...
├─ dataset\
│  ├─ mitbih_master\
│  │  ├─ mitbih_master_30s.npz
│  │  └─ splits\
│  └─ mit-bih-polysomnographic-database-1.0.0\
└─ EDR\
```

各路径的用途如下：

| 内容 | 实验室电脑当前路径 |
| --- | --- |
| 项目根目录 | `D:\CGY\MIT-BIH-2class` |
| 主数据文件 | `D:\CGY\dataset\mitbih_master\mitbih_master_30s.npz` |
| 受试者级划分 | `D:\CGY\dataset\mitbih_master\splits` |
| 原始 MIT-BIH 数据 | `D:\CGY\dataset\mit-bih-polysomnographic-database-1.0.0` |
| EDR 仓库 | `D:\CGY\EDR` |
| 实验输出 | `D:\CGY\MIT-BIH-2class\runs` |

两台电脑的项目绝对路径不要求一致，关键是使用内容一致的数据。共享 YAML 中的
`../dataset/...` 会以各自的项目根目录为基准解析；当前实验室项目位于
`D:\CGY\MIT-BIH-2class`，因此会正确解析到 `D:\CGY\dataset\...`。
`output_root: runs` 则会解析到当前项目内的 `D:\CGY\MIT-BIH-2class\runs`。

如果以后移动项目，只需确保配置仍能找到实际数据。不要为了让两台电脑的绝对路径
看起来相同而复制数据；应保证数据文件本身及划分内容一致。

首次使用前，在 PowerShell 中检查必要路径：

```powershell
$RequiredPaths = @(
    'D:\CGY\MIT-BIH-2class',
    'D:\CGY\dataset\mitbih_master\mitbih_master_30s.npz',
    'D:\CGY\dataset\mitbih_master\splits',
    'D:\CGY\dataset\mit-bih-polysomnographic-database-1.0.0',
    'D:\CGY\EDR'
)

$RequiredPaths | ForEach-Object {
    "{0} -> {1}" -f $_, (Test-Path -LiteralPath $_)
}
```

项目目录和主数据文件是常规训练的基础要求。`splits` 用于受试者级配置；原始 MIT-BIH
目录和 EDR 仓库主要用于重新提取扩展特征。运行某项任务前，其依赖路径必须显示 `True`。

先安装 Git 和 Git LFS。截图中 `D:\CGY\MIT-BIH-2class` 已经存在，因此不要再次直接
执行 `git clone`；先在 PowerShell 中检查它是否为本仓库：

```powershell
Set-Location D:\CGY\MIT-BIH-2class
git rev-parse --show-toplevel
git remote -v
```

正确结果应显示项目根目录和
`https://github.com/gyChen-3725/MIT-BIH-2class.git`。确认正确后执行：

```powershell
git lfs install
git pull --rebase
git lfs pull
git submodule update --init --recursive
git config pull.rebase true
git config fetch.prune true
```

如果现有目录不是 Git 仓库，不要直接覆盖。先把它改名备份，再执行：

```powershell
Set-Location D:\CGY
git lfs install
git clone --recurse-submodules https://github.com/gyChen-3725/MIT-BIH-2class.git
Set-Location D:\CGY\MIT-BIH-2class
git lfs pull
git config pull.rebase true
git config fetch.prune true
```

检查初始化结果：

```powershell
git status
git lfs status
git submodule status
```

正常情况下应位于 `main` 分支，工作区没有待提交文件，Git LFS 没有待推送对象。

## 三、每次开始工作

进入仓库后先执行：

```powershell
Set-Location D:\CGY\MIT-BIH-2class
git status
git pull --rebase
git lfs pull
git submodule update --init --recursive
```

只有 `git status` 显示工作区干净时，才直接执行 `git pull --rebase`。
如果存在本地修改，先判断这些修改是否需要保留：

- 需要保留：先完成提交，再拉取远程更新。
- 不确定：停止操作并检查文件，不要用 `git reset --hard` 清除。
- 只是缓存：确认它们已被 `.gitignore` 排除，不要手工加入提交。

需要临时保存尚未完成但必须跨电脑继续的工作时，优先创建明确的 WIP 提交：

```powershell
git add -A
git commit -m "wip(lab): save work before handoff"
git pull --rebase
git push
```

## 四、开始实验前

1. 确认已拉取最新 `main`。
2. 检查实验 YAML 中的 `name`，确保对应的 `runs/<name>/` 不会与另一台电脑冲突。
3. 建议先提交并推送本次使用的代码和配置，再启动长时间实验。
4. 在实验运行期间，不要在另一台电脑启动同名实验。

机器命名建议：

| 运行位置 | 实验名后缀 | 示例 |
| --- | --- | --- |
| 本机 | `_local_YYYYMMDD_vN` | `bafnet_local_20261111_v1` |
| 实验室电脑 | `_lab_YYYYMMDD_vN` | `bafnet_lab_20261111_v1` |

已有历史实验目录不要为了套用新规则而改名。新实验从现在开始使用唯一名称即可。

## 五、完成工作并上传

实验结束后执行：

```powershell
git status
git add -A
git status
git commit -m "exp(lab): add <实验名称> results"
git pull --rebase
git push
git status
git lfs status
```

第二次 `git status` 用来检查即将提交的范围，确认没有误加入数据集、临时文件或无关文件。
`*.npz`、`*.pt`、`*.h5` 会根据 `.gitattributes` 自动进入 Git LFS，无需手工逐个处理。

常用提交信息示例：

```text
exp(lab): add daeac five-fold results
exp(local): add bafnet ablation results
fix: correct fold aggregation metrics
docs: update reproduction notes
wip(lab): save work before handoff
```

## 六、在另一台电脑接续工作

先确认上一台电脑已经完成 `git push`，然后在当前电脑执行：

```powershell
git status
git pull --rebase
git lfs pull
git submodule update --init --recursive
```

不要用资源管理器、网盘或 U 盘直接覆盖整个仓库目录，也不要复制 `.git/`。如果需要传输 GitHub
之外的大型原始数据，只复制数据目录，并保持仓库中的相对路径约定不变。

## 七、推送被拒绝时

如果 `git push` 提示远程包含新的提交，说明另一台电脑先推送了内容。执行：

```powershell
git pull --rebase
git push
```

如果 rebase 没有冲突，Git 会自动整理提交顺序。如果出现冲突，按下一节处理，不要强制推送。

## 八、冲突处理

### 1. 文本文件冲突

文本文件包括 Python、YAML、JSON、Markdown 和 CSV。先查看冲突：

```powershell
git status
```

打开冲突文件，检查并处理 `<<<<<<<`、`=======`、`>>>>>>>` 标记，然后执行：

```powershell
git add <已解决的文件>
git rebase --continue
```

所有冲突解决后再执行 `git push`。如果无法判断应该保留哪一版，先执行
`git rebase --abort` 恢复到 rebase 前状态，再比较两台电脑的工作内容。

### 2. 模型权重或二进制结果冲突

`*.pt`、`*.npz`、`*.h5` 不能安全地逐行合并。出现此类冲突时：

1. 执行 `git rebase --abort`，不要随意选择覆盖。
2. 确认两份结果分别来自哪次实验。
3. 将其中一份结果放到新的唯一实验目录，并同步修改相关 YAML `name` 或说明文档。
4. 重新提交，再执行 `git pull --rebase` 和 `git push`。

如果两份文件其实是同一实验的不同版本，应人工决定保留哪一份，并先在仓库外备份被替换版本。

## 九、Git LFS 检查与故障处理

查看由 LFS 管理的文件：

```powershell
git lfs ls-files
git lfs status
```

拉取代码后只有很小的文本指针、没有实际模型文件时，执行：

```powershell
git lfs install
git lfs pull
```

上传中断时可以直接再次执行 `git push`。Git LFS 会复用已成功上传的对象，不要删除 `.git/lfs/`。

## 十、禁止操作清单

- 不要在两台电脑同时修改或生成同一个实验目录。
- 不要在有未提交修改时直接 `git pull`。
- 不要执行 `git push --force`、`git push -f` 或 `git reset --hard`。
- 不要删除 `.git/`、`.gitattributes` 或 `.gitmodules`。
- 不要绕过 Git LFS 手工重新添加大型权重文件。
- 不要从 GitHub 网页直接覆盖仓库中的实验结果。
- 不要把原始数据集、虚拟环境或缓存目录加入仓库。

## 十一、最短日常流程

开始工作：

```powershell
git status
git pull --rebase
git lfs pull
```

结束工作：

```powershell
git add -A
git status
git commit -m "exp(lab): describe this update"
git pull --rebase
git push
```

只要坚持“先拉取、唯一实验名、同一目录单机写入、结束后立即推送”，两台电脑之间通常不会产生结果冲突。
