# 手工特征产物

当前二分类实验的冻结347维输入位于 `expanded347/`：

- `features_30s_expanded.npz`：10,197个30秒窗口 × 347维；
- `manifest.json`：源文件、字段名称、维度和有限值统计。

347维由旧263维EDR/HRV输入和本仓库新增的84维特征组成：24维扩展HRV、
34维原始ECG统计与频谱、26维QRS形态。完整定义见
[`docs/features/扩展手工特征字典_347维.md`](../docs/features/扩展手工特征字典_347维.md)。

重新生成时，从仓库根目录运行 `scripts/features/extract_expanded_features.py`。
脚本默认自动识别两种布局：本机的相邻 `Projects/EDR`，以及服务器上的同级
`D:\CGY\EDR`。EDR只提供旧263维文件和逐记录Q/R/S定位；扩展特征实现、冻结产物、
选择流程及下游模型配置均由本仓库维护。

特征组合与选择仅规划、不拟合模型：

```powershell
python -B scripts\selection\optimize_feature_dimension.py --plan-only
```

正式组合搜索会拟合机器学习模型，必须由用户显式启动。
