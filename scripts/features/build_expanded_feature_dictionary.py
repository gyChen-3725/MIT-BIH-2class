"""生成347维扩展模型输入的Markdown与CSV特征字典。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.features.expanded import (
    ADVANCED_HRV_FEATURE_NAMES,
    ECG_FEATURE_NAMES,
    QRS_FEATURE_NAMES,
)

LEGACY_DICTIONARY = ROOT / "docs" / "features" / "handcrafted_feature_dictionary_263.csv"


def legacy_dictionary_rows(path: Path = LEGACY_DICTIONARY) -> list[dict[str, object]]:
    """从仓库内冻结的263维字典读取旧特征，避免运行时依赖相邻EDR仓库。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows: list[dict[str, object]] = list(csv.DictReader(stream))
    if len(rows) != 263:
        raise ValueError(f"旧特征字典应包含263行，实际为{len(rows)}行：{path}")
    for expected_index, row in enumerate(rows):
        index = int(str(row["index"]))
        if index != expected_index:
            raise ValueError(
                f"旧特征字典索引不连续：期望{expected_index}，实际{index}"
            )
        row["index"] = index
    return rows

ADVANCED_META = {
    "ectopic_fraction": ("窗口内非N型心搏占全部心搏的比例", "0–1"),
    "longest_normal_run_fraction": ("最长连续N型心搏段占窗口内心搏数的比例", "0–1"),
    "median_nn_seconds": ("有效NN间期中位数", "秒"),
    "iqr_nn_seconds": ("有效NN间期四分位距", "秒"),
    "mad_nn_seconds": ("有效NN间期相对中位数的绝对偏差中位数", "秒"),
    "min_nn_seconds": ("有效NN间期最小值", "秒"),
    "max_nn_seconds": ("有效NN间期最大值", "秒"),
    "nn_range_seconds": ("有效NN间期极差", "秒"),
    "cvnn": ("NN间期样本标准差除以均值", "无量纲"),
    "median_heart_rate_bpm": ("由有效NN间期换算的瞬时心率中位数", "次/分钟"),
    "minimum_heart_rate_bpm": ("有效瞬时心率最小值", "次/分钟"),
    "maximum_heart_rate_bpm": ("有效瞬时心率最大值", "次/分钟"),
    "heart_rate_range_bpm": ("有效瞬时心率极差", "次/分钟"),
    "heart_rate_slope_bpm_per_second": ("瞬时心率对窗口时间的一阶回归斜率", "次/分钟/秒"),
    "sdsd_seconds": ("连续有效NN间期差值的样本标准差", "秒"),
    "pnn20_fraction": ("相邻NN间期差绝对值超过20 ms的比例", "0–1"),
    "median_absolute_successive_difference_seconds": ("相邻NN间期绝对差中位数", "秒"),
    "successive_difference_iqr_seconds": ("相邻NN间期差的四分位距", "秒"),
    "acceleration_fraction": ("相邻NN间期缩短的比例", "0–1"),
    "deceleration_fraction": ("相邻NN间期延长的比例", "0–1"),
    "inflection_fraction": ("连续NN变化方向发生反转的比例", "0–1"),
    "poincare_sd1_seconds": ("Poincaré图垂直等值线方向的短时变异SD1", "秒"),
    "poincare_sd2_seconds": ("Poincaré图沿等值线方向的长期变异SD2", "秒"),
    "poincare_sd1_sd2_ratio": ("Poincaré SD1与SD2比值", "无量纲"),
}

ECG_META = {
    "finite_fraction": ("原始ECG有限值采样点比例", "0–1"),
    "bandpassed_mean": ("0.5–35 Hz ECG均值", "ECG物理单位"),
    "bandpassed_std": ("0.5–35 Hz ECG样本标准差", "ECG物理单位"),
    "bandpassed_rms": ("0.5–35 Hz ECG均方根", "ECG物理单位"),
    "bandpassed_median": ("0.5–35 Hz ECG中位数", "ECG物理单位"),
    "bandpassed_mad": ("0.5–35 Hz ECG绝对偏差中位数", "ECG物理单位"),
    "bandpassed_iqr": ("0.5–35 Hz ECG四分位距", "ECG物理单位"),
    "bandpassed_q05": ("0.5–35 Hz ECG的5%分位数", "ECG物理单位"),
    "bandpassed_q95": ("0.5–35 Hz ECG的95%分位数", "ECG物理单位"),
    "bandpassed_peak_to_peak": ("0.5–35 Hz ECG峰峰值", "ECG物理单位"),
    "bandpassed_skewness": ("0.5–35 Hz ECG偏度", "无量纲"),
    "bandpassed_kurtosis": ("0.5–35 Hz ECG Fisher峰度", "无量纲"),
    "line_length_per_second": ("相邻ECG差绝对值总和除以30秒", "ECG物理单位/秒"),
    "first_difference_rms": ("ECG一阶差分均方根", "ECG物理单位"),
    "second_difference_rms": ("ECG二阶差分均方根", "ECG物理单位"),
    "zero_crossings_per_second": ("带通ECG每秒过零次数", "次/秒"),
    "hjorth_mobility": ("一阶差分方差与信号方差比值的平方根", "无量纲"),
    "hjorth_complexity": ("一阶差分活动度与原信号活动度之比", "无量纲"),
    "histogram_entropy": ("32箱幅值直方图的归一化Shannon熵", "0–1"),
    "permutation_entropy_order3": ("50 Hz ECG的三阶归一化排列熵", "0–1"),
    "log_total_power_0p5_35": ("0.5–35 Hz总功率自然对数", "对数功率"),
    "log_power_0p5_5": ("0.5–5 Hz功率自然对数", "对数功率"),
    "log_power_5_15": ("5–15 Hz功率自然对数", "对数功率"),
    "log_power_15_35": ("15–35 Hz功率自然对数", "对数功率"),
    "relative_power_0p5_5": ("0.5–5 Hz功率占0.5–35 Hz总功率比例", "0–1"),
    "relative_power_5_15": ("5–15 Hz功率占0.5–35 Hz总功率比例", "0–1"),
    "relative_power_15_35": ("15–35 Hz功率占0.5–35 Hz总功率比例", "0–1"),
    "spectral_centroid_hz": ("0.5–35 Hz功率谱质心", "Hz"),
    "spectral_entropy": ("0.5–35 Hz归一化功率谱熵", "0–1"),
    "spectral_edge_90_hz": ("累计功率达到90%的频率", "Hz"),
    "dominant_frequency_hz": ("0.5–35 Hz最大谱功率频率", "Hz"),
    "baseline_wander_rms": ("0.5 Hz以下基线漂移均方根", "ECG物理单位"),
    "baseline_wander_fraction": ("基线漂移能量占基线与带通能量之和的比例", "0–1"),
    "flatline_fraction": ("一阶差分低于相对阈值的采样点比例", "0–1"),
}


def _qrs_meta(name: str) -> tuple[str, str]:
    explicit = {
        "valid_qrs_count": ("窗口内具有有效Q/R/S定位的正常QRS数量", "个"),
        "valid_qrs_fraction": ("有效QRS占窗口内正常心搏的比例", "0–1"),
    }
    if name in explicit:
        return explicit[name]
    statistics = {"median": "中位数", "iqr": "四分位距", "mad": "绝对偏差中位数"}
    quantities = {
        "qrs_duration": ("Q至S时限", "秒"),
        "qr_duration": ("Q至R时限", "秒"),
        "rs_duration": ("R至S时限", "秒"),
        "r_amplitude": ("极性校正R点幅度", "ECG物理单位"),
        "q_amplitude": ("极性校正Q点幅度", "ECG物理单位"),
        "s_amplitude": ("极性校正S点幅度", "ECG物理单位"),
        "qr_amplitude": ("R点减Q点幅度", "ECG物理单位"),
        "rs_amplitude": ("R点减S点幅度", "ECG物理单位"),
        "qrs_peak_to_peak": ("QRS片段峰峰值", "ECG物理单位"),
        "qrs_area": ("QRS相对Q-S连线的绝对面积", "ECG物理单位×秒"),
        "qrs_energy": ("QRS片段均方能量", "ECG物理单位²"),
        "qr_slope": ("Q-R段最大上升斜率", "ECG物理单位/秒"),
        "rs_slope": ("R-S段最大下降斜率绝对值", "ECG物理单位/秒"),
    }
    normalized_name = name.removesuffix("_seconds")
    for suffix, statistic in statistics.items():
        ending = f"_{suffix}"
        if normalized_name.endswith(ending):
            quantity = normalized_name[: -len(ending)]
            definition, unit = quantities[quantity]
            return f"窗口内正常心搏{definition}的{statistic}", unit
    raise KeyError(name)


def expanded_dictionary_rows() -> list[dict[str, object]]:
    rows = legacy_dictionary_rows()
    for prefix, names, metadata, category in (
        ("advanced_hrv", ADVANCED_HRV_FEATURE_NAMES, ADVANCED_META, "扩展节律/非线性HRV"),
        ("ecg", ECG_FEATURE_NAMES, ECG_META, "原始ECG统计与复杂度"),
    ):
        for name in names:
            definition, unit = metadata[name]
            rows.append({
                "feature": f"{prefix}.{name}",
                "source": prefix,
                "category": category,
                "definition": definition,
                "unit": unit,
                "context": "30秒",
            })
    for name in QRS_FEATURE_NAMES:
        definition, unit = _qrs_meta(name)
        rows.append({
            "feature": f"qrs.{name}",
            "source": "qrs",
            "category": "QRS形态分布",
            "definition": definition,
            "unit": unit,
            "context": "30秒",
        })
    for index, row in enumerate(rows):
        row["index"] = index
    if len(rows) != 347 or len({str(row["feature"]) for row in rows}) != 347:
        raise RuntimeError("扩展字典必须恰好包含347个唯一字段")
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# ECG扩展手工特征字典：模型输入347维",
        "",
        "旧263维EDR/HRV输入保持原顺序不变，随后追加24项扩展HRV、34项原始ECG和26项QRS形态特征。",
        "特征提取不读取分类标签，也不使用真实呼吸通道；所有窗口固定为30秒。",
        "",
        "## 维度构成",
        "",
        "| 特征族 | 维度 | 说明 |",
        "|---|---:|---|",
        "| 旧EDR模型输入 | 263 | 8项基础HRV＋10路EDR各25项＋5项跨路关系 |",
        "| 扩展HRV | 24 | 节律分布、心率趋势、Poincaré及变化方向 |",
        "| 原始ECG | 34 | 0.5–35 Hz时域、功率谱、熵、Hjorth及基线漂移 |",
        "| QRS形态 | 26 | QRS时限、幅值、面积、能量和斜率的稳健统计 |",
        "| 合计 | 347 | 旧维度可直接与扩展维度做消融 |",
        "",
        "30秒窗口不足以稳定估计常规LF/HF，因此本版有意不加入频域HRV。原始ECG采用整窗零相位滤波，",
        "适合当前离线实验，但不能直接宣称为实时因果特征。缺失值填补和标准化仍必须只在训练折拟合。",
        "",
        "## 完整索引",
        "",
        "| 索引 | 字段名 | 特征族 | 定义 | 单位/范围 | 时间范围 |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | `{row['feature']}` | {row['category']} | "
            f"{row['definition']} | {row['unit']} | {row['context']} |"
        )
    lines.extend([
        "",
        "## 代码位置",
        "",
        "- 特征函数：`src/mitbih_2class/features/expanded.py`",
        "- 全量提取：`scripts/features/extract_expanded_features.py`",
        "- 旧263维字典：`docs/features/手工特征字典_263维.md`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "features")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = expanded_dictionary_rows()
    markdown = output / "扩展手工特征字典_347维.md"
    csv_path = output / "expanded_feature_dictionary_347.csv"
    write_markdown(markdown, rows)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "index",
                "feature",
                "source",
                "category",
                "definition",
                "unit",
                "context",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {markdown}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
