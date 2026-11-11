"""把扩展手工特征的组合搜索结果渲染为可离线查看的单文件HTML仪表板。"""

from __future__ import annotations

import argparse
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

# 只生成文件，不依赖桌面图形环境。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "features" / "expanded347" / "selection_ml_grid_e3_v1"
DEFAULT_DAEAC_METRICS = (
    ROOT
    / "runs"
    / "daeac_filtered_rhythm_amplitude_e3_epoch5fold_2class"
    / "oof"
    / "metrics.json"
)

COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "slate": "#475569",
    "purple": "#7c3aed",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "macro_f1": "Macro F1",
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
}


def require_file(path: Path) -> Path:
    """在绘图前明确报告缺失产物，避免生成不完整页面。"""

    if not path.is_file():
        raise FileNotFoundError(f"缺少可视化输入：{path}")
    return path


def figure_svg(figure: plt.Figure) -> str:
    """将Matplotlib图直接嵌入HTML，保持单文件且无需外部图片。"""

    stream = io.StringIO()
    figure.savefig(stream, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    payload = stream.getvalue()
    return payload[payload.index("<svg") :]


def annotate_bars(axis: plt.Axes, bars, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_metric_overview(
    metrics: dict[str, object], baseline: dict[str, object] | None
) -> str:
    names = list(METRIC_LABELS)
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(11.5, 4.4))
    if baseline is None:
        bars = axis.bar(
            x,
            [float(metrics[name]) for name in names],
            color=COLORS["blue"],
            width=0.62,
            label="Feature selection + ML",
        )
        annotate_bars(axis, bars)
    else:
        width = 0.38
        selected_bars = axis.bar(
            x - width / 2,
            [float(metrics[name]) for name in names],
            width=width,
            color=COLORS["blue"],
            label="Feature selection + ML",
        )
        baseline_bars = axis.bar(
            x + width / 2,
            [float(baseline[name]) for name in names],
            width=width,
            color=COLORS["cyan"],
            label="DAEAC + rhythm/amplitude",
        )
        annotate_bars(axis, selected_bars)
        annotate_bars(axis, baseline_bars)
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel("Score")
    axis.set_xticks(x, [METRIC_LABELS[name] for name in names], rotation=18, ha="right")
    axis.set_title("OOF metric overview")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure_svg(figure)


def plot_confusion_matrix(metrics: dict[str, object]) -> str:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"二分类混淆矩阵应为2×2，实际为{matrix.shape}")
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(fractions, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(2):
        for column in range(2):
            color = "white" if fractions[row, column] > 0.55 else "#0f172a"
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{fractions[row, column]:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )
    axis.set_xticks([0, 1], ["Normal", "Apnea"])
    axis.set_yticks([0, 1], ["Normal", "Apnea"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("OOF confusion matrix · row-normalized")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure_svg(figure)


def plot_fold_performance(fold_best: pd.DataFrame) -> str:
    frame = fold_best.sort_values("fold")
    x = np.arange(len(frame))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    validation = axis.bar(
        x - width / 2,
        frame["val_accuracy"],
        width,
        label="Validation accuracy",
        color=COLORS["amber"],
    )
    testing = axis.bar(
        x + width / 2,
        frame["test_accuracy"],
        width,
        label="Test accuracy",
        color=COLORS["blue"],
    )
    annotate_bars(axis, validation)
    annotate_bars(axis, testing)
    lower = max(0.0, float(min(frame["val_accuracy"].min(), frame["test_accuracy"].min())) - 0.04)
    upper = min(1.0, float(max(frame["val_accuracy"].max(), frame["test_accuracy"].max())) + 0.045)
    axis.set_ylim(lower, upper)
    axis.set_xticks(x, [f"Fold {int(value)}" for value in frame["fold"]])
    axis.set_ylabel("Accuracy · zoomed axis")
    axis.set_title("Validation selection and held-out test performance")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="lower left")
    figure.tight_layout()
    return figure_svg(figure)


def group_screen_summary(validation_grid: pd.DataFrame) -> pd.DataFrame:
    screened = validation_grid.loc[validation_grid["stage"] == "group_screen"]
    return (
        screened.groupby(["feature_group", "model"], as_index=False)
        .agg(
            mean_val_accuracy=("val_accuracy", "mean"),
            mean_val_macro_f1=("val_macro_f1", "mean"),
            mean_val_roc_auc=("val_roc_auc", "mean"),
        )
    )


def plot_group_screen(summary: pd.DataFrame) -> str:
    group_order = (
        summary.groupby("feature_group")["mean_val_accuracy"]
        .max()
        .sort_values()
        .index.tolist()
    )
    model_styles = {
        "logistic": (COLORS["slate"], "o"),
        "random_forest": (COLORS["blue"], "s"),
        "hist_gb": (COLORS["green"], "^"),
    }
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    y = np.arange(len(group_order))
    offsets = np.linspace(-0.2, 0.2, len(model_styles))
    for offset, (model, (color, marker)) in zip(offsets, model_styles.items(), strict=True):
        subset = summary.loc[summary["model"] == model].set_index("feature_group")
        values = [float(subset.loc[group, "mean_val_accuracy"]) for group in group_order]
        axis.scatter(
            values,
            y + offset,
            label=model,
            color=color,
            marker=marker,
            s=46,
        )
    axis.set_yticks(y, group_order)
    axis.set_xlabel("Mean validation accuracy across five folds")
    axis.set_title("Full feature-group screening")
    axis.grid(axis="x", alpha=0.22)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure_svg(figure)


def plot_group_screen_by_fold(validation_grid: pd.DataFrame) -> str:
    """展示每折第一阶段结果，并用边框标出进入第二阶段的前三个特征组。"""

    screened = validation_grid.loc[validation_grid["stage"] == "group_screen"].copy()
    best_indices = screened.groupby(["fold", "feature_group"])["val_accuracy"].idxmax()
    best = screened.loc[best_indices]
    group_order = (
        best.groupby("feature_group")["val_accuracy"].mean().sort_values().index.tolist()
    )
    folds = sorted(int(value) for value in best["fold"].unique())
    values = np.full((len(group_order), len(folds)), np.nan)
    models = np.full((len(group_order), len(folds)), "", dtype=object)
    for row in best.itertuples(index=False):
        group_index = group_order.index(str(row.feature_group))
        fold_index = folds.index(int(row.fold))
        values[group_index, fold_index] = float(row.val_accuracy)
        models[group_index, fold_index] = {
            "logistic": "LR",
            "random_forest": "RF",
            "hist_gb": "HGB",
        }[str(row.model)]
    refined = validation_grid.loc[
        validation_grid["stage"] == "within_group_selection", ["fold", "feature_group"]
    ].drop_duplicates()
    refined_pairs = {
        (int(row.fold), str(row.feature_group)) for row in refined.itertuples(index=False)
    }

    figure, axis = plt.subplots(figsize=(8.7, 6.3))
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    image = axis.imshow(values, cmap="YlGnBu", aspect="auto", vmin=lower, vmax=upper)
    midpoint = (lower + upper) / 2
    for group_index, group in enumerate(group_order):
        for fold_index, fold in enumerate(folds):
            value = values[group_index, fold_index]
            axis.text(
                fold_index,
                group_index,
                f"{value:.3f}\n{models[group_index, fold_index]}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > midpoint else "#0f172a",
            )
            if (fold, group) in refined_pairs:
                axis.add_patch(
                    Rectangle(
                        (fold_index - 0.47, group_index - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor=COLORS["red"],
                        linewidth=2.2,
                    )
                )
    axis.set_xticks(range(len(folds)), [f"Fold {fold}" for fold in folds])
    axis.set_yticks(range(len(group_order)), group_order)
    axis.set_xlabel("Best model within each full feature group")
    axis.set_title("Stage 1 · group screening by fold")
    axis.legend(
        handles=[Patch(facecolor="none", edgecolor=COLORS["red"], linewidth=2, label="Top 3 → Stage 2")],
        loc="lower right",
    )
    figure.colorbar(image, ax=axis, label="Validation accuracy", fraction=0.035, pad=0.03)
    figure.tight_layout()
    return figure_svg(figure)


def plot_refined_search_by_fold(
    validation_grid: pd.DataFrame, fold_best: pd.DataFrame
) -> str:
    """把第二阶段压缩为每折的选择器×维度热图，红框表示最终选中的单元格。"""

    refined = validation_grid.loc[validation_grid["stage"] == "within_group_selection"].copy()
    dimensions = sorted(int(value) for value in refined["dimension"].unique())
    selector_order = ["anova", "mutual_info", "l1", "random_forest", "mrmr"]
    folds = sorted(int(value) for value in fold_best["fold"].unique())
    lower = float(refined["val_accuracy"].min())
    upper = float(refined["val_accuracy"].max())
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True, sharey=True)
    image = None
    for axis, fold in zip(axes.flat, folds, strict=False):
        subset = refined.loc[refined["fold"] == fold]
        best_indices = subset.groupby(["selector", "dimension"])["val_accuracy"].idxmax()
        cell_best = subset.loc[best_indices]
        values = np.full((len(selector_order), len(dimensions)), np.nan)
        for row in cell_best.itertuples(index=False):
            selector_index = selector_order.index(str(row.selector))
            dimension_index = dimensions.index(int(row.dimension))
            values[selector_index, dimension_index] = float(row.val_accuracy)
        masked = np.ma.masked_invalid(values)
        image = axis.imshow(masked, cmap="PuBuGn", aspect="auto", vmin=lower, vmax=upper)
        for selector_index in range(len(selector_order)):
            for dimension_index in range(len(dimensions)):
                value = values[selector_index, dimension_index]
                if np.isfinite(value):
                    axis.text(
                        dimension_index,
                        selector_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > (lower + upper) / 2 else "#0f172a",
                    )
        chosen = fold_best.loc[fold_best["fold"] == fold].iloc[0]
        chosen_selector = str(chosen["selector"])
        chosen_dimension = int(chosen["dimension"])
        if chosen_selector in selector_order and chosen_dimension in dimensions:
            axis.add_patch(
                Rectangle(
                    (
                        dimensions.index(chosen_dimension) - 0.47,
                        selector_order.index(chosen_selector) - 0.47,
                    ),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor=COLORS["red"],
                    linewidth=2.4,
                )
            )
            subtitle = f"winner: {chosen_selector} / {chosen_dimension}"
        else:
            subtitle = "winner: full group / no column selection"
        axis.set_title(f"Fold {fold}\n{subtitle}", fontsize=10)
        axis.set_xticks(range(len(dimensions)), dimensions, rotation=45)
        axis.set_yticks(range(len(selector_order)), selector_order)
    axes.flat[-1].axis("off")
    if image is not None:
        figure.colorbar(
            image,
            ax=[axis for axis in axes.flat[:-1]],
            label="Best validation accuracy in cell",
            fraction=0.025,
            pad=0.025,
        )
    figure.suptitle("Stage 2 · selector × dimension search by fold", y=0.98)
    figure.subplots_adjust(left=0.08, right=0.91, bottom=0.09, top=0.90, wspace=0.18, hspace=0.36)
    return figure_svg(figure)


def plot_dimension_trend(validation_grid: pd.DataFrame) -> str:
    refined = validation_grid.loc[validation_grid["stage"] == "within_group_selection"].copy()
    trend = (
        refined.groupby(["selector", "dimension"], as_index=False)["val_accuracy"]
        .mean()
        .sort_values("dimension")
    )
    colors = [
        COLORS["blue"],
        COLORS["cyan"],
        COLORS["purple"],
        COLORS["green"],
        COLORS["amber"],
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for color, (selector, rows) in zip(colors, trend.groupby("selector"), strict=True):
        axis.plot(
            rows["dimension"],
            rows["val_accuracy"],
            marker="o",
            linewidth=2,
            color=color,
            label=selector,
        )
    axis.set_xlabel("Selected feature dimension")
    axis.set_ylabel("Mean validation accuracy")
    axis.set_title("Dimension trend by selector · exploratory mean")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, loc="best")
    figure.tight_layout()
    return figure_svg(figure)


def plot_feature_frequency(frequency: pd.DataFrame, top_n: int) -> str:
    top = frequency.head(top_n).sort_values("selected_fold_count")
    figure, axis = plt.subplots(figsize=(9.4, max(5.0, top_n * 0.29)))
    bars = axis.barh(
        top["feature"],
        top["selected_fold_count"],
        color=COLORS["purple"],
        alpha=0.88,
    )
    axis.bar_label(bars, padding=3, fontsize=8)
    axis.set_xlim(0, max(5.5, float(top["selected_fold_count"].max()) + 0.7))
    axis.set_xlabel("Number of folds selected")
    axis.set_title(f"Top {top_n} repeatedly selected features")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    return figure_svg(figure)


def fold_table(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.sort_values("fold").itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.fold)}</td>"
            f"<td><code>{html.escape(str(row.feature_group))}</code></td>"
            f"<td><code>{html.escape(str(row.selector))}</code></td>"
            f"<td>{int(row.dimension)}</td>"
            f"<td><code>{html.escape(str(row.model))}</code></td>"
            f"<td>{float(row.val_accuracy):.4f}</td>"
            f"<td>{float(row.test_accuracy):.4f}</td>"
            f"<td>{float(row.test_macro_f1):.4f}</td>"
            f"<td>{float(row.test_roc_auc):.4f}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Fold</th><th>Feature group</th><th>Selector</th><th>Dimension</th>"
        "<th>Model</th><th>Val accuracy</th><th>Test accuracy</th>"
        "<th>Test macro F1</th><th>Test AUC</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def selection_funnel(validation_grid: pd.DataFrame, fold_best: pd.DataFrame) -> str:
    """用每折真实候选数量展示从特征组初筛到唯一方案的收缩过程。"""

    rows = []
    for fold in sorted(int(value) for value in fold_best["fold"].unique()):
        current = validation_grid.loc[validation_grid["fold"] == fold]
        screened = current.loc[current["stage"] == "group_screen"]
        refined = current.loc[current["stage"] == "within_group_selection"]
        group_count = int(screened["feature_group"].nunique())
        refined_group_count = int(refined["feature_group"].nunique())
        chosen = fold_best.loc[fold_best["fold"] == fold].iloc[0]
        winner = (
            f"{chosen['feature_group']} · {chosen['selector']} · "
            f"{int(chosen['dimension'])}D · {chosen['model']}"
        )
        rows.append(
            "<div class='funnel-row'>"
            f"<b>Fold {fold}</b>"
            f"<span class='stage'>{group_count} groups</span><i>→</i>"
            f"<span class='stage'>{len(screened)} full-group candidates</span><i>→</i>"
            f"<span class='stage accent'>{refined_group_count} groups enter Stage 2</span><i>→</i>"
            f"<span class='stage'>{len(refined)} refined candidates</span><i>→</i>"
            f"<span class='winner'>{html.escape(winner)}</span>"
            "</div>"
        )
    return "<div class='funnel-grid'>" + "".join(rows) + "</div>"


def _best_validation_candidate(frame: pd.DataFrame) -> pd.Series:
    """复用实验的验证指标优先级，为赛程图选择每个节点的代表方案。"""

    return frame.sort_values(
        [
            "val_accuracy",
            "val_macro_f1",
            "val_balanced_accuracy",
            "val_roc_auc",
            "dimension",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]


def _svg_node(
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    detail: str,
    fill: str,
    stroke: str,
    title_color: str = "#0f172a",
) -> str:
    """生成赛程图中的两行圆角节点。"""

    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
        f'<text x="{x + 12}" y="{y + 15}" font-size="11" font-weight="700" '
        f'fill="{title_color}">{html.escape(title)}</text>'
        f'<text x="{x + 12}" y="{y + 29}" font-size="9.5" fill="#64748b">'
        f"{html.escape(detail)}</text>"
    )


def tournament_brackets(validation_grid: pd.DataFrame, fold_best: pd.DataFrame) -> str:
    """生成五折赛程式晋级图：11个特征组→3个精筛组→1个最终方案。"""

    model_short = {"logistic": "LR", "random_forest": "RF", "hist_gb": "HGB"}
    bracket_cards = []
    for fold in sorted(int(value) for value in fold_best["fold"].unique()):
        current = validation_grid.loc[validation_grid["fold"] == fold]
        screened = current.loc[current["stage"] == "group_screen"]
        refined = current.loc[current["stage"] == "within_group_selection"]
        advanced_groups = set(refined["feature_group"].astype(str))
        chosen = fold_best.loc[fold_best["fold"] == fold].iloc[0]
        chosen_group = str(chosen["feature_group"])

        stage1_rows = []
        for feature_group, rows in screened.groupby("feature_group"):
            candidate = _best_validation_candidate(rows)
            stage1_rows.append(
                {
                    "feature_group": str(feature_group),
                    "model": str(candidate["model"]),
                    "accuracy": float(candidate["val_accuracy"]),
                }
            )
        stage1_rows.sort(key=lambda row: row["accuracy"], reverse=True)

        stage2_rows = []
        for feature_group in advanced_groups:
            candidate = _best_validation_candidate(
                refined.loc[refined["feature_group"] == feature_group]
            )
            stage2_rows.append(
                {
                    "feature_group": feature_group,
                    "selector": str(candidate["selector"]),
                    "dimension": int(candidate["dimension"]),
                    "model": str(candidate["model"]),
                    "accuracy": float(candidate["val_accuracy"]),
                }
            )
        stage2_rows.sort(key=lambda row: row["accuracy"], reverse=True)

        width, height = 1200, 550
        first_x, first_width, node_height = 30, 330, 36
        second_x, second_width = 485, 405
        champion_x, champion_width = 965, 215
        first_y = {row["feature_group"]: 50 + rank * 43 for rank, row in enumerate(stage1_rows)}
        second_y = {
            row["feature_group"]: 105 + rank * 155 for rank, row in enumerate(stage2_rows)
        }
        champion_y = 230
        elements = [
            (
                f'<svg viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="Fold {fold} feature selection tournament bracket">'
            ),
            (
                '<text x="30" y="25" font-size="13" font-weight="700" fill="#334155">'
                "ROUND 1 · FULL FEATURE GROUP SCREEN</text>"
            ),
            (
                '<text x="485" y="25" font-size="13" font-weight="700" fill="#334155">'
                "ROUND 2 · SELECTOR / DIMENSION / MODEL</text>"
            ),
            (
                '<text x="965" y="25" font-size="13" font-weight="700" fill="#334155">'
                "FOLD WINNER</text>"
            ),
        ]

        # 先画连线，保证节点覆盖在线条上方。
        for row in stage2_rows:
            group = row["feature_group"]
            source_center = first_y[group] + node_height / 2
            target_center = second_y[group] + 25
            color = COLORS["green"] if group == chosen_group else "#60a5fa"
            elements.append(
                f'<path d="M {first_x + first_width} {source_center} H 420 '
                f'V {target_center} H {second_x}" fill="none" stroke="{color}" '
                'stroke-width="2.2"/>'
            )
            elements.append(
                f'<path d="M {second_x + second_width} {target_center} H 925 '
                f'V {champion_y + 31} H {champion_x}" fill="none" stroke="{color}" '
                'stroke-width="2.2"/>'
            )

        for rank, row in enumerate(stage1_rows, start=1):
            group = row["feature_group"]
            advanced = group in advanced_groups
            winner = group == chosen_group
            fill = "#ecfdf5" if winner else "#eff6ff" if advanced else "#f8fafc"
            stroke = COLORS["green"] if winner else "#60a5fa" if advanced else "#cbd5e1"
            elements.append(
                _svg_node(
                    first_x,
                    first_y[group],
                    first_width,
                    node_height,
                    f"#{rank}  {group}",
                    f"best {model_short[row['model']]} · val accuracy {row['accuracy']:.4f}",
                    fill,
                    stroke,
                )
            )

        for row in stage2_rows:
            group = row["feature_group"]
            winner = group == chosen_group
            elements.append(
                _svg_node(
                    second_x,
                    second_y[group],
                    second_width,
                    50,
                    group,
                    (
                        f"{row['selector']} · {row['dimension']}D · "
                        f"{model_short[row['model']]} · best val {row['accuracy']:.4f}"
                    ),
                    "#ecfdf5" if winner else "#eff6ff",
                    COLORS["green"] if winner else "#60a5fa",
                )
            )

        elements.append(
            _svg_node(
                champion_x,
                champion_y,
                champion_width,
                62,
                chosen_group,
                (
                    f"{chosen['selector']} · {int(chosen['dimension'])}D · "
                    f"{model_short[str(chosen['model'])]}"
                ),
                "#dcfce7",
                COLORS["green"],
                "#065f46",
            )
        )
        elements.append(
            f'<text x="{champion_x + 12}" y="{champion_y + 46}" font-size="9.5" '
            f'fill="#047857">val {float(chosen["val_accuracy"]):.4f} · '
            f'test {float(chosen["test_accuracy"]):.4f}</text>'
        )
        elements.extend(
            [
                '<rect x="30" y="528" width="14" height="10" rx="2" fill="#f8fafc" stroke="#cbd5e1"/>',
                '<text x="51" y="537" font-size="9.5" fill="#64748b">eliminated after Stage 1</text>',
                '<rect x="205" y="528" width="14" height="10" rx="2" fill="#eff6ff" stroke="#60a5fa"/>',
                '<text x="226" y="537" font-size="9.5" fill="#64748b">advanced to Stage 2</text>',
                '<rect x="373" y="528" width="14" height="10" rx="2" fill="#ecfdf5" stroke="#059669"/>',
                '<text x="394" y="537" font-size="9.5" fill="#64748b">winning path</text>',
                "</svg>",
            ]
        )
        champion_label = (
            f"{chosen_group} · {chosen['selector']} · {int(chosen['dimension'])}D · "
            f"{chosen['model']}"
        )
        bracket_cards.append(
            f"<details class='bracket-card' {'open' if fold == 0 else ''}>"
            f"<summary>Fold {fold} champion · {html.escape(champion_label)}</summary>"
            + "".join(elements)
            + "</details>"
        )
    return "<div class='bracket-list'>" + "".join(bracket_cards) + "</div>"


def best_group_table(summary: pd.DataFrame) -> str:
    ordered = summary.sort_values("mean_val_accuracy", ascending=False)
    best = ordered.groupby("feature_group", as_index=False).first()
    best = best.sort_values("mean_val_accuracy", ascending=False)
    rows = [
        "<tr>"
        f"<td><code>{html.escape(str(row.feature_group))}</code></td>"
        f"<td><code>{html.escape(str(row.model))}</code></td>"
        f"<td>{float(row.mean_val_accuracy):.4f}</td>"
        f"<td>{float(row.mean_val_macro_f1):.4f}</td>"
        f"<td>{float(row.mean_val_roc_auc):.4f}</td>"
        "</tr>"
        for row in best.itertuples(index=False)
    ]
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Feature group</th><th>Best full-dimension model</th>"
        "<th>Mean val accuracy</th><th>Mean val macro F1</th><th>Mean val AUC</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_dashboard(
    result_dir: Path,
    metrics: dict[str, object],
    fold_best: pd.DataFrame,
    validation_grid: pd.DataFrame,
    frequency: pd.DataFrame,
    manifest: dict[str, object],
    baseline: dict[str, object] | None,
    top_n: int,
) -> str:
    summary = group_screen_summary(validation_grid)
    metric_cards = [
        ("Accuracy", float(metrics["accuracy"]), "blue"),
        ("Balanced accuracy", float(metrics["balanced_accuracy"]), "cyan"),
        ("Macro F1", float(metrics["macro_f1"]), "green"),
        ("ROC AUC", float(metrics["roc_auc"]), "purple"),
        ("Sensitivity", float(metrics["sensitivity"]), "amber"),
        ("Specificity", float(metrics["specificity"]), "blue"),
    ]
    cards = "".join(
        f"<div class='metric-card {color}'><span>{label}</span><strong>{value:.2%}</strong></div>"
        for label, value, color in metric_cards
    )
    selected_dimensions = ", ".join(
        str(int(value)) for value in fold_best.sort_values("fold")["dimension"]
    )
    generation_time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    baseline_note = (
        "指标总览同时显示现有 DAEAC＋节律/幅度五折结果。"
        if baseline is not None
        else "未提供DAEAC指标，本页仅显示特征选择＋机器学习结果。"
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>347维扩展手工特征 · 选择结果仪表板</title>
<style>
:root {{ color-scheme: light; --ink:#0f172a; --muted:#64748b; --line:#dbe4ef; --panel:#fff; --bg:#f4f7fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif; line-height:1.65; }}
.shell {{ width:min(1480px, calc(100% - 36px)); margin:28px auto 56px; }}
header {{ padding:30px 34px; border-radius:20px; background:linear-gradient(135deg,#0f172a,#1d4ed8); color:white; box-shadow:0 18px 48px #1e3a8a26; }}
header h1 {{ margin:0 0 8px; font-size:clamp(26px,3vw,40px); }}
header p {{ margin:4px 0; color:#dbeafe; }}
.metrics {{ display:grid; grid-template-columns:repeat(6,minmax(150px,1fr)); gap:14px; margin:20px 0; }}
.metric-card {{ background:var(--panel); border:1px solid var(--line); border-top:4px solid #2563eb; border-radius:14px; padding:15px 17px; box-shadow:0 8px 24px #0f172a0b; }}
.metric-card span {{ display:block; color:var(--muted); font-size:13px; }}
.metric-card strong {{ display:block; margin-top:4px; font-size:26px; }}
.metric-card.cyan {{ border-top-color:#0891b2; }} .metric-card.green {{ border-top-color:#059669; }}
.metric-card.amber {{ border-top-color:#d97706; }} .metric-card.purple {{ border-top-color:#7c3aed; }}
.grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; margin-top:18px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:20px; box-shadow:0 8px 24px #0f172a0b; overflow:hidden; }}
.panel.wide {{ grid-column:1 / -1; }}
.panel h2 {{ margin:0 0 5px; font-size:21px; }} .panel h3 {{ margin:22px 0 8px; }}
.hint {{ color:var(--muted); margin:0 0 12px; font-size:14px; }}
.chart svg {{ display:block; width:100%; height:auto; }}
.flow {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:16px 0; }}
.flow div {{ position:relative; padding:15px; border-radius:12px; background:#eff6ff; border:1px solid #bfdbfe; }}
.flow strong {{ display:block; color:#1d4ed8; }}
.funnel-grid {{ display:grid; gap:10px; }}
.funnel-row {{ display:grid; grid-template-columns:70px 110px 20px 175px 20px 175px 20px 165px 20px minmax(260px,1fr); align-items:center; gap:7px; }}
.funnel-row b {{ color:#1e3a8a; }} .funnel-row i {{ color:#94a3b8; font-style:normal; text-align:center; }}
.stage,.winner {{ padding:8px 10px; border-radius:9px; border:1px solid #dbe4ef; background:#f8fafc; font-size:12px; text-align:center; }}
.stage.accent {{ border-color:#fca5a5; background:#fef2f2; color:#991b1b; }}
.winner {{ background:#ecfdf5; border-color:#6ee7b7; color:#065f46; text-align:left; }}
.bracket-list {{ display:grid; gap:12px; }}
.bracket-card {{ border:1px solid var(--line); border-radius:12px; background:#fff; overflow:hidden; }}
.bracket-card summary {{ cursor:pointer; padding:13px 16px; background:#f8fafc; color:#1e3a8a; font-weight:700; }}
.bracket-card svg {{ display:block; width:100%; height:auto; padding:8px 12px 14px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th {{ position:sticky; top:0; background:#f8fafc; color:#334155; }}
th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(5),td:nth-child(5) {{ text-align:left; }}
code {{ color:#4338ca; background:#eef2ff; border-radius:5px; padding:2px 5px; }}
.callout {{ border-left:4px solid #d97706; background:#fffbeb; padding:14px 17px; border-radius:9px; }}
footer {{ color:var(--muted); text-align:center; margin-top:22px; font-size:13px; }}
@media (max-width:1200px) {{ .funnel-row {{ grid-template-columns:70px 1fr 20px 1.4fr 20px 1.4fr; }} .funnel-row .stage:nth-of-type(4),.funnel-row .winner {{ grid-column:2 / -1; }} }}
@media (max-width:1000px) {{ .metrics {{ grid-template-columns:repeat(3,1fr); }} .grid {{ grid-template-columns:1fr; }} .panel.wide {{ grid-column:auto; }} .flow {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width:620px) {{ .shell {{ width:min(100% - 18px,1480px); }} header {{ padding:22px; }} .metrics {{ grid-template-columns:1fr 1fr; }} .flow {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main class="shell">
<header>
  <h1>347维扩展手工特征 · 选择结果仪表板</h1>
  <p>10,197个30秒窗口 · 冻结样本级五折 · 验证accuracy选优 · 测试集每折只评价一次</p>
  <p>生成时间：{html.escape(generation_time)}</p>
</header>
<section class="metrics">{cards}</section>
<section class="grid">
  <article class="panel wide">
    <h2>阅读顺序与评价边界</h2>
    <p class="hint">本页先看最终OOF，再看五折波动和选择过程。{html.escape(baseline_note)}</p>
    <div class="flow">
      <div><strong>1 · Train</strong>拟合填补、特征排名和候选分类器</div>
      <div><strong>2 · Validation</strong>比较特征组、选择器、维度和模型</div>
      <div><strong>3 · Refit</strong>方案固定后使用Train＋Validation重新拟合</div>
      <div><strong>4 · Test</strong>只评价一次，汇总为OOF指标</div>
    </div>
    <div class="callout">五折最终维度依次为 <strong>{html.escape(selected_dimensions)}</strong>。维度趋势图是候选方案的探索性均值，不能越过每折验证选择，直接按测试结果重新挑一个“全局最佳维度”。</div>
  </article>
  <article class="panel wide"><h2>总体指标</h2><p class="hint">纵轴固定为0～1，避免放大细小差异。</p><div class="chart">{plot_metric_overview(metrics, baseline)}</div></article>
  <article class="panel"><h2>OOF混淆矩阵</h2><p class="hint">同时给出样本数和按真实类别归一化比例。</p><div class="chart">{plot_confusion_matrix(metrics)}</div></article>
  <article class="panel"><h2>五折稳定性</h2><p class="hint">纵轴局部放大，用于观察折间波动；具体数值见下表。</p><div class="chart">{plot_fold_performance(fold_best)}</div></article>
  <article class="panel wide"><h2>每折最终方案</h2>{fold_table(fold_best)}</article>
  <article class="panel wide"><h2>实际候选漏斗</h2><p class="hint">每折先比较11个完整特征组，再仅让验证表现前三的组进入列级搜索，最终保留一个方案。</p>{selection_funnel(validation_grid, fold_best)}</article>
  <article class="panel wide"><h2>五折特征选择赛程图</h2><p class="hint">左侧11个特征组参加初筛，蓝色路径晋级第二阶段，绿色路径最终夺冠。点击各折标题展开或收起赛程。</p>{tournament_brackets(validation_grid, fold_best)}</article>
  <article class="panel wide"><h2>第一阶段：特征组如何进入下一轮</h2><p class="hint">单元格为该折、该特征组表现最好的模型及验证accuracy；红框表示进入第二阶段的前三组。</p><div class="chart">{plot_group_screen_by_fold(validation_grid)}</div></article>
  <article class="panel wide"><h2>第二阶段：选择器与维度如何胜出</h2><p class="hint">每个单元格取该折对应选择器和维度下、跨特征组与模型的最高验证accuracy；红框表示最终方案。Fold 3由第一阶段的完整92维组合直接胜出。</p><div class="chart">{plot_refined_search_by_fold(validation_grid, fold_best)}</div></article>
  <article class="panel"><h2>完整特征组合初筛</h2><p class="hint">每个点是同一特征组、同一模型在五折验证集上的平均accuracy。</p><div class="chart">{plot_group_screen(summary)}</div></article>
  <article class="panel"><h2>选择器与维度趋势</h2><p class="hint">只统计进入第二阶段的组合，因此用于趋势观察，不是公平的全局排行榜。</p><div class="chart">{plot_dimension_trend(validation_grid)}</div></article>
  <article class="panel wide"><h2>各特征组最佳完整维度模型</h2><p class="hint">每个特征组只保留三类模型中平均验证accuracy最高者。</p>{best_group_table(summary)}</article>
  <article class="panel wide"><h2>跨折高频特征</h2><p class="hint">计数表示该特征出现在多少折的最终输入中；折内仍需独立选择，不能把该汇总直接当成无泄漏的全局掩码。</p><div class="chart">{plot_feature_frequency(frequency, top_n)}</div></article>
  <article class="panel wide">
    <h2>数据来源与复现信息</h2>
    <ul>
      <li>结果目录：<code>{html.escape(str(result_dir))}</code></li>
      <li>特征文件：<code>{html.escape(str(manifest.get("feature_file", "unknown")))}</code></li>
      <li>选择指标：<code>{html.escape(str(manifest.get("selection_metric", "unknown")))}</code></li>
      <li>选择器：<code>{html.escape(", ".join(map(str, manifest.get("selectors", []))))}</code></li>
      <li>模型：<code>{html.escape(", ".join(map(str, manifest.get("models", []))))}</code></li>
    </ul>
  </article>
</section>
<footer>Standalone HTML · all charts embedded · no network dependency</footer>
</main></body></html>"""
    return html_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--daeac-metrics",
        type=Path,
        default=DEFAULT_DAEAC_METRICS,
        help="可选的DAEAC OOF指标，用于同图对照；文件不存在时自动跳过",
    )
    parser.add_argument("--top-features", type=int, default=25)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    output = args.output.resolve() if args.output else result_dir / "feature_selection_dashboard.html"
    if args.top_features < 1:
        raise ValueError("top-features必须是正整数")
    metrics = json.loads(
        require_file(result_dir / "oof_metrics.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        require_file(result_dir / "manifest.json").read_text(encoding="utf-8")
    )
    fold_best = pd.read_csv(require_file(result_dir / "fold_best.csv"))
    validation_grid = pd.read_csv(require_file(result_dir / "validation_grid.csv"))
    frequency = pd.read_csv(require_file(result_dir / "selected_feature_frequency.csv"))
    baseline = None
    if args.daeac_metrics.is_file():
        baseline = json.loads(args.daeac_metrics.read_text(encoding="utf-8"))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_dashboard(
            result_dir,
            metrics,
            fold_best,
            validation_grid,
            frequency,
            manifest,
            baseline,
            min(args.top_features, len(frequency)),
        ),
        encoding="utf-8",
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
