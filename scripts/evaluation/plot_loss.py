"""根据训练历史绘制单折或五折的训练集与验证集 loss 曲线。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# 服务器通常没有图形桌面；在导入 pyplot 前固定使用文件渲染后端。
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mitbih_2class.config import default_config_path, load_config


def load_fold(run_dir: Path) -> tuple[list[dict[str, float | int]], int, str]:
    """读取历史、checkpoint 轮次及其选择指标，并检查绘图字段。"""

    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"训练历史为空: {run_dir}")
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError(f"训练 epoch 不连续: {run_dir}")
    best_epoch = int(metrics["best_epoch"])
    if best_epoch not in epochs:
        raise ValueError(f"最佳 checkpoint epoch 不在训练历史中: {run_dir}")
    for row in history:
        float(row["train_loss"])
        float(row["val_loss"])
    # 旧结果尚未记录选择指标，当时统一按 macro F1 选 checkpoint。
    checkpoint_metric = str(metrics.get("checkpoint_metric", "val_macro_f1"))
    return history, best_epoch, checkpoint_metric


def draw_fold(
    ax: plt.Axes, history: list[dict[str, float | int]], best_epoch: int, checkpoint_metric: str
) -> None:
    """使用统一的颜色和标记绘制一折，方便跨折比较。"""

    epochs = [int(row["epoch"]) for row in history]
    ax.plot(epochs, [float(row["train_loss"]) for row in history], label="Train loss")
    ax.plot(epochs, [float(row["val_loss"]) for row in history], label="Validation loss")
    # 最佳轮次按配置中的验证指标选择，不一定落在验证 loss 最低处。
    metric_label = "accuracy" if checkpoint_metric == "val_accuracy" else "macro F1"
    ax.axvline(
        best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best {metric_label} epoch"
    )
    ax.set_xlabel("Epoch")
    # 有些模型会在交叉熵之外加入重建损失，纵轴统一标为实际训练目标。
    ax.set_ylabel("Objective loss")
    ax.grid(alpha=0.25)


def save_fold_curve(
    run_dir: Path,
    fold: int,
    history: list[dict[str, float | int]],
    best_epoch: int,
    checkpoint_metric: str,
) -> Path:
    """把一折的 loss 曲线保存到该折目录，供单折筛选和五折查看共用。"""

    figure, ax = plt.subplots(figsize=(8, 4.5))
    draw_fold(ax, history, best_epoch, checkpoint_metric)
    ax.set_title(f"Fold {fold} loss")
    ax.legend(loc="best")
    figure.tight_layout()
    output_path = run_dir / "loss_curve.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    """指定 fold 时仅画一折；否则画五折各图及总览图。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--run-root", type=Path, help="指定已完成实验的结果目录")
    args = parser.parse_args()
    config = load_config(args.config)
    # 服务器结果复制到本机时，目录名可能与本机 YAML 的实验名不同。
    # 显式指定结果目录即可直接读取其中的 history.json 和 metrics.json。
    run_root = args.run_root.resolve() if args.run_root else config.output_root / config.name

    if args.fold is not None:
        # 单折实验不要求其他四折存在，也不生成可能误导为完整 OOF 的总览图。
        run_dir = run_root / f"fold_{args.fold}"
        history, best_epoch, checkpoint_metric = load_fold(run_dir)
        print(save_fold_curve(run_dir, args.fold, history, best_epoch, checkpoint_metric))
        return

    # 每折训练长度可能不同，因此每个子图保留自己的横轴范围；纵轴共享以便比较。
    overview, axes = plt.subplots(3, 2, figsize=(12, 11), sharey=True)
    for fold, ax in enumerate(axes.flat[:5]):
        run_dir = run_root / f"fold_{fold}"
        history, best_epoch, checkpoint_metric = load_fold(run_dir)
        draw_fold(ax, history, best_epoch, checkpoint_metric)
        ax.set_title(f"Fold {fold} · {len(history)} epochs · best {best_epoch}")

        print(save_fold_curve(run_dir, fold, history, best_epoch, checkpoint_metric))

    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    overview.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.96))
    overview.suptitle(f"{config.name}: loss curves", y=0.99)
    overview.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir = run_root / "oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "loss_curves.png"
    overview.savefig(overview_path, dpi=160)
    plt.close(overview)
    print(overview_path)


if __name__ == "__main__":
    main()
