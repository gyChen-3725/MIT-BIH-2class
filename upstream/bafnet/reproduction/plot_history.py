"""分别显示 BAFNet 两阶段的训练/验证目标损失和分类准确率。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "artifacts" / "paper_100plus5_seed42")
    args = parser.parse_args()
    history = json.loads((args.run / "history.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, stage in enumerate(("stage1", "stage2")):
        values = history[stage]
        epochs = range(1, len(values["loss"]) + 1)
        axes[row, 0].plot(epochs, values["loss"], marker="o", label="Train")
        axes[row, 0].plot(epochs, values["val_loss"], marker="o", label="Validation")
        axes[row, 0].set_title(f"{stage}: total loss (BCE + 2 MSE + L2)")
        axes[row, 1].plot(epochs, values["outputs_accuracy"], marker="o", label="Train")
        axes[row, 1].plot(epochs, values["val_outputs_accuracy"], marker="o", label="Validation")
        axes[row, 1].set_title(f"{stage}: classification accuracy")
        for axis in axes[row]:
            axis.set_xlabel("Epoch within stage")
            axis.grid(alpha=0.25)
            axis.legend()
    figure.tight_layout()
    output = args.run / "training_curves.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
