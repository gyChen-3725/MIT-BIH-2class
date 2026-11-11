"""按配置指定的划分执行一折二分类训练，并保存可复核的实验产物。

每个 epoch 只使用训练集更新参数，验证集用于选最佳 checkpoint 和早停，
测试集仅在训练结束后用最佳 checkpoint 评估一次。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mitbih_2class.config import ExperimentConfig, default_config_path, load_config
from mitbih_2class.data import (
    build_dataloaders,
    class_counts,
    load_master,
    load_split,
)
from mitbih_2class.metrics import compute_metrics
from mitbih_2class.models import build_model


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机源，尽量减少重复运行差异。

CUDA/cuDNN 的确定性设置可能降低速度；不同硬件或库版本仍可能有微小差异。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    """解析 auto/cpu/cuda；明确请求 CUDA 却不可用时立即报错。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def build_criterion(config: ExperimentConfig, counts: tuple[int, int], device: torch.device) -> nn.Module:
    """根据训练集类别数量创建交叉熵损失。

balanced 权重使用 total/(2*class_count)，使少数类获得更大权重。
counts 必须来自训练集，不能利用验证或测试标签计算权重。
    """

    if config.class_weight == "none":
        return nn.CrossEntropyLoss()
    if min(counts) <= 0:
        raise ValueError(f"Both classes must occur in the training set, got {counts}")
    total = sum(counts)
    weights = torch.tensor([total / (2 * count) for count in counts], dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights.to(device))


def validation_checkpoint_score(result: dict[str, Any], metric: str) -> float:
    """按预先配置的验证指标取分数；数值越大越好。

    准确率来自完整验证集，不读取测试集。旧配置默认使用 macro F1；
    准确率实验使用同一份验证结果，不改变训练损失或预测阈值。
    """

    if metric == "val_accuracy":
        return float(result["accuracy"])
    if metric == "val_macro_f1":
        return float(result["metrics"]["macro_f1"])
    raise ValueError(f"不支持的 checkpoint 指标: {metric}")


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    limit_batches: int | None = None,
    collect: bool = False,
) -> dict[str, Any]:
    """运行一个集合，返回按样本数加权的 loss 与准确率。

传入 optimizer 时执行反向传播；否则关闭梯度用于验证或测试。
collect=True 会逐样本保留索引、标签和阳性概率，以便计算指标与汇总 OOF。
limit_batches 仅供 debug 快速检查，不应用于正式结果。
    """

    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    n = 0
    rows: list[dict[str, Any]] = []

    # 验证和测试阶段不构建梯度图，节省内存且保证不会更新模型参数。
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            features = batch["features"].to(device, non_blocking=True)
            aux_features = batch.get("aux_features")
            if aux_features is not None:
                aux_features = aux_features.to(device, non_blocking=True)
            handcrafted_features = batch.get("handcrafted_features")
            if handcrafted_features is not None:
                handcrafted_features = handcrafted_features.to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).long()
            if training:
                optimizer.zero_grad(set_to_none=True)
            auxiliary_forward = getattr(model, "forward_with_aux", None)
            if callable(auxiliary_forward):
                # 论文网络可提供辅助重建损失；分类预测始终只从 logits 计算。
                logits, auxiliary_loss = auxiliary_forward(features)
                loss = criterion(logits, labels) + auxiliary_loss
            else:
                if aux_features is None and handcrafted_features is None:
                    logits = model(features)
                else:
                    logits = model(
                        features,
                        aux_features=aux_features,
                        handcrafted_features=handcrafted_features,
                    )
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            probabilities = torch.softmax(logits.detach(), dim=1)
            predictions = probabilities.argmax(dim=1)
            batch_n = len(labels)
            # 先乘 batch 大小，最后除以样本总数，避免尾批较小时被过度加权。
            loss_sum += float(loss.detach().cpu()) * batch_n
            correct += int((predictions == labels).sum().cpu())
            n += batch_n
            if collect:
                # 保留 master 原始行号，后续可核查预测是否恰好覆盖冻结测试折。
                for index in range(batch_n):
                    rows.append(
                        {
                            "dataset_index": int(batch["dataset_index"][index]),
                            "subject_id": str(batch["subject_id"][index]),
                            "record": str(batch["record"][index]),
                            "true_label": int(labels[index].cpu()),
                            "pred_label": int(predictions[index].cpu()),
                            "prob_apnea": float(probabilities[index, 1].cpu()),
                        }
                    )
    if n == 0:
        raise RuntimeError("No batches were processed")
    result: dict[str, Any] = {"loss": loss_sum / n, "accuracy": correct / n, "n": n}
    if collect:
        result["rows"] = rows
        result["metrics"] = compute_metrics(
            [row["true_label"] for row in rows],
            [row["pred_label"] for row in rows],
            [row["prob_apnea"] for row in rows],
        )
    return result


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """在目标设备读取最佳权重，兼容未提供 weights_only 的旧版 PyTorch。"""

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # 旧版 PyTorch 不支持 weights_only 参数。
        return torch.load(path, map_location=device)


def _write_predictions(path: Path, rows: list[dict[str, Any]], fold: int) -> None:
    """保存当前测试折的逐样本预测，供受试者审计和五折 OOF 汇总。"""

    fields = [
        "fold",
        "dataset_index",
        "subject_id",
        "record",
        "true_label",
        "pred_label",
        "prob_apnea",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"fold": fold, **row})


def train_fold(config: ExperimentConfig, debug: bool = False) -> dict[str, object]:
    """完成单折训练、最佳模型选择和独立测试集评估。

    debug 使用单独目录、CPU、单个 epoch 和少量 batch，只检查链路是否可运行；
    生成的指标不能与完整训练结果混用。
    """

    if debug:
        debug_name = config.name if config.name.endswith("_debug") else f"{config.name}_debug"
        config = config.with_overrides(
            name=debug_name, max_epochs=1, device="cpu", num_workers=0
        )
    seed_everything(config.seed + config.fold)
    device = resolve_device(config.device)
    config.write_snapshot()

    master = load_master(config)
    split = load_split(config.split_path)
    train_loader, val_loader, test_loader = build_dataloaders(config, master, split)
    # 手工特征的填补与缩放只由训练折拟合，和模型权重一起保存以供推理复现。
    feature_normalizer = train_loader.dataset.feature_normalizer
    if feature_normalizer is not None:
        np.savez(
            config.run_dir / "feature_normalizer.npz",
            median=feature_normalizer.median,
            mean=feature_normalizer.mean,
            scale=feature_normalizer.scale,
        )
    counts = class_counts(master, split["train_indices"])
    model = build_model(config).to(device)
    criterion = build_criterion(config, counts, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    limit_batches = 2 if debug else None
    checkpoint_path = config.run_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_score = -float("inf")
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        train_result = _run_epoch(
            model, train_loader, criterion, device, optimizer, limit_batches=limit_batches
        )
        val_result = _run_epoch(
            model, val_loader, criterion, device, limit_batches=limit_batches, collect=True
        )
        val_accuracy = float(val_result["accuracy"])
        val_macro_f1 = float(val_result["metrics"]["macro_f1"])
        val_score = validation_checkpoint_score(val_result, config.checkpoint_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_result["loss"]),
                "train_accuracy": float(train_result["accuracy"]),
                "val_loss": float(val_result["loss"]),
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
            }
        )
        print(
            f"fold={config.fold} epoch={epoch:03d} "
            f"train_loss={train_result['loss']:.5f} val_loss={val_result['loss']:.5f} "
            f"val_accuracy={val_accuracy:.5f} val_macro_f1={val_macro_f1:.5f} "
            f"{config.checkpoint_metric}={val_score:.5f}"
        )
        if val_score > best_score:
            # checkpoint 与早停共用配置指标；同分保留更早的 checkpoint。
            best_score = val_score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "checkpoint_metric": config.checkpoint_metric,
                    "best_val_score": val_score,
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    (config.run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    # 恢复最佳验证权重后才看测试集，避免把最后一个 epoch 当作最终模型。
    test_result = _run_epoch(
        model, test_loader, criterion, device, limit_batches=limit_batches, collect=True
    )
    metrics = dict(test_result["metrics"])
    handcrafted_gate_logit = getattr(model, "handcrafted_gate_logit", None)
    metrics.update(
        {
            "fold": config.fold,
            "split_unit": config.split_unit,
            "test_loss": float(test_result["loss"]),
            "best_epoch": int(checkpoint["epoch"]),
            "checkpoint_metric": config.checkpoint_metric,
            "best_val_score": float(checkpoint["best_val_score"]),
            "best_val_accuracy": float(checkpoint["val_accuracy"]),
            "best_val_macro_f1": float(checkpoint["val_macro_f1"]),
            "debug": debug,
        }
    )
    if isinstance(handcrafted_gate_logit, torch.Tensor):
        # 保存最佳checkpoint实际学到的门控值，便于判断手工分支被放大还是被抑制。
        metrics["handcrafted_gate"] = float(
            handcrafted_gate_logit.detach().sigmoid().cpu()
        )
    (config.run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_predictions(config.run_dir / "predictions.csv", test_result["rows"], config.fold)
    provenance = {
        # 记录关键软件、设备和参数规模，方便在本机与服务器之间核对运行条件。
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_class_counts": list(counts),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (config.run_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"completed: {config.run_dir}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """定义单折训练命令行参数，详细超参数从 YAML 读取。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--debug", action="store_true")
    return parser


def cli_main() -> None:
    """命令行入口：先解析并校验配置，再启动指定折训练。"""

    args = build_parser().parse_args()
    config = load_config(args.config, {"fold": args.fold, "device": args.device})
    train_fold(config, debug=args.debug)


if __name__ == "__main__":
    cli_main()

