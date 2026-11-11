"""按 BAFNet notebook 的固定划分、两阶段训练与逐记录评价运行实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
from pathlib import Path

# Windows 上的 K-means/MKL 在小批量数据下会为线程数发出警告并浪费资源。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import sklearn
import tensorflow as tf
from model import create_model
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "artifacts" / "bafnet_5min.npz"
DEFAULT_RUN = HERE / "artifacts" / "paper_100plus5_seed42"
ADDITIONAL_INFO = HERE.parent / "source" / "utils" / "code_for_calculating_per-recording" / "dataset" / "additional-information.txt"


def _targets(features: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    """保留 notebook fit 调用的目标顺序，包括两路重建目标互换的原始行为。"""

    return [
        tf.keras.utils.to_categorical(labels, 2),
        features[:, :, 1:2],
        features[:, :, 0:1],
    ]


def hard_sample_indices(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """按作者的五簇 K-means 与簇内少数类方案选取第二阶段样本。

    原 notebook 遍历列表时就删除少数类项；这里保留这一步的选择语义，
    只给随机源加种子，使复跑能获得同一批样本。
    """

    cluster_ids = KMeans(n_clusters=5, n_init=10, random_state=seed).fit_predict(features[:, :, 0])
    rng = random.Random(seed)
    selected: list[int] = []
    details: dict[str, dict[str, int]] = {}
    # notebook 用字典按簇首次出现的顺序遍历；训练样本的顺序会影响后续 shuffle。
    groups: dict[int, list[tuple[int, int]]] = {}
    for index, cluster in enumerate(cluster_ids):
        groups.setdefault(int(cluster), []).append((int(labels[index]), index))
    for cluster, group in groups.items():
        positive = sum(label for label, _ in group)
        negative = len(group) - positive
        minority = 1 if negative > positive else 0
        count = min(positive, negative)
        for item in group:
            if item[0] == minority:
                selected.append(item[1])
                group.remove(item)
        selected.extend(index for _, index in rng.sample(group, count))
        details[str(cluster)] = {
            "normal": negative, "apnea": positive, "minority": minority,
            "selected": len(selected) - sum(value["selected"] for value in details.values()),
        }
    return np.asarray(selected, dtype=np.int64), details


def compile_model(model: tf.keras.Model) -> None:
    """使用论文初始学习率与 notebook 三个输出的损失和分类准确率。"""

    model.compile(
        loss={"outputs": "binary_crossentropy", "rri": "mean_squared_error", "ampl": "mean_squared_error"},
        loss_weights={"outputs": 1, "rri": 1, "ampl": 1},
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics={"outputs": "accuracy"},
    )


def lr_schedule(epoch: int, learning_rate: float) -> float:
    """保留作者 notebook 的学习率更新时点。

    Keras 的 epoch 从 0 开始计数；原条件在索引 71、81、91 首次触发，
    也就是第 72、82、92 个 epoch 开始时将当前学习率乘以 0.1。
    第二阶段重新调用 fit，epoch 索引归零，延续第一阶段结束时的学习率。
    """

    if epoch > 70 and (epoch - 1) % 10 == 0:
        return learning_rate * 0.1
    return learning_rate


def evaluate_segments(model: tf.keras.Model, features: np.ndarray, labels: np.ndarray,
                      records: np.ndarray, minutes: np.ndarray, output_csv: Path) -> dict:
    """用最终模型的分类头输出逐分钟分数，与原 notebook 的测试口径一致。"""

    # 作者在每一阶段先 evaluate 三头总损失，再用分类头预测并统计测试指标。
    keras_metrics = model.evaluate(features, _targets(features, labels), verbose=0, return_dict=True)
    scores = model.predict(features, batch_size=1024, verbose=0)[0][:, 1]
    predicted = (scores > 0.5).astype(np.int8)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["y_true", "y_score", "subject", "minute"])
        writer.writerows(zip(labels, scores, records, minutes))
    tn, fp, fn, tp = matrix.ravel()
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "keras_evaluate": {name: float(value) for name, value in keras_metrics.items()},
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_records(scores_csv: Path) -> dict:
    """复现作者 evaluate.py 的每小时预测事件率与 >5 次/小时判别规则。"""

    by_record: dict[str, list[float]] = {}
    with scores_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_record.setdefault(row["subject"], []).append(float(row["y_score"]))
    prediction_rate = {
        record: float(np.mean(np.asarray(scores) > 0.5) * 60)
        for record, scores in by_record.items()
    }
    reference_rate: dict[str, float] = {}
    for line in ADDITIONAL_INFO.read_text(encoding="utf-8").splitlines():
        columns = line.strip().split("\t")
        if len(columns) == 12 and columns[0].startswith("x"):
            reference_rate[columns[0]] = float(columns[3]) / float(columns[1]) * 60
    if set(prediction_rate) != set(reference_rate):
        raise ValueError("逐记录预测与附加信息的 x 记录未一一对应")
    records = sorted(prediction_rate)
    truth = np.asarray([reference_rate[name] > 5 for name in records])
    pred = np.asarray([prediction_rate[name] > 5 for name in records])
    reference_values = np.asarray([reference_rate[name] for name in records])
    predicted_values = np.asarray([prediction_rate[name] for name in records])
    matrix = confusion_matrix(truth, pred, labels=[False, True])
    tn, fp, fn, tp = matrix.ravel()
    correlation = (
        float(np.corrcoef(reference_values, predicted_values)[0, 1])
        if reference_values.std() > 0 and predicted_values.std() > 0
        else None
    )
    return {
        "n_records": len(records),
        "accuracy": float(accuracy_score(truth, pred)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(truth, [prediction_rate[name] for name in records])),
        "correlation": correlation,
        "confusion_matrix": matrix.tolist(),
        "predicted_events_per_hour": prediction_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--dataset-name", default="Apnea-ECG")
    parser.add_argument(
        "--record-eval", choices=("apnea_ecg", "none"), default="apnea_ecg",
        help="Apnea-ECG 才使用作者的每小时呼吸暂停指数规则",
    )
    parser.add_argument("--smoke", action="store_true", help="缩小样本和 epoch，仅验证链路")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    with np.load(args.data, allow_pickle=False) as data:
        features = data["features"]
        labels = data["labels"].astype(np.int64)
        records = data["record"].astype(str)
        minutes = data["minute"]
        split = data["split"]
    if features.ndim != 3 or features.shape[1:] != (900, 2):
        raise ValueError("BAFNet 需要 [N, 900, 2] 特征")
    if not (set(np.unique(split)) == {0, 1} and set(np.unique(labels)) == {0, 1}):
        raise ValueError("数据必须包含训练与盲测集合及两个类别")
    train_pool = np.flatnonzero(split == 0)
    test_indices = np.flatnonzero(split == 1)
    train_indices, val_indices = train_test_split(
        train_pool, test_size=0.3, random_state=args.seed, stratify=labels[train_pool]
    )
    if args.smoke:
        train_indices = train_indices[:256]
        val_indices = val_indices[:128]
        test_indices = test_indices[:128]
        args.stage1_epochs = args.stage2_epochs = 1
    x_train, y_train = features[train_indices], labels[train_indices]
    x_val, y_val = features[val_indices], labels[val_indices]
    x_test, y_test = features[test_indices], labels[test_indices]
    args.output.mkdir(parents=True, exist_ok=True)

    model = create_model()
    compile_model(model)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(args.output / "best.weights.h5"), monitor="val_outputs_accuracy",
        mode="max", save_best_only=True, save_weights_only=True,
    )
    scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    stage1_steps = min(92, math.ceil(len(x_train) / args.batch_size))
    history1 = model.fit(
        x_train, _targets(x_train, y_train), batch_size=args.batch_size,
        epochs=args.stage1_epochs, steps_per_epoch=stage1_steps,
        validation_data=(x_val, _targets(x_val, y_val)),
        callbacks=[scheduler, checkpoint], verbose=2,
    )
    stage1_metrics = evaluate_segments(
        model, x_test, y_test, records[test_indices], minutes[test_indices],
        args.output / "stage1_test_scores.csv",
    )
    print("stage1 test", stage1_metrics, flush=True)

    selected, clusters = hard_sample_indices(x_train, y_train, args.seed)
    x_hard, y_hard = x_train[selected], y_train[selected]
    history2 = model.fit(
        x_hard, _targets(x_hard, y_hard), batch_size=args.batch_size,
        epochs=args.stage2_epochs,
        validation_data=(x_val, _targets(x_val, y_val)),
        callbacks=[scheduler, checkpoint], verbose=2,
    )
    model.save_weights(args.output / "final.weights.h5")
    stage2_csv = args.output / "stage2_test_scores.csv"
    stage2_metrics = evaluate_segments(
        model, x_test, y_test, records[test_indices], minutes[test_indices], stage2_csv
    )
    record_metrics = (
        evaluate_records(stage2_csv)
        if args.record_eval == "apnea_ecg" and not args.smoke else None
    )
    results = {
        "protocol": (
            f"BAFNet two-stage flow, {args.stage1_epochs}+{args.stage2_epochs} epochs, "
            f"5-minute RRI/RPA, dataset={args.dataset_name}"
        ),
        "dataset_name": args.dataset_name,
        "record_eval": args.record_eval,
        "data": str(args.data.resolve()),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "initial_learning_rate": 0.001,
        "learning_rate_decay_epoch_indices": {
            "stage1": list(range(71, args.stage1_epochs, 10)),
            "stage2": list(range(71, args.stage2_epochs, 10)),
        },
        "parameter_count": model.count_params(),
        "runtime": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "split_counts": {"train": len(x_train), "val": len(x_val), "test": len(x_test)},
        "stage1_steps_per_epoch": stage1_steps,
        "hard_sample_count": len(selected),
        "hard_sample_class_counts": np.bincount(y_hard, minlength=2).tolist(),
        "clusters": clusters,
        "stage1_test": stage1_metrics,
        "stage2_test": stage2_metrics,
        "record_test": record_metrics,
        "smoke": args.smoke,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "history.json").write_text(
        json.dumps(
            {"stage1": history1.history, "stage2": history2.history},
            indent=2, default=float,
        ),
        encoding="utf-8",
    )
    print("completed", args.output, flush=True)


if __name__ == "__main__":
    main()
