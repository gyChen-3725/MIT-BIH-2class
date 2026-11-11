"""计算二分类测试指标，固定 Apnea 为阳性类（标签 1）。

混淆矩阵的行是真实标签、列是预测标签，顺序始终为 [Normal, Apnea]。
概率指标使用模型对 Apnea 的预测概率；其余分类指标使用最终预测类别。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    targets: Sequence[int], predictions: Sequence[int], apnea_probabilities: Sequence[float]
) -> dict[str, object]:
    """计算准确率、召回率、特异度、F1、MCC、ROC AUC 等指标。

    sensitivity 是阳性类召回率 TP/(TP+FN)，specificity 是阴性类召回率
    TN/(TN+FP)。若当前样本只包含一个真实类别，AUC/AP 和 balanced accuracy
    无法按完整二分类口径解释，相关字段返回 None 或按现有零分母规则处理。
    """

    y_true = np.asarray(targets, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    y_score = np.asarray(apnea_probabilities, dtype=np.float64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # 固定标签顺序后，即使某个小批次缺类，混淆矩阵仍保持 2×2。
    tn, fp, _, _ = matrix.ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred)) if both_classes else None
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(y_true, y_score)) if both_classes else None
        ),
        "confusion_matrix": matrix.tolist(),
    }

