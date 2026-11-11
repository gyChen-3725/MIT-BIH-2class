"""检查配置的验证指标确实决定 checkpoint 排序。"""

from mitbih_2class.training import validation_checkpoint_score


def test_accuracy_selection_can_differ_from_macro_f1_selection() -> None:
    # 第二轮准确率更高，第一轮 macro F1 更高；两种规则应选择不同轮次。
    validation = [
        {"accuracy": 0.80, "metrics": {"macro_f1": 0.78}},
        {"accuracy": 0.82, "metrics": {"macro_f1": 0.76}},
    ]
    accuracy_best = max(range(2), key=lambda i: validation_checkpoint_score(validation[i], "val_accuracy"))
    f1_best = max(range(2), key=lambda i: validation_checkpoint_score(validation[i], "val_macro_f1"))
    assert accuracy_best == 1
    assert f1_best == 0
