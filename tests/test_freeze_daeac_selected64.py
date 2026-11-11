"""验证347维组合搜索到DAEAC折内64维掩码的冻结规则。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "selection"))

from freeze_daeac_selected64 import (
    choose_validation_candidate,
    map_group_selection_to_full_indices,
)


def test_validation_candidate_uses_accuracy_then_macro_f1() -> None:
    frame = pd.DataFrame(
        [
            {
                "fold": 0,
                "stage": "within_group_selection",
                "dimension": 64,
                "feature_group": "a",
                "selector": "anova",
                "model": "logistic",
                "val_accuracy": 0.8,
                "val_macro_f1": 0.7,
                "val_balanced_accuracy": 0.75,
                "val_roc_auc": 0.82,
            },
            {
                "fold": 0,
                "stage": "within_group_selection",
                "dimension": 64,
                "feature_group": "b",
                "selector": "l1",
                "model": "hist_gb",
                "val_accuracy": 0.8,
                "val_macro_f1": 0.72,
                "val_balanced_accuracy": 0.74,
                "val_roc_auc": 0.81,
            },
        ]
    )
    selected = choose_validation_candidate(frame, fold=0)
    assert selected["feature_group"] == "b"
    assert selected["selector"] == "l1"


def test_group_ranking_maps_back_to_full_347_order() -> None:
    full_names = tuple(f"feature_{index}" for index in range(347))
    group_names = ("feature_9", "feature_3", "feature_20")
    ranking = np.asarray([1, 2, 0])
    indices, names = map_group_selection_to_full_indices(
        group_names, ranking, full_names, dimension=2
    )
    assert indices == [3, 20]
    assert names == ["feature_3", "feature_20"]
