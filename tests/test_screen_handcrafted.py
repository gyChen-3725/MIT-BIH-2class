"""检查受试者内 AUC 和混合事件的筛选边界。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "features"))

from screen_handcrafted_features import apnea_subtype, binary_auc


def test_binary_auc_handles_tied_feature_values() -> None:
    labels = np.asarray([False, False, True, True])
    scores = np.asarray([0.0, 1.0, 1.0, 2.0])
    assert binary_auc(labels, scores) == 0.875


def test_mixed_event_is_not_a_single_apnea_subtype() -> None:
    assert apnea_subtype("2 CA") == "Central"
    assert apnea_subtype("2 X") == "Obstructive"
    assert apnea_subtype("2 H CA") == "Other"
