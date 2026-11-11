"""检查机器学习实验的数据对齐、划分隔离与二分类指标。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "training"))

from train_handcrafted_2class import evaluate, load_design_matrix, load_fold


def test_fold_requires_disjoint_complete_indices(tmp_path: Path) -> None:
    path = tmp_path / "fold_0.json"
    path.write_text(
        json.dumps({
            "split_unit": "epoch",
            "train_indices": [0, 1], "val_indices": [2], "test_indices": [3],
        }),
        encoding="utf-8",
    )
    arrays = load_fold(path, 4)
    assert arrays["train"].tolist() == [0, 1]
    path.write_text(
        json.dumps({
            "split_unit": "epoch",
            "train_indices": [0, 1], "val_indices": [1], "test_indices": [3],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="互斥"):
        load_fold(path, 4)


def test_binary_metrics_use_apnea_as_positive_class() -> None:
    result = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9]))
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == 0.75
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.5


def test_expanded_file_exposes_separate_ablation_groups(tmp_path: Path) -> None:
    feature_path = tmp_path / "expanded.npz"
    master_path = tmp_path / "master.npz"
    n = 2
    np.savez(
        feature_path,
        record=np.asarray(["r1", "r2"]),
        subject_id=np.asarray(["p1", "p2"]),
        start_sample=np.asarray([0, 7500]),
        class_label=np.asarray(["Normal", "Apnea"]),
        method_names=np.asarray([
            "am_area", "am_qr", "am_r", "am_q", "sm_angle",
            "sm_qr", "sm_rs", "fm_qs", "fm_rr", "bm_qrs",
        ]),
        feature_names=np.asarray([f"f{i}" for i in range(25)]),
        method_features=np.ones((n, 10, 25), dtype=np.float32),
        hrv_feature_names=np.asarray([f"h{i}" for i in range(8)]),
        hrv_features=np.ones((n, 8), dtype=np.float32),
        cross_feature_names=np.asarray([f"c{i}" for i in range(5)]),
        cross_features=np.ones((n, 5), dtype=np.float32),
        advanced_hrv_feature_names=np.asarray([f"a{i}" for i in range(24)]),
        advanced_hrv_features=np.ones((n, 24), dtype=np.float32),
        ecg_feature_names=np.asarray([f"e{i}" for i in range(34)]),
        ecg_features=np.ones((n, 34), dtype=np.float32),
        qrs_feature_names=np.asarray([f"q{i}" for i in range(26)]),
        qrs_features=np.ones((n, 26), dtype=np.float32),
    )
    np.savez(
        master_path,
        record=np.asarray(["r1", "r2"]),
        start_sample=np.asarray([0, 7500]),
        raw_label=np.asarray(["Normal", "OA"]),
    )
    matrices, _, _, names = load_design_matrix(feature_path, master_path)

    assert matrices["advanced_hrv"].shape == (2, 24)
    assert matrices["ecg"].shape == (2, 34)
    assert matrices["qrs"].shape == (2, 26)
    assert matrices["non_edr"].shape == (2, 84)
    assert matrices["hrv_all"].shape == (2, 32)
    assert matrices["ecg_qrs"].shape == (2, 60)
    assert matrices["non_edr_with_hrv"].shape == (2, 92)
    assert matrices["edr_all"].shape == (2, 255)
    assert matrices["fusion3_expanded"].shape == (2, 172)
    assert matrices["expanded_all"].shape == (2, 347)
    assert len(names["expanded_all"]) == 347
