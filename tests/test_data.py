from pathlib import Path

import numpy as np
import pytest

from mitbih_2class.config import load_config
from mitbih_2class.data import (
    EpochDataset,
    FeatureNormalizer,
    MasterArrays,
    apply_handcrafted_selection,
    binary_label,
    load_handcrafted_all,
    load_master,
    validate_split,
)

ROOT = Path(__file__).resolve().parents[1]


def test_binary_label_mapping() -> None:
    assert binary_label("Normal") == 0
    for value in ("H", "HA", "X", "OA", "CA", "CAA"):
        assert binary_label(value) == 1


def test_split_rejects_subject_overlap() -> None:
    master = MasterArrays(
        signals=np.zeros((3, 4), dtype=np.float32),
        labels=np.asarray([0, 1, 0]),
        subjects=np.asarray(["P1", "P2", "P3"]),
        records=np.asarray(["r1", "r2", "r3"]),
    )
    split = {
        "train_indices": [0],
        "val_indices": [1],
        "test_indices": [2],
        "train_subjects": ["P1"],
        "val_subjects": ["P2", "P1"],
        "test_subjects": ["P3"],
    }
    with pytest.raises(ValueError, match="Subject overlap"):
        validate_split(master, split)


def test_epoch_split_allows_subject_overlap_but_not_index_overlap() -> None:
    master = MasterArrays(
        signals=np.zeros((3, 4), dtype=np.float32),
        labels=np.asarray([0, 1, 0]),
        subjects=np.asarray(["P1", "P1", "P2"]),
        records=np.asarray(["r1", "r1", "r2"]),
    )
    split = {
        "split_unit": "epoch",
        "train_indices": [0],
        "val_indices": [1],
        "test_indices": [2],
        "train_subjects": ["P1"],
        "val_subjects": ["P1"],
        "test_subjects": ["P2"],
    }
    validate_split(master, split, split_unit="epoch")
    with pytest.raises(ValueError, match="声明"):
        validate_split(master, split, split_unit="subject")
    split["test_indices"] = [0]
    split["test_subjects"] = ["P1"]
    with pytest.raises(ValueError, match="Sample index overlap"):
        validate_split(master, split, split_unit="epoch")


def test_three_modalities_use_same_master_row(tmp_path: Path) -> None:
    path = tmp_path / "master.npz"
    np.savez(
        path,
        ecg_raw=np.asarray([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32),
        ecg_filtered=np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32),
        rri_256=np.stack((np.ones(256), np.full(256, 0.8))),
        r_amp_256=np.stack((np.ones(256), np.full(256, 3.0))),
        raw_label=np.asarray(["Normal", "OA"]),
        subject_id=np.asarray(["P1", "P2"]),
        record=np.asarray(["r1", "r2"]),
    )
    config = load_config(ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_e3_epoch5fold.yaml")
    config = config.with_overrides(data_path=path, expected_input_length=4)
    master = load_master(config)
    sample = EpochDataset(master, [1])[0]
    assert sample["dataset_index"] == 1
    assert sample["label"] == 1
    assert sample["features"].shape == (1, 4)
    assert sample["features"][0, 0].item() > 1.0
    assert sample["aux_features"].shape == (2, 256)
    assert sample["aux_features"][0, 0].item() == pytest.approx(0.8)
    assert sample["aux_features"][1, 0].item() == pytest.approx(np.arcsinh(3.0))


def test_handcrafted_all_aligns_rows_and_uses_263_columns(tmp_path: Path) -> None:
    master_path = tmp_path / "master.npz"
    features_path = tmp_path / "features.npz"
    np.savez(
        master_path,
        raw_label=np.asarray(["Normal", "OA"]),
        subject_id=np.asarray(["P1", "P2"]),
        record=np.asarray(["r1", "r2"]),
        start_sample=np.asarray([0, 7500]),
    )
    names = np.asarray([f"feature_{i}" for i in range(25)] + [
        "coverage_fraction", "longest_uncovered_seconds", "baseline_coverage_fraction"
    ])
    np.savez(
        features_path,
        record=np.asarray(["r1", "r2"]),
        start_sample=np.asarray([0, 7500]),
        class_label=np.asarray(["Normal", "OA"]),
        method_names=np.asarray([f"method_{i}" for i in range(10)]),
        feature_names=names,
        method_features=np.ones((2, 10, 28), dtype=np.float32),
        hrv_feature_names=np.asarray([f"hrv_{i}" for i in range(8)]),
        hrv_features=np.arange(16, dtype=np.float32).reshape(2, 8),
        cross_feature_names=np.asarray([f"cross_{i}" for i in range(5)]),
        cross_features=np.ones((2, 5), dtype=np.float32),
    )
    config = load_config(ROOT / "configs" / "models" / "handcrafted_mlp_all_e3_epoch5fold.yaml")
    config = config.with_overrides(data_path=master_path, handcrafted_path=features_path)
    master = load_master(config)
    assert master.signals.shape == (2, 263)
    assert master.signals[1, 0] == 8
    assert master.aux_signals is None
    with np.load(features_path) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["start_sample"] = np.asarray([1, 7500])
    np.savez(features_path, **payload)
    with pytest.raises(ValueError, match="start_sample"):
        load_master(config)


def test_handcrafted_normalizer_only_uses_fit_rows() -> None:
    train = np.asarray([[1, np.nan], [3, 4]], dtype=np.float32)
    normalizer = FeatureNormalizer.fit(train)
    np.testing.assert_allclose(normalizer.median, [2, 4])
    np.testing.assert_allclose(normalizer.mean, [2, 4])
    np.testing.assert_allclose(normalizer.scale, [1, 1])
    np.testing.assert_allclose(normalizer.transform(np.asarray([100, np.inf])), [98, 0])


def test_expanded_handcrafted_matrix_uses_frozen_all_feature_order(tmp_path: Path) -> None:
    master_path = tmp_path / "master.npz"
    feature_path = tmp_path / "expanded.npz"
    np.savez(
        master_path,
        raw_label=np.asarray(["Normal", "OA"]),
        record=np.asarray(["r1", "r2"]),
        start_sample=np.asarray([0, 7500]),
    )
    matrix = np.arange(694, dtype=np.float32).reshape(2, 347)
    names = np.asarray([f"expanded.{index}" for index in range(347)])
    np.savez(
        feature_path,
        record=np.asarray(["r1", "r2"]),
        start_sample=np.asarray([0, 7500]),
        class_label=np.asarray(["Normal", "Apnea"]),
        all_features=matrix,
        all_feature_names=names,
    )
    with np.load(master_path) as master:
        loaded, loaded_names = load_handcrafted_all(feature_path, master)

    np.testing.assert_array_equal(loaded, matrix)
    assert loaded_names == tuple(names)


def test_frozen_handcrafted_selection_checks_names(tmp_path: Path) -> None:
    path = tmp_path / "fold_0.json"
    path.write_text(
        '{"source_feature_count": 3, "selected_indices": [2, 0], '
        '"selected_names": ["c", "a"]}',
        encoding="utf-8",
    )
    matrix = np.arange(12).reshape(4, 3)
    selected, names = apply_handcrafted_selection(matrix, ("a", "b", "c"), path)
    np.testing.assert_array_equal(selected, matrix[:, [2, 0]])
    assert names == ("c", "a")

