from pathlib import Path

import pytest

from mitbih_2class.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_resolves_external_dataset() -> None:
    config = load_config(ROOT / "configs" / "models" / "smoke_cnn.yaml")
    assert config.data_path.name == "mitbih_master_30s.npz"
    assert config.labels == ("Normal", "Apnea")
    assert config.split_path.name == "fold_0.json"
    assert config.checkpoint_metric == "val_macro_f1"


def test_accuracy_checkpoint_config_is_validated() -> None:
    config = load_config(
        ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_aspp3_accuracy_patience50_e3_epoch5fold.yaml"
    )
    assert config.checkpoint_metric == "val_accuracy"
    with pytest.raises(ValueError, match="checkpoint_metric"):
        config.with_overrides(checkpoint_metric="test_accuracy")


def test_daeac_handcrafted_fusion_config_is_consistent() -> None:
    config = load_config(
        ROOT / "configs" / "models" / "daeac_ecg_rhythm_amplitude_handcrafted_e3_epoch5fold.yaml"
    )
    assert config.model == "daeac"
    assert config.model_params["use_rhythm_amplitude"] is True
    assert config.model_params["use_handcrafted"] is True
    assert config.model_params["handcrafted_dim"] == 263


def test_daeac_expanded_selected64_config_uses_fold_mask() -> None:
    config = load_config(
        ROOT
        / "configs"
        / "models"
        / "daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml"
    )
    assert config.handcrafted_selection_dir is not None
    assert config.model_params["handcrafted_dim"] == 64
    assert config.model_params["handcrafted_hidden_dim"] == 64
    assert config.handcrafted_path.name == "features_30s_expanded.npz"

