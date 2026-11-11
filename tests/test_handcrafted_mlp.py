from pathlib import Path

import pytest
import torch

from mitbih_2class.config import load_config
from mitbih_2class.models import build_model
from mitbih_2class.models.handcrafted_mlp import HandcraftedMLP

ROOT = Path(__file__).resolve().parents[1]


def test_handcrafted_mlp_shape_and_parameter_count() -> None:
    config = load_config(ROOT / "configs" / "models" / "handcrafted_mlp_all_e3_epoch5fold.yaml")
    model = build_model(config)
    assert isinstance(model, HandcraftedMLP)
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_050
    assert model(torch.randn(8, 1, 263)).shape == (8, 2)


def test_handcrafted_mlp_rejects_non_feature_input() -> None:
    model = HandcraftedMLP()
    with pytest.raises(ValueError, match=r"\[batch, 1, 263\]"):
        model(torch.randn(2, 1, 7500))
