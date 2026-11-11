from pathlib import Path

import pytest
import torch

from mitbih_2class.config import load_config
from mitbih_2class.models import build_model
from mitbih_2class.models.daeac import DAEACMACNNSE

ROOT = Path(__file__).resolve().parents[1]


def test_daeac_binary_shape_and_reference_parameter_count() -> None:
    config = load_config(ROOT / "configs" / "models" / "daeac.yaml")
    model = build_model(config).eval()
    assert isinstance(model, DAEACMACNNSE)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_414_530

    sample = torch.randn(1, 1, 7500)
    with torch.no_grad():
        features = model.extract_features(sample)
        logits = model(sample)
    assert features.shape == (1, 1024)
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_daeac_rejects_wrong_input_channel_count() -> None:
    model = DAEACMACNNSE()
    with pytest.raises(ValueError, match=r"\[batch, 1, length\]"):
        model(torch.randn(1, 2, 128))


def test_daeac_fuses_saved_rhythm_and_amplitude() -> None:
    model = DAEACMACNNSE(base_channels=4, use_rhythm_amplitude=True).eval()
    ecg = torch.randn(1, 1, 256)
    aux = torch.randn(1, 2, 256)
    with torch.no_grad():
        first = model.extract_features(ecg, aux)
        changed = model.extract_features(ecg, aux + 1.0)
    assert first.shape == (1, 384)
    assert not torch.allclose(first[:, 256:], changed[:, 256:])
    with pytest.raises(ValueError, match="aux_features"):
        model(ecg)


def test_optional_third_aspp_receives_gradient_without_changing_output_shape() -> None:
    # 用较短波形检查新增块确实参与计算，不运行任何训练流程。
    old_model = DAEACMACNNSE(base_channels=4, use_rhythm_amplitude=True)
    model = DAEACMACNNSE(
        base_channels=4, use_rhythm_amplitude=True, use_third_aspp=True
    ).eval()
    assert model.aspp3 is not None
    assert len(model.aspp3.branches) == 4
    assert sum(p.numel() for p in model.parameters()) > sum(
        p.numel() for p in old_model.parameters()
    )

    logits = model(torch.randn(1, 1, 256), torch.randn(1, 2, 256))
    assert logits.shape == (1, 2)
    logits.sum().backward()
    gradient = model.aspp3.branches[0].conv.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0


def test_daeac_fuses_handcrafted_embedding_before_classifier() -> None:
    model = DAEACMACNNSE(
        base_channels=4,
        use_rhythm_amplitude=True,
        use_handcrafted=True,
        handcrafted_dim=263,
        handcrafted_hidden_dim=32,
    ).eval()
    ecg = torch.randn(2, 1, 256)
    aux = torch.randn(2, 2, 256)
    handcrafted = torch.randn(2, 263)
    embeddings = model.extract_features(ecg, aux, handcrafted)
    assert embeddings.shape == (2, 256 + 128 + 32)
    logits = model(ecg, aux, handcrafted)
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert model.handcrafted_branch[0].weight.grad is not None
    assert model.aspp1.branches[0].conv.weight.grad is not None
    with pytest.raises(ValueError, match="handcrafted_features"):
        model(ecg, aux)


def test_expanded_selected64_config_uses_lightweight_mlp_fusion() -> None:
    config = load_config(
        ROOT
        / "configs"
        / "models"
        / "daeac_ecg_rhythm_amplitude_handcrafted_expanded_selected64_e3_epoch5fold.yaml"
    )
    model = build_model(config).eval()
    assert isinstance(model, DAEACMACNNSE)
    assert model.handcrafted_dim == 64
    assert model.feature_dim == 1024 + 128 + 64
    assert model.handcrafted_branch[0].in_features == 64
    assert model.handcrafted_branch[0].out_features == 64
    with torch.no_grad():
        embeddings = model.extract_features(
            torch.randn(1, 1, 256),
            torch.randn(1, 2, 256),
            torch.randn(1, 64),
        )
    assert embeddings.shape == (1, 1216)


def test_handcrafted_gate_starts_small_and_receives_gradient() -> None:
    """门控应保留输出形状，并能从分类损失中学习手工分支权重。"""

    model = DAEACMACNNSE(
        base_channels=4,
        use_rhythm_amplitude=True,
        use_handcrafted=True,
        handcrafted_dim=128,
        handcrafted_hidden_dim=64,
        use_handcrafted_gate=True,
        handcrafted_gate_init=0.05,
    ).eval()
    assert model.handcrafted_gate_logit is not None
    assert float(model.handcrafted_gate_logit.sigmoid().detach()) == pytest.approx(0.05)

    logits = model(
        torch.randn(2, 1, 256),
        torch.randn(2, 2, 256),
        torch.randn(2, 128),
    )
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert model.handcrafted_gate_logit.grad is not None
    assert torch.isfinite(model.handcrafted_gate_logit.grad)


def test_handcrafted_gate_rejects_invalid_initial_value() -> None:
    with pytest.raises(ValueError, match="handcrafted_gate_init"):
        DAEACMACNNSE(
            use_handcrafted=True,
            use_handcrafted_gate=True,
            handcrafted_gate_init=1.0,
        )


def test_mi128_fold0_config_enables_small_gate() -> None:
    config = load_config(
        ROOT
        / "configs"
        / "models"
        / "daeac_ecg_rhythm_amplitude_handcrafted_mi128_gated_fold0.yaml"
    )
    model = build_model(config)
    assert isinstance(model, DAEACMACNNSE)
    assert model.handcrafted_dim == 128
    assert model.handcrafted_branch[0].in_features == 128
    assert model.handcrafted_branch[0].out_features == 64
    assert model.feature_dim == 1024 + 128 + 64
    assert model.handcrafted_gate_logit is not None
    assert float(model.handcrafted_gate_logit.sigmoid().detach()) == pytest.approx(0.05)
