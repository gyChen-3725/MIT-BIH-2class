from pathlib import Path

import torch

from mitbih_2class.config import load_config
from mitbih_2class.data import EpochDataset, load_master
from mitbih_2class.models import build_model

ROOT = Path(__file__).resolve().parents[1]


def test_bafnet_dual_stream_data_and_auxiliary_gradients() -> None:
    """双路特征必须对齐同一 master 行，重建头和分类头都要收到梯度。"""

    config = load_config(ROOT / "configs" / "models" / "bafnet_e3_epoch5fold.yaml")
    master = load_master(config)
    assert master.signals.shape == (10197, 2, 256)
    dataset = EpochDataset(master, [0, 1], config.normalization)
    features = torch.stack([dataset[0]["features"], dataset[1]["features"]])
    assert features.shape == (2, 2, 256)
    assert bool(torch.isfinite(features).all())
    assert 0 <= float(features.min()) <= float(features.max()) <= 1

    model = build_model(config)
    logits, auxiliary_loss = model.forward_with_aux(features)
    assert logits.shape == (2, 2)
    assert bool(torch.isfinite(auxiliary_loss))
    (logits.square().mean() + auxiliary_loss).backward()
    assert model.classifier.weight.grad is not None
    assert model.rri_decoder.conv3.weight.grad is not None
    assert model.amp_decoder.conv3.weight.grad is not None
