from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mitbih_2class.config import load_config
from mitbih_2class.models import SmokeCNN, build_model

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_cnn_output_shape() -> None:
    model = SmokeCNN()
    output = model(torch.randn(2, 1, 7500))
    assert output.shape == (2, 2)


def test_model_module_is_selected_by_config_name() -> None:
    config = load_config(ROOT / "configs" / "models" / "smoke_cnn.yaml")
    assert isinstance(build_model(config), SmokeCNN)


def test_missing_model_module_has_actionable_error() -> None:
    config = load_config(ROOT / "configs" / "models" / "smoke_cnn.yaml").with_overrides(
        model="missing_paper"
    )
    with pytest.raises(ValueError, match="missing_paper"):
        build_model(config)


def test_new_paper_module_needs_no_registry_change(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(ROOT / "configs" / "models" / "smoke_cnn.yaml").with_overrides(
        model="paper_example", model_params={"channels": 32}
    )
    loaded: list[str] = []

    def fake_import(name: str) -> SimpleNamespace:
        loaded.append(name)
        return SimpleNamespace(build_model=lambda selected: SmokeCNN(len(selected.labels)))

    monkeypatch.setattr("mitbih_2class.models.import_module", fake_import)
    assert isinstance(build_model(config), SmokeCNN)
    assert loaded == ["mitbih_2class.models.paper_example"]
    assert config.model_params == {"channels": 32}

