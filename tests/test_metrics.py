import pytest

from mitbih_2class.metrics import compute_metrics


def test_binary_metrics_use_apnea_as_positive_class() -> None:
    metrics = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.8, 0.7, 0.9])
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]

