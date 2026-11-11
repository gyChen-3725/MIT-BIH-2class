"""验证组合搜索的特征排名和验证集选优规则。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "selection"))

from optimize_feature_dimension import SELECTORS, candidate_key, fit_feature_ranking


def test_each_selector_returns_a_complete_feature_permutation() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(120, 8))
    labels = (values[:, 0] + 0.2 * values[:, 1] > 0).astype(np.int64)
    values[::11, 3] = np.nan

    for selector in SELECTORS:
        ranking = fit_feature_ranking(values, labels, selector, seed=9)
        assert ranking.shape == (8,)
        np.testing.assert_array_equal(np.sort(ranking), np.arange(8))
        assert ranking[0] in {0, 1}


def test_validation_tie_prefers_smaller_dimension() -> None:
    metrics = {
        "accuracy": 0.8,
        "balanced_accuracy": 0.75,
        "macro_f1": 0.76,
        "roc_auc": 0.84,
    }
    smaller = {
        "selector": "anova",
        "dimension": 32,
        "model": "logistic",
        "validation_metrics": metrics,
    }
    larger = {
        "selector": "anova",
        "dimension": 64,
        "model": "logistic",
        "validation_metrics": metrics,
    }

    assert candidate_key(smaller, "accuracy") > candidate_key(larger, "accuracy")
