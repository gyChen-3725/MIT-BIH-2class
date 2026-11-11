"""核对列联表统计量与受试者条件置换的关键边界。"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "clustering"))

from analyze_daeac_cluster_significance import (
    contingency,
    holm_adjust,
    pearson_chi_square,
    within_subject_permutation_p,
)


def test_pearson_chi_square_matches_scipy() -> None:
    ids = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    labels = np.array([0, 0, 1, 0, 1, 1, 1, 1])
    table = contingency(ids, labels, 2)
    statistic, minimum_expected = pearson_chi_square(table)
    reference = chi2_contingency(table, correction=False)
    np.testing.assert_allclose(statistic, reference.statistic)
    np.testing.assert_allclose(minimum_expected, reference.expected_freq.min())


def test_subject_conditioning_does_not_call_subject_identity_a_cluster_signal() -> None:
    # 两位受试者各自只有一类：普通卡方会显著，受试者内重排则无法改变标签。
    labels = np.array([0] * 20 + [1] * 20)
    subjects = np.array(["A"] * 20 + ["B"] * 20)
    cluster_ids = np.array([0] * 20 + [1] * 20)
    assignments = {name: cluster_ids for name in ("rri", "amplitude", "both")}
    statistic, _ = pearson_chi_square(contingency(cluster_ids, labels, 2))
    observed = {name: statistic for name in assignments}
    for mode in ("shuffle", "circular_shift"):
        p_values, _ = within_subject_permutation_p(
            assignments, labels, subjects, 2, observed, 99, 42, mode=mode
        )
        assert p_values == {name: 1.0 for name in assignments}


def test_holm_adjustment_respects_order_and_monotonicity() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}
