import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "clustering"))

from cluster_handcrafted_features import choose_k, contingency


def test_contingency_has_fixed_cluster_and_label_order() -> None:
    table = contingency(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]), 2)
    np.testing.assert_array_equal(table, [[1, 1], [0, 2]])


def test_choose_k_uses_composite_internal_rank() -> None:
    rows = [
        {"k": 2, "silhouette": 0.5, "calinski_harabasz": 10, "davies_bouldin": 0.8, "stability_ari": 0.9},
        {"k": 3, "silhouette": 0.4, "calinski_harabasz": 8, "davies_bouldin": 1.0, "stability_ari": 0.7},
    ]
    assert choose_k(rows) == 2
    assert all("composite_rank" in row for row in rows)
