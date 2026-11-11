"""验证层次化聚类的父簇选择与叶节点编码。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "clustering"))

from cluster_handcrafted_hierarchical import compose_leaf_ids, select_most_mixed_parent


def test_select_most_mixed_parent_uses_training_fraction() -> None:
    parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    selected, rows = select_most_mixed_parent(parent, labels, n_parent_clusters=2)

    assert selected == 0
    assert rows[0]["apnea_fraction"] == 0.5
    assert rows[1]["apnea_fraction"] == 0.25


def test_compose_leaf_ids_keeps_other_parent_as_one_leaf() -> None:
    parent = np.array([0, 0, 1, 1, 0])
    local = np.array([1, 0, -1, -1, 1])
    leaves, names = compose_leaf_ids(parent, refined_parent=0, local_ids=local, local_k=2)

    np.testing.assert_array_equal(leaves, np.array([2, 1, 0, 0, 2]))
    assert names == ["P1", "P0/S0", "P0/S1"]
