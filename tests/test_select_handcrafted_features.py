import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "features"))

from select_handcrafted_features import select_features


def test_selection_keeps_stronger_representative_from_correlated_pair() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    strong = np.asarray([0, 1, 2, 3, 5, 6, 7, 8], dtype=float)
    duplicate = strong * 2.0
    complementary = np.asarray([0, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    constant = np.ones(8)
    values = np.column_stack((strong, duplicate, complementary, constant))
    result = select_features(
        values,
        labels,
        ("strong", "duplicate", "complementary", "constant"),
        max_features=2,
        correlation_threshold=0.90,
    )
    selected = result["selected_names"]
    assert len(set(selected) & {"strong", "duplicate"}) == 1
    assert "complementary" in selected
    assert result["quality_excluded_indices"] == [3]
