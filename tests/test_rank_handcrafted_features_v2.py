import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "features"))

from rank_handcrafted_features_v2 import rank_features_v2


def test_v2_ranking_covers_all_features_and_defers_correlated_copy() -> None:
    labels = np.tile([0, 0, 1, 1], 20)
    subjects = np.repeat(["A", "B"], 40)
    strong = labels.astype(float) + np.tile([0.0, 0.1, 0.0, 0.1], 20)
    duplicate = strong * 2.0
    weaker = np.tile([0.0, 1.0, 0.0, 1.0], 20)
    values = np.column_stack((strong, duplicate, weaker))
    result = rank_features_v2(
        values,
        labels,
        subjects,
        ("strong", "duplicate", "weaker"),
        correlation_threshold=0.95,
    )
    assert sorted(result["ranked_indices"]) == [0, 1, 2]
    assert result["nonredundant_representative_count"] == 2
    representatives = set(result["ranked_names"][:2])
    assert "weaker" in representatives
    assert len(representatives & {"strong", "duplicate"}) == 1
