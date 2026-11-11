import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "features"))

from compare_handcrafted_selection import selection_stability


def test_selection_stability_counts_shared_features() -> None:
    result = selection_stability([{0, 1}, {0, 2}, {0, 1}])
    assert result["selected_in_all_folds"] == 1
    assert result["selected_in_at_least_four_folds"] == 0
    assert 0 < result["pairwise_jaccard_mean"] < 1
