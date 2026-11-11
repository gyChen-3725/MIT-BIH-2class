"""验证本实验仓库维护的347维特征字典。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "features"))

from build_expanded_feature_dictionary import expanded_dictionary_rows


def test_expanded_dictionary_appends_new_families_after_legacy_order() -> None:
    rows = expanded_dictionary_rows()

    assert len(rows) == 347
    assert rows[262]["feature"] == "cross.attenuation_difference_am_qr_minus_sm_angle"
    assert rows[263]["feature"] == "advanced_hrv.ectopic_fraction"
    assert rows[287]["feature"] == "ecg.finite_fraction"
    assert rows[321]["feature"] == "qrs.valid_qrs_count"
    assert rows[-1]["index"] == 346
