"""验证非EDR扩展特征的维度、方向和30秒窗口边界。"""

import numpy as np

from mitbih_2class.features.expanded import (
    ADVANCED_HRV_FEATURE_NAMES,
    ECG_FEATURE_NAMES,
    QRS_FEATURE_NAMES,
    epoch_advanced_hrv_features,
    epoch_ecg_features,
    epoch_qrs_morphology_features,
    expanded_feature_names,
)


def test_expanded_feature_family_counts_are_frozen() -> None:
    assert len(ADVANCED_HRV_FEATURE_NAMES) == 24
    assert len(ECG_FEATURE_NAMES) == 34
    assert len(QRS_FEATURE_NAMES) == 26
    names = expanded_feature_names(tuple(f"legacy.{index}" for index in range(263)))
    assert len(names) == 347
    assert names[263] == "advanced_hrv.ectopic_fraction"
    assert names[-1] == "qrs.rs_slope_iqr"


def test_regular_rhythm_has_expected_advanced_hrv() -> None:
    beat_times = np.arange(0.5, 30.0, 1.0)
    normal = np.ones(len(beat_times), dtype=bool)
    features = epoch_advanced_hrv_features(beat_times, normal, 0.0, 30.0)

    assert features["ectopic_fraction"] == 0.0
    assert features["longest_normal_run_fraction"] == 1.0
    assert features["median_nn_seconds"] == 1.0
    assert features["median_heart_rate_bpm"] == 60.0
    assert features["pnn20_fraction"] == 0.0


def test_ecg_spectrum_recovers_ten_hz_component() -> None:
    fs = 250.0
    time = np.arange(round(30 * fs)) / fs
    waveform = np.sin(2 * np.pi * 10.0 * time)
    features = epoch_ecg_features(waveform, fs)

    assert features["finite_fraction"] == 1.0
    assert abs(features["dominant_frequency_hz"] - 10.0) < 0.3
    assert features["relative_power_5_15"] > 0.95
    assert 0.0 <= features["permutation_entropy_order3"] <= 1.0


def test_qrs_morphology_uses_only_beats_inside_epoch() -> None:
    fs = 250.0
    waveform = np.zeros(round(30 * fs))
    r = np.asarray([1000, 2000, 3000, 8000])
    q = r - 10
    s = r + 15
    for peak in r[:3]:
        waveform[peak - 10] = -0.2
        waveform[peak] = 1.0
        waveform[peak + 15] = -0.3
    features = epoch_qrs_morphology_features(
        waveform,
        q,
        r,
        s,
        np.ones(len(r), dtype=bool),
        1.0,
        0,
        len(waveform),
        fs,
    )

    assert features["valid_qrs_count"] == 3.0
    assert features["valid_qrs_fraction"] == 1.0
    assert abs(features["qrs_duration_median_seconds"] - 0.1) < 1e-12
    assert features["r_amplitude_median"] == 1.0
