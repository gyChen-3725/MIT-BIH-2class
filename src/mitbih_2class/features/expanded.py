"""不依赖呼吸参考、且不限于EDR的30秒ECG扩展手工特征。"""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

ADVANCED_HRV_FEATURE_NAMES = (
    "ectopic_fraction",
    "longest_normal_run_fraction",
    "median_nn_seconds",
    "iqr_nn_seconds",
    "mad_nn_seconds",
    "min_nn_seconds",
    "max_nn_seconds",
    "nn_range_seconds",
    "cvnn",
    "median_heart_rate_bpm",
    "minimum_heart_rate_bpm",
    "maximum_heart_rate_bpm",
    "heart_rate_range_bpm",
    "heart_rate_slope_bpm_per_second",
    "sdsd_seconds",
    "pnn20_fraction",
    "median_absolute_successive_difference_seconds",
    "successive_difference_iqr_seconds",
    "acceleration_fraction",
    "deceleration_fraction",
    "inflection_fraction",
    "poincare_sd1_seconds",
    "poincare_sd2_seconds",
    "poincare_sd1_sd2_ratio",
)

ECG_FEATURE_NAMES = (
    "finite_fraction",
    "bandpassed_mean",
    "bandpassed_std",
    "bandpassed_rms",
    "bandpassed_median",
    "bandpassed_mad",
    "bandpassed_iqr",
    "bandpassed_q05",
    "bandpassed_q95",
    "bandpassed_peak_to_peak",
    "bandpassed_skewness",
    "bandpassed_kurtosis",
    "line_length_per_second",
    "first_difference_rms",
    "second_difference_rms",
    "zero_crossings_per_second",
    "hjorth_mobility",
    "hjorth_complexity",
    "histogram_entropy",
    "permutation_entropy_order3",
    "log_total_power_0p5_35",
    "log_power_0p5_5",
    "log_power_5_15",
    "log_power_15_35",
    "relative_power_0p5_5",
    "relative_power_5_15",
    "relative_power_15_35",
    "spectral_centroid_hz",
    "spectral_entropy",
    "spectral_edge_90_hz",
    "dominant_frequency_hz",
    "baseline_wander_rms",
    "baseline_wander_fraction",
    "flatline_fraction",
)

QRS_FEATURE_NAMES = (
    "valid_qrs_count",
    "valid_qrs_fraction",
    "qrs_duration_median_seconds",
    "qrs_duration_iqr_seconds",
    "qrs_duration_mad_seconds",
    "qr_duration_median_seconds",
    "rs_duration_median_seconds",
    "r_amplitude_median",
    "r_amplitude_iqr",
    "r_amplitude_mad",
    "q_amplitude_median",
    "s_amplitude_median",
    "qr_amplitude_median",
    "qr_amplitude_iqr",
    "rs_amplitude_median",
    "rs_amplitude_iqr",
    "qrs_peak_to_peak_median",
    "qrs_peak_to_peak_iqr",
    "qrs_area_median",
    "qrs_area_iqr",
    "qrs_energy_median",
    "qrs_energy_iqr",
    "qr_slope_median",
    "qr_slope_iqr",
    "rs_slope_median",
    "rs_slope_iqr",
)


def lowpass_filter(
    values: np.ndarray, fs: float, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    """对原始ECG执行零相位Butterworth低通，供QRS形态统计复用。"""

    waveform = np.asarray(values, dtype=np.float64).reshape(-1)
    if not 0.0 < cutoff_hz < fs / 2.0:
        raise ValueError("低通截止频率无效")
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, waveform)


def _empty(names: tuple[str, ...]) -> dict[str, float]:
    return {name: np.nan for name in names}


def _longest_true_fraction(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=np.int8)
    if not len(values):
        return np.nan
    transitions = np.diff(np.pad(values, (1, 1)))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    longest = int(np.max(stops - starts)) if len(starts) else 0
    return float(longest / len(values))


def epoch_advanced_hrv_features(
    beat_times: np.ndarray,
    normal_beat: np.ndarray,
    start_seconds: float,
    stop_seconds: float,
) -> dict[str, float]:
    """提取超短时节律、Poincaré和心率动态特征，不估计不可靠的LF/HF。"""

    result = _empty(ADVANCED_HRV_FEATURE_NAMES)
    times = np.asarray(beat_times, dtype=np.float64)
    normal = np.asarray(normal_beat, dtype=bool)
    if times.shape != normal.shape:
        raise ValueError("心搏时间与心搏类型长度不一致")
    within = (times >= start_seconds) & (times < stop_seconds)
    local_times = times[within]
    local_normal = normal[within]
    if not len(local_times):
        return result
    result["ectopic_fraction"] = float(1.0 - np.mean(local_normal))
    result["longest_normal_run_fraction"] = _longest_true_fraction(local_normal)
    if len(local_times) < 2:
        return result

    intervals = np.diff(local_times)
    valid = local_normal[:-1] & local_normal[1:] & (intervals > 0.3) & (intervals < 2.0)
    nn = intervals[valid]
    if len(nn) < 3:
        return result
    median_nn = float(np.median(nn))
    mean_nn = float(np.mean(nn))
    result.update({
        "median_nn_seconds": median_nn,
        "iqr_nn_seconds": float(np.subtract(*np.quantile(nn, [0.75, 0.25]))),
        "mad_nn_seconds": float(np.median(np.abs(nn - median_nn))),
        "min_nn_seconds": float(np.min(nn)),
        "max_nn_seconds": float(np.max(nn)),
        "nn_range_seconds": float(np.ptp(nn)),
        "cvnn": float(np.std(nn, ddof=1) / mean_nn),
    })
    heart_rate = 60.0 / nn
    result.update({
        "median_heart_rate_bpm": float(np.median(heart_rate)),
        "minimum_heart_rate_bpm": float(np.min(heart_rate)),
        "maximum_heart_rate_bpm": float(np.max(heart_rate)),
        "heart_rate_range_bpm": float(np.ptp(heart_rate)),
    })
    interval_midpoints = (local_times[:-1] + local_times[1:]) / 2.0
    if len(heart_rate) >= 3 and np.ptp(interval_midpoints[valid]) > 0:
        result["heart_rate_slope_bpm_per_second"] = float(
            np.polyfit(interval_midpoints[valid] - start_seconds, heart_rate, 1)[0]
        )

    # 相邻差值只使用连续三个正常心搏，不跨异位搏或无效间期连接。
    adjacent = valid[:-1] & valid[1:]
    differences = np.diff(intervals)[adjacent]
    if len(differences):
        absolute = np.abs(differences)
        result.update({
            "sdsd_seconds": float(np.std(differences, ddof=1))
            if len(differences) > 1 else 0.0,
            "pnn20_fraction": float(np.mean(absolute > 0.02)),
            "median_absolute_successive_difference_seconds": float(np.median(absolute)),
            "successive_difference_iqr_seconds": float(
                np.subtract(*np.quantile(differences, [0.75, 0.25]))
            ),
            "acceleration_fraction": float(np.mean(differences < 0)),
            "deceleration_fraction": float(np.mean(differences > 0)),
        })
        left = intervals[:-1][adjacent]
        right = intervals[1:][adjacent]
        sd1 = float(np.std((right - left) / np.sqrt(2.0), ddof=1)) if len(left) > 1 else 0.0
        sd2 = float(np.std((right + left) / np.sqrt(2.0), ddof=1)) if len(left) > 1 else 0.0
        result["poincare_sd1_seconds"] = sd1
        result["poincare_sd2_seconds"] = sd2
        result["poincare_sd1_sd2_ratio"] = float(sd1 / sd2) if sd2 > 1e-12 else np.nan

    triple_valid = valid[:-2] & valid[1:-1] & valid[2:]
    first_change = np.diff(intervals)[:-1][triple_valid]
    second_change = np.diff(intervals)[1:][triple_valid]
    if len(first_change):
        result["inflection_fraction"] = float(
            np.mean(np.sign(first_change) * np.sign(second_change) < 0)
        )
    return result


def _permutation_entropy_order3(values: np.ndarray) -> float:
    """计算三阶排列熵；只依赖相对次序，因此不受ECG绝对幅值影响。"""

    windows = np.lib.stride_tricks.sliding_window_view(values, 3)
    if not len(windows):
        return np.nan
    orders = np.argsort(windows, axis=1, kind="stable")
    # 三阶排列只有6种；把排列映射到0..26后统计非零模式即可。
    codes = orders[:, 0] * 9 + orders[:, 1] * 3 + orders[:, 2]
    counts = np.unique(codes, return_counts=True)[1].astype(np.float64)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(6.0))


def epoch_ecg_features(ecg: np.ndarray, fs: float) -> dict[str, float]:
    """提取原始ECG的时域、频域、复杂度和基线漂移特征。"""

    result = _empty(ECG_FEATURE_NAMES)
    waveform = np.asarray(ecg, dtype=np.float64).reshape(-1)
    result["finite_fraction"] = float(np.mean(np.isfinite(waveform)))
    if len(waveform) != round(30 * fs) or result["finite_fraction"] < 0.99:
        return result
    if not np.all(np.isfinite(waveform)):
        finite = np.flatnonzero(np.isfinite(waveform))
        waveform = np.interp(np.arange(len(waveform)), finite, waveform[finite])
    detrended = signal.detrend(waveform)
    band_sos = signal.butter(4, (0.5, 35.0), btype="bandpass", fs=fs, output="sos")
    baseline_sos = signal.butter(2, 0.5, btype="lowpass", fs=fs, output="sos")
    bandpassed = signal.sosfiltfilt(band_sos, detrended)
    baseline = signal.sosfiltfilt(baseline_sos, detrended)
    median = float(np.median(bandpassed))
    first = np.diff(bandpassed)
    second = np.diff(bandpassed, n=2)
    variance = float(np.var(bandpassed))
    first_variance = float(np.var(first))
    mobility = float(np.sqrt(first_variance / variance)) if variance > 1e-20 else np.nan
    second_mobility = (
        float(np.sqrt(np.var(second) / first_variance)) if first_variance > 1e-20 else np.nan
    )
    result.update({
        "bandpassed_mean": float(np.mean(bandpassed)),
        "bandpassed_std": float(np.std(bandpassed, ddof=1)),
        "bandpassed_rms": float(np.sqrt(np.mean(bandpassed**2))),
        "bandpassed_median": median,
        "bandpassed_mad": float(np.median(np.abs(bandpassed - median))),
        "bandpassed_iqr": float(np.subtract(*np.quantile(bandpassed, [0.75, 0.25]))),
        "bandpassed_q05": float(np.quantile(bandpassed, 0.05)),
        "bandpassed_q95": float(np.quantile(bandpassed, 0.95)),
        "bandpassed_peak_to_peak": float(np.ptp(bandpassed)),
        "bandpassed_skewness": float(stats.skew(bandpassed, bias=False)),
        "bandpassed_kurtosis": float(stats.kurtosis(bandpassed, fisher=True, bias=False)),
        "line_length_per_second": float(np.sum(np.abs(first)) / 30.0),
        "first_difference_rms": float(np.sqrt(np.mean(first**2))),
        "second_difference_rms": float(np.sqrt(np.mean(second**2))),
        "zero_crossings_per_second": float(
            np.count_nonzero(np.diff(np.signbit(bandpassed))) / 30.0
        ),
        "hjorth_mobility": mobility,
        "hjorth_complexity": float(second_mobility / mobility)
        if np.isfinite(mobility) and mobility > 1e-12 else np.nan,
    })
    histogram = np.histogram(bandpassed, bins=32)[0].astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    result["histogram_entropy"] = float(
        -np.sum(probabilities * np.log(probabilities)) / np.log(32.0)
    )
    # 250 Hz降到50 Hz后计算排列熵，降低相邻过采样点造成的虚假确定性。
    complexity_wave = signal.resample_poly(bandpassed, 1, max(1, round(fs / 50.0)))
    result["permutation_entropy_order3"] = _permutation_entropy_order3(complexity_wave)

    frequencies, power = signal.welch(
        bandpassed, fs=fs, nperseg=min(len(bandpassed), round(4 * fs)), window="hann"
    )
    selected = (frequencies >= 0.5) & (frequencies <= 35.0)
    selected_frequencies = frequencies[selected]
    selected_power = power[selected]
    total = float(np.trapezoid(selected_power, selected_frequencies))
    eps = 1e-20
    result["log_total_power_0p5_35"] = float(np.log(max(total, eps)))
    band_definitions = ((0.5, 5.0, "0p5_5"), (5.0, 15.0, "5_15"), (15.0, 35.0, "15_35"))
    for low, high, suffix in band_definitions:
        mask = (frequencies >= low) & (frequencies < high if high < 35.0 else frequencies <= high)
        band_power = float(np.trapezoid(power[mask], frequencies[mask]))
        result[f"log_power_{suffix}"] = float(np.log(max(band_power, eps)))
        result[f"relative_power_{suffix}"] = float(band_power / total) if total > eps else np.nan
    if np.sum(selected_power) > eps:
        weights = selected_power / np.sum(selected_power)
        result["spectral_centroid_hz"] = float(np.dot(selected_frequencies, weights))
        result["spectral_entropy"] = float(
            -np.sum(weights * np.log(weights + eps)) / np.log(len(weights))
        )
        cumulative = np.cumsum(selected_power)
        edge_index = int(np.searchsorted(cumulative, 0.9 * cumulative[-1]))
        result["spectral_edge_90_hz"] = float(selected_frequencies[min(edge_index, len(selected_frequencies) - 1)])
        result["dominant_frequency_hz"] = float(selected_frequencies[int(np.argmax(selected_power))])

    baseline_rms = float(np.sqrt(np.mean(baseline**2)))
    band_rms = float(np.sqrt(np.mean(bandpassed**2)))
    result["baseline_wander_rms"] = baseline_rms
    result["baseline_wander_fraction"] = float(
        baseline_rms**2 / (baseline_rms**2 + band_rms**2 + eps)
    )
    flat_threshold = max(1e-8, 1e-3 * float(np.std(bandpassed)))
    result["flatline_fraction"] = float(np.mean(np.abs(first) < flat_threshold))
    return result


def _distribution(values: np.ndarray) -> tuple[float, float, float]:
    median = float(np.median(values))
    iqr = float(np.subtract(*np.quantile(values, [0.75, 0.25])))
    mad = float(np.median(np.abs(values - median)))
    return median, iqr, mad


def epoch_qrs_morphology_features(
    filtered_ecg: np.ndarray,
    q_samples: np.ndarray,
    r_samples: np.ndarray,
    s_samples: np.ndarray,
    normal_beat: np.ndarray,
    polarity: float,
    start_sample: int,
    stop_sample: int,
    fs: float,
) -> dict[str, float]:
    """汇总当前窗口正常心搏的QRS时限、振幅、面积、能量和斜率分布。"""

    result = _empty(QRS_FEATURE_NAMES)
    q = np.asarray(q_samples, dtype=np.int64)
    r = np.asarray(r_samples, dtype=np.int64)
    s = np.asarray(s_samples, dtype=np.int64)
    normal = np.asarray(normal_beat, dtype=bool)
    if not (q.shape == r.shape == s.shape == normal.shape):
        raise ValueError("Q/R/S位置与心搏类型长度不一致")
    within = (r >= start_sample) & (r < stop_sample)
    eligible = within & normal
    valid = eligible & (q >= start_sample) & (s < stop_sample) & (q < r) & (r < s)
    result["valid_qrs_count"] = float(np.sum(valid))
    result["valid_qrs_fraction"] = float(np.sum(valid) / np.sum(eligible)) if np.any(eligible) else np.nan
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return result
    canonical = np.asarray(filtered_ecg, dtype=np.float64) * float(polarity)
    qrs_duration = (s[indices] - q[indices]) / fs
    qr_duration = (r[indices] - q[indices]) / fs
    rs_duration = (s[indices] - r[indices]) / fs
    q_amp = canonical[q[indices]]
    r_amp = canonical[r[indices]]
    s_amp = canonical[s[indices]]
    qr_amp = r_amp - q_amp
    rs_amp = r_amp - s_amp
    peak_to_peak = np.empty(len(indices), dtype=np.float64)
    area = np.empty(len(indices), dtype=np.float64)
    energy = np.empty(len(indices), dtype=np.float64)
    qr_slope = np.empty(len(indices), dtype=np.float64)
    rs_slope = np.empty(len(indices), dtype=np.float64)
    for offset, index in enumerate(indices):
        segment = canonical[q[index] : s[index] + 1]
        peak_to_peak[offset] = np.ptp(segment)
        line = np.linspace(segment[0], segment[-1], len(segment))
        area[offset] = np.trapezoid(np.abs(segment - line), dx=1.0 / fs)
        energy[offset] = np.mean(segment**2)
        qr_slope[offset] = np.max(np.gradient(canonical[q[index] : r[index] + 1], 1.0 / fs))
        rs_slope[offset] = -np.min(np.gradient(canonical[r[index] : s[index] + 1], 1.0 / fs))

    qrs_median, qrs_iqr, qrs_mad = _distribution(qrs_duration)
    result.update({
        "qrs_duration_median_seconds": qrs_median,
        "qrs_duration_iqr_seconds": qrs_iqr,
        "qrs_duration_mad_seconds": qrs_mad,
        "qr_duration_median_seconds": float(np.median(qr_duration)),
        "rs_duration_median_seconds": float(np.median(rs_duration)),
        "r_amplitude_median": _distribution(r_amp)[0],
        "r_amplitude_iqr": _distribution(r_amp)[1],
        "r_amplitude_mad": _distribution(r_amp)[2],
        "q_amplitude_median": float(np.median(q_amp)),
        "s_amplitude_median": float(np.median(s_amp)),
        "qr_amplitude_median": _distribution(qr_amp)[0],
        "qr_amplitude_iqr": _distribution(qr_amp)[1],
        "rs_amplitude_median": _distribution(rs_amp)[0],
        "rs_amplitude_iqr": _distribution(rs_amp)[1],
        "qrs_peak_to_peak_median": _distribution(peak_to_peak)[0],
        "qrs_peak_to_peak_iqr": _distribution(peak_to_peak)[1],
        "qrs_area_median": _distribution(area)[0],
        "qrs_area_iqr": _distribution(area)[1],
        "qrs_energy_median": _distribution(energy)[0],
        "qrs_energy_iqr": _distribution(energy)[1],
        "qr_slope_median": _distribution(qr_slope)[0],
        "qr_slope_iqr": _distribution(qr_slope)[1],
        "rs_slope_median": _distribution(rs_slope)[0],
        "rs_slope_iqr": _distribution(rs_slope)[1],
    })
    return result


def expanded_feature_names(
    legacy_names: tuple[str, ...],
) -> tuple[str, ...]:
    """保持旧263维顺序不变，再追加三组非EDR特征。"""

    names = (
        *legacy_names,
        *(f"advanced_hrv.{name}" for name in ADVANCED_HRV_FEATURE_NAMES),
        *(f"ecg.{name}" for name in ECG_FEATURE_NAMES),
        *(f"qrs.{name}" for name in QRS_FEATURE_NAMES),
    )
    if len(names) != len(set(names)):
        raise ValueError("扩展特征名称存在重复")
    return tuple(names)
