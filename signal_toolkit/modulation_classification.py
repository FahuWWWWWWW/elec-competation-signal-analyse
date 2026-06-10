"""Modulation type identification using instantaneous features.

Supports AM, FM, PM, DSB, SSB, CW, and FSK classification.
Used in 2023-D 信号调制方式识别与参数估计装置.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy import signal as scipy_signal


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mx = np.max(np.abs(x))
    return x / mx if mx > 0 else x


def extract_features(signal: np.ndarray, fs: float) -> dict:
    x = _normalize(signal)
    analytic = scipy_signal.hilbert(x)
    inst_amplitude = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
    N = len(x)
    trim = int(N * 0.1)
    if trim >= N // 2:
        trim = N // 4
    amp_mid = inst_amplitude[trim:N - trim] if N > 2 * trim else inst_amplitude
    freq_mid = inst_freq[trim:N - trim] if N > 2 * trim else inst_freq
    gamma_max = np.max(np.abs(inst_amplitude - np.mean(inst_amplitude)))
    sigma_ap = np.std(amp_mid) / np.mean(amp_mid) if np.mean(amp_mid) > 0 else 0
    sigma_dp = np.std(np.diff(inst_phase[trim:N - trim])) if N > 2 * trim else np.std(np.diff(inst_phase))
    sigma_af = np.std(freq_mid) / np.mean(np.abs(freq_mid)) if np.mean(np.abs(freq_mid)) > 0 else 0
    p = x ** 2
    p_total = np.sum(p)
    p_carrier = np.sum(np.cos(2 * np.pi * np.arange(len(x)) * np.mean(freq_mid) / fs) ** 2)
    p_ratio = p_total / (p_carrier + 1e-12)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(x).astype(int))))
    return {
        'gamma_max': gamma_max,
        'sigma_ap': sigma_ap,
        'sigma_dp': sigma_dp,
        'sigma_af': sigma_af,
        'p_ratio': p_ratio,
        'zero_crossings': zero_crossings / len(x),
    }


def _high_order_cumulant(signal: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(signal)
    x = x.real if np.iscomplexobj(x) else x
    x = x.astype(float)
    x -= np.mean(x)
    m20 = np.mean(x ** 2)
    m21 = np.mean(np.abs(x) ** 2)
    m40 = np.mean(x ** 4)
    m41 = np.mean(x ** 3 * np.conj(x))
    m42 = np.mean(np.abs(x) ** 4)
    c20 = m20
    c21 = m21
    c40 = m40 - 3 * m20 ** 2
    c41 = m41 - 3 * m21 * m20
    c42 = m42 - abs(m20) ** 2 - 2 * m21 ** 2
    denom = c21 ** 2 + 1e-12
    c40_norm = abs(c40) / denom
    c41_norm = abs(c41) / denom
    c42_norm = abs(c42) / denom
    return c40_norm, c41_norm, c42_norm


def classify_modulation(signal: np.ndarray, fs: float) -> str:
    features = extract_features(signal, fs)
    x = _normalize(signal)
    analytic = scipy_signal.hilbert(x)
    c40, c41, c42 = _high_order_cumulant(analytic)

    if features['gamma_max'] > 0.3:
        if features['sigma_ap'] > 0.15:
            return 'AM'
        if features['sigma_dp'] > 0.3:
            return 'PM'
        if c40 < 0.5 and c42 < 0.5:
            return 'SSB'
        return 'DSB'
    else:
        if features['sigma_af'] > 0.05:
            return 'FM'
        if features['zero_crossings'] > 0.5:
            return 'FSK'
        return 'CW'
