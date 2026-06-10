"""Waveform characterization and parameter extraction.

Used in 2021-J 周期信号波形识别及参数测量装置.
"""

from typing import Optional
import numpy as np
from numpy.fft import rfft, rfftfreq


def zero_crossing_rate(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float)
    x -= np.mean(x)
    sign = np.signbit(x).astype(int)
    return float(np.sum(np.abs(np.diff(sign))) / max(len(x) - 1, 1))


def count_peaks(signal: np.ndarray, threshold: float = 0.1) -> int:
    x = np.asarray(signal, dtype=float)
    x_norm = x / (np.max(np.abs(x)) + 1e-12)
    peaks = 0
    for i in range(1, len(x) - 1):
        if x_norm[i] > threshold and x_norm[i] >= x_norm[i - 1] and x_norm[i] >= x_norm[i + 1]:
            peaks += 1
    return peaks


def duty_cycle(signal: np.ndarray, threshold: float = 0.0) -> float:
    x = np.asarray(signal, dtype=float)
    x -= np.mean(x)
    above = np.sum(x > threshold)
    return float(above / max(len(x), 1))


def symmetry_ratio(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=float)
    N = len(x)
    half = N // 2
    if half == 0:
        return 1.0
    pos = x[:half]
    neg = x[-half:][::-1]
    corr = np.corrcoef(pos, neg)[0, 1]
    return float(corr if not np.isnan(corr) else 0.0)


def estimate_frequency(signal: np.ndarray, fs: float) -> float:
    x = np.asarray(signal, dtype=float)
    x -= np.mean(x)
    sign_changes = np.where(np.diff(np.signbit(x).astype(int)) != 0)[0]
    if len(sign_changes) < 2:
        return 0.0
    crossings = sign_changes.astype(float)
    for i in range(len(crossings) - 1):
        idx = int(crossings[i])
        if idx + 1 < len(x):
            frac = -x[idx] / (x[idx + 1] - x[idx])
            crossings[i] = idx + frac
    periods = np.diff(crossings[::2])
    if len(periods) == 0:
        return 0.0
    T = np.mean(periods) / fs
    return 1.0 / T if T > 0 else 0.0


def classify_waveform(signal: np.ndarray, fs: float) -> str:
    x = np.asarray(signal, dtype=float)
    zcr = zero_crossing_rate(x)
    sym = symmetry_ratio(x)
    f0 = estimate_frequency(x, fs)

    if f0 <= 0:
        amp_balance = min(np.sum(x > 0), np.sum(x < 0)) / len(x)
        if amp_balance > 0.05:
            fft_mag = np.abs(rfft(x))
            fft_mag[0] = 0
            if np.max(fft_mag) > 0:
                freqs = rfftfreq(len(x), 1 / fs)
                f0 = freqs[np.argmax(fft_mag)]
        if f0 <= 0:
            if zcr < 0.05 and amp_balance > 0.3:
                return 'square'
            return 'noise'

    N = int(round(fs / f0))
    x_period = x[:N] if N < len(x) else x
    if len(x_period) < 4:
        return 'unknown'
    x_period = x_period - np.mean(x_period)
    with np.errstate(divide='ignore', invalid='ignore'):
        x_period = x_period / (np.max(np.abs(x_period)) + 1e-12)

    t = np.arange(len(x_period))
    sine_fit = np.sin(2 * np.pi * t / N)
    sine_err = float(np.sqrt(np.mean((x_period - sine_fit) ** 2)))

    tri = 2 * abs(2 * (t / N - np.floor(t / N + 0.5))) - 1
    tri_err = float(np.sqrt(np.mean((x_period - tri) ** 2)))

    min_err = min(sine_err, tri_err)

    if zcr > 0.3 and min_err > 0.5 and abs(sym) < 0.5:
        return 'noise'

    if min_err < 0.4:
        if sine_err < tri_err:
            return 'sine'
        return 'triangle'

    above = np.sum(x > np.std(x) * 0.5)
    if 2 * above / len(x) > 0.3:
        return 'square'
    return 'pulse'
