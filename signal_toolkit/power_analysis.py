"""AC power analysis for single-phase power measurements."""

from typing import Tuple
import numpy as np


def rms(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=float)
    return float(np.sqrt(np.mean(signal ** 2)))


def active_power(voltage: np.ndarray, current: np.ndarray) -> float:
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    return float(np.mean(v * i))


def apparent_power(v_rms: float, i_rms: float) -> float:
    return v_rms * i_rms


def reactive_power(
    voltage: np.ndarray,
    current: np.ndarray,
    fundamental_freq: float = 50.0,
    fs: float = 10000.0,
) -> float:
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    t = np.arange(len(v)) / fs
    i_shifted = np.roll(i, int(round(fs / fundamental_freq / 4)))
    return float(np.mean(v * i_shifted))


def power_factor(voltage: np.ndarray, current: np.ndarray) -> float:
    p = active_power(voltage, current)
    v_rms = rms(voltage)
    i_rms = rms(current)
    s = apparent_power(v_rms, i_rms)
    if s == 0.0:
        return 0.0
    pf = p / s
    return float(np.clip(pf, -1.0, 1.0))


def compute_power_parameters(
    voltage: np.ndarray,
    current: np.ndarray,
    fundamental_freq: float = 50.0,
    fs: float = 10000.0,
) -> dict:
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    v_rms = rms(v)
    i_rms = rms(i)
    p = active_power(v, i)
    s = apparent_power(v_rms, i_rms)
    q = reactive_power(v, i, fundamental_freq, fs)
    pf = power_factor(v, i)
    phase_angle = np.arccos(np.clip(pf, -1.0, 1.0))
    if pf < 0:
        phase_angle = -phase_angle
    return {
        'v_rms': v_rms,
        'i_rms': i_rms,
        'p': p,
        's': s,
        'q': q,
        'pf': pf,
        'phase': phase_angle,
    }
