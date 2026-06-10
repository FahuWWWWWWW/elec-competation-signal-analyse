"""Impedance, LCR measurement analysis."""

from typing import Tuple, Optional
import numpy as np


def impedance_vi(
    voltage: np.ndarray,
    current: np.ndarray,
    fs: float,
    f_target: float,
) -> Tuple[float, float]:
    N = len(voltage)
    t = np.arange(N) / fs
    cos_w = np.cos(2 * np.pi * f_target * t)
    sin_w = -np.sin(2 * np.pi * f_target * t)
    I_v = 2 * np.mean(voltage * cos_w)
    Q_v = 2 * np.mean(voltage * sin_w)
    I_i = 2 * np.mean(current * cos_w)
    Q_i = 2 * np.mean(current * sin_w)
    v_complex = I_v + 1j * Q_v
    i_complex = I_i + 1j * Q_i
    z = v_complex / i_complex if abs(i_complex) > 1e-12 else complex(0, 0)
    return float(np.abs(z)), float(np.angle(z))


def parallel_equivalent(rp: float, xp: float) -> Tuple[float, float]:
    if rp == 0.0:
        return 0.0, 0.0
    z_sq = rp ** 2 + xp ** 2
    rs = rp * z_sq / (rp ** 2 + z_sq)
    xs = xp * z_sq / (rp ** 2 + z_sq)
    return rs, xs


def quality_factor(z_mag: float, z_phase: float) -> float:
    if abs(z_mag) < 1e-12:
        return 0.0
    r = z_mag * np.cos(z_phase)
    x = z_mag * np.sin(z_phase)
    if abs(r) < 1e-12:
        return float('inf')
    return float(abs(x / r))


def lcr_from_impedance(
    z_mag: float,
    z_phase: float,
    f_target: float,
) -> dict:
    if f_target == 0:
        return {'resistance': z_mag, 'reactance': 0.0, 'type': 'unknown'}
    r = z_mag * np.cos(z_phase)
    x = z_mag * np.sin(z_phase)
    omega = 2 * np.pi * f_target
    result = {'resistance': r, 'reactance': x}
    if abs(x) < 1e-12:
        result['type'] = 'resistance'
    elif x > 0:
        L = x / omega
        result['type'] = 'inductance'
        result['inductance'] = L
        result['q'] = quality_factor(z_mag, z_phase)
    else:
        C = -1.0 / (x * omega)
        result['type'] = 'capacitance'
        result['capacitance'] = C
        result['q'] = quality_factor(z_mag, z_phase)
    return result


def series_resonance(
    L: float = 0.0,
    C: float = 0.0,
    R: float = 0.0,
) -> dict:
    if L <= 0.0 or C <= 0.0:
        return {'f_resonance': 0.0, 'q': 0.0}
    f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
    q = (1.0 / R) * np.sqrt(L / C) if R > 0 else float('inf')
    return {'f_resonance': f0, 'q': q}


def parallel_resonance(L: float = 0.0, C: float = 0.0, R: float = float('inf')) -> dict:
    if L <= 0.0 or C <= 0.0:
        return {'f_resonance': 0.0, 'q': 0.0}
    f0 = 1.0 / (2 * np.pi * np.sqrt(L * C))
    q = R * np.sqrt(C / L) if R < float('inf') else float('inf')
    return {'f_resonance': f0, 'q': q}


def detect_resonance(
    freqs: np.ndarray,
    impedance_mag: np.ndarray,
) -> Optional[float]:
    if len(freqs) < 3:
        return None
    idx = np.argmin(impedance_mag)
    if idx == 0 or idx == len(freqs) - 1:
        idx2 = np.argmax(impedance_mag)
        if idx2 == 0 or idx2 == len(freqs) - 1:
            return None
        return float(freqs[idx2])
    return float(freqs[idx])
