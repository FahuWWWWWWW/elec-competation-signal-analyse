"""TDR (Time Domain Reflectometry) analysis for cable testing."""

from typing import Dict, Tuple, Union
import numpy as np

c: float = 299792458


class TDR:
    """TDR cable fault detection and analysis.

    Supports both real-time sampling and equivalent-time sampling.
    Used in 2023-B (coaxial cable) and 2025-D (twisted pair) problems.
    """

    def __init__(self, fs: float, vf: float = 0.67):
        self.fs = fs
        self.dt = 1.0 / fs
        self.vf = vf
        self.v = c * vf

    def equivalent_sampling(
        self,
        real_samples: int,
        m: int,
        trigger_period: float,
    ) -> Tuple[np.ndarray, float]:
        dt_real = 1.0 / self.fs
        dt_eff = dt_real / m
        t_eff = np.arange(0, real_samples * dt_real, dt_eff)
        self.fs_eff = 1.0 / dt_eff
        return t_eff, dt_eff

    def detect_reflection(
        self,
        signal: np.ndarray,
        threshold: float = 0.05,
        min_distance: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find incident and reflected pulse peaks using envelope detection."""
        envelope = np.abs(signal)
        env_min = envelope.min()
        env_max = envelope.max()
        if env_max - env_min < 1e-12:
            return np.array([], dtype=int), np.array([])
        envelope = (envelope - env_min) / (env_max - env_min)
        padded = np.pad(envelope, 1, mode='constant', constant_values=0)
        raw_mid = np.zeros_like(envelope, dtype=bool)
        for i in range(len(envelope)):
            pi = i + 1
            if (padded[pi] > threshold
                    and padded[pi] >= padded[pi - 1]
                    and padded[pi] >= padded[pi + 1]):
                raw_mid[i] = True
        if min_distance > 0:
            indices = np.where(raw_mid)[0]
            if len(indices) == 0:
                return np.array([], dtype=int), np.array([])
            filtered = [indices[0]]
            for idx in indices[1:]:
                if idx - filtered[-1] >= min_distance:
                    filtered.append(idx)
            indices = np.array(filtered)
        else:
            indices = np.where(raw_mid)[0]
        amplitudes = signal[indices]
        return indices, amplitudes

    def compute_distance(self, delay_samples: int) -> float:
        t = delay_samples / self.fs
        return c * self.vf * t / 2.0

    @staticmethod
    def reflection_coefficient(z_load: float, z0: float = 50) -> float:
        if np.isinf(z_load):
            return 1.0
        if z_load == 0:
            return -1.0
        return (z_load - z0) / (z_load + z0)

    def cable_model(
        self,
        length_m: float,
        fault_type: str = 'open',
        z0: float = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic TDR waveform with oversampling for accuracy.

        Internal oversampling factor = 10 to ensure smooth pulse edges.
        """
        osr = 10
        fs_int = self.fs * osr
        dt_int = 1.0 / fs_int
        v = self.v
        t_round = 2 * length_m / v
        duration = max(t_round * 4, 50e-9)
        n_int = int(round(duration * fs_int))
        n_int = max(n_int, 100)
        t_int = np.arange(n_int) / fs_int
        tr = max(2.0 / fs_int, 1e-9)
        sigma = tr / 2.355
        tau = 3 * sigma
        incident = np.exp(-((t_int - tau) ** 2) / (2 * sigma ** 2))
        delay_idx = int(round(t_round / dt_int))
        gamma_map: Dict[str, float] = {
            'open': 1.0,
            'short': -1.0,
            'load_75ohm': (75 - z0) / (75 + z0),
            'load_150ohm': (150 - z0) / (150 + z0),
        }
        gamma = gamma_map.get(fault_type, 0.0)
        reflected = np.zeros(n_int)
        if delay_idx < n_int:
            n_ref = n_int - delay_idx
            ref_pulse = gamma * np.exp(-((t_int[:n_ref] - tau) ** 2) / (2 * sigma ** 2))
            reflected[delay_idx:] = ref_pulse
        sig_int = incident + reflected
        sig = sig_int[::osr]
        n_out = len(sig)
        t = np.arange(n_out) / self.fs
        return sig, t


def equivalent_sampling(
    real_waveform: np.ndarray,
    m: int,
    trigger_period: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(real_waveform)
    dt_real = trigger_period / n
    dt_eff = dt_real / m
    t_eff = np.arange(0, trigger_period, dt_eff)
    t_real = np.linspace(0, trigger_period, n, endpoint=False)
    equiv = np.interp(t_eff, t_real, real_waveform)
    return equiv, t_eff


def detect_reflection(
    waveform: np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    envelope = np.abs(np.asarray(waveform, dtype=float))
    envelope -= envelope.min()
    envelope /= envelope.max() + 1e-12
    peaks: list = []
    for i in range(1, len(envelope) - 1):
        if (envelope[i] > threshold and envelope[i] >= envelope[i - 1]
                and envelope[i] >= envelope[i + 1]):
            peaks.append(i)
    return np.array(peaks)


def compute_distance(time_delay: float, vf: float = 0.67) -> float:
    return c * vf * time_delay / 2.0


def cable_model(
    length: float,
    fault_type: str = 'open',
    vf: float = 0.67,
    fs: float = 100e6,
) -> Tuple[np.ndarray, np.ndarray]:
    tdr = TDR(fs=fs, vf=vf)
    return tdr.cable_model(length, fault_type=fault_type)
