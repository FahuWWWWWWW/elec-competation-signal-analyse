"""DDS (Direct Digital Synthesis) signal generation."""

from typing import Tuple
import numpy as np


class DDS:
    """DDS signal generator with phase accumulator.

    Models a hardware DDS (e.g. AD9834, AD9850) with:
    - N-bit phase accumulator (default 32-bit)
    - M-bit amplitude lookup table (default 12-bit)

    Frequency resolution: f_clk / 2^N
    """

    def __init__(
        self,
        f_clk: float = 125e6,
        phase_bits: int = 32,
        amp_bits: int = 12,
        fs: float = None,
    ):
        self.f_clk = f_clk
        self.phase_bits = phase_bits
        self.amp_bits = amp_bits
        self.fs = fs if fs is not None else f_clk
        self._phase_acc: float = 0
        self._ftw: int = 0

    @property
    def frequency_resolution(self) -> float:
        return self.f_clk / (1 << self.phase_bits)

    def frequency_to_ftw(self, f_out: float) -> int:
        return int(round(f_out * (1 << self.phase_bits) / self.f_clk))

    def ftw_to_frequency(self, ftw: int) -> float:
        return ftw * self.f_clk / (1 << self.phase_bits)

    def set_frequency(self, f_out: float) -> int:
        self._ftw = self.frequency_to_ftw(f_out)
        return self._ftw

    def generate(self, duration: float, phase: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = int(round(duration * self.fs))
        t = np.arange(n_samples) / self.fs
        ftw = self._ftw
        phase_inc = ftw / (1 << self.phase_bits)
        accum = self._phase_acc + phase_inc * np.arange(n_samples)
        self._phase_acc = (self._phase_acc + phase_inc * n_samples) % 1.0
        amp = (1 << (self.amp_bits - 1)) - 1
        y = amp * np.sin(2 * np.pi * np.mod(accum, 1.0) + phase)
        return t, y


def generate_sine(
    fs: float,
    f_out: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    y = amplitude * np.sin(2 * np.pi * f_out * t + phase)
    return t, y


def generate_am(
    fs: float,
    f_carrier: float,
    f_mod: float,
    depth: float = 0.5,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    y = amplitude * (1 + depth * np.cos(2 * np.pi * f_mod * t)) * np.cos(2 * np.pi * f_carrier * t)
    return t, y


def generate_fm(
    fs: float,
    f_carrier: float,
    f_dev: float,
    mod_freq: float,
    duration: float = 1.0,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    beta = f_dev / mod_freq
    y = amplitude * np.cos(2 * np.pi * f_carrier * t + beta * np.sin(2 * np.pi * mod_freq * t))
    return t, y


def generate_sweep(
    fs: float,
    f_start: float,
    f_stop: float,
    duration: float = 1.0,
    amplitude: float = 1.0,
    method: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    if method == 'linear':
        f_inst = f_start + (f_stop - f_start) * t / duration
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
    elif method == 'log':
        f_inst = f_start * (f_stop / f_start) ** (t / duration)
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
    else:
        raise ValueError(f"Unknown sweep method: {method}")
    y = amplitude * np.sin(phase)
    return t, y


def generate_pulse(
    fs: float,
    duration: float,
    tr: float = 1e-9,
    pw: float = 10e-9,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    tr_samples = int(round(tr * fs))
    pw_samples = int(round(pw * fs))
    if tr_samples < 2:
        tr_samples = 2
    y = np.zeros(n)
    rise = amplitude / (1 + np.exp(-np.linspace(-6, 6, tr_samples)))
    fall = amplitude / (1 + np.exp(np.linspace(-6, 6, tr_samples)))
    rise_end = tr_samples
    fall_start = rise_end + pw_samples
    fall_end = fall_start + tr_samples
    if fall_end <= n:
        y[:tr_samples] = rise
        y[tr_samples:fall_start] = amplitude
        y[fall_start:fall_end] = fall
    elif fall_start < n:
        y[:tr_samples] = rise
        y[tr_samples:n] = amplitude
    else:
        y[:n] = amplitude * np.ones(n) if n <= tr_samples else np.concatenate([
            amplitude / (1 + np.exp(-np.linspace(-6, 6, n)))
        ])
    return t, y
