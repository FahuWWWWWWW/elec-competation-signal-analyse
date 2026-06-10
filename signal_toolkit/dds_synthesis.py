"""DDS (Direct Digital Synthesis) signal generation."""

import numpy as np


class DDS:
    """DDS signal generator with phase accumulator.

    Models a hardware DDS (e.g. AD9834, AD9850) with:
    - N-bit phase accumulator (default 32-bit)
    - M-bit amplitude lookup table (default 12-bit)

    Frequency resolution: f_clk / 2^N
    """

    def __init__(self, f_clk=125e6, phase_bits=32, amp_bits=12, fs=None):
        self.f_clk = f_clk
        self.phase_bits = phase_bits
        self.amp_bits = amp_bits
        self.fs = fs if fs is not None else f_clk
        self._phase_acc = 0
        self._ftw = 0

    @property
    def frequency_resolution(self):
        return self.f_clk / (1 << self.phase_bits)

    def frequency_to_ftw(self, f_out):
        return int(round(f_out * (1 << self.phase_bits) / self.f_clk))

    def ftw_to_frequency(self, ftw):
        return ftw * self.f_clk / (1 << self.phase_bits)

    def set_frequency(self, f_out):
        self._ftw = self.frequency_to_ftw(f_out)
        return self._ftw

    def generate(self, duration, phase=0):
        n_samples = int(round(duration * self.fs))
        t = np.arange(n_samples) / self.fs
        ftw = self._ftw
        phase_inc = ftw / (1 << self.phase_bits)
        phase_rad = 2 * np.pi * (phase + phase * (1 >> self.phase_bits))
        accum = self._phase_acc + phase_inc * np.arange(n_samples)
        self._phase_acc = (self._phase_acc + phase_inc * n_samples) % 1.0
        amp = (1 << (self.amp_bits - 1)) - 1
        y = amp * np.sin(2 * np.pi * np.mod(accum, 1.0) + phase)
        return t, y


def generate_sine(fs, f_out, duration, amplitude=1.0, phase=0):
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    y = amplitude * np.sin(2 * np.pi * f_out * t + phase)
    return t, y


def generate_am(fs, f_carrier, f_mod, depth=0.5, duration=1.0, amplitude=1.0):
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    y = amplitude * (1 + depth * np.cos(2 * np.pi * f_mod * t)) * np.cos(2 * np.pi * f_carrier * t)
    return t, y


def generate_fm(fs, f_carrier, f_dev, mod_freq, duration=1.0, amplitude=1.0):
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    beta = f_dev / mod_freq
    y = amplitude * np.cos(2 * np.pi * f_carrier * t + beta * np.sin(2 * np.pi * mod_freq * t))
    return t, y


def generate_sweep(fs, f_start, f_stop, duration=1.0, amplitude=1.0, method='linear'):
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


def generate_pulse(fs, duration, tr=1e-9, pw=10e-9, amplitude=1.0):
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
