"""Goertzel algorithm for efficient single-frequency DFT computation."""

import numpy as np


def goertzel(samples, target_freq, fs):
    samples = np.asarray(samples, dtype=float)
    N = len(samples)
    if N == 0:
        return 0.0, 0.0
    omega = 2.0 * np.pi * target_freq / fs
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)
    coeff = 2.0 * cos_w
    v1 = 0.0
    v2 = 0.0
    for s in samples:
        v0 = s + coeff * v1 - v2
        v2 = v1
        v1 = v0
    real = v1 - v2 * cos_w
    imag = v2 * sin_w
    magnitude = np.sqrt(real ** 2 + imag ** 2)
    amplitude = 2.0 * magnitude / N
    phase = np.arctan2(imag, real)
    return amplitude, phase


def goertzel_bank(samples, freq_list, fs):
    samples = np.asarray(samples, dtype=float)
    results = []
    for f in freq_list:
        amp, phase = goertzel(samples, f, fs)
        results.append((f, amp, phase))
    return results


def detect_frequencies(samples, fs, f_min, f_max, step=10, threshold=0.01):
    samples = np.asarray(samples, dtype=float)
    N_freqs = int(np.floor((f_max - f_min) / step)) + 1
    freqs = np.linspace(f_min, f_max, N_freqs)
    magnitudes = np.array([goertzel(samples, f, fs)[0] for f in freqs],
                          dtype=float)
    max_mag = np.max(magnitudes)
    if max_mag == 0.0:
        return []
    peaks = []
    for i in range(1, N_freqs - 1):
        if (magnitudes[i] > magnitudes[i - 1]
                and magnitudes[i] >= magnitudes[i + 1]
                and magnitudes[i] > threshold * max_mag):
            peaks.append((freqs[i], magnitudes[i]))
    return peaks
