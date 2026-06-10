"""Filter design utilities for circuit modeling and signal conditioning."""

import numpy as np
from scipy import signal as scipy_signal


def design_lpf(f_cutoff, fs, order=4, rp=0.5):
    nyquist = 0.5 * fs
    normal_cutoff = f_cutoff / nyquist
    b, a = scipy_signal.butter(order, normal_cutoff, btype='low')
    return b, a


def design_bpf(f_low, f_high, fs, order=4):
    nyquist = 0.5 * fs
    normal_low = f_low / nyquist
    normal_high = f_high / nyquist
    b, a = scipy_signal.butter(order, [normal_low, normal_high],
                               btype='band')
    return b, a


def apply_filter(signal, b, a):
    signal = np.asarray(signal, dtype=float)
    return scipy_signal.filtfilt(b, a, signal)


def sallen_key_transfer(R1, R2, C1, C2):
    wn = 1.0 / np.sqrt(R1 * R2 * C1 * C2)
    zeta = 0.5 * wn * (R1 + R2) * C2
    Q = 1.0 / (2.0 * zeta) if zeta > 0.0 else float('inf')
    return wn, zeta, Q


def butterworth_order(f_pass, f_stop, fs, A_pass=3, A_stop=40):
    ratio = f_stop / f_pass
    if ratio <= 1.0:
        return 0
    numerator = 10.0 ** (A_stop / 10.0) - 1.0
    denominator = 10.0 ** (A_pass / 10.0) - 1.0
    n = 0.5 * np.log10(numerator / denominator) / np.log10(ratio)
    return int(np.ceil(n))
