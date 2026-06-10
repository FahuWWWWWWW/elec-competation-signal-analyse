"""Common utility functions for signal processing toolkit."""

import numpy as np


def dbm_to_vpp(dbm, impedance=50):
    p_w = 10.0 ** (np.asarray(dbm, dtype=float) / 10.0) * 1e-3
    vrms = np.sqrt(p_w * impedance)
    vpp = 2.0 * np.sqrt(2.0) * vrms
    return vpp


def vpp_to_dbm(vpp, impedance=50):
    vrms = np.asarray(vpp, dtype=float) / (2.0 * np.sqrt(2.0))
    p_w = vrms ** 2 / impedance
    dbm = 10.0 * np.log10(p_w / 1e-3)
    return dbm


def snr(signal, noise):
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    p_signal = np.mean(signal ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0.0:
        return float('inf')
    return 10.0 * np.log10(p_signal / p_noise)


def next_power_of_two(n):
    return 1 << (int(np.ceil(np.log2(n))))


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    w = np.ones(window) / window
    return np.convolve(x, w, mode='same')
