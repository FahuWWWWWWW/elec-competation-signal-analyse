"""I/Q demodulation and coherent detection."""

import numpy as np
from scipy import signal as scipy_signal


def _design_lpf(cutoff, fs, order=5):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = scipy_signal.butter(order, norm_cutoff, btype='low')
    return b, a


def _apply_lpf(x, cutoff, fs, order=5):
    b, a = _design_lpf(cutoff, fs, order)
    return scipy_signal.filtfilt(b, a, x)


def iq_demodulate(signal, f_carrier, fs):
    t = np.arange(len(signal)) / fs
    lo_i = np.cos(2 * np.pi * f_carrier * t)
    lo_q = -np.sin(2 * np.pi * f_carrier * t)
    i_mix = signal * lo_i
    q_mix = signal * lo_q
    cutoff = f_carrier * 0.1 if f_carrier > 0 else fs * 0.01
    if cutoff > fs * 0.5:
        cutoff = fs * 0.4
    I = _apply_lpf(i_mix, cutoff, fs)
    Q = _apply_lpf(q_mix, cutoff, fs)
    return I, Q


def demodulate_am(signal, fs, f_carrier=None):
    if f_carrier is not None:
        I, Q = iq_demodulate(signal, f_carrier, fs)
        envelope = np.sqrt(I ** 2 + Q ** 2)
    else:
        analytic = scipy_signal.hilbert(signal)
        envelope = np.abs(analytic)
    env_mean = np.mean(envelope)
    env_min = np.min(envelope)
    env_max = np.max(envelope)
    if abs(env_max + env_min) > 1e-12:
        modulation_index = (env_max - env_min) / (env_max + env_min)
    else:
        modulation_index = 0.0
    return envelope, modulation_index


def demodulate_fm(signal, fs, f_carrier=None, f_dev=None):
    if f_carrier is not None:
        I, Q = iq_demodulate(signal, f_carrier, fs)
        inst_phase = np.unwrap(np.arctan2(Q, I))
        demod = np.diff(inst_phase) / (2 * np.pi) * fs
        demod = np.concatenate([[demod[0]], demod])
        demod = _apply_lpf(demod, fs * 0.1, fs, order=3)
    else:
        analytic = scipy_signal.hilbert(signal)
        inst_phase = np.unwrap(np.angle(analytic))
        demod = np.diff(inst_phase) / (2 * np.pi) * fs
        demod = np.concatenate([[demod[0]], demod])
        demod = _apply_lpf(demod, fs * 0.1, fs, order=3)
    if f_carrier is not None:
        demod = demod - f_carrier
    else:
        demod = demod - np.mean(demod)
    if f_dev is not None:
        freq_deviation = np.std(demod)
    else:
        freq_deviation = np.std(demod)
    return demod, freq_deviation


def demodulate_pm(signal, fs, f_carrier=None):
    if f_carrier is not None:
        I, Q = iq_demodulate(signal, f_carrier, fs)
        inst_phase = np.unwrap(np.arctan2(Q, I))
    else:
        analytic = scipy_signal.hilbert(signal)
        inst_phase = np.unwrap(np.angle(analytic))
    inst_phase -= np.mean(inst_phase)
    return inst_phase


def coherent_demodulate(signal, f_carrier, f_target, fs, phase=0):
    t = np.arange(len(signal)) / fs
    ref = np.cos(2 * np.pi * f_target * t + phase)
    mixed = signal * ref
    cutoff = abs(f_carrier - f_target) * 0.5
    if cutoff <= 0 or cutoff > fs * 0.5:
        cutoff = min(abs(f_target) * 0.1, fs * 0.4)
    baseband = _apply_lpf(mixed, cutoff, fs)
    n = len(signal)
    I = 2 * np.mean(baseband[int(n * 0.25):]) if n > 10 else np.mean(baseband)
    ref_q = -np.sin(2 * np.pi * f_target * t + phase)
    mixed_q = signal * ref_q
    baseband_q = _apply_lpf(mixed_q, cutoff, fs)
    Q = 2 * np.mean(baseband_q[int(n * 0.25):]) if n > 10 else np.mean(baseband_q)
    amplitude = np.sqrt(I ** 2 + Q ** 2)
    phase_out = np.arctan2(Q, I)
    reconstructed = amplitude * np.cos(2 * np.pi * f_target * t + phase_out)
    return amplitude, phase_out, reconstructed


def separate_signals(mixed_signal, freq_list, fs):
    n = len(mixed_signal)
    t = np.arange(n) / fs
    components = []
    amplitudes = []
    for f in freq_list:
        _, phase, recon = coherent_demodulate(mixed_signal, f, f, fs, 0)
        amp = np.sqrt(2) * np.std(recon)
        amplitudes.append(amp)
        components.append(recon)
    return components, amplitudes
