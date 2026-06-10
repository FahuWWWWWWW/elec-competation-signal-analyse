"""I/Q demodulation and coherent detection."""

import numpy as np
from scipy import signal as scipy_signal


def _design_lpf(cutoff, fs, order=5):
    nyq = 0.5 * fs
    norm_cutoff = min(cutoff / nyq, 0.99)
    b, a = scipy_signal.butter(order, norm_cutoff, btype='low')
    return b, a


def _apply_lpf(x, cutoff, fs, order=5):
    b, a = _design_lpf(cutoff, fs, order)
    return scipy_signal.filtfilt(b, a, x)


def _trim_signal(x, trim_ratio=0.15):
    """Remove start/end transients by trimming given ratio from both ends."""
    n = len(x)
    start = int(n * trim_ratio)
    end = n - start
    return x[start:end] if end > start else x


def iq_demodulate(signal, f_carrier, fs, cutoff=None):
    t = np.arange(len(signal)) / fs
    lo_i = np.cos(2 * np.pi * f_carrier * t)
    lo_q = -np.sin(2 * np.pi * f_carrier * t)
    i_mix = signal * lo_i
    q_mix = signal * lo_q
    if cutoff is None:
        cutoff = min(f_carrier * 0.2, fs * 0.4)
    if cutoff > fs * 0.49:
        cutoff = fs * 0.49
    I = _apply_lpf(i_mix, cutoff, fs)
    Q = _apply_lpf(q_mix, cutoff, fs)
    return I, Q


def demodulate_am(signal, fs, f_carrier=None, cutoff=None):
    if f_carrier is not None:
        I, Q = iq_demodulate(signal, f_carrier, fs, cutoff=cutoff)
        envelope = np.sqrt(I ** 2 + Q ** 2)
    else:
        analytic = scipy_signal.hilbert(signal)
        envelope = np.abs(analytic)
    env_trimmed = _trim_signal(envelope, trim_ratio=0.15)
    env_low = np.percentile(env_trimmed, 5)
    env_high = np.percentile(env_trimmed, 95)
    if abs(env_high + env_low) > 1e-12:
        modulation_index = (env_high - env_low) / (env_high + env_low)
    else:
        modulation_index = 0.0
    return envelope, modulation_index


def demodulate_fm(signal, fs, f_carrier=None, f_dev=None, cutoff=None):
    if f_carrier is not None:
        if cutoff is None:
            if f_dev is not None:
                cutoff = f_dev * 2.5
            else:
                cutoff = f_carrier * 0.4
        I, Q = iq_demodulate(signal, f_carrier, fs, cutoff=cutoff)
        inst_phase = np.unwrap(np.arctan2(Q, I))
        demod = np.diff(inst_phase) / (2 * np.pi) * fs
        demod = np.concatenate([[demod[0]], demod])
        demod = demod - f_carrier
    else:
        analytic = scipy_signal.hilbert(signal)
        inst_phase = np.unwrap(np.angle(analytic))
        demod = np.diff(inst_phase) / (2 * np.pi) * fs
        demod = np.concatenate([[demod[0]], demod])
        if f_carrier is not None:
            demod = demod - f_carrier
        else:
            demod = demod - np.mean(demod)
    demod_trimmed = _trim_signal(demod, 0.15)
    freq_deviation = np.std(demod_trimmed) * np.sqrt(2)
    return demod, freq_deviation


def demodulate_pm(signal, fs, f_carrier=None):
    if f_carrier is not None:
        I, Q = iq_demodulate(signal, f_carrier, fs)
        inst_phase = np.unwrap(np.arctan2(Q, I))
    else:
        analytic = scipy_signal.hilbert(signal)
        inst_phase = np.unwrap(np.angle(analytic))
    inst_phase_trimmed = _trim_signal(inst_phase, 0.15)
    inst_phase -= np.mean(inst_phase_trimmed)
    return inst_phase


def coherent_demodulate(signal, f_carrier, f_target, fs, phase=0):
    t = np.arange(len(signal)) / fs
    ref = np.cos(2 * np.pi * f_target * t + phase)
    mixed = signal * ref
    delta_f = abs(f_carrier - f_target)
    if delta_f < 1:
        cutoff = min(abs(f_target) * 0.15, fs * 0.4)
    else:
        cutoff = min(delta_f * 0.5, fs * 0.4)
    if cutoff < 1:
        cutoff = 100
    baseband = _apply_lpf(mixed, cutoff, fs)
    n = len(signal)
    trim = int(n * 0.2)
    valid = baseband[trim:n - trim] if n > 2 * trim else baseband
    ref_q = -np.sin(2 * np.pi * f_target * t + phase)
    mixed_q = signal * ref_q
    baseband_q = _apply_lpf(mixed_q, cutoff, fs)
    valid_q = baseband_q[trim:n - trim] if n > 2 * trim else baseband_q
    I = 2 * np.mean(valid) if n > 2 * trim else 0
    Q = 2 * np.mean(valid_q) if n > 2 * trim else 0
    amplitude = np.sqrt(I ** 2 + Q ** 2)
    phase_out = np.arctan2(Q, I)
    reconstructed = amplitude * np.cos(2 * np.pi * f_target * t + phase_out)
    return amplitude, phase_out, reconstructed


def separate_signals(mixed_signal, freq_list, fs):
    n = len(mixed_signal)
    t = np.arange(n) / fs
    components = []
    amplitudes = []
    residual = mixed_signal.copy()
    for f in sorted(freq_list):
        amp, phase, recon = coherent_demodulate(mixed_signal, f, f, fs, 0)
        amplitudes.append(amp)
        components.append(recon)
        residual = residual - recon
    return components, amplitudes
