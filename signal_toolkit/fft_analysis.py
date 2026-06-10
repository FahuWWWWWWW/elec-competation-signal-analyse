"""FFT analysis, window functions and THD computation."""

import numpy as np
from scipy import signal as scipy_signal


def flat_top_window(N):
    n = np.arange(N)
    a0, a1, a2, a3, a4 = 0.2156, 0.4160, 0.2778, 0.0836, 0.0068
    return (a0
            - a1 * np.cos(2.0 * np.pi * n / (N - 1))
            + a2 * np.cos(4.0 * np.pi * n / (N - 1))
            - a3 * np.cos(6.0 * np.pi * n / (N - 1))
            + a4 * np.cos(8.0 * np.pi * n / (N - 1)))


def hanning_window(N):
    return np.hanning(N)


def hamming_window(N):
    return np.hamming(N)


def blackman_harris_window(N):
    n = np.arange(N)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    return (a0
            - a1 * np.cos(2.0 * np.pi * n / (N - 1))
            + a2 * np.cos(4.0 * np.pi * n / (N - 1))
            - a3 * np.cos(6.0 * np.pi * n / (N - 1)))


def _get_window(window, N):
    if window is None:
        return np.ones(N)
    if callable(window):
        return window(N)
    w_map = {
        'hanning': hanning_window,
        'hamming': hamming_window,
        'flat_top': flat_top_window,
        'blackman_harris': blackman_harris_window,
    }
    return w_map[window](N)


def compute_spectrum(signal, fs, window='hanning', n_fft=None, detrend=True):
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    if n_fft is None:
        n_fft = 1 << (int(np.ceil(np.log2(N))))

    x = signal.copy()
    if detrend:
        x -= np.mean(x)

    w = _get_window(window, N)
    x = x * w

    X = np.fft.rfft(x, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    mag = np.abs(X) / np.sum(w)
    mag[1:] *= 2.0
    if n_fft % 2 == 0:
        mag[-1] /= 2.0

    phase = np.angle(X)

    return freqs, mag, phase


def find_fundamental(freqs, mag, f_range=(None, None)):
    lo = f_range[0] if f_range[0] is not None else freqs[0]
    hi = f_range[1] if f_range[1] is not None else freqs[-1]
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return None
    indices = np.where(mask)[0]
    return indices[np.argmax(mag[indices])]


def find_harmonics(freqs, mag, fundamental_freq, num_harmonics=5):
    harmonics = {}
    delta_f = freqs[1] - freqs[0]
    for order in range(2, num_harmonics + 2):
        target = order * fundamental_freq
        tol = max(delta_f * 3.0, fundamental_freq * 0.02)
        mask = (freqs >= target - tol) & (freqs <= target + tol)
        if np.any(mask):
            idx = np.where(mask)[0][np.argmax(mag[mask])]
            harmonics[order] = mag[idx]
        else:
            harmonics[order] = 0.0
    return harmonics


def compute_thd(signal, fs, fundamental_freq, num_harmonics=5, window='flat_top'):
    freqs, mag, _ = compute_spectrum(signal, fs, window=window)
    tol = fundamental_freq * 0.1
    f0_idx = find_fundamental(freqs, mag,
                              f_range=(fundamental_freq - tol,
                                       fundamental_freq + tol))
    if f0_idx is None:
        return 0.0, {}
    f0_actual = freqs[f0_idx]
    h1 = mag[f0_idx]
    if h1 <= 0.0:
        return 0.0, {}
    harmonics = find_harmonics(freqs, mag, f0_actual, num_harmonics)
    h_sq_sum = sum(v ** 2 for v in harmonics.values())
    thd = 100.0 * np.sqrt(h_sq_sum) / h1
    return thd, harmonics
