"""Example: THD computation and spectrum analysis (2021-A)."""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from signal_toolkit import fft_analysis as fa
from signal_toolkit import dds_synthesis as dds

fs = 1_000_000
f0 = 100_000
t, sig = dds.generate_sine(fs, f0, duration=0.01)
harmonics = [2, 3, 4, 5]
for h, amp in zip(harmonics, [0.03, 0.015, 0.008, 0.004]):
    _, h_sig = dds.generate_sine(fs, f0 * h, duration=0.01, amplitude=amp)
    sig += h_sig

thd_pct, h_data = fa.compute_thd(sig, fs, f0, num_harmonics=5)
print(f"THD = {thd_pct:.3f}%")
print(f"Harmonics: { {k: f'{v:.5f}' for k, v in h_data.items()} }")

freqs, mag, phase = fa.compute_spectrum(sig, fs, window='flat_top')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(t[:200], sig[:200])
ax1.set_title("Time Domain (200 samples)")
ax1.set_xlabel("Time (s)")
mask = freqs <= 600_000
ax2.stem(freqs[mask], mag[mask], basefmt=' ')
ax2.set_title("Flat-top Window Spectrum")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'example_fft.png'), dpi=100)
print("Saved example_fft.png")
