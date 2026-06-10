"""Example: I/Q demodulation and signal separation (2023-H, 2022-F)."""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from signal_toolkit import iq_demodulation as iq
from signal_toolkit import dds_synthesis as dds

fs = 500_000

print("=== IQ Demodulation: AM Signal ===")
t, am_sig = dds.generate_am(fs, f_carrier=100_000, f_mod=5000, depth=0.5, duration=0.005)
envelope, m_est = iq.demodulate_am(am_sig, fs, f_carrier=100_000)
print(f"  Actual m=0.50, Estimated m={m_est:.3f}")

print("\n=== IQ Demodulation: FM Signal ===")
t, fm_sig = dds.generate_fm(fs, f_carrier=100_000, f_dev=20000, mod_freq=3000, duration=0.005)
demod, f_dev_est = iq.demodulate_fm(fm_sig, fs, f_carrier=100_000)
print(f"  Actual dev=20kHz, Estimated dev={f_dev_est/1000:.1f}kHz")

print("\n=== Coherent Demodulation: Signal Separation ===")
f_sep = [23000, 55000, 82000]
amps_sep = [1.0, 0.6, 0.3]
mixed = np.zeros(int(fs * 0.02))
for f, a in zip(f_sep, amps_sep):
    _, sig = dds.generate_sine(fs, f, duration=0.02, amplitude=a)
    mixed += sig

components, amplitudes = iq.separate_signals(mixed, f_sep, fs)
print(f"  Actual amps: {amps_sep}")
print(f"  Extracted amps: {[f'{a:.3f}' for a in amplitudes]}")

fig, axes = plt.subplots(len(f_sep) + 1, 1, figsize=(10, 8))
axes[0].plot(t[:500], mixed[:500])
axes[0].set_title("Mixed Signal (3 components)")
for i, (comp, f) in enumerate(zip(components, f_sep)):
    axes[i + 1].plot(t[:500], comp[:500])
    axes[i + 1].set_title(f"Separated: {f/1000:.0f}kHz (amp={amplitudes[i]:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'example_iq.png'), dpi=100)
print("Saved example_iq.png")
