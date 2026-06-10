"""Example: Goertzel frequency detection (2023-H)."""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from signal_toolkit import goertzel as gz
from signal_toolkit import dds_synthesis as dds

fs = 500_000

f_actual = [23000, 47000, 81000]
amps = [1.0, 0.7, 0.4]

mixed = np.zeros(int(fs * 0.05))
for f, a in zip(f_actual, amps):
    _, sig = dds.generate_sine(fs, f, duration=0.05, amplitude=a)
    mixed += sig

print("=== Goertzel Targeted Detection ===")
test_freqs = [20000, 23000, 25000, 40000, 47000, 50000, 80000, 81000, 82000]
results = gz.goertzel_bank(mixed, test_freqs, fs)
for freq, mag, phase in results:
    marker = " <<<" if abs(mag) > 0.3 else ""
    print(f"  f={freq:5d}Hz: mag={mag:.4f}, phase={phase:.3f}rad{marker}")

print("\n=== Goertzel Frequency Scan ===")
detected = gz.detect_frequencies(mixed, fs, f_min=10000, f_max=100000, step=500, threshold=0.1)
for freq, mag in detected:
    print(f"  Detected: {freq:.0f}Hz (mag={mag:.3f})")

fig, ax = plt.subplots(figsize=(10, 4))
f_scan = np.arange(10000, 100001, 500)
mags = []
for f in f_scan:
    mag, _ = gz.goertzel(mixed[:min(len(mixed), 2000)], f, fs)
    mags.append(mag)
ax.plot(f_scan, mags)
ax.set_title("Goertzel Frequency Scan (10-100kHz)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
for f in f_actual:
    ax.axvline(x=f, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'example_goertzel.png'), dpi=100)
print("Saved example_goertzel.png")
