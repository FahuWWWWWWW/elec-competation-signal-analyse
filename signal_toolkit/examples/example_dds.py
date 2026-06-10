"""Example: DDS signal generation (2022-F, 2024-C, 2025-G)."""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from signal_toolkit import dds_synthesis as dds

fs = 10_000_000
fig, axes = plt.subplots(3, 2, figsize=(12, 8))

t, sine = dds.generate_sine(fs, 500_000, duration=0.0001)
axes[0, 0].plot(t * 1e6, sine)
axes[0, 0].set_title("Sine 500kHz")
axes[0, 0].set_xlabel("Time (us)")

t, am = dds.generate_am(fs, f_carrier=2_000_000, f_mod=50_000, depth=0.5, duration=0.0002)
axes[0, 1].plot(t * 1e6, am)
axes[0, 1].set_title("AM 2MHz/50kHz m=0.5")
axes[0, 1].set_xlabel("Time (us)")

t, fm = dds.generate_fm(fs, f_carrier=2_000_000, f_dev=50_000, mod_freq=10_000, duration=0.0003)
axes[1, 0].plot(t * 1e6, fm)
axes[1, 0].set_title("FM 2MHz/50kHz dev")
axes[1, 0].set_xlabel("Time (us)")

t, sweep = dds.generate_sweep(fs, 100_000, 1_000_000, duration=0.001)
axes[1, 1].specgram(sweep, NFFT=256, Fs=fs, noverlap=200)
axes[1, 1].set_title("Sweep 100k-1MHz")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Frequency (Hz)")

t, pulse = dds.generate_pulse(fs, duration=0.000002, tr=1e-9, pw=10e-9)
axes[2, 0].plot(t * 1e9, pulse)
axes[2, 0].set_title("TDR Pulse tr=1ns pw=10ns")
axes[2, 0].set_xlabel("Time (ns)")

dds_obj = dds.DDS(f_clk=125e6, phase_bits=32)
ftw = dds_obj.frequency_to_ftw(1_000_000)
f_out = dds_obj.ftw_to_frequency(ftw)
axes[2, 1].text(0.1, 0.6, f"DDS @125MHz clk\n32-bit accumulator\nFTW for 1MHz: {ftw}\nf_out from FTW: {f_out:.3f}Hz\nResolution: {dds_obj.frequency_resolution:.4f}Hz",
                fontsize=10, transform=axes[2, 1].transAxes)
axes[2, 1].axis('off')
axes[2, 1].set_title("DDS Parameter Verification")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'example_dds.png'), dpi=100)
print("Saved example_dds.png")
