"""Example: TDR cable analysis (2023-B, 2025-D)."""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from signal_toolkit import tdr_analysis as tdr

fs = 100_000_000
vf = 0.67

analyzer = tdr.TDR(fs=fs, vf=vf)

print("=== Equivalent Sampling Verification ===")
t_eff, res = analyzer.equivalent_sampling(100, 40, 1e-6)
f_eff = 1.0 / res
print(f"  Real ADC: {fs/1e6:.0f}MSPS, M=40")
print(f"  Effective rate: {f_eff/1e9:.1f}GSPS")
print(f"  Time resolution: {res*1e12:.1f}ps")
print(f"  Spatial resolution: {3e8*vf*res/2*100:.1f}cm")

print("\n=== Reflection Coefficient ===")
for z_load in [1e6, 0.1, 75, 50]:
    gamma = analyzer.reflection_coefficient(z_load)
    print(f"  Z_load={z_load:8.1f}ohm: Gamma={gamma:+.3f}")

print("\n=== Cable Fault Simulation ===")
faults = ['open', 'short', 'load_75ohm', 'load_150ohm']
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

for ax, fault in zip(axes.flatten(), faults):
    t, signal = analyzer.cable_model(length_m=5.0, fault_type=fault)
    ax.plot(t * 1e9, signal)
    ax.set_title(f"Fault: {fault} @5m")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")

    idx, amp = analyzer.detect_reflection(signal)
    if len(idx) > 1:
        dist = analyzer.compute_distance(idx[1] - idx[0])
        ax.axvline(x=t[idx[1]] * 1e9, color='r', linestyle='--', alpha=0.5)
        ax.annotate(f"{dist:.1f}m", xy=(t[idx[1]] * 1e9, 0))

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'example_tdr.png'), dpi=100)
print("Saved example_tdr.png")
