"""
2023年电赛C题「电感电容测量装置」阻抗测量仿真
目标: 验证阻抗电压法测量C/L/D/Q的可行性和精度
技术: 恒流源驱动 + DFT相位检测 + 阻抗计算
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 输出目录
output_dir = 'simulation_output'
os.makedirs(output_dir, exist_ok=True)

print("=== 2023-C LCR Measurement Simulation ===")
print("Technique: VI Method + DFT Phase Detection")
print()


def generate_sine_wave(t, f, A, phi=0):
    """生成正弦波"""
    return A * np.sin(2 * np.pi * f * t + phi)


def dft_single_bin(samples, fs, f_test, N):
    """DFT提取单一频率分量 (矩形窗)"""
    k = int(round(f_test / fs * N))
    n = np.arange(N)
    # 复数相关
    real = np.sum(samples * np.cos(2 * np.pi * k * n / N))
    imag = -np.sum(samples * np.sin(2 * np.pi * k * n / N))
    return (real + 1j * imag) / N


def measure_impedance(v_samples, i_samples, fs, f_test):
    """通过DFT测量阻抗"""
    N = len(v_samples)
    V = dft_single_bin(v_samples, fs, f_test, N)
    I = dft_single_bin(i_samples, fs, f_test, N)
    
    Z = V / I if abs(I) > 1e-15 else np.inf
    Z_mag = np.abs(Z)
    Z_phase = np.angle(Z, deg=True)
    
    return Z_mag, Z_phase


# ==================== Test 1: 电容阻抗频率特性 ====================
print("Test 1: Capacitor Impedance vs Frequency")

C_values = [1e-9, 10e-9, 100e-9]  # 1nF, 10nF, 100nF
D_values = [0.01, 0.05, 0.1]  # D值
freq_range = np.logspace(3, 6, 500)  # 1kHz ~ 1MHz

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1a: 容抗 vs 频率 (不同C值)
ax = axes[0, 0]
for C in C_values:
    Xc = 1 / (2 * np.pi * freq_range * C)
    ax.loglog(freq_range, Xc, linewidth=2, label=f'C = {C*1e9:.0f} nF')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('|Zc| = 1/(ωC) (Ω)')
ax.set_title('(a) Capacitive Reactance vs Frequency')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# 1b: 相位 vs 频率 (不同D值)
ax = axes[0, 1]
C_fixed = 10e-9  # 10nF
for D in D_values:
    # 相位角 = -arctan(1/D) = -90° + arctan(D) (低D近似)
    # 更精确: tan(δ) = D, δ是损耗角, 相位角 = -(90° - δ)
    phase = -(90 - np.arctan(D) * 180 / np.pi) * np.ones_like(freq_range)
    ax.semilogx(freq_range, phase, linewidth=2, label=f'D = {D:.3f}')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Phase Angle (°)')
ax.set_title('(b) Capacitor Phase vs Frequency (C=10nF)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# 1c: 阻抗实部(Rs) vs 频率
ax = axes[1, 0]
for C in C_values:
    for D in [0.01, 0.1]:
        Rs = D / (2 * np.pi * freq_range * C)
        ax.loglog(freq_range, Rs, linewidth=1.5, label=f'C={C*1e9:.0f}nF, D={D}')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Equivalent Series Resistance Rs (Ω)')
ax.set_title('(c) ESR vs Frequency')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# 1d: 阻抗虚部(Xc) vs 频率
ax = axes[1, 1]
for C in C_values:
    Xc = -1 / (2 * np.pi * freq_range * C)
    ax.semilogx(freq_range, Xc, linewidth=2, label=f'C = {C*1e9:.0f} nF')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Reactance Xc (Ω)')
ax.set_title('(d) Capacitive Reactance (Imaginary Part)')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('Test 1: Capacitor Impedance Characteristics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_Capacitor_Impedance.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test1_Capacitor_Impedance.png")


# ==================== Test 2: D值测量精度验证 ====================
print("\nTest 2: Capacitor D Value Measurement Accuracy")

fs = 500e3  # ADC采样率 500kHz
N = 1024    # DFT点数
f_test = 10e3  # 测试频率 10kHz (测量100nF电容)
t = np.arange(N) / fs

C_test_values = np.array([1, 10, 50, 100]) * 1e-9  # 1~100nF
D_test_values = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

D_measured_results = {}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: 不同D值的电压波形对比 (C=10nF)
ax = axes[0, 0]
C_fixed = 10e-9
I_drive = 1e-3  # 1mA恒流

for D in [0.01, 0.1, 1.0]:
    omega = 2 * np.pi * f_test
    Rs = D / (omega * C_fixed)
    Xc = -1 / (omega * C_fixed)
    Z_mag = np.sqrt(Rs**2 + Xc**2)
    Z_phase = np.arctan2(Xc, Rs) * 180 / np.pi
    
    V_mag = I_drive * Z_mag
    v_waveform = generate_sine_wave(t, f_test, V_mag, np.radians(Z_phase))
    v_waveform = v_waveform + 0.001 * np.random.randn(N)  # ADC噪声
    
    ax.plot(t * 1e3, v_waveform * 1e3, linewidth=1.5, label=f'D = {D:.3f}, V = {V_mag*1e3:.2f}mV')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Voltage (mV)')
ax.set_title(f'(a) Voltage Waveforms (C=10nF, I=1mA)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2b: D值测量误差 vs 真实D值
ax = axes[0, 1]
C_fixed = 10e-9
D_errors = []

for D_true in D_test_values:
    omega = 2 * np.pi * f_test
    Rs = D_true / (omega * C_fixed)
    Xc = -1 / (omega * C_fixed)
    Z = Rs + 1j * Xc
    
    # 生成电压电流波形
    i_waveform = generate_sine_wave(t, f_test, I_drive, 0)
    v_waveform = I_drive * np.abs(Z) * np.sin(2 * np.pi * f_test * t + np.angle(Z))
    v_waveform = v_waveform + 0.0005 * np.random.randn(N)  # 低噪声
    i_waveform = i_waveform + 0.0001 * np.random.randn(N)
    
    # DFT测量
    Z_mag_m, Z_phase_m = measure_impedance(v_waveform, i_waveform, fs, f_test)
    
    # 计算D
    Rs_meas = Z_mag_m * np.cos(np.radians(Z_phase_m))
    Xc_meas = -Z_mag_m * np.sin(np.radians(Z_phase_m))
    D_meas = Rs_meas / abs(Xc_meas)
    
    D_errors.append(abs(D_meas - D_true) / D_true * 100)

ax.plot(D_test_values, D_errors, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5%')
ax.set_xlabel('True D Value')
ax.set_ylabel('D Measurement Error (%)')
ax.set_title('(b) D Measurement Error vs True D')
ax.legend()
ax.grid(True, alpha=0.3)

# 2c: 相位分辨率对D值精度的影响
ax = axes[1, 0]
phase_errors = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # 相位测量误差(度)
D_true_fixed = 0.01

for D_tf in [0.005, 0.01, 0.05]:
    D_err_vs_phase = []
    for pe in phase_errors:
        # D = tan(δ) ≈ δ (小角度), δ = 90° - |θ|
        # ΔD/D ≈ Δθ/δ (当δ很小时，精度要求极高)
        delta_rad = np.arctan(D_tf)
        D_err = pe * np.pi / 180 / delta_rad * 100
        D_err_vs_phase.append(min(D_err, 100))  # 上限100%
    ax.plot(phase_errors, D_err_vs_phase, 'o-', linewidth=2, markersize=8, label=f'D = {D_tf}')

ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5%')
ax.set_xlabel('Phase Measurement Error (°)')
ax.set_ylabel('D Value Error (%)')
ax.set_title('(c) D Error vs Phase Resolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 2d: 不同容值的C测量误差
ax = axes[1, 1]
C_meas_errors = []
for C_true in C_test_values:
    omega = 2 * np.pi * f_test
    Xc = -1 / (omega * C_true)
    Z = 1j * Xc  # 理想电容
    
    v_waveform = I_drive * abs(Xc) * np.sin(2 * np.pi * f_test * t - np.pi/2)
    v_waveform = v_waveform + 0.001 * np.random.randn(N)
    i_waveform = generate_sine_wave(t, f_test, I_drive, 0)
    
    Z_mag_m, Z_phase_m = measure_impedance(v_waveform, i_waveform, fs, f_test)
    Xc_meas = -Z_mag_m * np.sin(np.radians(Z_phase_m))
    C_meas = 1 / (omega * abs(Xc_meas))
    
    C_meas_errors.append(abs(C_meas - C_true) / C_true * 100)

ax.bar([f'{c*1e9:.0f}' for c in C_test_values], C_meas_errors, color='steelblue', edgecolor='black')
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5%')
ax.set_xlabel('Capacitance (nF)')
ax.set_ylabel('Measurement Error (%)')
ax.set_title('(d) Capacitance Measurement Error')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 2: Capacitor C and D Measurement Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_Capacitor_D_Measurement.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Capacitance measurement: max error = {max(C_meas_errors):.2f}%")
print(f"  D value measurement: max error = {max(D_errors):.2f}%")


# ==================== Test 3: 电感阻抗频率特性 ====================
print("\nTest 3: Inductor Impedance vs Frequency")

L_values = [10e-6, 50e-6, 100e-6]  # 10μH, 50μH, 100μH
Q_values = [10, 50, 100, 200]
freq_range = np.logspace(3, 6, 500)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a: 感抗 vs 频率
ax = axes[0, 0]
for L in L_values:
    Xl = 2 * np.pi * freq_range * L
    ax.loglog(freq_range, Xl, linewidth=2, label=f'L = {L*1e6:.0f} μH')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Xl = ωL (Ω)')
ax.set_title('(a) Inductive Reactance vs Frequency')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# 3b: Q值 vs 频率 (假设L=50μH, R=0.5Ω固定)
ax = axes[0, 1]
L_fixed = 50e-6
R_fixed = 0.5
Q_calc = 2 * np.pi * freq_range * L_fixed / R_fixed
ax.semilogx(freq_range, Q_calc, 'b-', linewidth=2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Q = ωL/R')
ax.set_title(f'(b) Q Factor vs Frequency (L=50μH, R={R_fixed}Ω)')
ax.grid(True, which='both', alpha=0.3)

# 3c: 不同Q值的相位
ax = axes[1, 0]
f_test_ind = 50e3  # 50kHz
for Q in Q_values:
    phase = np.arctan(Q) * 180 / np.pi
    ax.bar(str(Q), phase, color='steelblue', edgecolor='black')
ax.set_xlabel('Q Value')
ax.set_ylabel('Phase Angle (°)')
ax.set_title(f'(c) Phase Angle vs Q (f={f_test_ind/1e3:.0f}kHz)')
ax.grid(True, alpha=0.3, axis='y')

# 3d: 阻抗实部(ESR) vs 频率
ax = axes[1, 1]
for L in L_values:
    for Q in [10, 100]:
        R = 2 * np.pi * freq_range * L / Q
        ax.loglog(freq_range, R, linewidth=1.5, label=f'L={L*1e6:.0f}μH, Q={Q}')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Equivalent Series Resistance (Ω)')
ax.set_title('(d) ESR vs Frequency')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('Test 3: Inductor Impedance Characteristics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_Inductor_Impedance.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test3_Inductor_Impedance.png")


# ==================== Test 4: Q值测量精度 ====================
print("\nTest 4: Inductor Q Value Measurement Accuracy")

fs = 500e3
N = 2048  # 更多点数以提高相位精度
f_test = 50e3  # 50kHz
I_drive = 10e-3  # 10mA (电感阻抗低，需要更大电流)
t = np.arange(N) / fs

L_test_values = np.array([10, 50, 100]) * 1e-6  # 10~100μH
Q_test_values = [1, 10, 50, 100, 200]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a: 不同Q值的电压波形 (L=50μH)
ax = axes[0, 0]
L_fixed = 50e-6
for Q in [1, 10, 200]:
    omega = 2 * np.pi * f_test
    R = omega * L_fixed / Q
    Xl = omega * L_fixed
    Z = R + 1j * Xl
    
    V_mag = I_drive * np.abs(Z)
    Z_phase = np.angle(Z, deg=True)
    
    v_waveform = generate_sine_wave(t, f_test, V_mag, np.radians(Z_phase))
    v_waveform = v_waveform + 0.0005 * np.random.randn(N)
    
    ax.plot(t[:100] * 1e3, v_waveform[:100] * 1e3, linewidth=1.5, label=f'Q={Q}, V={V_mag*1e3:.2f}mV')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Voltage (mV)')
ax.set_title('(a) Voltage Waveforms (L=50μH, I=10mA)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4b: Q值测量误差
ax = axes[0, 1]
Q_meas_errors = []
for Q_true in Q_test_values:
    omega = 2 * np.pi * f_test
    R = omega * L_fixed / Q_true
    Xl = omega * L_fixed
    Z = R + 1j * Xl
    
    v_waveform = I_drive * np.abs(Z) * np.sin(2 * np.pi * f_test * t + np.angle(Z))
    v_waveform = v_waveform + 0.0005 * np.random.randn(N)
    i_waveform = generate_sine_wave(t, f_test, I_drive, 0)
    
    Z_mag_m, Z_phase_m = measure_impedance(v_waveform, i_waveform, fs, f_test)
    
    R_meas = Z_mag_m * np.cos(np.radians(Z_phase_m))
    Xl_meas = Z_mag_m * np.sin(np.radians(Z_phase_m))
    Q_meas = Xl_meas / R_meas if R_meas > 1e-6 else np.inf
    
    err = abs(Q_meas - Q_true) / Q_true * 100 if Q_true < 1e6 else 0
    Q_meas_errors.append(min(err, 50))  # 上限50%

ax.plot(Q_test_values, Q_meas_errors, 'ro-', linewidth=2, markersize=8)
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5%')
ax.set_xlabel('True Q Value')
ax.set_ylabel('Q Measurement Error (%)')
ax.set_title('(b) Q Value Measurement Error')
ax.legend()
ax.grid(True, alpha=0.3)

# 4c: 不同电感值的L测量误差
ax = axes[1, 0]
L_meas_errors = []
for L_true in L_test_values:
    omega = 2 * np.pi * f_test
    R = omega * L_true / 50  # 假设Q=50
    Xl = omega * L_true
    Z = R + 1j * Xl
    
    v_waveform = I_drive * np.abs(Z) * np.sin(2 * np.pi * f_test * t + np.angle(Z))
    v_waveform = v_waveform + 0.0005 * np.random.randn(N)
    i_waveform = generate_sine_wave(t, f_test, I_drive, 0)
    
    Z_mag_m, Z_phase_m = measure_impedance(v_waveform, i_waveform, fs, f_test)
    Xl_meas = Z_mag_m * np.sin(np.radians(Z_phase_m))
    L_meas = Xl_meas / omega
    
    L_meas_errors.append(abs(L_meas - L_true) / L_true * 100)

ax.bar([f'{l*1e6:.0f}' for l in L_test_values], L_meas_errors, color='steelblue', edgecolor='black')
ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5%')
ax.set_xlabel('Inductance (μH)')
ax.set_ylabel('Measurement Error (%)')
ax.set_title('(c) Inductance Measurement Error')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4d: 相位分辨率对Q精度的影响
ax = axes[1, 1]
phase_errs = np.array([0.05, 0.1, 0.5, 1.0])
for Q_tf in [10, 50, 200]:
    Q_err_vs_phase = []
    for pe in phase_errs:
        # Q = tan(θ), ΔQ/Q ≈ Δθ/sin(θ)cos(θ) = Δθ·2Q/sin(2θ)
        theta = np.arctan(Q_tf)
        dQ = pe * np.pi / 180 * (1 + Q_tf**2)
        Q_err_vs_phase.append(dQ / Q_tf * 100)
    ax.plot(phase_errs, Q_err_vs_phase, 'o-', linewidth=2, markersize=8, label=f'Q = {Q_tf}')

ax.axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5%')
ax.set_xlabel('Phase Error (°)')
ax.set_ylabel('Q Value Error (%)')
ax.set_title('(d) Q Error vs Phase Resolution')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Test 4: Inductor L and Q Measurement Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_Inductor_Q_Measurement.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Inductance measurement: max error = {max(L_meas_errors):.2f}%")
print(f"  Q value measurement: max error = {max(Q_meas_errors):.2f}%")


# ==================== Test 5: 恒流源驱动与量程选择 ====================
print("\nTest 5: Constant Current Source and Range Selection")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 5a: 不同被测阻抗下的输出电压
ax = axes[0, 0]
Z_range = np.logspace(1, 5, 100)  # 10Ω ~ 100kΩ
I_values = [0.1e-3, 1e-3, 10e-3]  # 0.1mA, 1mA, 10mA

for I in I_values:
    V_out = I * Z_range
    ax.loglog(Z_range, V_out * 1e3, linewidth=2, label=f'I = {I*1e3:.1f}mA')

ax.axvline(x=100, color='k', linestyle=':', alpha=0.5, label='|Z| = 100Ω')
ax.axvline(x=10000, color='k', linestyle='--', alpha=0.5, label='|Z| = 10kΩ')
ax.set_xlabel('|Z| (Ω)')
ax.set_ylabel('Voltage (mV)')
ax.set_title('(a) Output Voltage vs |Z|')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# 5b: 最佳测量频率选择
ax = axes[0, 1]
freqs = np.logspace(3, 5, 100)
C_vals = [1e-9, 10e-9, 100e-9]
L_vals = [10e-6, 100e-6]

for C in C_vals:
    Zc = 1 / (2 * np.pi * freqs * C)
    ax.loglog(freqs, Zc, '--', linewidth=1.5, label=f'C={C*1e9:.0f}nF')

for L in L_vals:
    Zl = 2 * np.pi * freqs * L
    ax.loglog(freqs, Zl, '-', linewidth=1.5, label=f'L={L*1e6:.0f}μH')

ax.axhspan(100, 10000, alpha=0.2, color='green', label='Optimal |Z| range')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('|Z| (Ω)')
ax.set_title('(b) Optimal Frequency Selection')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

# 5c: 量程切换策略
ax = axes[1, 0]
C_range = np.array([1, 10, 100]) * 1e-9
L_range = np.array([10, 50, 100]) * 1e-6

# 推荐频率和增益
rec_freq = [100e3, 10e3, 10e3]
rec_gain = [10, 1, 1]

x_pos = np.arange(len(C_range))
ax.bar(x_pos - 0.2, np.array(rec_freq)/1e3, 0.4, label='Recommended Freq (kHz)', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x_pos + 0.2, rec_gain, 0.4, label='PGA Gain', color='coral')

ax.set_xlabel('Capacitance (nF)')
ax.set_ylabel('Frequency (kHz)', color='steelblue')
ax2.set_ylabel('PGA Gain', color='coral')
ax.set_title('(c) Auto-Range Strategy for Capacitor')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c*1e9:.0f}' for c in C_range])
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# 5d: 恒流源电路输出阻抗
ax = axes[1, 1]
R_out_ideal = np.ones(100) * 1e6  # 理想恒流源输出阻抗1MΩ
R_out_practical = np.ones(100) * 100e3  # 实际运放恒流源100kΩ
f_out = np.logspace(0, 6, 100)

ax.semilogx(f_out, R_out_ideal, 'g--', linewidth=2, label='Ideal: 1MΩ')
ax.semilogx(f_out, R_out_practical, 'b-', linewidth=2, label='Practical: 100kΩ')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Output Impedance (Ω)')
ax.set_title('(d) Constant Current Source Output Impedance')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('Test 5: Current Source and Measurement Range Design', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_Current_Source_Range.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test5_Current_Source_Range.png")


# ==================== Test 6: DFT相位检测 vs 噪声 ====================
print("\nTest 6: DFT Phase Detection under Noise")

fs = 500e3
N = 1024
f_test = 10e3
t = np.arange(N) / fs

SNR_values = [20, 40, 60]  # dB
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6a: 不同SNR下的波形
ax = axes[0, 0]
A_signal = 1.0
for snr_db in SNR_values:
    noise_rms = A_signal / (10**(snr_db/20))
    signal = generate_sine_wave(t, f_test, A_signal, 0)
    noisy = signal + noise_rms * np.random.randn(N)
    ax.plot(t[:100] * 1e3, noisy[:100], linewidth=1, alpha=0.8, label=f'SNR = {snr_db}dB')

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (V)')
ax.set_title('(a) Waveforms at Different SNR')
ax.legend()
ax.grid(True, alpha=0.3)

# 6b: 相位测量精度 vs SNR
ax = axes[0, 1]
snr_range = np.arange(20, 81, 5)
phase_errs_snr = []

for snr_db in snr_range:
    noise_rms = A_signal / (10**(snr_db/20))
    
    # 多次测量取平均
    phase_errs_single = []
    for _ in range(50):
        signal = generate_sine_wave(t, f_test, A_signal, 0)
        noisy = signal + noise_rms * np.random.randn(N)
        
        # DFT测量相位
        X = dft_single_bin(noisy, fs, f_test, N)
        phase_meas = np.angle(X, deg=True)
        phase_errs_single.append(abs(phase_meas))
    
    phase_errs_snr.append(np.std(phase_errs_single))

ax.plot(snr_range, phase_errs_snr, 'bo-', linewidth=2, markersize=8)
ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='Target: 0.1°')
ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Acceptable: 1.0°')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Phase Error Std (°)')
ax.set_title('(b) Phase Precision vs SNR')
ax.legend()
ax.grid(True, alpha=0.3)

# 6c: DFT点数对频率分辨率的影响
ax = axes[1, 0]
N_values = [128, 256, 512, 1024, 2048]
f_res = [fs/n for n in N_values]
ax.bar([str(n) for n in N_values], np.array(f_res), color='steelblue', edgecolor='black')
ax.set_xlabel('DFT Points N')
ax.set_ylabel('Frequency Resolution (Hz)')
ax.set_title('(c) DFT Frequency Resolution')
ax.grid(True, alpha=0.3, axis='y')

# 6d: 窗函数对频谱泄漏的影响
ax = axes[1, 1]
N_win = 256
signal_off = generate_sine_wave(np.arange(N_win)/fs, f_test*1.05, 1.0, 0)  # 频率偏移5%

rect_win = signal_off
hanning_win = signal_off * np.hanning(N_win)
blackman_win = signal_off * np.blackman(N_win)

# 计算频谱
f_bins = np.fft.fftfreq(N_win, 1/fs)[:N_win//2]
spec_rect = np.abs(np.fft.fft(rect_win))[:N_win//2]
spec_hann = np.abs(np.fft.fft(hanning_win))[:N_win//2]
spec_black = np.abs(np.fft.fft(blackman_win))[:N_win//2]

ax.semilogy(f_bins/1e3, spec_rect + 1e-10, linewidth=1.5, label='Rectangular')
ax.semilogy(f_bins/1e3, spec_hann + 1e-10, linewidth=1.5, label='Hanning')
ax.semilogy(f_bins/1e3, spec_black + 1e-10, linewidth=1.5, label='Blackman')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('Magnitude')
ax.set_title('(d) Window Function Effect on Spectral Leakage')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

fig.suptitle('Test 6: DFT Phase Detection Performance', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_DFT_Phase_Detection.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: Test6_DFT_Phase_Detection.png")
print(f"  SNR=60dB → Phase error ≈ {phase_errs_snr[-1]:.3f}°")


# ==================== Test 7: Monte Carlo误差预算 ====================
print("\nTest 7: Monte Carlo Error Budget")

num_mc = 100
fs_mc = 500e3
N_mc = 2048
f_test_cap = 10e3  # 电容测试频率
f_test_ind = 50e3  # 电感测试频率
I_drive_cap = 1e-3
I_drive_ind = 10e-3

t_cap = np.arange(N_mc) / fs_mc
t_ind = np.arange(N_mc) / fs_mc

C_errors_mc = []
D_errors_mc = []
L_errors_mc = []
Q_errors_mc = []

for _ in range(num_mc):
    # 误差源
    fs_err = fs_mc * (1 + 0.001 * np.random.randn())  # 0.1%采样率误差
    R_ref_err = 1 + 0.005 * np.random.randn()  # 参考电阻0.5%误差
    adc_noise = 10**(-50/20)  # SNR=50dB
    
    # === 电容测量 (C=10nF, D=0.01) ===
    C_true = 10e-9
    D_true = 0.01
    omega = 2 * np.pi * f_test_cap
    Rs = D_true / (omega * C_true)
    Xc = -1 / (omega * C_true)
    Z = Rs + 1j * Xc
    
    v = I_drive_cap * np.abs(Z) * np.sin(2 * np.pi * f_test_cap * t_cap + np.angle(Z))
    v = v + adc_noise * np.random.randn(N_mc)
    i = generate_sine_wave(t_cap, f_test_cap, I_drive_cap * R_ref_err, 0)
    
    Zm, Zp = measure_impedance(v, i, fs_err, f_test_cap)
    Xc_m = -Zm * np.sin(np.radians(Zp))
    C_m = 1 / (omega * abs(Xc_m))
    Rs_m = Zm * np.cos(np.radians(Zp))
    D_m = abs(Rs_m / Xc_m)
    
    C_errors_mc.append(abs(C_m - C_true) / C_true * 100)
    D_errors_mc.append(abs(D_m - D_true) / D_true * 100)
    
    # === 电感测量 (L=50μH, Q=50) ===
    L_true = 50e-6
    Q_true = 50
    omega = 2 * np.pi * f_test_ind
    R = omega * L_true / Q_true
    Xl = omega * L_true
    Z = R + 1j * Xl
    
    v = I_drive_ind * np.abs(Z) * np.sin(2 * np.pi * f_test_ind * t_ind + np.angle(Z))
    v = v + adc_noise * np.random.randn(N_mc)
    i = generate_sine_wave(t_ind, f_test_ind, I_drive_ind * R_ref_err, 0)
    
    Zm, Zp = measure_impedance(v, i, fs_err, f_test_ind)
    Xl_m = Zm * np.sin(np.radians(Zp))
    L_m = Xl_m / omega
    R_m = Zm * np.cos(np.radians(Zp))
    Q_m = Xl_m / R_m if R_m > 1e-6 else 0
    
    L_errors_mc.append(abs(L_m - L_true) / L_true * 100)
    Q_errors_mc.append(abs(Q_m - Q_true) / Q_true * 100 if Q_m > 0 else 100)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(C_errors_mc, bins=15, color='blue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=5, color='r', linestyle='--', linewidth=2, label='Req: 5%')
axes[0, 0].set_title(f'Capacitance Error\nMean={np.mean(C_errors_mc):.2f}%, 95%CI=[{np.percentile(C_errors_mc,2.5):.2f}, {np.percentile(C_errors_mc,97.5):.2f}]%')
axes[0, 0].set_xlabel('Error (%)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(D_errors_mc, bins=15, color='red', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=5, color='r', linestyle='--', linewidth=2, label='Req: 5%')
axes[0, 1].set_title(f'D Value Error\nMean={np.mean(D_errors_mc):.2f}%, 95%CI=[{np.percentile(D_errors_mc,2.5):.2f}, {np.percentile(D_errors_mc,97.5):.2f}]%')
axes[0, 1].set_xlabel('Error (%)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(L_errors_mc, bins=15, color='green', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=5, color='r', linestyle='--', linewidth=2, label='Req: 5%')
axes[1, 0].set_title(f'Inductance Error\nMean={np.mean(L_errors_mc):.2f}%, 95%CI=[{np.percentile(L_errors_mc,2.5):.2f}, {np.percentile(L_errors_mc,97.5):.2f}]%')
axes[1, 0].set_xlabel('Error (%)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(Q_errors_mc, bins=15, color='purple', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=5, color='r', linestyle='--', linewidth=2, label='Req: 5%')
axes[1, 1].set_title(f'Q Value Error\nMean={np.mean(Q_errors_mc):.2f}%, 95%CI=[{np.percentile(Q_errors_mc,2.5):.2f}, {np.percentile(Q_errors_mc,97.5):.2f}]%')
axes[1, 1].set_xlabel('Error (%)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 7: Monte Carlo Error Budget (100 Runs)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_MonteCarlo_ErrorBudget.png'), dpi=150, bbox_inches='tight')
plt.close()

C_ci = [np.percentile(C_errors_mc, 2.5), np.percentile(C_errors_mc, 97.5)]
D_ci = [np.percentile(D_errors_mc, 2.5), np.percentile(D_errors_mc, 97.5)]
L_ci = [np.percentile(L_errors_mc, 2.5), np.percentile(L_errors_mc, 97.5)]
Q_ci = [np.percentile(Q_errors_mc, 2.5), np.percentile(Q_errors_mc, 97.5)]

print(f"  Capacitance: Mean={np.mean(C_errors_mc):.2f}%, 95%CI=[{C_ci[0]:.2f}, {C_ci[1]:.2f}]% (Req≤5%) {'✓' if C_ci[1]<=5 else '✗'}")
print(f"  D Value: Mean={np.mean(D_errors_mc):.2f}%, 95%CI=[{D_ci[0]:.2f}, {D_ci[1]:.2f}]% (Req≤5%) {'✓' if D_ci[1]<=5 else '✗'}")
print(f"  Inductance: Mean={np.mean(L_errors_mc):.2f}%, 95%CI=[{L_ci[0]:.2f}, {L_ci[1]:.2f}]% (Req≤5%) {'✓' if L_ci[1]<=5 else '✗'}")
print(f"  Q Value: Mean={np.mean(Q_errors_mc):.2f}%, 95%CI=[{Q_ci[0]:.2f}, {Q_ci[1]:.2f}]% (Req≤5%) {'✓' if Q_ci[1]<=5 else '✗'}")

print("\n=== Simulation Complete ===")
print(f"Output: {os.path.abspath(output_dir)}")
