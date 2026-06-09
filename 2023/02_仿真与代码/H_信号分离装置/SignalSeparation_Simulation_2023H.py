"""
2023年电赛H题「信号分离装置」信号处理仿真
目标: 验证FFT频率检测 + I/Q正交解调分离多频信号的可行性
技术: Goertzel频率检测 + I/Q解调 + 波形重建
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 输出目录
output_dir = 'simulation_output'
os.makedirs(output_dir, exist_ok=True)

print("=== 2023-H Signal Separation Simulation ===")
print("Technique: FFT Detection + I/Q Demodulation + Waveform Reconstruction")
print()

# 全局参数
fs = 500e3  # ADC采样率 500kHz
N = 4096    # FFT点数
t_max = N / fs
t = np.arange(N) / fs


def generate_sine(t, f, A, phi=0):
    return A * np.sin(2 * np.pi * f * t + np.radians(phi))


def generate_triangle(t, f, A, phi=0):
    """生成三角波"""
    phase = 2 * np.pi * f * t + np.radians(phi)
    return A * (2 / np.pi) * np.arcsin(np.sin(phase))


def goertzel(samples, target_freq, sample_rate):
    """Goertzel算法检测单一频率功率"""
    N = len(samples)
    k = int(0.5 + N * target_freq / sample_rate)
    w = 2 * np.pi * k / N
    cosine = np.cos(w)
    coeff = 2 * cosine
    
    s_prev = 0
    s_prev2 = 0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return power


def detect_two_frequencies(samples, fs, f_min=20e3, f_max=100e3, step=1e3):
    """检测混合信号中的两个主要频率"""
    freqs = np.arange(f_min, f_max + step, step)
    powers = np.array([goertzel(samples, f, fs) for f in freqs])
    
    # 找第一个峰值
    peak1_idx = np.argmax(powers)
    f1 = freqs[peak1_idx]
    
    # 排除f1附近±5kHz，找第二个峰值
    mask = np.abs(freqs - f1) > 5e3
    powers_masked = powers * mask
    peak2_idx = np.argmax(powers_masked)
    f2 = freqs[peak2_idx]
    
    return min(f1, f2), max(f1, f2), freqs, powers


def iq_demodulate(samples, fs, f_target, t):
    """I/Q正交解调"""
    lo_cos = np.cos(2 * np.pi * f_target * t)
    lo_sin = np.sin(2 * np.pi * f_target * t)
    
    i_mix = samples * lo_cos
    q_mix = samples * lo_sin
    
    # 低通滤波 (移动平均)
    cutoff = 5e3  # 5kHz截止
    window = int(fs / (2 * cutoff))
    if window < 3:
        window = 3
    
    i_filtered = np.convolve(i_mix, np.ones(window)/window, mode='same')
    q_filtered = np.convolve(q_mix, np.ones(window)/window, mode='same')
    
    # 取稳态平均值
    i_avg = np.mean(i_filtered[N//2:])
    q_avg = np.mean(q_filtered[N//2:])
    
    # C = A*sin(ωt+φ) → i_avg = A/2*sin(φ), q_avg = A/2*cos(φ)
    # phase = arctan2(sin(φ), cos(φ)) = arctan2(i_avg, q_avg) = φ
    amplitude = 2 * np.sqrt(i_avg**2 + q_avg**2)
    phase = np.arctan2(i_avg, q_avg) * 180 / np.pi
    
    return amplitude, phase


# ==================== Test 1: 混合信号C = A + B ====================
print("Test 1: Mixed Signal C = A + B")

# 信号参数
fA = 50e3   # 50kHz
fB = 100e3  # 100kHz
A_amp = 0.5  # 幅度0.5V (峰峰值1V)
B_amp = 0.5

signal_A = generate_sine(t, fA, A_amp, 0)
signal_B = generate_sine(t, fB, B_amp, 45)
signal_C = signal_A + signal_B

# FFT频谱
fft_C = np.fft.fft(signal_C)
freqs_fft = np.fft.fftfreq(N, 1/fs)[:N//2]
magnitude = 2/N * np.abs(fft_C[:N//2])

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1a: 时域波形 A
axes[0, 0].plot(t[:200] * 1e3, signal_A[:200], 'b-', linewidth=1.5)
axes[0, 0].set_title(f'(a) Signal A: {fA/1e3:.0f}kHz Sine')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude (V)')
axes[0, 0].grid(True, alpha=0.3)

# 1b: 时域波形 B
axes[0, 1].plot(t[:200] * 1e3, signal_B[:200], 'r-', linewidth=1.5)
axes[0, 1].set_title(f'(b) Signal B: {fB/1e3:.0f}kHz Sine (45° phase)')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude (V)')
axes[0, 1].grid(True, alpha=0.3)

# 1c: 混合信号C
axes[1, 0].plot(t[:200] * 1e3, signal_C[:200], 'g-', linewidth=1.5)
axes[1, 0].set_title('(c) Mixed Signal C = A + B')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Amplitude (V)')
axes[1, 0].grid(True, alpha=0.3)

# 1d: C的频谱
axes[1, 1].plot(freqs_fft[:200]/1e3, magnitude[:200], 'k-', linewidth=1.5)
axes[1, 1].axvline(x=fA/1e3, color='b', linestyle='--', label=f'fA={fA/1e3:.0f}kHz')
axes[1, 1].axvline(x=fB/1e3, color='r', linestyle='--', label=f'fB={fB/1e3:.0f}kHz')
axes[1, 1].set_title('(d) Spectrum of C')
axes[1, 1].set_xlabel('Frequency (kHz)')
axes[1, 1].set_ylabel('Magnitude (V)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 1e: 加法器电路概念图 (用文字表示)
axes[2, 0].text(0.5, 0.5, 'Gain=1 Adder:\nC = A + B\n\nUsing Inverting Summer:\nR1=R2=Rf\nVout = -(Vin1+Vin2)', 
                ha='center', va='center', fontsize=14, transform=axes[2, 0].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[2, 0].set_xlim(0, 1)
axes[2, 0].set_ylim(0, 1)
axes[2, 0].axis('off')
axes[2, 0].set_title('(e) Adder Circuit Concept')

# 1f: 分离电路框图
axes[2, 1].text(0.5, 0.5, 'Separation Circuit:\n\n1. ADC采样 C(t)\n2. FFT/Goertzel检测 fA, fB\n3. I/Q解调提取幅度/相位\n4. 波形重建 A\', B\'\n5. DAC输出', 
                ha='center', va='center', fontsize=12, transform=axes[2, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[2, 1].set_xlim(0, 1)
axes[2, 1].set_ylim(0, 1)
axes[2, 1].axis('off')
axes[2, 1].set_title('(f) Separation Algorithm Flow')

fig.suptitle('Test 1: Mixed Signal Generation and Spectrum', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_Mixed_Signal_Spectrum.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test1_Mixed_Signal_Spectrum.png")


# ==================== Test 2: FFT频率检测 ====================
print("\nTest 2: Frequency Detection from Mixed Signal")

# 测试多组频率
test_cases = [
    (50e3, 100e3, 'sine', 'sine'),
    (30e3, 80e3, 'sine', 'sine'),
    (40e3, 90e3, 'triangle', 'sine'),
]

fig, axes = plt.subplots(len(test_cases), 2, figsize=(14, 10))

for idx, (fA_test, fB_test, typeA, typeB) in enumerate(test_cases):
    if typeA == 'sine':
        A = generate_sine(t, fA_test, 0.5, 0)
    else:
        A = generate_triangle(t, fA_test, 0.5, 0)
    
    if typeB == 'sine':
        B = generate_sine(t, fB_test, 0.5, 30)
    else:
        B = generate_triangle(t, fB_test, 0.5, 30)
    
    C = A + B + 0.01 * np.random.randn(N)  # 添加噪声
    
    # 频率检测
    fA_det, fB_det, freqs_scan, powers = detect_two_frequencies(C, fs)
    
    # 绘图
    ax1 = axes[idx, 0]
    ax1.plot(freqs_scan/1e3, powers, 'b-', linewidth=1.5)
    ax1.axvline(x=fA_test/1e3, color='b', linestyle='--', label=f'True fA={fA_test/1e3:.0f}kHz')
    ax1.axvline(x=fB_test/1e3, color='r', linestyle='--', label=f'True fB={fB_test/1e3:.0f}kHz')
    ax1.axvline(x=fA_det/1e3, color='b', linestyle=':', alpha=0.7, label=f'Det fA={fA_det/1e3:.0f}kHz')
    ax1.axvline(x=fB_det/1e3, color='r', linestyle=':', alpha=0.7, label=f'Det fB={fB_det/1e3:.0f}kHz')
    ax1.set_title(f'Case {idx+1}: {typeA}+{typeB}, fA={fA_test/1e3:.0f}kHz, fB={fB_test/1e3:.0f}kHz')
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Goertzel Power')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[idx, 1]
    # 显示检测误差
    err_A = abs(fA_det - fA_test) / fA_test * 100
    err_B = abs(fB_det - fB_test) / fB_test * 100
    
    categories = ['fA Error', 'fB Error']
    errors = [err_A, err_B]
    colors = ['green' if e < 1 else 'red' for e in errors]
    ax2.bar(categories, errors, color=colors, edgecolor='black')
    ax2.axhline(y=1, color='r', linestyle='--', label='1% target')
    ax2.set_ylabel('Detection Error (%)')
    ax2.set_title(f'Detection Accuracy (Step=1kHz)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 2: Frequency Detection via Goertzel Algorithm', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_Frequency_Detection.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test2_Frequency_Detection.png")


# ==================== Test 3: I/Q正交解调分离正弦波 ====================
print("\nTest 3: I/Q Demodulation for Sine Wave Separation")

fA = 50e3
fB = 100e3
A = generate_sine(t, fA, 0.5, 20)
B = generate_sine(t, fB, 0.5, 60)
C = A + B + 0.005 * np.random.randn(N)

# I/Q解调
ampA, phaseA = iq_demodulate(C, fs, fA, t)
ampB, phaseB = iq_demodulate(C, fs, fB, t)

# 重建信号
A_recon = ampA * np.sin(2 * np.pi * fA * t + np.radians(phaseA))
B_recon = ampB * np.sin(2 * np.pi * fB * t + np.radians(phaseB))

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 原始A和重建A'
axes[0, 0].plot(t[:300] * 1e3, A[:300], 'b-', linewidth=2, label='Original A')
axes[0, 0].plot(t[:300] * 1e3, A_recon[:300], 'r--', linewidth=1.5, label='Reconstructed A\'' )
axes[0, 0].set_title(f'A: Original vs Reconstructed (fA={fA/1e3:.0f}kHz)')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude (V)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 原始B和重建B'
axes[0, 1].plot(t[:300] * 1e3, B[:300], 'b-', linewidth=2, label='Original B')
axes[0, 1].plot(t[:300] * 1e3, B_recon[:300], 'r--', linewidth=1.5, label='Reconstructed B\'' )
axes[0, 1].set_title(f'B: Original vs Reconstructed (fB={fB/1e3:.0f}kHz)')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude (V)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 误差分析
A_error = A - A_recon
B_error = B - B_recon

axes[1, 0].plot(t[:300] * 1e3, A_error[:300] * 1e3, 'g-', linewidth=1.5)
axes[1, 0].set_title(f'A Separation Error (RMS={np.std(A_error)*1e3:.2f}mV)')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Error (mV)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t[:300] * 1e3, B_error[:300] * 1e3, 'g-', linewidth=1.5)
axes[1, 1].set_title(f'B Separation Error (RMS={np.std(B_error)*1e3:.2f}mV)')
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Error (mV)')
axes[1, 1].grid(True, alpha=0.3)

# 幅度和相位误差 (I/Q解调提取的是峰值幅度，原始A/B峰值=0.5V)
A_amp_err = abs(ampA - 0.5) / 0.5 * 100
A_phase_err = abs(phaseA - 20)
if A_phase_err > 180:
    A_phase_err = 360 - A_phase_err
B_amp_err = abs(ampB - 0.5) / 0.5 * 100
B_phase_err = abs(phaseB - 60)
if B_phase_err > 180:
    B_phase_err = 360 - B_phase_err

categories = ['A Amp', 'A Phase', 'B Amp', 'B Phase']
errors = [A_amp_err, A_phase_err, B_amp_err, B_phase_err]
colors = ['green' if (i%2==0 and e<5) or (i%2==1 and e<5) else 'orange' for i, e in enumerate(errors)]

axes[2, 0].bar(categories, errors, color=colors, edgecolor='black')
axes[2, 0].axhline(y=5, color='r', linestyle='--', linewidth=2, label='5% target')
axes[2, 0].set_ylabel('Error')
axes[2, 0].set_title('Parameter Estimation Errors')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3, axis='y')

# 频谱对比
fft_A = np.fft.fft(A)
fft_A_recon = np.fft.fft(A_recon)
axes[2, 1].plot(freqs_fft[:200]/1e3, 2/N*np.abs(fft_A[:200]), 'b-', linewidth=1.5, label='Original A')
axes[2, 1].plot(freqs_fft[:200]/1e3, 2/N*np.abs(fft_A_recon[:200]), 'r--', linewidth=1.5, label='Reconstructed A\'')
axes[2, 1].set_title('Spectrum: Original vs Reconstructed')
axes[2, 1].set_xlabel('Frequency (kHz)')
axes[2, 1].set_ylabel('Magnitude (V)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

fig.suptitle('Test 3: Sine Wave Separation via I/Q Demodulation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_Sine_Separation_IQ.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  A: Amp error={A_amp_err:.2f}%, Phase error={A_phase_err:.2f}°")
print(f"  B: Amp error={B_amp_err:.2f}%, Phase error={B_phase_err:.2f}°")


# ==================== Test 4: 三角波识别与分离 ====================
print("\nTest 4: Triangle Wave Detection and Separation")

fA = 40e3
fB = 90e3

A_tri = generate_triangle(t, fA, 0.5, 0)
B_sin = generate_sine(t, fB, 0.5, 0)
C = A_tri + B_sin + 0.005 * np.random.randn(N)

# 谐波分析检测波形类型
def detect_waveform_type(samples, fs, f_target):
    """通过谐波含量判断波形类型"""
    power_fund = goertzel(samples, f_target, fs)
    power_3rd = goertzel(samples, 3*f_target, fs)
    power_5th = goertzel(samples, 5*f_target, fs)
    
    harmonic_ratio = (power_3rd + power_5th) / (power_fund + 1e-10)
    return 'TRIANGLE' if harmonic_ratio > 0.02 else 'SINE', harmonic_ratio

type_A, ratio_A = detect_waveform_type(C, fs, fA)
type_B, ratio_B = detect_waveform_type(C, fs, fB)

# I/Q解调
ampA, phaseA = iq_demodulate(C, fs, fA, t)
ampB, phaseB = iq_demodulate(C, fs, fB, t)

# 重建
A_recon_tri = generate_triangle(t, fA, ampA/2, phaseA)  # 三角波重建
B_recon_sin = ampB * np.sin(2 * np.pi * fB * t + np.radians(phaseB))

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 三角波频谱特征
fft_tri = np.fft.fft(A_tri)
fft_sin = np.fft.fft(B_sin)
axes[0, 0].plot(freqs_fft[:300]/1e3, 2/N*np.abs(fft_tri[:300]), 'b-', linewidth=1.5, label='Triangle A')
axes[0, 0].plot(freqs_fft[:300]/1e3, 2/N*np.abs(fft_sin[:300]), 'r-', linewidth=1.5, label='Sine B')
axes[0, 0].axvline(x=fA/1e3, color='b', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=fB/1e3, color='r', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=3*fA/1e3, color='b', linestyle=':', alpha=0.3, label='3rd harmonic')
axes[0, 0].axvline(x=5*fA/1e3, color='b', linestyle=':', alpha=0.3, label='5th harmonic')
axes[0, 0].set_title('(a) Spectrum: Triangle vs Sine')
axes[0, 0].set_xlabel('Frequency (kHz)')
axes[0, 0].set_ylabel('Magnitude (V)')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 波形类型检测
axes[0, 1].bar(['A (40kHz)', 'B (90kHz)'], [ratio_A, ratio_B], color=['coral', 'steelblue'], edgecolor='black')
axes[0, 1].axhline(y=0.02, color='r', linestyle='--', label='Threshold = 0.02')
axes[0, 1].set_ylabel('Harmonic Ratio')
axes[0, 1].set_title(f'(b) Waveform Detection: A={type_A}, B={type_B}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 原始和重建的A (三角波)
axes[1, 0].plot(t[:300] * 1e3, A_tri[:300], 'b-', linewidth=2, label='Original A (Triangle)')
axes[1, 0].plot(t[:300] * 1e3, A_recon_tri[:300], 'r--', linewidth=1.5, label='Reconstructed A\' (Triangle)')
axes[1, 0].set_title('(c) Triangle Wave Separation')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Amplitude (V)')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 原始和重建的B (正弦波)
axes[1, 1].plot(t[:300] * 1e3, B_sin[:300], 'b-', linewidth=2, label='Original B (Sine)')
axes[1, 1].plot(t[:300] * 1e3, B_recon_sin[:300], 'r--', linewidth=1.5, label='Reconstructed B\' (Sine)')
axes[1, 1].set_title('(d) Sine Wave Separation')
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Amplitude (V)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# 三角波重建误差 (如果用正弦滤波)
A_recon_wrong = ampA * np.sin(2 * np.pi * fA * t + np.radians(phaseA))  # 错误地用正弦重建
tri_error_correct = A_tri - A_recon_tri
tri_error_wrong = A_tri - A_recon_wrong

axes[2, 0].plot(t[:300] * 1e3, tri_error_correct[:300] * 1e3, 'g-', linewidth=1.5, label='Triangle reconstruction')
axes[2, 0].plot(t[:300] * 1e3, tri_error_wrong[:300] * 1e3, 'r-', linewidth=1.5, label='Sine reconstruction (wrong!)')
axes[2, 0].set_title('(e) Triangle Reconstruction Error')
axes[2, 0].set_xlabel('Time (ms)')
axes[2, 0].set_ylabel('Error (mV)')
axes[2, 0].legend(fontsize=8)
axes[2, 0].grid(True, alpha=0.3)

# 误差统计
axes[2, 1].bar(['Correct\nReconstruction', 'Wrong\n(Sine filter)'], 
               [np.std(tri_error_correct)*1e3, np.std(tri_error_wrong)*1e3], 
               color=['green', 'red'], edgecolor='black')
axes[2, 1].set_ylabel('RMS Error (mV)')
axes[2, 1].set_title('(f) Reconstruction Quality Comparison')
axes[2, 1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 4: Triangle Wave Detection and Separation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_Triangle_Wave_Separation.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Waveform detection: A={type_A} (ratio={ratio_A:.3f}), B={type_B} (ratio={ratio_B:.3f})")
print(f"  Triangle recon error: {np.std(tri_error_correct)*1e3:.2f}mV")
print(f"  Sine-filter error: {np.std(tri_error_wrong)*1e3:.2f}mV")


# ==================== Test 5: 分离质量评估 ====================
print("\nTest 5: Separation Quality Metrics")

snr_values = np.arange(20, 61, 5)
separation_quality = []

for snr_db in snr_values:
    noise_rms = 0.5 / (10**(snr_db/20))
    
    A = generate_sine(t, 50e3, 0.5, 0)
    B = generate_sine(t, 100e3, 0.5, 0)
    C = A + B + noise_rms * np.random.randn(N)
    
    # 分离
    ampA, phaseA = iq_demodulate(C, fs, 50e3, t)
    ampB, phaseB = iq_demodulate(C, fs, 100e3, t)
    A_recon = ampA * np.sin(2 * np.pi * 50e3 * t + np.radians(phaseA))
    B_recon = ampB * np.sin(2 * np.pi * 100e3 * t + np.radians(phaseB))
    
    # 计算SNR改善
    A_snr = 20 * np.log10(np.std(A) / np.std(A - A_recon))
    separation_quality.append(A_snr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(snr_values, separation_quality, 'bo-', linewidth=2, markersize=8)
axes[0].plot(snr_values, snr_values, 'k--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel('Input SNR (dB)')
axes[0].set_ylabel('Output SNR (dB)')
axes[0].set_title('(a) Separation Performance vs Input SNR')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 不同频率间隔的分离难度
freq_spacings = [5e3, 10e3, 20e3, 30e3, 50e3]
spacing_errors = []

for spacing in freq_spacings:
    fA_s = 50e3
    fB_s = 50e3 + spacing
    
    A = generate_sine(t, fA_s, 0.5, 0)
    B = generate_sine(t, fB_s, 0.5, 0)
    C = A + B + 0.01 * np.random.randn(N)
    
    ampA, phaseA = iq_demodulate(C, fs, fA_s, t)
    ampB, phaseB = iq_demodulate(C, fs, fB_s, t)
    A_recon = ampA * np.sin(2 * np.pi * fA_s * t + np.radians(phaseA))
    
    err = np.std(A - A_recon) / np.std(A) * 100
    spacing_errors.append(err)

axes[1].plot(np.array(freq_spacings)/1e3, spacing_errors, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Frequency Spacing (kHz)')
axes[1].set_ylabel('Separation Error (%)')
axes[1].set_title('(b) Separation Error vs Frequency Spacing')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Test 5: Signal Separation Quality Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_Separation_Quality.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test5_Separation_Quality.png")


# ==================== Test 6: 相位控制精度 ====================
print("\nTest 6: Phase Control Accuracy")

fA = 50e3
fB = 100e3  # fB = 2*fA
A = generate_sine(t, fA, 0.5, 0)
B = generate_sine(t, fB, 0.5, 0)
C = A + B

# 目标相位差
target_phases = np.arange(0, 181, 5)  # 0°~180°, 5°步进
measured_phases = []
phase_errors = []

for target_phi in target_phases:
    # 直接从C中I/Q解调B的幅度和原始相位
    ampB, phaseB = iq_demodulate(C, fs, fB, t)
    
    # 设置相位差: B' = sin(ωBt + φ_target)
    # A的相位是0°，所以B'与A'的相位差就是φ_target
    B_recon_phase = ampB * np.sin(2 * np.pi * fB * t + np.radians(target_phi))
    
    # 测量重建信号B'的实际相位
    _, actual_phase = iq_demodulate(B_recon_phase, fs, fB, t)
    actual_phi = actual_phase % 360
    if actual_phi > 180:
        actual_phi -= 360
    
    measured_phases.append(actual_phi)
    err = abs(actual_phi - target_phi)
    if err > 180:
        err = 360 - err
    phase_errors.append(err)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(target_phases, measured_phases, 'bo-', linewidth=2, markersize=6, label='Measured')
axes[0].plot(target_phases, target_phases, 'k--', linewidth=1.5, label='Target')
axes[0].set_xlabel('Target Phase Difference (°)')
axes[0].set_ylabel('Measured Phase Difference (°)')
axes[0].set_title('(a) Phase Control: Target vs Measured')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(target_phases, phase_errors, 'ro-', linewidth=2, markersize=6)
axes[1].axhline(y=5, color='r', linestyle='--', linewidth=2, label='Requirement: 5°')
axes[1].set_xlabel('Target Phase Difference (°)')
axes[1].set_ylabel('Phase Error (°)')
axes[1].set_title('(b) Phase Control Error')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Test 6: Phase Difference Control (0°~180°, 5° Resolution)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_Phase_Control.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Max phase error: {max(phase_errors):.2f}° (Requirement: ≤5°) {'✓' if max(phase_errors)<=5 else '✗'}")


# ==================== Test 7: Monte Carlo综合测试 ====================
print("\nTest 7: Monte Carlo System Test")

num_mc = 100
freq_errors = []
amp_errors = []
phase_errors_mc = []
separation_success = []

for _ in range(num_mc):
    # 随机频率 (5kHz整数倍)
    fA_mc = np.random.choice(np.arange(20e3, 100e3, 5e3))
    fB_mc = np.random.choice(np.arange(fA_mc + 5e3, 100e3 + 5e3, 5e3))
    
    # 随机相位
    phiA = np.random.uniform(0, 360)
    phiB = np.random.uniform(0, 360)
    
    # 生成信号
    A = generate_sine(t, fA_mc, 0.5, phiA)
    B = generate_sine(t, fB_mc, 0.5, phiB)
    C = A + B + 0.01 * np.random.randn(N)
    
    # 频率检测
    fA_det, fB_det, _, _ = detect_two_frequencies(C, fs)
    freq_err = abs(fA_det - fA_mc) + abs(fB_det - fB_mc)
    freq_errors.append(freq_err)
    
    # I/Q解调
    ampA, phaseA = iq_demodulate(C, fs, fA_mc, t)
    ampB, phaseB = iq_demodulate(C, fs, fB_mc, t)
    
    # 幅度误差 (ampA是峰值幅度，原始峰值=0.5V)
    amp_err = abs(ampA - 0.5) / 0.5 * 100
    amp_errors.append(amp_err)
    
    # 相位误差
    phase_err = abs(phaseA - phiA)
    if phase_err > 180:
        phase_err = 360 - phase_err
    phase_errors_mc.append(phase_err)
    
    # 分离成功判定 (幅度误差<10%, 频率误差<2kHz)
    success = (amp_err < 10) and (abs(fA_det - fA_mc) < 2e3)
    separation_success.append(success)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(freq_errors, bins=15, color='blue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f'Frequency Detection Error\nMean={np.mean(freq_errors):.0f}Hz')
axes[0, 0].set_xlabel('Error (Hz)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(amp_errors, bins=15, color='red', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=5, color='r', linestyle='--', linewidth=2, label='Req: 5%')
axes[0, 1].set_title(f'Amplitude Error\nMean={np.mean(amp_errors):.2f}%')
axes[0, 1].set_xlabel('Error (%)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(phase_errors_mc, bins=15, color='green', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=5, color='r', linestyle='--', linewidth=2, label='Req: 5°')
axes[1, 0].set_title(f'Phase Error\nMean={np.mean(phase_errors_mc):.2f}°')
axes[1, 0].set_xlabel('Error (°)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

success_rate = np.sum(separation_success) / len(separation_success) * 100
axes[1, 1].bar(['Success', 'Failure'], [success_rate, 100-success_rate], 
               color=['green', 'red'], edgecolor='black')
axes[1, 1].set_ylabel('Percentage (%)')
axes[1, 1].set_title(f'Separation Success Rate: {success_rate:.1f}%')
axes[1, 1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 7: Monte Carlo System Test (100 Runs, Random Frequencies)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_MonteCarlo_System.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Frequency error: Mean={np.mean(freq_errors):.0f}Hz")
print(f"  Amplitude error: Mean={np.mean(amp_errors):.2f}%")
print(f"  Phase error: Mean={np.mean(phase_errors_mc):.2f}°")
print(f"  Success rate: {success_rate:.1f}%")

print("\n=== Simulation Complete ===")
print(f"Output: {os.path.abspath(output_dir)}")
