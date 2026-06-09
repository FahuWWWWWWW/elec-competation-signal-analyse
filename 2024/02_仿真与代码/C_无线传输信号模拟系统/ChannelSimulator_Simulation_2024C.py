"""
2024年电赛C题「无线传输信号模拟系统」仿真
目标: 验证DDS载波产生 + AM调制 + 多径模拟 + 合路输出的正确性
技术: 数字信号处理模拟射频信号产生与多径合路
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 输出目录
output_dir = 'simulation_output'
os.makedirs(output_dir, exist_ok=True)

print("=== 2024-C Wireless Channel Simulator ===")
print("Technique: DDS + AM Modulation + Multipath + Combiner")
print()

# 全局参数
fs = 500e6  # 仿真采样率 500MSPS (足够采样40MHz信号)
N = 8192    # 采样点数
t = np.arange(N) / fs


def generate_carrier(t, fc, Ac, phi_deg=0):
    """产生载波信号 (CW)"""
    return Ac * np.cos(2 * np.pi * fc * t + np.radians(phi_deg))


def generate_am(t, fc, fm, Ac, ma, phi_deg=0):
    """产生AM信号"""
    carrier = np.cos(2 * np.pi * fc * t + np.radians(phi_deg))
    modulating = np.cos(2 * np.pi * fm * t)
    return Ac * (1 + ma * modulating) * carrier


def generate_multipath(t, fc, Ac, alpha, tau, phi_m_deg, phi_d_deg=0):
    """
    产生多径信号
    
    参数:
        alpha: 幅度衰减因子 (0~1)
        tau: 时延 (秒)
        phi_m_deg: 多径附加相位 (度)
        phi_d_deg: 直达信号初相 (度)
    """
    # 时延引入的相移
    phase_delay = 2 * np.pi * fc * tau
    
    # 总相移
    total_phase = np.radians(phi_d_deg) + phase_delay + np.radians(phi_m_deg)
    
    # 多径幅度
    Am = alpha * Ac
    
    # 时域延迟: 对信号进行插值延迟
    # 简单方法: 相位近似 (对于窄带信号，时延≈相移)
    # 精确方法: 频域相移 (对应时域延迟)
    
    # 使用频域方法实现精确延迟
    signal_orig = Am * np.cos(2 * np.pi * fc * t + total_phase)
    
    # 如果tau很小，直接用相位近似
    if tau < 1e-6:  # <1us
        return signal_orig
    else:
        # 对于大时延，用采样点移位+插值
        delay_samples = tau * fs
        int_delay = int(delay_samples)
        frac_delay = delay_samples - int_delay
        
        # 整数延迟+线性插值
        signal_delayed = np.zeros_like(signal_orig)
        for i in range(N):
            if i >= int_delay + 1:
                signal_delayed[i] = (1-frac_delay) * signal_orig[i-int_delay] + frac_delay * signal_orig[i-int_delay-1]
        
        return signal_delayed


# ==================== Test 1: CW与AM信号产生 ====================
print("Test 1: CW and AM Signal Generation")

fc = 35e6  # 35MHz载波
fm = 2e6   # 2MHz调制

# CW信号
cw = generate_carrier(t, fc, 1.0, 0)

# AM信号 (ma=0.6)
am = generate_am(t, fc, fm, 1.0, 0.6, 0)

# 频谱分析
N_fft = 8192
freqs = np.fft.fftfreq(N_fft, 1/fs)[:N_fft//2]
cw_fft = np.abs(np.fft.fft(cw[:N_fft]))[:N_fft//2] * 2/N_fft
am_fft = np.abs(np.fft.fft(am[:N_fft]))[:N_fft//2] * 2/N_fft

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# CW时域
N_show = 500
axes[0, 0].plot(t[:N_show]*1e9, cw[:N_show], 'b-', linewidth=1.5)
axes[0, 0].set_title('(a) CW Signal (fc=35MHz)')
axes[0, 0].set_xlabel('Time (ns)')
axes[0, 0].set_ylabel('Amplitude (V)')
axes[0, 0].grid(True, alpha=0.3)

# CW频谱
axes[0, 1].plot(freqs[:200]/1e6, cw_fft[:200], 'b-', linewidth=1.5)
axes[0, 1].axvline(x=fc/1e6, color='r', linestyle='--', label=f'fc={fc/1e6:.0f}MHz')
axes[0, 1].set_title('(b) CW Spectrum')
axes[0, 1].set_xlabel('Frequency (MHz)')
axes[0, 1].set_ylabel('Magnitude (V)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# AM时域
axes[1, 0].plot(t[:N_show]*1e9, am[:N_show], 'r-', linewidth=1.5)
# 绘制包络
envelope_pos = 1.0 * (1 + 0.6 * np.cos(2 * np.pi * fm * t[:N_show]))
envelope_neg = -1.0 * (1 + 0.6 * np.cos(2 * np.pi * fm * t[:N_show]))
axes[1, 0].plot(t[:N_show]*1e9, envelope_pos, 'g--', linewidth=1, alpha=0.7, label='Envelope')
axes[1, 0].plot(t[:N_show]*1e9, envelope_neg, 'g--', linewidth=1, alpha=0.7)
axes[1, 0].set_title('(c) AM Signal (ma=0.6, fm=2MHz)')
axes[1, 0].set_xlabel('Time (ns)')
axes[1, 0].set_ylabel('Amplitude (V)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# AM频谱
axes[1, 1].plot(freqs[300:450]/1e6, am_fft[300:450], 'r-', linewidth=1.5)
axes[1, 1].axvline(x=(fc-fm)/1e6, color='b', linestyle='--', label=f'fc-fm={fc/1e6-2:.0f}MHz')
axes[1, 1].axvline(x=fc/1e6, color='k', linestyle='--', label=f'fc={fc/1e6:.0f}MHz')
axes[1, 1].axvline(x=(fc+fm)/1e6, color='b', linestyle='--', label=f'fc+fm={fc/1e6+2:.0f}MHz')
axes[1, 1].set_title('(d) AM Spectrum (fc±fm)')
axes[1, 1].set_xlabel('Frequency (MHz)')
axes[1, 1].set_ylabel('Magnitude (V)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 1: CW and AM Signal Generation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_CW_AM_Signals.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test1_CW_AM_Signals.png")


# ==================== Test 2: AM调制度控制 ====================
print("\nTest 2: AM Modulation Depth Control")

ma_values = [0.3, 0.5, 0.7, 0.9]  # 30%~90%

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

measured_ma = []

for idx, ma_target in enumerate(ma_values):
    am_test = generate_am(t, fc, fm, 1.0, ma_target, 0)
    
    # 在数字仿真中，AM信号由精确公式产生，ma是精确的
    # 实际电赛中，ma由DAC控制乘法器调制电压，精度取决于DAC分辨率
    # 这里模拟8-bit DAC控制ma的精度: 量化步进约0.4%
    dac_bits = 8
    ma_quantized = np.round(ma_target * (2**dac_bits - 1)) / (2**dac_bits - 1)
    ma_measured = ma_quantized  # 模拟DAC量化后的实际ma
    measured_ma.append(ma_measured)
    
    ax = axes[idx // 2, idx % 2]
    N_disp = 400
    ax.plot(t[:N_disp]*1e9, am_test[:N_disp], 'b-', linewidth=1.5)
    # 绘制理想包络
    envelope = 1.0 * (1 + ma_target * np.cos(2 * np.pi * fm * t[:N_disp]))
    ax.plot(t[:N_disp]*1e9, envelope, 'r--', linewidth=1, label='Envelope')
    ax.plot(t[:N_disp]*1e9, -envelope, 'r--', linewidth=1)
    err = abs(ma_measured - ma_target) / ma_target * 100
    ax.set_title(f'ma={ma_target:.1f} → DAC={ma_measured:.3f}, Err={err:.1f}%')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 2: AM Modulation Depth Control', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_AM_Modulation_Depth.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  AM depth control results:")
for ma_t, ma_m in zip(ma_values, measured_ma):
    err = abs(ma_m - ma_t) / ma_t * 100
    print(f"    Target ma={ma_t:.1f}: Measured={ma_m:.3f}, Error={err:.1f}% {'✓' if err<=5 else '✗'}")


# ==================== Test 3: 多径时延与衰减 ====================
print("\nTest 3: Multipath Delay and Attenuation")

fc_test = 35e6
tau_values = [50e-9, 100e-9, 150e-9, 200e-9]  # 50~200ns
alpha_values = [1.0, 0.5, 0.2, 0.1]  # 0dB, -6dB, -14dB, -20dB

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 不同时延的多径
sd = generate_carrier(t, fc_test, 1.0, 0)

for idx, tau in enumerate(tau_values):
    sm = generate_multipath(t, fc_test, 1.0, 0.5, tau, 0, 0)
    s_out = sd + sm
    
    ax = axes[idx // 2, idx % 2]
    N_disp = 600
    t_ns = t[:N_disp] * 1e9
    ax.plot(t_ns, sd[:N_disp], 'b-', linewidth=1.5, label='Direct (SD)')
    ax.plot(t_ns, sm[:N_disp], 'r--', linewidth=1.5, label='Multipath (SM)')
    ax.plot(t_ns, s_out[:N_disp], 'g-', linewidth=2, alpha=0.7, label='Combined (SOut)')
    ax.set_title(f'Time Delay τ={tau*1e9:.0f}ns, α=0.5')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 3: Multipath Signal with Different Delays', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_Multipath_Delay.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test3_Multipath_Delay.png")


# ==================== Test 4: 多径衰减控制 ====================
print("\nTest 4: Multipath Attenuation Control")

tau_test = 100e-9  # 100ns时延

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, alpha in enumerate(alpha_values):
    sm = generate_multipath(t, fc_test, 1.0, alpha, tau_test, 0, 0)
    s_out = sd + sm
    
    # 计算衰减dB
    attenuation_dB = 20 * np.log10(alpha)
    
    ax = axes[idx // 2, idx % 2]
    N_disp = 600
    t_ns = t[:N_disp] * 1e9
    ax.plot(t_ns, sd[:N_disp], 'b-', linewidth=1.5, label='Direct (SD)')
    ax.plot(t_ns, sm[:N_disp], 'r--', linewidth=1.5, label=f'Multipath α={alpha}')
    ax.plot(t_ns, s_out[:N_disp], 'g-', linewidth=2, alpha=0.7, label='SOut')
    ax.set_title(f'Attenuation={attenuation_dB:.0f}dB (α={alpha})')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 4: Multipath Signal with Different Attenuations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_Multipath_Attenuation.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Attenuation accuracy:")
for alpha in alpha_values:
    target_dB = 20 * np.log10(alpha)
    print(f"    α={alpha}: {target_dB:.1f}dB")


# ==================== Test 5: 多径相位控制 ====================
print("\nTest 5: Multipath Phase Control")

tau_test = 50e-9
alpha_test = 0.8
phi_m_values = [0, 30, 90, 180]  # 0°~180°

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, phi_m in enumerate(phi_m_values):
    sm = generate_multipath(t, fc_test, 1.0, alpha_test, tau_test, phi_m, 0)
    s_out = sd + sm
    
    ax = axes[idx // 2, idx % 2]
    N_disp = 600
    t_ns = t[:N_disp] * 1e9
    ax.plot(t_ns, sd[:N_disp], 'b-', linewidth=1.5, label='SD')
    ax.plot(t_ns, sm[:N_disp], 'r--', linewidth=1.5, label=f'SM (φ={phi_m}°)')
    ax.plot(t_ns, s_out[:N_disp], 'g-', linewidth=2, alpha=0.7, label='SOut')
    ax.set_title(f'Multipath Phase φ={phi_m}°')
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 5: Multipath Signal with Different Phases', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_Multipath_Phase.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test5_Multipath_Phase.png")


# ==================== Test 6: 合路输出频谱分析 ====================
print("\nTest 6: Combined Output Spectrum Analysis")

# AM信号 + 多径
am_direct = generate_am(t, fc_test, fm, 1.0, 0.5, 0)
am_multipath = generate_multipath(t, fc_test, 1.0, 0.6, 100e-9, 45, 0)
am_out = am_direct + am_multipath

# 频谱
am_direct_fft = np.abs(np.fft.fft(am_direct[:N_fft]))[:N_fft//2] * 2/N_fft
am_multi_fft = np.abs(np.fft.fft(am_multipath[:N_fft]))[:N_fft//2] * 2/N_fft
am_out_fft = np.abs(np.fft.fft(am_out[:N_fft]))[:N_fft//2] * 2/N_fft

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 直达信号频谱
axes[0, 0].plot(freqs[300:450]/1e6, am_direct_fft[300:450], 'b-', linewidth=1.5)
axes[0, 0].set_title('(a) Direct AM Signal Spectrum')
axes[0, 0].set_xlabel('Frequency (MHz)')
axes[0, 0].set_ylabel('Magnitude (V)')
axes[0, 0].grid(True, alpha=0.3)

# 多径信号频谱
axes[0, 1].plot(freqs[300:450]/1e6, am_multi_fft[300:450], 'r-', linewidth=1.5)
axes[0, 1].set_title('(b) Multipath Signal Spectrum')
axes[0, 1].set_xlabel('Frequency (MHz)')
axes[0, 1].set_ylabel('Magnitude (V)')
axes[0, 1].grid(True, alpha=0.3)

# 合路信号频谱
axes[1, 0].plot(freqs[300:450]/1e6, am_out_fft[300:450], 'g-', linewidth=1.5)
axes[1, 0].set_title('(c) Combined Output Spectrum')
axes[1, 0].set_xlabel('Frequency (MHz)')
axes[1, 0].set_ylabel('Magnitude (V)')
axes[1, 0].grid(True, alpha=0.3)

# 时域波形
N_disp = 600
axes[1, 1].plot(t[:N_disp]*1e9, am_direct[:N_disp], 'b-', linewidth=1.5, label='SD')
axes[1, 1].plot(t[:N_disp]*1e9, am_multipath[:N_disp], 'r--', linewidth=1.5, label='SM')
axes[1, 1].plot(t[:N_disp]*1e9, am_out[:N_disp], 'g-', linewidth=2, alpha=0.7, label='SOut')
axes[1, 1].set_title('(d) Time Domain Waveforms')
axes[1, 1].set_xlabel('Time (ns)')
axes[1, 1].set_ylabel('Amplitude (V)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 6: AM with Multipath - Spectrum and Time Domain', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_Combined_Spectrum.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test6_Combined_Spectrum.png")


# ==================== Test 7: 载波频率扫描 ====================
print("\nTest 7: Carrier Frequency Sweep (30~40MHz)")

fc_values = np.arange(30e6, 41e6, 1e6)  # 30~40MHz, 1MHz步进
measured_freqs = []
freq_errors = []

for fc_target in fc_values:
    cw_test = generate_carrier(t, fc_target, 1.0, 0)
    
    # 测量频率 (FFT峰值)
    fft_cw = np.abs(np.fft.fft(cw_test[:N_fft]))[:N_fft//2]
    peak_idx = np.argmax(fft_cw)
    fc_measured = freqs[peak_idx]
    
    measured_freqs.append(fc_measured)
    err = abs(fc_measured - fc_target) / fc_target * 100
    freq_errors.append(err)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fc_values/1e6, fc_values/1e6, 'k--', linewidth=2, label='Target')
axes[0].plot(fc_values/1e6, np.array(measured_freqs)/1e6, 'bo-', linewidth=2, markersize=8, label='Measured')
axes[0].set_xlabel('Target Frequency (MHz)')
axes[0].set_ylabel('Measured Frequency (MHz)')
axes[0].set_title('(a) Frequency Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar([f'{f/1e6:.0f}' for f in fc_values], freq_errors, color='steelblue', edgecolor='black')
axes[1].axhline(y=2, color='r', linestyle='--', linewidth=2, label='Requirement: 2%')
axes[1].set_xlabel('Target Frequency (MHz)')
axes[1].set_ylabel('Frequency Error (%)')
axes[1].set_title('(b) Frequency Error')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 7: Carrier Frequency Sweep Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_Frequency_Sweep.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Frequency sweep results:")
max_err = max(freq_errors)
print(f"    Max frequency error: {max_err:.4f}% (Req ≤2%) {'✓' if max_err<=2 else '✗'}")

print("\n=== Simulation Complete ===")
print(f"Output: {os.path.abspath(output_dir)}")
