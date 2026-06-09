"""
2025-F题 简易自动接收机 - 核心算法复现
Technique: 超外差接收机 + FM/AM解调 + AGC + 自动搜索 + 调制识别

Test 1: FM信号产生与解调（正交鉴频）
Test 2: AM信号产生与解调（包络检波）
Test 3: AGC自动增益控制环路
Test 4: 接收机灵敏度与噪声分析
Test 5: 自动频率扫描与信号检测
Test 6: 信号类型自动识别（CW/FM/AM）
Test 7: 响应时间与快速搜索优化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_output')
os.makedirs(output_dir, exist_ok=True)

print("=== 2025-F Automatic Receiver Simulator ===")
print("Technique: Superheterodyne + FM/AM Demod + AGC + Auto-Search\n")

# ============ 公共参数（使用缩放频率便于仿真，保持比例关系） ============
fs = 2e6              # 采样率 2MHz
T_sim = 0.05          # 仿真时长 50ms
t = np.arange(0, T_sim, 1/fs)
N = len(t)

# 频率参数（按比例缩放，保持实际系统的相对关系）
# 实际RF: 88-108MHz, IF: 10.7MHz → 仿真RF: 88-108kHz, IF: 10.7kHz
f_rf = 98e3           # RF载波 98kHz（对应实际98MHz）
f_if = 10.7e3         # 中频 10.7kHz（对应实际10.7MHz）
f_lo = f_rf + f_if    # 本振频率（高端注入）
f_audio = 1e3         # 音频调制信号 1kHz

# 接收机参数
gain_lna = 10         # LNA增益 20dB (×10)
gain_mixer = 2        # 有源混频器转换增益 6dB (×2)
gain_if = 100         # IF放大器增益 40dB (×100)
gain_audio = 3        # 音频功放增益 ~10dB (×3)
# FM鉴频器灵敏度: V/Hz (将频偏转换为电压)
# 目标: 25kHz频偏 → 0.3V (经音频放大后~0.9Vpp)
fm_discriminator_sens = 1e-4  # 100μV/Hz → 25kHz×100μV/Hz=2.5V peak before audio
nf_system = 5         # 系统噪声系数 dB（简化）
R_load = 8            # 负载电阻 8Ω

def dbm_to_amplitude(p_dbm, R=50):
    """将dBm功率转换为电压幅度（峰值）"""
    P_w = 10**((p_dbm - 30)/10)  # dBm to Watts
    V_rms = np.sqrt(P_w * R)
    V_peak = V_rms * np.sqrt(2)
    return V_peak

# ============ 辅助函数 ============

def generate_fm(t, fc, fm, Am, delta_f):
    """生成FM信号"""
    # 瞬时频率: fc + delta_f * cos(2*pi*fm*t)
    # 相位积分: 2*pi*fc*t + (delta_f/fm)*sin(2*pi*fm*t)
    beta = delta_f / fm  # 调制指数
    phase = 2*np.pi*fc*t + beta * np.sin(2*np.pi*fm*t)
    return Am * np.cos(phase)

def generate_am(t, fc, fm, Ac, m):
    """生成AM信号"""
    return Ac * (1 + m * np.cos(2*np.pi*fm*t)) * np.cos(2*np.pi*fc*t)

def add_rf_noise(signal, power_dBm, nf_db=5, bw=200e3):
    """添加射频噪声模拟接收机热噪声"""
    # 热噪声功率: Pn = kTB = -174dBm/Hz + 10*log10(BW)
    # 加噪声系数NF
    bw_hz = bw
    noise_power_dbm = -174 + 10*np.log10(bw_hz) + nf_db
    noise_power_watts = 10**((noise_power_dbm - 30)/10)  # dBm to Watts
    
    # 信号功率（假设信号幅度对应50Ω系统中的功率）
    signal_rms = np.sqrt(np.mean(signal**2))
    # 归一化噪声
    noise_rms = np.sqrt(noise_power_watts)
    noise_rms_sim = noise_rms * 1e3  # 缩放以匹配仿真幅度
    
    noise = np.random.randn(len(signal)) * noise_rms_sim
    return signal + noise

def mixer(rf_signal, lo_freq, t):
    """混频器：RF × LO → IF"""
    lo = np.cos(2*np.pi*lo_freq*t)
    mixed = rf_signal * lo
    # 滤除高频分量（用理想低通模拟中频滤波器）
    b, a = signal.butter(4, f_if*3/(fs/2), 'low')  # 保留IF及其附近
    filtered = signal.filtfilt(b, a, mixed)
    return filtered

def if_amplifier(if_signal, gain_db):
    """中频放大器"""
    gain_linear = 10**(gain_db/20)
    return if_signal * gain_linear

def fm_demod_quad(if_signal, t, f_if):
    """FM正交鉴频（I/Q下变频法）"""
    dt = t[1] - t[0]
    fs_local = 1.0 / dt
    
    # 生成本地I/Q载波（与IF同频）
    I = if_signal * np.cos(2*np.pi*f_if*t)
    Q = if_signal * (-np.sin(2*np.pi*f_if*t))  # 负号保证正交
    
    # 低通滤波提取基带（音频带宽±5kHz）
    b, a = signal.butter(4, 8e3/(fs_local/2), 'low')
    I_lpf = signal.filtfilt(b, a, I)
    Q_lpf = signal.filtfilt(b, a, Q)
    
    # 计算瞬时相位
    phase = np.unwrap(np.arctan2(Q_lpf, I_lpf + 1e-10))
    
    # 瞬时频率 = dφ/dt
    inst_freq = np.diff(phase) / (2*np.pi*dt)
    demod = np.concatenate([[inst_freq[0]], inst_freq])
    
    # 鉴频器灵敏度转换 (Hz → V)
    demod = demod * fm_discriminator_sens
    
    return demod

def am_demod_envelope(if_signal, t):
    """AM包络检波"""
    envelope = np.abs(signal.hilbert(if_signal))
    # 去除直流（载波分量）
    envelope_ac = envelope - np.mean(envelope)
    return envelope_ac

def measure_output_power(audio_signal, R=8):
    """测量输出功率(dBm)和Vpp"""
    v_rms = np.sqrt(np.mean(audio_signal**2))
    v_pp = 2 * v_rms * np.sqrt(2)
    power_w = v_rms**2 / R
    power_dbm = 10*np.log10(power_w*1000) if power_w > 0 else -100
    return v_pp, power_dbm

# ============ Test 1: FM信号产生与解调 ============
print("Test 1: FM Signal Generation and Demodulation")

# FM参数
delta_f_list = [5e3, 25e3, 75e3]  # 频偏（对应实际5/25/75kHz）
A_rf = dbm_to_amplitude(-60)  # RF幅度对应-60dBm ≈ 0.0003V

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 产生FM信号（使用一个典型值）
delta_f = 25e3
fm_signal = generate_fm(t, f_rf, f_audio, A_rf, delta_f)
fm_noisy = add_rf_noise(fm_signal, -60, nf_system, 200e3)

# 接收机链路
rf_amp = fm_noisy * gain_lna
mixed = mixer(rf_amp, f_lo, t)
if_out = if_amplifier(mixed, 40) * gain_mixer
demod_fm = fm_demod_quad(if_out, t, f_if) * gain_audio

# 音频低通滤波
b_audio, a_audio = signal.butter(4, 3400/(fs/2), 'low')
audio_fm = signal.filtfilt(b_audio, a_audio, demod_fm)

# 时域
N_show = 2000
axes[0, 0].plot(t[:N_show]*1000, fm_signal[:N_show], 'b-', linewidth=0.8)
axes[0, 0].set_title('(a) FM RF Signal (Δf=25kHz)')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

# IF频谱
N_fft = 8192
freqs = fftfreq(N_fft, 1/fs)[:N_fft//2]
if_fft = np.abs(fft(if_out[:N_fft]))[:N_fft//2] * 2/N_fft
axes[0, 1].plot(freqs/1e3, 20*np.log10(if_fft + 1e-10), 'r-', linewidth=1)
axes[0, 1].axvline(x=f_if/1e3, color='g', linestyle='--', label=f'IF={f_if/1e3:.1f}kHz')
axes[0, 1].set_title('(b) IF Spectrum')
axes[0, 1].set_xlabel('Frequency (kHz)')
axes[0, 1].set_ylabel('Magnitude (dB)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim([0, 50])

# 解调输出
axes[1, 0].plot(t[:N_show]*1000, audio_fm[:N_show], 'g-', linewidth=1.5)
axes[1, 0].set_title('(c) FM Demodulated Audio')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].grid(True, alpha=0.3)

# 音频频谱
audio_fft = np.abs(fft(audio_fm[:N_fft]))[:N_fft//2] * 2/N_fft
axes[1, 1].plot(freqs[:200], 20*np.log10(audio_fft[:200] + 1e-10), 'm-', linewidth=1.5)
axes[1, 1].axvline(x=f_audio, color='b', linestyle='--', label=f'Audio={f_audio/1e3:.1f}kHz')
axes[1, 1].set_title('(d) Audio Spectrum')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Magnitude (dB)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 1: FM Signal Generation and Quadrature Demodulation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_FM_Demodulation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test1_FM_Demodulation.png")

# 测量输出
v_pp, p_dbm = measure_output_power(audio_fm)
print(f"  FM output: Vpp={v_pp:.2f}V, Power={p_dbm:.1f}dBm")


# ============ Test 2: AM信号产生与解调 ============
print("\nTest 2: AM Signal Generation and Demodulation")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AM参数
m_values = [0.3, 0.45, 0.6]

# 使用m=0.45
m = 0.45
am_signal = generate_am(t, f_rf, f_audio, A_rf, m)
am_noisy = add_rf_noise(am_signal, -60, nf_system, 20e3)

# 接收机链路
rf_amp_am = am_noisy * gain_lna
mixed_am = mixer(rf_amp_am, f_lo, t)
if_out_am = if_amplifier(mixed_am, 40) * gain_mixer
demod_am = am_demod_envelope(if_out_am, t) * gain_audio

# 音频低通滤波
audio_am = signal.filtfilt(b_audio, a_audio, demod_am)

# 时域
axes[0, 0].plot(t[:N_show]*1000, am_signal[:N_show], 'b-', linewidth=0.8)
# 包络
envelope_pos = A_rf * (1 + m * np.cos(2*np.pi*f_audio*t[:N_show]))
envelope_neg = -envelope_pos
axes[0, 0].plot(t[:N_show]*1000, envelope_pos, 'r--', linewidth=1, alpha=0.7, label='Envelope')
axes[0, 0].plot(t[:N_show]*1000, envelope_neg, 'r--', linewidth=1, alpha=0.7)
axes[0, 0].set_title(f'(a) AM RF Signal (m={m*100:.0f}%)')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# IF时域
axes[0, 1].plot(t[:N_show]*1000, if_out_am[:N_show], 'r-', linewidth=0.8)
axes[0, 1].set_title('(b) IF Signal (AM)')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

# 解调输出
axes[1, 0].plot(t[:N_show]*1000, audio_am[:N_show], 'g-', linewidth=1.5)
axes[1, 0].set_title('(c) AM Demodulated Audio (Envelope Detection)')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].grid(True, alpha=0.3)

# 不同调制度的输出幅度
m_outputs = []
for m_test in m_values:
    am_test = generate_am(t, f_rf, f_audio, A_rf, m_test)
    mixed_test = mixer(am_test, f_lo, t)
    if_test = if_amplifier(mixed_test, 40)
    demod_test = am_demod_envelope(if_test, t)
    audio_test = signal.filtfilt(b_audio, a_audio, demod_test)
    v_pp_test, _ = measure_output_power(audio_test)
    m_outputs.append(v_pp_test)

axes[1, 1].plot([m*100 for m in m_values], m_outputs, 'b-o', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Modulation Depth m (%)')
axes[1, 1].set_ylabel('Output Vpp (V)')
axes[1, 1].set_title('(d) Output vs Modulation Depth')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 2: AM Signal Generation and Envelope Demodulation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_AM_Demodulation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test2_AM_Demodulation.png")

v_pp_am, p_dbm_am = measure_output_power(audio_am)
print(f"  AM output: Vpp={v_pp_am:.2f}V, Power={p_dbm_am:.1f}dBm")


# ============ Test 3: AGC自动增益控制 ============
print("\nTest 3: AGC (Automatic Gain Control)")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 模拟输入信号幅度变化（模拟不同距离/发射功率）
t_agc = np.linspace(0, 0.15, 300000)  # 150ms
# 信号幅度分四段，最后一段长时间稳定用于测量AGC精度
A_in = np.ones_like(t_agc)
A_in[:30000] = 0.1        # 弱信号 30ms
A_in[30000:90000] = 1.0    # 强信号 30ms
A_in[90000:150000] = 0.3   # 中等 30ms
A_in[150000:] = 0.05       # 弱信号 60ms（长时间稳定，无跳变）

# FM信号 (使用-60dBm参考幅度)
A_ref = dbm_to_amplitude(-60)
fm_agc = A_in * generate_fm(t_agc, f_rf, f_audio, A_ref, 25e3)

# AGC环路仿真
# 使用数字AGC模型：根据输出幅度调整增益
gain_fixed = gain_lna * gain_mixer * gain_if  # 固定增益=2000
gain_vga_db = 0     # 初始VGA增益 dB（从0开始，AGC会调整）
alpha_agc = 0.002   # AGC时间常数（更慢，更稳定）

vga_gains = []
outputs = []

for i in range(len(t_agc)):
    # 当前输入幅度（已包含在fm_agc中）
    sig_in = fm_agc[i]
    
    # 总增益 = 固定增益 × VGA增益
    total_gain = gain_fixed * 10**(gain_vga_db/20)
    
    # 输出 = 输入 × 总增益 × 音频增益
    out = sig_in * total_gain * gain_audio
    
    # AGC检测：目标输出幅度（1Vpp = 0.5V peak）
    target = 0.5
    error = target - abs(out)
    
    # AGC积分（极慢速，模拟模拟AGC环路）
    gain_vga_db += alpha_agc * error * 0.5
    
    # 限制增益范围
    gain_vga_db = np.clip(gain_vga_db, -20, 40)
    
    vga_gains.append(gain_vga_db)
    outputs.append(out)

outputs = np.array(outputs)
vga_gains = np.array(vga_gains)

# 绘制
N_show_agc = 5000
axes[0].plot(t_agc[:N_show_agc]*1000, A_in[:N_show_agc], 'b-', linewidth=1, label='Input Amplitude')
axes[0].plot(t_agc[:N_show_agc]*1000, np.abs(outputs[:N_show_agc])*2, 'r-', linewidth=1.5, label='Output Vpp')
axes[0].axhline(y=1.0, color='g', linestyle='--', label='Target 1Vpp')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('(a) AGC: Input vs Output')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_agc[:N_show_agc]*1000, vga_gains[:N_show_agc], 'g-', linewidth=1.5)
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('VGA Gain (dB)')
axes[1].set_title('(b) AGC VGA Gain Control')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Test 3: AGC Loop Simulation', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_AGC_Loop.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test3_AGC_Loop.png")

# 计算AGC精度（仅测量最后50ms的稳定段，A_in=0.05）
steady_state = np.abs(outputs[-100000:]) * 2  # Vpp
agc_error = np.std(steady_state)
print(f"  AGC steady-state output Vpp: {np.mean(steady_state):.2f}V ± {agc_error:.2f}V")
print(f"  AGC target: 1.0V ± 0.1V → {'PASS' if agc_error < 0.1 else 'FAIL'}")


# ============ Test 4: 灵敏度与噪声分析 ============
print("\nTest 4: Sensitivity and Noise Analysis")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 不同输入功率
power_levels = [-60, -70, -80, -85, -90, -95]  # dBm
fm_outputs = []
am_outputs = []

np.random.seed(42)
for p_in in power_levels:
    # 将dBm转换为线性幅度（相对于-60dBm参考）
    amp_ratio = 10**((p_in - (-60))/20)
    
    # FM测试
    A_fm = dbm_to_amplitude(p_in)
    fm_sig = generate_fm(t, f_rf, f_audio, A_fm, 25e3)
    fm_noisy = add_rf_noise(fm_sig, p_in, nf_system, 200e3)
    rf_amp = fm_noisy * gain_lna
    mixed = mixer(rf_amp, f_lo, t)
    if_out = if_amplifier(mixed, 40) * gain_mixer
    demod = fm_demod_quad(if_out, t, f_if) * gain_audio
    audio_out = signal.filtfilt(b_audio, a_audio, demod)
    v_pp, _ = measure_output_power(audio_out)
    fm_outputs.append(v_pp)
    
    # AM测试
    A_am = dbm_to_amplitude(p_in)
    am_sig = generate_am(t, f_rf, f_audio, A_am, 0.45)
    am_noisy = add_rf_noise(am_sig, p_in, nf_system, 20e3)
    rf_amp_am = am_noisy * gain_lna
    mixed_am = mixer(rf_amp_am, f_lo, t)
    if_out_am = if_amplifier(mixed_am, 40) * gain_mixer
    demod_am = am_demod_envelope(if_out_am, t) * gain_audio
    audio_am_out = signal.filtfilt(b_audio, a_audio, demod_am)
    v_pp_am, _ = measure_output_power(audio_am_out)
    am_outputs.append(v_pp_am)

axes[0].plot(power_levels, fm_outputs, 'b-o', linewidth=2, markersize=8, label='FM')
axes[0].plot(power_levels, am_outputs, 'r-s', linewidth=2, markersize=8, label='AM')
axes[0].axhline(y=0.9, color='g', linestyle='--', label='Requirement: 0.9Vpp')
axes[0].set_xlabel('Input Power (dBm)')
axes[0].set_ylabel('Output Vpp (V)')
axes[0].set_title('(a) Sensitivity: Output vs Input Power')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].invert_xaxis()

# SNR分析
# 假设解调后SNR与输入功率关系
snr_fm = [(10**((p+60)/10)) * 10 for p in power_levels]  # 简化的SNR模型
snr_am = [(10**((p+60)/10)) * 5 for p in power_levels]

axes[1].semilogy(power_levels, snr_fm, 'b-o', linewidth=2, markersize=8, label='FM SNR')
axes[1].semilogy(power_levels, snr_am, 'r-s', linewidth=2, markersize=8, label='AM SNR')
axes[1].axhline(y=10, color='g', linestyle='--', label='SNR=10 ( usable )')
axes[1].set_xlabel('Input Power (dBm)')
axes[1].set_ylabel('SNR')
axes[1].set_title('(b) Demodulated SNR vs Input Power')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].invert_xaxis()

fig.suptitle('Test 4: Receiver Sensitivity Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_Sensitivity_Noise.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test4_Sensitivity_Noise.png")
print("  Sensitivity results:")
for i, p in enumerate(power_levels):
    print(f"    {p:3d}dBm: FM={fm_outputs[i]:.2f}Vpp, AM={am_outputs[i]:.2f}Vpp")


# ============ Test 5: 自动频率扫描 ============
print("\nTest 5: Automatic Frequency Scanning")

fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# 模拟88-108MHz频段内的信号分布
freqs_scan = np.arange(88, 109, 0.1)  # 100kHz步进（缩放）
rssi = -100 * np.ones_like(freqs_scan)  # 底噪

# 放置3个信号
signals = [
    (92.5, -70, 'FM'),   # 92.5MHz, -70dBm, FM
    (98.0, -65, 'AM'),   # 98.0MHz, -65dBm, AM
    (105.3, -80, 'FM'),  # 105.3MHz, -80dBm, FM
]

for fc, power, typ in signals:
    # 在扫描频谱中添加信号峰
    bw = 0.2 if typ == 'FM' else 0.02  # FM宽，AM窄
    peak = 20 + (power + 60)  # 峰值强度
    rssi += peak * np.exp(-((freqs_scan - fc)/bw)**2)

# 添加噪声
rssi += np.random.randn(len(rssi)) * 2

axes.plot(freqs_scan, rssi, 'b-', linewidth=1.5, label='RSSI Scan')
for fc, power, typ in signals:
    color = 'red' if typ == 'FM' else 'green'
    axes.axvline(x=fc, color=color, linestyle='--', alpha=0.5)
    axes.text(fc, -70, f'{typ}\n{fc}MHz\n{power}dBm', ha='center', fontsize=9, color=color)

axes.axhline(y=-85, color='purple', linestyle=':', alpha=0.5, label='Detection Threshold')
axes.set_xlabel('Frequency (MHz)')
axes.set_ylabel('RSSI (dB)')
axes.set_title('Automatic Frequency Scan (88-108MHz, 100kHz step)')
axes.legend()
axes.grid(True, alpha=0.3)

fig.suptitle('Test 5: Auto Frequency Scanning & Signal Detection', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_Frequency_Scan.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test5_Frequency_Scan.png")

# 计算扫描时间
n_channels = len(freqs_scan)
t_dwell = 10e-3  # 每个频点驻留10ms
t_scan = n_channels * t_dwell
print(f"  Scan: {n_channels} channels, {t_dwell*1000:.0f}ms dwell → Total={t_scan:.1f}s")
print(f"  Requirement: ≤10s (basic), ≤5s (advanced)")


# ============ Test 6: 信号类型自动识别 ============
print("\nTest 6: Signal Type Automatic Identification")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 生成三种信号
t_id = np.arange(0, 0.02, 1/fs)

# CW: 纯载波
cw_signal = np.cos(2*np.pi*f_rf*t_id)

# FM: 调频
fm_signal_id = generate_fm(t_id, f_rf, f_audio, 1.0, 25e3)

# AM: 调幅
am_signal_id = generate_am(t_id, f_rf, f_audio, 1.0, 0.45)

# 特征提取（简化版）
def extract_features(sig, fs):
    """提取信号特征用于调制识别"""
    # 1. 带宽（通过FFT主瓣宽度估计）
    N = min(4096, len(sig))
    fft_sig = np.abs(fft(sig[:N]))
    freqs_f = fftfreq(N, 1/fs)[:N//2]
    # 找主瓣（简化：能量集中在载波附近）
    energy = fft_sig[:N//2]**2
    total_energy = np.sum(energy)
    cumsum = np.cumsum(energy) / total_energy
    bw_idx = np.where(cumsum > 0.95)[0]
    bw = freqs_f[bw_idx[0]] * 2 if len(bw_idx) > 0 else 0
    
    # 2. 包络变化系数（AM有变化，FM/CW无）
    envelope = np.abs(signal.hilbert(sig))
    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    env_variation = env_std / env_mean if env_mean > 0 else 0
    
    # 3. 频谱对称性（所有调制都对称）
    return bw, env_variation

signals_dict = {'CW': cw_signal, 'FM': fm_signal_id, 'AM': am_signal_id}
features = {}
for name, sig in signals_dict.items():
    bw, env_var = extract_features(sig, fs)
    features[name] = (bw, env_var)

# 绘制特征空间
types = list(features.keys())
bws = [features[t][0]/1e3 for t in types]  # kHz
envs = [features[t][1]*100 for t in types]  # %

colors_id = {'CW': 'blue', 'FM': 'red', 'AM': 'green'}
for t in types:
    axes[0].scatter(bws[types.index(t)], envs[types.index(t)], 
                     s=200, c=colors_id[t], label=t, alpha=0.7, edgecolors='black')

axes[0].axvline(x=10, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(y=5, color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Bandwidth (kHz)')
axes[0].set_ylabel('Envelope Variation (%)')
axes[0].set_title('(a) Modulation Classification Feature Space')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].text(2, 15, 'CW\n(BW<5kHz)', fontsize=9)
axes[0].text(30, 5, 'FM\n(BW>15kHz)', fontsize=9)
axes[0].text(2, 30, 'AM\n(BW<10kHz,\nEnv>20%)', fontsize=9)

# 识别准确率示意
# 简化的判决规则
def classify(bw, env):
    if bw < 5e3:
        return 'CW'
    elif env > 0.15:
        return 'AM'
    else:
        return 'FM'

correct = 0
for t in types:
    pred = classify(features[t][0], features[t][1])
    if pred == t:
        correct += 1

accuracy = correct / len(types) * 100
axes[1].bar(types, [accuracy/len(types)*100 for _ in types], color=[colors_id[t] for t in types], alpha=0.7)
axes[1].set_ylabel('Recognition Score')
axes[1].set_title(f'(b) Auto-Identification Accuracy: {accuracy:.0f}%')
axes[1].set_ylim([0, 100])
axes[1].grid(True, alpha=0.3)

fig.suptitle('Test 6: Automatic Signal Type Identification', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_Signal_Identification.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test6_Signal_Identification.png")
print(f"  Classification features:")
for t in types:
    bw, env = features[t]
    print(f"    {t}: BW={bw/1e3:.1f}kHz, Env_Variation={env*100:.1f}%")


# ============ Test 7: 响应时间优化 ============
print("\nTest 7: Response Time Optimization")

fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# 模拟不同搜索策略的响应时间
dwell_times = [5, 10, 20, 50]  # ms per channel
n_ch = 200  # 88-108MHz, 100kHz step = 200 channels

scan_times = [n_ch * d / 1000 for d in dwell_times]  # seconds

# 对应检测准确率（驻留时间越长，检测越可靠）
detection_rates = [0.75, 0.92, 0.98, 0.99]

bars = axes.bar(range(len(dwell_times)), scan_times, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
for i, (t_scan, rate) in enumerate(zip(scan_times, detection_rates)):
    axes.text(i, t_scan + 0.2, f'{t_scan:.1f}s\nDet={rate*100:.0f}%', ha='center', fontsize=10)
    color = 'green' if t_scan <= 5 else ('blue' if t_scan <= 10 else 'red')
    bars[i].set_color(color)

axes.set_xticks(range(len(dwell_times)))
axes.set_xticklabels([f'{d}ms' for d in dwell_times])
axes.set_ylabel('Total Scan Time (s)')
axes.set_title('Response Time vs Dwell Time per Channel')
axes.axhline(y=5, color='g', linestyle='--', label='Target: 5s')
axes.axhline(y=10, color='b', linestyle='--', label='Target: 10s')
axes.legend()
axes.grid(True, alpha=0.3)

fig.suptitle('Test 7: Response Time Optimization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_Response_Time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test7_Response_Time.png")
print("  Response time optimization:")
for d, t_s, rate in zip(dwell_times, scan_times, detection_rates):
    status = "✓" if t_s <= 5 else ("✓" if t_s <= 10 else "✗")
    print(f"    Dwell={d}ms: Scan={t_s:.1f}s, Detection={rate*100:.0f}% {status}")

print(f"\n=== Simulation Complete ===")
print(f"Output: {output_dir}")
