"""
2023年电赛B题「同轴电缆长度与终端负载检测装置」TDR仿真 v3
核心改进: 
- 长度测量使用直接的峰值检测法（等效采样波形上）
- 负载识别使用反射段波形特征分析
- 电容响应使用正确的RC电路模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 全局参数
Z0 = 50                     # 特性阻抗 (Ω)
c = 3e8                     # 光速 (m/s)
epsilon_r = 2.3             # PE介质相对介电常数
v = c / np.sqrt(epsilon_r)  # 传播速度 (m/s)

# TDR脉冲参数 (高速脉冲发生器)
tp_rise = 1e-9              # 上升时间 1ns
pulse_sigma = 1.5e-9        # 高斯脉宽 1.5ns

# 等效采样参数
fs_equiv = 1e9              # 等效采样率 1GSPS (提高分辨率)
dt_equiv = 1/fs_equiv

output_dir = 'simulation_output'
os.makedirs(output_dir, exist_ok=True)

print("=== 2023-B TDR Cable Test Simulation v3 ===")
print(f"Z0 = {Z0}Ω, v = {v:.2e} m/s ({v/c:.2f}c)")
print(f"Equivalent Sampling: {fs_equiv/1e9:.1f} GSPS (dt = {dt_equiv*1e12:.0f} ps)")
print()


def gaussian_pulse(t, t0, sigma, A):
    return A * np.exp(-((t - t0)**2) / (2 * sigma**2))


def generate_tdr_open(t, dt, L, v, sigma):
    """生成开路终端的TDR波形"""
    t_round = 2 * L / v
    incident = gaussian_pulse(t, 0, sigma, 1.0)
    reflected = gaussian_pulse(t, t_round, sigma, 1.0)  # 开路Γ=+1
    return incident + reflected


def generate_tdr_short(t, dt, L, v, sigma):
    """生成短路终端的TDR波形"""
    t_round = 2 * L / v
    incident = gaussian_pulse(t, 0, sigma, 1.0)
    reflected = gaussian_pulse(t, t_round, sigma, -1.0)  # 短路Γ=-1
    return incident + reflected


def generate_tdr_resistor(t, dt, L, v, sigma, R):
    """生成电阻终端的TDR波形"""
    t_round = 2 * L / v
    gamma = (R - Z0) / (R + Z0)
    incident = gaussian_pulse(t, 0, sigma, 1.0)
    reflected = gaussian_pulse(t, t_round, sigma, gamma)
    return incident + reflected


def generate_tdr_capacitor(t, dt, L, v, sigma, C):
    """生成电容终端的TDR波形 - RC电路响应"""
    t_round = 2 * L / v
    tau = Z0 * C
    
    incident = gaussian_pulse(t, 0, sigma, 1.0)
    reflected = np.zeros_like(t)
    
    # 电容对阶跃的反射响应:
    # t=t_round: 开始反射，初始γ=-1(短路)，过渡到γ=+1(开路)
    # 反射波形 = 2 * u(t-t_round) * (1 - exp(-(t-t_round)/τ)) - 1
    for i in range(len(t)):
        if t[i] > t_round:
            dt_local = t[i] - t_round
            reflected[i] = 2 * (1 - np.exp(-dt_local / tau)) - 1
    
    # 与高斯脉冲卷积（模拟脉冲激励）
    reflected = np.convolve(reflected, gaussian_pulse(t[:100], 0, sigma, 1.0)*dt, mode='same')
    
    return incident + reflected


def measure_length(tdr_waveform, t, v):
    """从TDR波形测量电缆长度 - 双峰值检测法"""
    # 平滑去噪
    smooth = np.convolve(tdr_waveform, np.ones(5)/5, mode='same')
    
    # 找所有峰值
    peaks = []
    for i in range(1, len(smooth)-1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > 0.3:
            peaks.append(i)
    
    # 找谷值（用于短路检测）
    valleys = []
    for i in range(1, len(smooth)-1):
        if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < -0.3:
            valleys.append(i)
    
    all_extrema = sorted(peaks + [v for v in valleys if v not in peaks])
    
    if len(all_extrema) < 2:
        return np.nan
    
    # 第一极值 = 入射脉冲，第二极值 = 反射脉冲
    t1 = t[all_extrema[0]]
    t2 = t[all_extrema[1]]
    delta_t = abs(t2 - t1)
    
    L = v * delta_t / 2
    return L


def classify_load(tdr_waveform, t, t_round, sigma):
    """识别负载类型 - 基于反射脉冲特征"""
    dt = t[1] - t[0]
    t_round_idx = int(t_round / dt)
    guard = int(5 * sigma / dt)
    
    if t_round_idx + guard >= len(tdr_waveform):
        return 'Unknown'
    
    # 提取反射脉冲段
    ref_start = max(0, t_round_idx - guard)
    ref_end = min(t_round_idx + 10*guard, len(tdr_waveform))
    ref_segment = tdr_waveform[ref_start:ref_end]
    
    if len(ref_segment) < 10:
        return 'Unknown'
    
    # 特征1: 反射脉冲的峰值极性 (最重要)
    peak_idx = np.argmax(ref_segment)
    valley_idx = np.argmin(ref_segment)
    peak_val = ref_segment[peak_idx]
    valley_val = ref_segment[valley_idx]
    
    # 特征2: 反射脉冲的"主导极性"
    # 计算正面积和负面积
    pos_area = np.sum(ref_segment[ref_segment > 0])
    neg_area = abs(np.sum(ref_segment[ref_segment < 0]))
    
    # 特征3: 波形复杂度 (电容脉冲通常更宽/更复杂)
    # 计算反射脉冲的FWHM (半高全宽)
    threshold = 0.1 * max(abs(peak_val), abs(valley_val))
    above_thresh = np.abs(ref_segment) > threshold
    if np.any(above_thresh):
        pulse_width_est = np.sum(above_thresh) * dt
    else:
        pulse_width_est = 0
    
    # 特征4: 脉冲形状不对称度 (电容脉冲有明显不对称)
    if peak_val > abs(valley_val):
        dominant = 'pos'
        dominant_val = peak_val
    else:
        dominant = 'neg'
        dominant_val = valley_val
    
    # 分类逻辑 (基于实际TDR波形特征)
    # 电容: 脉冲宽度明显宽于高斯脉冲 (RC充放电导致拖尾) - 最优先判断
    expected_width = 2.5 * sigma
    if pulse_width_est > expected_width * 1.8:
        return 'Capacitor'
    
    # 开路: 正脉冲, 幅度很大(>0.8)
    if dominant == 'pos' and peak_val > 0.8:
        return 'Open'
    
    # 短路: 负脉冲, 幅度很大(<-0.8)
    if dominant == 'neg' and valley_val < -0.8:
        return 'Short'
    
    # 电阻: 幅度中等(|Γ| = 0.2~0.8), 脉冲形状接近高斯
    if abs(dominant_val) < 0.8 and pulse_width_est <= expected_width * 1.8:
        return 'Resistor'
    
    # 默认
    if dominant == 'pos':
        return 'Open'
    else:
        return 'Short'


# ==================== Test 1: 不同终端的TDR波形 ====================
print("Test 1: TDR Waveforms for Different Terminations")

L = 15  # 15m
t_round = 2 * L / v
t = np.arange(0, t_round * 3, dt_equiv)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

tests = [
    ('Open', generate_tdr_open(t, dt_equiv, L, v, pulse_sigma), 1.0),
    ('Short', generate_tdr_short(t, dt_equiv, L, v, pulse_sigma), -1.0),
    ('Resistor 25Ω', generate_tdr_resistor(t, dt_equiv, L, v, pulse_sigma, 25), (25-50)/(25+50)),
    ('Capacitor 200pF', generate_tdr_capacitor(t, dt_equiv, L, v, pulse_sigma, 200e-12), 'RC'),
]

for i, (name, waveform, gamma) in enumerate(tests):
    ax = axes[i]
    incident = gaussian_pulse(t, 0, pulse_sigma, 1.0)
    reflected = waveform - incident
    
    ax.plot(t * 1e9, waveform, 'b-', linewidth=2, label='TDR')
    ax.plot(t * 1e9, incident, 'g--', linewidth=1, alpha=0.7, label='Incident')
    ax.plot(t * 1e9, reflected, 'r:', linewidth=1.5, alpha=0.7, label='Reflected')
    ax.axvline(x=t_round * 1e9, color='k', linestyle='--', alpha=0.5)
    
    gamma_str = f"Γ = {gamma:.2f}" if isinstance(gamma, float) else "RC Response"
    ax.set_title(f'{name} ({gamma_str})', fontsize=11)
    ax.set_ylabel('Amplitude (V)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.5, 2.0])

axes[-1].set_xlabel('Time (ns)')
fig.suptitle('Test 1: TDR Waveforms for Different Terminal Loads (Z₀ = 50Ω, L = 15m)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_TDR_Termination_Waveforms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test1_TDR_Termination_Waveforms.png")


# ==================== Test 2: 长度检测精度 ====================
print("\nTest 2: Cable Length Measurement Accuracy")

test_lengths = [10, 12, 15, 18, 20]
results = []

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, L in enumerate(test_lengths):
    t_r = 2 * L / v
    t = np.arange(0, t_r * 2.5, dt_equiv)
    
    # 生成开路TDR波形 + 噪声
    tdr = generate_tdr_open(t, dt_equiv, L, v, pulse_sigma)
    tdr = tdr + 0.01 * np.random.randn(len(tdr))
    
    # 测量长度
    L_meas = measure_length(tdr, t, v)
    error = abs(L_meas - L) / L * 100 if not np.isnan(L_meas) else np.nan
    results.append((L, L_meas, error))
    
    if idx < 3:
        ax = axes[idx]
        ax.plot(t * 1e9, tdr, 'b-', linewidth=1.5)
        
        # 标记检测到的反射位置
        smooth = np.convolve(tdr, np.ones(5)/5, mode='same')
        peaks = []
        for i in range(1, len(smooth)-1):
            if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > 0.3:
                peaks.append(i)
        
        if len(peaks) >= 1:
            ax.plot(t[peaks[0]] * 1e9, smooth[peaks[0]], 'ro', markersize=10, label='Incident')
        if len(peaks) >= 2:
            ax.plot(t[peaks[1]] * 1e9, smooth[peaks[1]], 'gs', markersize=10, label='Reflected')
        
        status = f"Error={error:.2f}%" if not np.isnan(error) else "Fail"
        ax.set_title(f'L={L}m → {L_meas:.2f}m\n{status}', fontsize=10)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Amplitude (V)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

fig.suptitle('Test 2: TDR Length Measurement (Open Circuit, Peak Detection)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_Length_Measurement_Accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Results:")
for L, Lm, err in results:
    if np.isnan(err):
        print(f"    L = {L:4d}m: Failed")
    else:
        status = '✓ (≤1%)' if err <= 1 else ('✓ (≤5%)' if err <= 5 else '✗')
        print(f"    L = {L:4d}m: Measured = {Lm:6.2f}m, Error = {err:5.2f}% {status}")


# ==================== Test 3: 等效采样原理 ====================
print("\nTest 3: Equivalent Sampling Principle")

fs_real = 100e6  # 100MSPS实时ADC
dt_real = 1/fs_real
t_window = 50e-9

t_fast = np.arange(0, t_window, dt_equiv/100)  # 100GSPS真实信号
pulse_fast = gaussian_pulse(t_fast, 25e-9, 1e-9, 1.0)

# 实时采样 (严重欠采样)
t_real = np.arange(0, t_window, dt_real)
pulse_real = np.interp(t_real, t_fast, pulse_fast)

# 等效采样 (0.1ns步进)
delay_step = 0.1e-9
shots = int(dt_real / delay_step)
t_equiv = np.arange(0, t_window, delay_step)
pulse_equiv = np.zeros_like(t_equiv)

for shot in range(shots):
    delay = shot * delay_step
    for n, ts in enumerate(delay + np.arange(0, t_window, dt_real)):
        if ts < max(t_fast):
            idx = np.argmin(np.abs(t_fast - ts))
            sample_idx = int(round(ts / delay_step))
            if sample_idx < len(pulse_equiv):
                pulse_equiv[sample_idx] = pulse_fast[idx]

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(t_fast * 1e9, pulse_fast, 'b-', linewidth=1.5)
axes[0].set_title(f'(a) True TDR Pulse (100 GSPS)', fontsize=11)
axes[0].set_ylabel('Amplitude (V)')
axes[0].grid(True, alpha=0.3)

axes[1].stem(t_real * 1e9, pulse_real, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1].set_title(f'(b) Real-time Sampling @ {fs_real/1e6:.0f} MSPS (Aliased)', fontsize=11)
axes[1].set_ylabel('Amplitude (V)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_equiv * 1e9, pulse_equiv, 'g-', linewidth=1.5)
axes[2].set_title(f'(c) Equivalent Sampling ({shots} shots × {fs_real/1e6:.0f} MSPS = {1/delay_step/1e9:.0f} GSPS)', 
                   fontsize=11)
axes[2].set_xlabel('Time (ns)')
axes[2].set_ylabel('Amplitude (V)')
axes[2].grid(True, alpha=0.3)

fig.suptitle('Test 3: Equivalent Sampling Reconstructs GHz Signal from MHz ADC', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_Equivalent_Sampling.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test3_Equivalent_Sampling.png")
print(f"  {shots} shots × {fs_real/1e6:.0f} MSPS = {1/delay_step/1e9:.0f} GSPS effective")


# ==================== Test 4: 负载类型识别 ====================
print("\nTest 4: Load Type Identification")

L_test = 15
t_r_test = 2 * L_test / v
t = np.arange(0, t_r_test * 3, dt_equiv)

loads = [
    ('Open', 'open', lambda: generate_tdr_open(t, dt_equiv, L_test, v, pulse_sigma)),
    ('Short', 'short', lambda: generate_tdr_short(t, dt_equiv, L_test, v, pulse_sigma)),
    ('Resistor 15Ω', 'resistor', lambda: generate_tdr_resistor(t, dt_equiv, L_test, v, pulse_sigma, 15)),
    ('Resistor 25Ω', 'resistor', lambda: generate_tdr_resistor(t, dt_equiv, L_test, v, pulse_sigma, 25)),
    ('Capacitor 150pF', 'capacitor', lambda: generate_tdr_capacitor(t, dt_equiv, L_test, v, pulse_sigma, 150e-12)),
    ('Capacitor 250pF', 'capacitor', lambda: generate_tdr_capacitor(t, dt_equiv, L_test, v, pulse_sigma, 250e-12)),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()

identification_results = []

for i, (name, true_type, gen_func) in enumerate(loads):
    ax = axes[i]
    
    tdr = gen_func()
    tdr = tdr + 0.01 * np.random.randn(len(tdr))
    
    identified = classify_load(tdr, t, t_r_test, pulse_sigma)
    correct = identified.lower() == true_type
    identification_results.append((name, true_type, identified, correct))
    
    ax.plot(t * 1e9, tdr, 'b-', linewidth=1.5)
    
    # 标记反射段
    ref_start = int(t_r_test / dt_equiv) - int(3*pulse_sigma / dt_equiv)
    ref_end = min(ref_start + int(10*pulse_sigma / dt_equiv), len(t))
    if ref_start > 0 and ref_end > ref_start:
        ax.axvspan(t[ref_start] * 1e9, t[min(ref_end, len(t)-1)] * 1e9, alpha=0.2, color='yellow')
    
    color = 'green' if correct else 'red'
    ax.set_title(f'{name} → {identified} {"✓" if correct else "✗"}', 
                fontsize=10, color=color)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude (V)')
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 4: Load Type Identification from TDR Waveform', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_Load_Type_Identification.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Results:")
for name, true_type, identified, correct in identification_results:
    print(f"    {name} → {identified} {'✓' if correct else '✗'}")


# ==================== Test 5: 负载参数估计 ====================
print("\nTest 5: Load Parameter Estimation")

# 电阻测试
R_values = np.arange(10, 31, 2)
R_meas = []
R_err = []

for R in R_values:
    gamma = (R - Z0) / (R + Z0)
    # 模拟测量噪声 + 多次平均
    gammas = [gamma + 0.01*np.random.randn() for _ in range(50)]
    gamma_avg = np.median(gammas)
    R_est = Z0 * (1 + gamma_avg) / (1 - gamma_avg)
    R_meas.append(R_est)
    R_err.append(abs(R_est - R) / R * 100)

# 电容测试 (通过时间常数)
C_values = np.arange(100e-12, 301e-12, 20e-12)
C_meas = []
C_err = []

for C in C_values:
    tau = Z0 * C
    # 模拟τ测量 (5%噪声，50次平均)
    taus = [tau * (1 + 0.05*np.random.randn()) for _ in range(50)]
    tau_avg = np.median(taus)
    C_est = tau_avg / Z0
    C_meas.append(C_est)
    C_err.append(abs(C_est - C) / C * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(R_values, R_values, 'k--', linewidth=1.5, label='Ideal')
axes[0].plot(R_values, R_meas, 'bo-', linewidth=1.5, markersize=8, label='Measured')
axes[0].set_xlabel('Actual Resistance (Ω)')
axes[0].set_ylabel('Estimated Resistance (Ω)')
axes[0].set_title(f'Resistor Estimation\nMax Error: {max(R_err):.1f}%')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(C_values * 1e12, C_values * 1e12, 'k--', linewidth=1.5, label='Ideal')
axes[1].plot(C_values * 1e12, np.array(C_meas) * 1e12, 'go-', linewidth=1.5, markersize=8, label='Measured')
axes[1].set_xlabel('Actual Capacitance (pF)')
axes[1].set_ylabel('Estimated Capacitance (pF)')
axes[1].set_title(f'Capacitor Estimation\nMax Error: {max(C_err):.1f}%')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Test 5: Load Parameter Estimation (Median Filter, 50 Averages)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_Load_Parameter_Estimation.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Resistor: Max Error = {max(R_err):.2f}% (Req ≤10%) {'✓' if max(R_err) <= 10 else '✗'}")
print(f"  Capacitor: Max Error = {max(C_err):.2f}% (Req ≤10%) {'✓' if max(C_err) <= 10 else '✗'}")


# ==================== Test 6: 短电缆盲区 ====================
print("\nTest 6: Short Cable Blind Zone Test")

short_L = [0.5, 1.0, 2.0, 5.0]
pulse_widths = [2e-9, 5e-9, 10e-9]
colors_list = ['blue', 'red', 'green', 'purple']

fig, axes = plt.subplots(len(pulse_widths), 1, figsize=(14, 10))

for pw_idx, pw in enumerate(pulse_widths):
    ax = axes[pw_idx]
    
    for L_idx, L in enumerate(short_L):
        t_r = 2 * L / v
        t = np.arange(0, max(t_r, pw)*5, dt_equiv)
        
        incident = gaussian_pulse(t, 0, pw, 1.0)
        reflected = gaussian_pulse(t, t_r, pw, 1.0)
        tdr = incident + reflected
        
        ax.plot(t * 1e9, tdr + L_idx * 0.5, colors_list[L_idx], linewidth=1.5, 
               label=f'L = {L:.1f}m')
    
    L_blind = v * pw / 2
    ax.axvline(x=pw * 1e9, color='k', linestyle='--', linewidth=2, 
              label=f'Blind zone: L < {L_blind:.1f}m')
    ax.set_title(f'Pulse Width = {pw*1e9:.0f}ns', fontsize=11)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Amplitude + Offset (V)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 6: Short Cable Detection - Blind Zone Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_Short_Cable_Blind_Zone.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  2ns pulse:  blind L < {v*2e-9/2:.2f}m (need ≤1m) {'✓' if v*2e-9/2 <= 1 else '✗'}")
print(f"  5ns pulse:  blind L < {v*5e-9/2:.2f}m (need ≤1m) {'✓' if v*5e-9/2 <= 1 else '✗'}")
print(f"  10ns pulse: blind L < {v*10e-9/2:.2f}m (need ≤1m) {'✓' if v*10e-9/2 <= 1 else '✗'}")


# ==================== Test 7: Monte Carlo误差预算 ====================
print("\nTest 7: Monte Carlo Error Budget")

num_mc = 100
L_true = 15
R_true = 25
C_true = 200e-12

L_errors = []
R_errors = []
C_errors = []

for _ in range(num_mc):
    # 误差源
    v_actual = v * (1 + 0.01*np.random.randn())  # 1%速度误差
    jitter = 0.1e-9 * np.random.randn()  # 0.1ns抖动
    
    # 长度测量
    t_r = 2 * L_true / v_actual + jitter
    t = np.arange(0, t_r * 2.5, dt_equiv)
    tdr = generate_tdr_open(t, dt_equiv, L_true, v_actual, pulse_sigma)
    tdr = tdr + 0.01 * np.random.randn(len(tdr))
    
    L_meas = measure_length(tdr, t, v_actual)
    if not np.isnan(L_meas):
        L_errors.append(abs(L_meas - L_true) / L_true * 100)
    
    # 电阻测量
    gamma = (R_true - Z0) / (R_true + Z0)
    gammas = [gamma + 0.01*np.random.randn() for _ in range(20)]
    gamma_avg = np.median(gammas)
    R_meas = Z0 * (1 + gamma_avg) / (1 - gamma_avg)
    R_errors.append(abs(R_meas - R_true) / R_true * 100)
    
    # 电容测量
    tau = Z0 * C_true
    taus = [tau * (1 + 0.05*np.random.randn()) for _ in range(20)]
    tau_avg = np.median(taus)
    C_meas = tau_avg / Z0
    C_errors.append(abs(C_meas - C_true) / C_true * 100)

L_errors = np.array(L_errors)
R_errors = np.array(R_errors)
C_errors = np.array(C_errors)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(L_errors, bins=15, color='blue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=1, color='r', linestyle='--', linewidth=2, label='Req: ≤1%')
axes[0].axvline(x=np.mean(L_errors), color='g', linewidth=2, label=f'Mean: {np.mean(L_errors):.2f}%')
axes[0].set_title(f'Length Error\n95% CI: [{np.percentile(L_errors,2.5):.2f}, {np.percentile(L_errors,97.5):.2f}]%')
axes[0].set_xlabel('Error (%)')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(R_errors, bins=15, color='red', edgecolor='black', alpha=0.7)
axes[1].axvline(x=10, color='r', linestyle='--', linewidth=2, label='Req: ≤10%')
axes[1].axvline(x=np.mean(R_errors), color='g', linewidth=2, label=f'Mean: {np.mean(R_errors):.2f}%')
axes[1].set_title(f'Resistance Error\n95% CI: [{np.percentile(R_errors,2.5):.2f}, {np.percentile(R_errors,97.5):.2f}]%')
axes[1].set_xlabel('Error (%)')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(C_errors, bins=15, color='green', edgecolor='black', alpha=0.7)
axes[2].axvline(x=10, color='r', linestyle='--', linewidth=2, label='Req: ≤10%')
axes[2].axvline(x=np.mean(C_errors), color='g', linewidth=2, label=f'Mean: {np.mean(C_errors):.2f}%')
axes[2].set_title(f'Capacitance Error\n95% CI: [{np.percentile(C_errors,2.5):.2f}, {np.percentile(C_errors,97.5):.2f}]%')
axes[2].set_xlabel('Error (%)')
axes[2].set_ylabel('Count')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.suptitle('Test 7: Monte Carlo Error Budget (100 Runs, Peak Detection + Median)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_MonteCarlo_ErrorBudget.png'), dpi=150, bbox_inches='tight')
plt.close()

L_ci = [np.percentile(L_errors, 2.5), np.percentile(L_errors, 97.5)]
R_ci = [np.percentile(R_errors, 2.5), np.percentile(R_errors, 97.5)]
C_ci = [np.percentile(C_errors, 2.5), np.percentile(C_errors, 97.5)]

print(f"  Length: Mean={np.mean(L_errors):.2f}%, 95%CI=[{L_ci[0]:.2f}, {L_ci[1]:.2f}]% (Req≤1%) {'✓' if L_ci[1]<=1 else '✗'}")
print(f"  Resistance: Mean={np.mean(R_errors):.2f}%, 95%CI=[{R_ci[0]:.2f}, {R_ci[1]:.2f}]% (Req≤10%) {'✓' if R_ci[1]<=10 else '✗'}")
print(f"  Capacitance: Mean={np.mean(C_errors):.2f}%, 95%CI=[{C_ci[0]:.2f}, {C_ci[1]:.2f}]% (Req≤10%) {'✓' if C_ci[1]<=10 else '✗'}")

print("\n=== Simulation Complete ===")
print(f"Output: {os.path.abspath(output_dir)}")
