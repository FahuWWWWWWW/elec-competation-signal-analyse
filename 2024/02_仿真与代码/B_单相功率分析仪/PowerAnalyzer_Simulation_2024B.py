"""
2024年电赛B题「单相功率分析仪」仿真
目标: 验证交流电压/电流/功率/THD测量算法的精度
技术: ADC采样 + RMS计算 + FFT谐波分析
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 输出目录
output_dir = 'simulation_output'
os.makedirs(output_dir, exist_ok=True)

print("=== 2024-B Single-Phase Power Analyzer Simulation ===")
print("Technique: ADC Sampling + RMS + FFT Harmonic Analysis")
print()

# 电网参数
f_base = 50  # 基波频率 50Hz
T_base = 1 / f_base  # 周期 20ms

# ADC参数
fs = 2560  # 采样率 2.56kHz (50Hz的51.2倍，便于FFT)
N = 2560   # 采样2560点 = 1秒 (50个周期)
t = np.arange(N) / fs


def generate_voltage(t, U_rms=220, harmonic_spec=None):
    """生成电压波形，可添加谐波"""
    u = np.sqrt(2) * U_rms * np.sin(2 * np.pi * f_base * t)
    if harmonic_spec:
        for n, amp_percent in harmonic_spec.items():
            u += np.sqrt(2) * U_rms * (amp_percent/100) * np.sin(2 * np.pi * n * f_base * t)
    return u


def generate_current(t, I_rms=4, phi_deg=0, harmonic_spec=None, crest_factor=1.414):
    """生成电流波形"""
    i = crest_factor * I_rms * np.sin(2 * np.pi * f_base * t + np.radians(phi_deg))
    if harmonic_spec:
        for n, amp_percent in harmonic_spec.items():
            i += crest_factor * I_rms * (amp_percent/100) * np.sin(2 * np.pi * n * f_base * t + np.radians(phi_deg*n))
    return i


# 传感器参数
PT_RATIO = 100  # PT变比 100:1 (220V → 2.2V)
CT_RATIO = 1000  # CT变比 1000:1
BURDEN_R = 100  # burden电阻 100Ω

def adc_sample(u_raw, i_raw, fs_adc, bits=12, v_ref=3.3, noise_db=60):
    """
    模拟ADC采样过程 (含传感器缩放)
    
    输入: u_raw为实际电压(V), i_raw为实际电流(A)
    处理: 
      - 电压经PT缩放: u_sensor = u_raw / PT_RATIO
      - 电流经CT+burden: i_sensor = i_raw / CT_RATIO * BURDEN_R
      - 加偏置、量化、加噪声
    返回: 传感器输出的测量值 (需外部乘变比还原)
    """
    # 传感器转换
    u_sensor = u_raw / PT_RATIO  # PT副边电压
    i_sensor = i_raw / CT_RATIO * BURDEN_R  # burden电阻电压
    
    # 缩放至ADC量程 (假设传感器输出最大约3Vpp，对应220V/4A)
    # 对电压进一步分压到1.1Vrms (峰峰值约3.1V，加1.65V偏置后范围0.1~3.2V)
    u_sensor = u_sensor / 2  # 分压1:1
    
    # 添加ADC噪声
    snr_total = 6.02 * bits + 1.76 + noise_db
    noise_rms_u = np.std(u_sensor) / (10**(snr_total/20))
    noise_rms_i = np.std(i_sensor) / (10**(snr_total/20))
    
    u_noisy = u_sensor + noise_rms_u * np.random.randn(len(u_sensor))
    i_noisy = i_sensor + noise_rms_i * np.random.randn(len(i_sensor))
    
    # 偏置到1.65V (3.3V/2)
    u_adc = u_noisy + v_ref / 2
    i_adc = i_noisy + v_ref / 2
    
    # 量化
    u_adc = np.clip(u_adc, 0, v_ref)
    i_adc = np.clip(i_adc, 0, v_ref)
    
    u_codes = np.round(u_adc / v_ref * (2**bits - 1))
    i_codes = np.round(i_adc / v_ref * (2**bits - 1))
    
    # 转回电压值 (传感器输出)
    u_measured = u_codes / (2**bits - 1) * v_ref - v_ref/2
    i_measured = i_codes / (2**bits - 1) * v_ref - v_ref/2
    
    return u_measured, i_measured


def calculate_rms(samples):
    """计算有效值"""
    return np.sqrt(np.mean(samples**2))


def calculate_power(u_samples, i_samples):
    """计算有功功率"""
    return np.mean(u_samples * i_samples)


def harmonic_analysis(samples, fs, f_base=50, max_harmonic=10):
    """
    FFT谐波分析
    返回: 基波幅度, 各次谐波幅度, THD
    """
    N = len(samples)
    
    # 去直流
    ac = samples - np.mean(samples)
    
    # 加窗减少泄漏
    window = np.hanning(N)
    ac_windowed = ac * window
    
    # FFT
    fft_result = np.fft.fft(ac_windowed)
    magnitude = 2/N * np.abs(fft_result[:N//2])
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    
    # 提取谐波 (考虑窗函数补偿)
    harmonics = []
    for n in range(1, max_harmonic + 1):
        target_freq = n * f_base
        # 在目标频率附近搜索峰值
        mask = (freqs >= target_freq - 2) & (freqs <= target_freq + 2)
        if np.any(mask):
            idx = np.argmax(magnitude[mask])
            harmonics.append(np.max(magnitude[mask]))
        else:
            harmonics.append(0)
    
    harmonics = np.array(harmonics)
    I1 = harmonics[0]
    I2_10 = harmonics[1:]
    
    # THD
    thd = np.sqrt(np.sum(I2_10**2)) / I1 * 100 if I1 > 0 else 0
    
    return I1, I2_10, thd, freqs, magnitude


# ==================== Test 1: 不同负载类型的电压电流波形 ====================
print("Test 1: Voltage and Current Waveforms for Different Load Types")

# 定义不同负载
loads = [
    ('Resistive (PF=1.0)', generate_voltage(t), generate_current(t, phi_deg=0)),
    ('Inductive (PF=0.7)', generate_voltage(t), generate_current(t, phi_deg=45)),
    ('Rectifier (THD~80%)', generate_voltage(t), 
     generate_current(t, harmonic_spec={3: 60, 5: 30, 7: 15, 9: 10})),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for idx, (name, u, i) in enumerate(loads):
    ax_u = axes[idx, 0]
    ax_i = axes[idx, 1]
    
    # 显示前4个周期
    N_show = int(4 * T_base * fs)
    t_show = t[:N_show] * 1e3  # ms
    
    ax_u.plot(t_show, u[:N_show], 'b-', linewidth=1.5)
    ax_u.set_title(f'{name} - Voltage')
    ax_u.set_xlabel('Time (ms)')
    ax_u.set_ylabel('Voltage (V)')
    ax_u.grid(True, alpha=0.3)
    
    ax_i.plot(t_show, i[:N_show], 'r-', linewidth=1.5)
    ax_i.set_title(f'{name} - Current')
    ax_i.set_xlabel('Time (ms)')
    ax_i.set_ylabel('Current (A)')
    ax_i.grid(True, alpha=0.3)

fig.suptitle('Test 1: Voltage/Current Waveforms for Different Load Types', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_Load_Waveforms.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test1_Load_Waveforms.png")


# ==================== Test 2: RMS测量精度验证 ====================
print("\nTest 2: RMS Measurement Accuracy")

# 不同有效值的电压/电流
test_U = [110, 220, 230]  # Vrms
test_I = [0.5, 2, 4]      # Arms
test_phi = [0, 30, 60]    # 相位角

results_rms = []

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 电压RMS误差
U_true_list = []
U_meas_list = []
U_err_list = []

for U_true in test_U:
    u = generate_voltage(t, U_rms=U_true)
    u_adc, _ = adc_sample(u, u, fs)  # 模拟ADC
    U_meas = calculate_rms(u_adc) * PT_RATIO * 2
    err = abs(U_meas - U_true) / U_true * 100
    U_true_list.append(U_true)
    U_meas_list.append(U_meas)
    U_err_list.append(err)

axes[0, 0].plot(U_true_list, U_true_list, 'k--', linewidth=2, label='Ideal')
axes[0, 0].plot(U_true_list, U_meas_list, 'bo-', linewidth=2, markersize=8, label='Measured')
axes[0, 0].set_xlabel('True Voltage RMS (V)')
axes[0, 0].set_ylabel('Measured Voltage RMS (V)')
axes[0, 0].set_title('(a) Voltage RMS Measurement')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar([str(u) for u in U_true_list], U_err_list, color='steelblue', edgecolor='black')
axes[0, 1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Requirement: 1%')
axes[0, 1].set_xlabel('True Voltage RMS (V)')
axes[0, 1].set_ylabel('Error (%)')
axes[0, 1].set_title('(b) Voltage RMS Error')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 电流RMS误差
I_true_list = []
I_meas_list = []
I_err_list = []

for I_true in test_I:
    i = generate_current(t, I_rms=I_true)
    _, i_adc = adc_sample(i, i, fs)
    I_meas = calculate_rms(i_adc) / BURDEN_R * CT_RATIO
    err = abs(I_meas - I_true) / I_true * 100
    I_true_list.append(I_true)
    I_meas_list.append(I_meas)
    I_err_list.append(err)

axes[1, 0].plot(I_true_list, I_true_list, 'k--', linewidth=2, label='Ideal')
axes[1, 0].plot(I_true_list, I_meas_list, 'ro-', linewidth=2, markersize=8, label='Measured')
axes[1, 0].set_xlabel('True Current RMS (A)')
axes[1, 0].set_ylabel('Measured Current RMS (A)')
axes[1, 0].set_title('(c) Current RMS Measurement')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar([str(i) for i in I_true_list], I_err_list, color='coral', edgecolor='black')
axes[1, 1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Requirement: 1%')
axes[1, 1].set_xlabel('True Current RMS (A)')
axes[1, 1].set_ylabel('Error (%)')
axes[1, 1].set_title('(d) Current RMS Error')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 2: RMS Measurement Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_RMS_Accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Voltage RMS: max error = {max(U_err_list):.3f}% (Req ≤1%) {'✓' if max(U_err_list)<=1 else '✗'}")
print(f"  Current RMS: max error = {max(I_err_list):.3f}% (Req ≤1%) {'✓' if max(I_err_list)<=1 else '✗'}")


# ==================== Test 3: 功率与功率因数测量 ====================
print("\nTest 3: Power and Power Factor Measurement")

test_cases_power = [
    (220, 4, 0, 'Resistive'),
    (220, 4, 30, 'Inductive'),
    (220, 4, 60, 'Highly Inductive'),
    (220, 2, 45, 'Medium Load'),
]

P_true_list = []
P_meas_list = []
PF_true_list = []
PF_meas_list = []
P_err_list = []
PF_err_list = []

for U_t, I_t, phi, name in test_cases_power:
    u = generate_voltage(t, U_rms=U_t)
    i = generate_current(t, I_rms=I_t, phi_deg=phi)
    
    # 真实值
    P_true = U_t * I_t * np.cos(np.radians(phi))
    PF_true = np.cos(np.radians(phi))
    
    # ADC采样+测量
    u_adc, i_adc = adc_sample(u, i, fs)
    P_meas = calculate_power(u_adc, i_adc) * (PT_RATIO * 2) * (CT_RATIO / BURDEN_R)
    U_meas = calculate_rms(u_adc) * PT_RATIO * 2
    I_meas = calculate_rms(i_adc) / BURDEN_R * CT_RATIO
    PF_meas = P_meas / (U_meas * I_meas) if (U_meas * I_meas) > 0 else 0
    
    P_err = abs(P_meas - P_true) / P_true * 100
    PF_err = abs(PF_meas - PF_true) / PF_true * 100 if PF_true > 0 else 0
    
    P_true_list.append(P_true)
    P_meas_list.append(P_meas)
    PF_true_list.append(PF_true)
    PF_meas_list.append(PF_meas)
    P_err_list.append(P_err)
    PF_err_list.append(PF_err)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 有功功率
axes[0, 0].plot(P_true_list, P_true_list, 'k--', linewidth=2, label='Ideal')
axes[0, 0].plot(P_true_list, P_meas_list, 'bo-', linewidth=2, markersize=8, label='Measured')
axes[0, 0].set_xlabel('True Power (W)')
axes[0, 0].set_ylabel('Measured Power (W)')
axes[0, 0].set_title('(a) Active Power Measurement')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(range(len(test_cases_power)), P_err_list, color='steelblue', edgecolor='black')
axes[0, 1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Requirement: 1%')
axes[0, 1].set_xticks(range(len(test_cases_power)))
axes[0, 1].set_xticklabels([c[3] for c in test_cases_power], rotation=15, ha='right')
axes[0, 1].set_ylabel('Power Error (%)')
axes[0, 1].set_title('(b) Active Power Error')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 功率因数
axes[1, 0].plot(PF_true_list, PF_true_list, 'k--', linewidth=2, label='Ideal')
axes[1, 0].plot(PF_true_list, PF_meas_list, 'ro-', linewidth=2, markersize=8, label='Measured')
axes[1, 0].set_xlabel('True PF')
axes[1, 0].set_ylabel('Measured PF')
axes[1, 0].set_title('(c) Power Factor Measurement')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar(range(len(test_cases_power)), PF_err_list, color='coral', edgecolor='black')
axes[1, 1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Requirement: 1%')
axes[1, 1].set_xticks(range(len(test_cases_power)))
axes[1, 1].set_xticklabels([c[3] for c in test_cases_power], rotation=15, ha='right')
axes[1, 1].set_ylabel('PF Error (%)')
axes[1, 1].set_title('(d) Power Factor Error')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 3: Power and Power Factor Measurement', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_Power_PF_Accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Power: max error = {max(P_err_list):.3f}% (Req ≤1%) {'✓' if max(P_err_list)<=1 else '✗'}")
print(f"  PF: max error = {max(PF_err_list):.3f}% (Req ≤1%) {'✓' if max(PF_err_list)<=1 else '✗'}")


# ==================== Test 4: 谐波分析THD ====================
print("\nTest 4: Harmonic Analysis and THD")

# 不同THD的负载
thd_cases = [
    ('Pure Sine (THD=0%)', {}),
    ('Rectifier (THD~55%)', {3: 55, 5: 25, 7: 12}),
    ('LED Driver (THD~80%)', {3: 70, 5: 35, 7: 20, 9: 10}),
    ('Heater (THD~30%)', {3: 25, 5: 10, 7: 5}),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (name, harm_spec) in enumerate(thd_cases):
    u = generate_voltage(t)
    i = generate_current(t, harmonic_spec=harm_spec)
    
    # 模拟ADC
    _, i_adc = adc_sample(i, i, fs)
    
    # 谐波分析
    I1, I_harm, thd_meas, freqs, mag = harmonic_analysis(i_adc, fs, max_harmonic=10)
    
    # 计算真实THD
    if harm_spec:
        I_true_harm = np.array([harm_spec.get(n, 0) for n in range(2, 11)])
        thd_true = np.sqrt(np.sum(I_true_harm**2)) / 100 * 100  # 基波=100%
    else:
        thd_true = 0
    
    thd_err = abs(thd_meas - thd_true)
    
    # 绘图 (频谱)
    ax = axes[idx // 2, idx % 2]
    harmonics_x = range(1, 11)
    ax.bar(harmonics_x, [I1/np.sqrt(2)/BURDEN_R*CT_RATIO] + list(I_harm/np.sqrt(2)/BURDEN_R*CT_RATIO),
           color=['blue'] + ['red']*9, edgecolor='black')
    ax.set_xlabel('Harmonic Order')
    ax.set_ylabel('Current (Arms)')
    ax.set_title(f'{name}\nTrue THD={thd_true:.1f}%, Meas THD={thd_meas:.1f}%, Err={thd_err:.1f}%')
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Test 4: Harmonic Spectrum and THD for Different Loads', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_THD_Harmonics.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test4_THD_Harmonics.png")


# ==================== Test 5: FFT窗函数与频谱泄漏 ====================
print("\nTest 5: Window Function Effect on Spectral Leakage")

# 生成含谐波的电流信号
i_test = generate_current(t, harmonic_spec={3: 30, 5: 15, 7: 8})
_, i_adc = adc_sample(i_test, i_test, fs)

# 不同窗函数
windows = {
    'Rectangular': np.ones(N),
    'Hanning': np.hanning(N),
    'Blackman': np.blackman(N),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 时域窗函数
ax = axes[0, 0]
N_show = 512
for name, w in windows.items():
    ax.plot(t[:N_show]*1e3, w[:N_show], linewidth=2, label=name)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title('(a) Window Functions')
ax.legend()
ax.grid(True, alpha=0.3)

# 频谱对比 (Rectangular)
ax = axes[0, 1]
fft_rect = np.fft.fft(i_adc * windows['Rectangular'])
freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
mag_rect = 2/N * np.abs(fft_rect[:N//2])
ax.semilogy(freqs[:300], mag_rect[:300] + 1e-10, 'b-', linewidth=1.5, label='Rectangular')
fft_hann = np.fft.fft(i_adc * windows['Hanning'])
mag_hann = 2/N * np.abs(fft_hann[:N//2])
ax.semilogy(freqs[:300], mag_hann[:300] + 1e-10, 'r-', linewidth=1.5, label='Hanning')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (log)')
ax.set_title('(b) Spectrum: Rectangular vs Hanning')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# 基波幅度测量误差
ax = axes[1, 0]
harmonic_errors = {}
for name, w in windows.items():
    I1_w, _, thd_w, _, _ = harmonic_analysis(i_adc, fs, max_harmonic=10)
    harmonic_errors[name] = thd_w

ax.bar(harmonic_errors.keys(), harmonic_errors.values(), 
       color=['steelblue', 'coral', 'lightgreen'], edgecolor='black')
ax.set_ylabel('THD (%)')
ax.set_title('(c) THD Measurement with Different Windows')
ax.grid(True, alpha=0.3, axis='y')

# 频谱分辨率
ax = axes[1, 1]
# 比较不同FFT点数
Ns = [512, 1024, 2560]
for N_test in Ns:
    t_test = np.arange(N_test) / fs
    i_test = generate_current(t_test, harmonic_spec={3: 30})
    fft_test = np.fft.fft(i_test * np.hanning(N_test))
    freqs_test = np.fft.fftfreq(N_test, 1/fs)[:N_test//2]
    mag_test = 2/N_test * np.abs(fft_test[:N_test//2])
    ax.plot(freqs_test[:50], mag_test[:50], linewidth=1.5, label=f'N={N_test}')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('(d) Frequency Resolution vs FFT Points')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Test 5: FFT Window and Resolution Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_FFT_Window_Leakage.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test5_FFT_Window_Leakage.png")


# ==================== Test 6: CT多匝缠绕灵敏度分析 ====================
print("\nTest 6: CT Multi-Turn Winding Sensitivity")

turns = [1, 2, 5, 10]
I_primary = 4  # 4A原方电流
ct_ratio = 1000  # CT变比1000:1
burden_r = 100  # burden电阻100Ω

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 灵敏度 vs 匝数
sensitivities = []
for n in turns:
    I_equiv = I_primary * n  # 等效原方电流
    I_secondary = I_equiv / ct_ratio  # 副方电流
    V_burden = I_secondary * burden_r  # burden电压
    sensitivities.append(V_burden * 1e3)  # mV

axes[0, 0].bar([str(n) for n in turns], sensitivities, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Primary Turns (Np)')
axes[0, 0].set_ylabel('Burden Voltage (mV)')
axes[0, 0].set_title('(a) Sensitivity vs Primary Turns')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 不同匝数下的量程变化
ranges = []
for n in turns:
    I_max = 4  # 原方最大4A
    I_equiv_max = I_max / n  # 考虑匝数后的等效量程
    ranges.append(I_equiv_max)

axes[0, 1].bar([str(n) for n in turns], ranges, color='coral', edgecolor='black')
axes[0, 1].axhline(y=4, color='r', linestyle='--', linewidth=2, label='Required: 4A')
axes[0, 1].set_xlabel('Primary Turns (Np)')
axes[0, 1].set_ylabel('Effective Current Range (A)')
axes[0, 1].set_title('(b) Effective Range vs Primary Turns')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 不同匝数下的ADC分辨率
adc_resolutions = []
adc_bits = 12
v_ref = 3.3
for n in turns:
    I_equiv = 4 * n
    I_sec = I_equiv / ct_ratio
    V_burden = I_sec * burden_r
    lsb = v_ref / (2**adc_bits)
    current_res = lsb / burden_r * ct_ratio / n
    adc_resolutions.append(current_res * 1e3)  # mA

axes[1, 0].bar([str(n) for n in turns], adc_resolutions, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Primary Turns (Np)')
axes[1, 0].set_ylabel('Current Resolution (mA)')
axes[1, 0].set_title('(c) ADC Current Resolution vs Turns')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 实际测量误差模拟
axes[1, 1].text(0.5, 0.5, 'CT Multi-Turn Summary:\n\nNp=1: 400mV burden, 4A range\nNp=5: 2V burden, 0.8A range\n\nTrade-off:\nMore turns → Higher sensitivity\nBut → Smaller current range\n\nRecommended: Np=1 for 4A range\nUse burden R tuning for gain', 
                ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')
axes[1, 1].set_title('(d) Design Trade-offs')

fig.suptitle('Test 6: CT Multi-Turn Winding Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_CT_MultiTurn.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  Saved: Test6_CT_MultiTurn.png")


# ==================== Test 7: Monte Carlo综合误差分析 ====================
print("\nTest 7: Monte Carlo Error Budget")

num_mc = 100
U_errors = []
I_errors = []
P_errors = []
PF_errors = []
THD_errors = []

for _ in range(num_mc):
    # 随机参数
    U_true = np.random.uniform(200, 240)
    I_true = np.random.uniform(0.5, 4)
    phi = np.random.uniform(0, 60)
    
    # 随机谐波 (0~60% THD)
    thd_level = np.random.uniform(0, 60)
    harm_spec = {}
    if thd_level > 5:
        harm_spec[3] = thd_level * 0.8
        harm_spec[5] = thd_level * 0.3
    
    u = generate_voltage(t, U_rms=U_true)
    i = generate_current(t, I_rms=I_true, phi_deg=phi, harmonic_spec=harm_spec)
    
    # ADC采样
    u_adc, i_adc = adc_sample(u, i, fs)
    
    # 测量
    U_meas = calculate_rms(u_adc) * PT_RATIO * 2
    I_meas = calculate_rms(i_adc) / BURDEN_R * CT_RATIO
    P_meas = calculate_power(u_adc, i_adc) * (PT_RATIO * 2) * (CT_RATIO / BURDEN_R)
    PF_meas = P_meas / (U_meas * I_meas) if (U_meas * I_meas) > 0 else 0
    
    # 真实值
    I_true_total = calculate_rms(i)
    P_true = calculate_power(u, i)
    PF_true = P_true / (U_true * I_true_total) if (U_true * I_true_total) > 0 else 0
    
    # 谐波
    _, _, thd_meas, _, _ = harmonic_analysis(i_adc, fs)
    
    # 真实THD
    if harm_spec:
        I_h = np.array([harm_spec.get(n, 0) for n in range(2, 11)])
        thd_true = np.sqrt(np.sum(I_h**2))
    else:
        thd_true = 0
    
    U_errors.append(abs(U_meas - U_true) / U_true * 100)
    I_errors.append(abs(I_meas - I_true_total) / I_true_total * 100)
    P_errors.append(abs(P_meas - P_true) / P_true * 100)
    PF_errors.append(abs(PF_meas - PF_true) / PF_true * 100 if PF_true > 0.1 else 0)
    THD_errors.append(abs(thd_meas - thd_true))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

metrics = [
    (U_errors, 'Voltage', 'blue', 1),
    (I_errors, 'Current', 'red', 1),
    (P_errors, 'Power', 'green', 1),
    (PF_errors, 'Power Factor', 'purple', 1),
    (THD_errors, 'THD', 'orange', 2),
]

for idx, (errors, name, color, req) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    ax.hist(errors, bins=15, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(x=req, color='r', linestyle='--', linewidth=2, label=f'Req: {req}%')
    mean_err = np.mean(errors)
    p95 = np.percentile(errors, 95)
    ax.axvline(x=mean_err, color='g', linewidth=2, label=f'Mean: {mean_err:.2f}%')
    ax.set_title(f'{name} Error\n95%CI: {p95:.2f}%')
    ax.set_xlabel('Error (%)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 空位放总结
ax = axes[1, 2]
summary_text = f"""Monte Carlo Summary (100 runs):

Voltage Error:
  Mean={np.mean(U_errors):.2f}%, 95%={np.percentile(U_errors,95):.2f}%
  Requirement: ≤1% {'✓' if np.percentile(U_errors,95)<=1 else '✗'}

Current Error:
  Mean={np.mean(I_errors):.2f}%, 95%={np.percentile(I_errors,95):.2f}%
  Requirement: ≤1% {'✓' if np.percentile(I_errors,95)<=1 else '✗'}

Power Error:
  Mean={np.mean(P_errors):.2f}%, 95%={np.percentile(P_errors,95):.2f}%
  Requirement: ≤1% {'✓' if np.percentile(P_errors,95)<=1 else '✗'}

PF Error:
  Mean={np.mean(PF_errors):.2f}%, 95%={np.percentile(PF_errors,95):.2f}%
  Requirement: ≤1% {'✓' if np.percentile(PF_errors,95)<=1 else '✗'}

THD Error:
  Mean={np.mean(THD_errors):.2f}%, 95%={np.percentile(THD_errors,95):.2f}%
  Requirement: ≤2% {'✓' if np.percentile(THD_errors,95)<=2 else '✗'}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, transform=ax.transAxes, 
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Summary')

fig.suptitle('Test 7: Monte Carlo Error Budget (100 Runs)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_MonteCarlo_ErrorBudget.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"  Voltage: 95%CI={np.percentile(U_errors,95):.2f}% (Req≤1%) {'✓' if np.percentile(U_errors,95)<=1 else '✗'}")
print(f"  Current: 95%CI={np.percentile(I_errors,95):.2f}% (Req≤1%) {'✓' if np.percentile(I_errors,95)<=1 else '✗'}")
print(f"  Power:   95%CI={np.percentile(P_errors,95):.2f}% (Req≤1%) {'✓' if np.percentile(P_errors,95)<=1 else '✗'}")
print(f"  PF:      95%CI={np.percentile(PF_errors,95):.2f}% (Req≤1%) {'✓' if np.percentile(PF_errors,95)<=1 else '✗'}")
print(f"  THD:     95%CI={np.percentile(THD_errors,95):.2f}% (Req≤2%) {'✓' if np.percentile(THD_errors,95)<=2 else '✗'}")

print("\n=== Simulation Complete ===")
print(f"Output: {os.path.abspath(output_dir)}")
