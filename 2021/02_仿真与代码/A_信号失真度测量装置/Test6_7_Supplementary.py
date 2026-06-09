"""
2021-A题 信号失真度测量装置 - Test6 & Test7 补充可视化
生成缺失的两张仿真图
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_output')
os.makedirs(output_dir, exist_ok=True)

print("=== 2021-A THD Test 6 & 7 Supplementary Plots ===")

# ============ Test 6: 相干采样频率搜索算法 ============
print("\nTest 6: Coherent Sampling Frequency Search")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 参数
f_target = 10e3
N_coherent = 8192
fs_fixed = 200e3
df = fs_fixed / N_coherent
M_opt = round(f_target / df)
f_estimated = M_opt * df
error_freq = abs(f_estimated - f_target)

# 左图：固定fs下的频率误差
methods = ['Target\nFreq', 'Estimated\nFreq']
values = [f_target, f_estimated]
errors = [0, error_freq]

bars = axes[0].bar(methods, values, color=['blue', 'orange'], alpha=0.7)
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_title(f'(a) Fixed fs Method: fs={fs_fixed/1e3:.0f}kHz, N={N_coherent}')
axes[0].set_ylim([9900, 10100])
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 5, f'{val:.3f}Hz', 
                 ha='center', fontsize=10, fontweight='bold')
axes[0].text(0.5, 0.95, f'Frequency Resolution: {df:.3f}Hz\nError: {error_freq:.3f}Hz ({error_freq/f_target*100:.4f}%)',
             transform=axes[0].transAxes, ha='center', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].grid(True, alpha=0.3)

# 右图：自适应fs搜索 - 不同周期数K对应的最佳fs和THD误差
M_search = 523
K_cycles = np.arange(1, 11)
fs_required = M_search * f_target / K_cycles
valid_mask = (fs_required >= 200e3) & (fs_required <= 10e6)

axes[1].plot(K_cycles[valid_mask], fs_required[valid_mask]/1e3, 'b-o', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Cycles (K)')
axes[1].set_ylabel('Required fs (kHz)', color='blue')
axes[1].tick_params(axis='y', labelcolor='blue')
axes[1].set_title('(b) Adaptive fs Search for Coherent Sampling')
axes[1].grid(True, alpha=0.3)

# 标注有效范围
axes[1].axhspan(200, 10000, alpha=0.1, color='green', label='Valid range')
axes[1].legend()

fig.suptitle('Test 6: Coherent Sampling Frequency Search Algorithm', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_Coherent_Frequency_Search.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test6_Coherent_Frequency_Search.png")


# ============ Test 7: 窗函数特性对比 ============
print("\nTest 7: Window Function Characteristics")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

windows = {
    'Rectangular': signal.windows.boxcar(1024),
    'Hanning': signal.windows.hann(1024),
    'Hamming': signal.windows.hamming(1024),
    'Flat-top': signal.windows.flattop(1024),
    'Blackman-Harris': signal.windows.blackmanharris(1024)
}

colors = ['blue', 'green', 'red', 'purple', 'orange']

# 第一行：时域波形
for i, (name, win) in enumerate(windows.items()):
    ax = axes[0, i] if i < 3 else None
    if i < 3:
        ax.plot(win, color=colors[i], linewidth=1.5)
        ax.set_title(f'({chr(97+i)}) {name} (Time Domain)')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_ylim([-0.1, 1.1])
        ax.grid(True, alpha=0.3)

# 第二行：频域特性（前3个窗）
for i, (name, win) in enumerate(list(windows.items())[:3]):
    ax = axes[1, i]
    # FFT of window
    W = np.abs(np.fft.fft(win, 65536))
    W = W / np.max(W)
    W_db = 20 * np.log10(W + 1e-10)
    freqs = np.fft.fftfreq(65536, 1)[:32768]
    
    ax.plot(freqs[:2000] * 1024, W_db[:2000], color=colors[i], linewidth=1.5)
    ax.set_title(f'({chr(100+i)}) {name} Spectrum')
    ax.set_xlabel('Frequency (bins)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim([-120, 0])
    ax.axhline(y=-3, color='r', linestyle='--', alpha=0.5, label='-3dB')
    ax.axhline(y=-40, color='gray', linestyle=':', alpha=0.5, label='-40dB')
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle('Test 7: Window Function Characteristics Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_Window_Characteristics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test7_Window_Characteristics.png")

# ============ 窗函数对比表格图 ============
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# 计算各窗函数参数
win_data = []
for name, win in windows.items():
    W = np.abs(np.fft.fft(win, 65536))
    W = W / np.max(W)
    W_db = 20 * np.log10(W + 1e-10)
    
    # 3dB bandwidth
    idx_3db = np.where(W_db[:32768] < -3)[0]
    bw_3db = 2 * idx_3db[0] / 65536 * 1024 if len(idx_3db) > 0 else 0
    
    # Scallop loss
    scallop = W_db[0]
    
    # First sidelobe level
    peaks, _ = signal.find_peaks(W_db[:2000])
    first_sidelobe = W_db[peaks[0]] if len(peaks) > 0 else -100
    
    # THD error estimate (non-coherent)
    thd_err = {'Rectangular': '>4%', 'Hanning': '~1.5%', 'Hamming': '~1.2%', 
               'Flat-top': '<0.02%', 'Blackman-Harris': '<0.1%'}[name]
    
    win_data.append([name, f'{bw_3db:.2f}', f'{scallop:.2f}', f'{first_sidelobe:.1f}', thd_err])

table = ax.table(cellText=win_data,
                colLabels=['Window', '3dB BW (bins)', 'Scallop Loss (dB)', '1st Sidelobe (dB)', 'THD Error (non-coherent)'],
                cellLoc='center',
                loc='center',
                colColours=['#4472C4']*5)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Color cells
for i in range(5):
    table[(0, i)].set_text_props(color='white', fontweight='bold')

fig.suptitle('Test 7: Window Function Comparison Table', fontsize=13, fontweight='bold')
plt.savefig(os.path.join(output_dir, 'Test7_Window_Comparison_Table.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test7_Window_Comparison_Table.png")

print(f"\n=== Supplementary Plots Complete ===")
print(f"Output: {output_dir}")
