"""
2024-G题 简易录音屏蔽系统 - 核心算法复现
Technique: 超声波干扰 + 麦克风非线性建模 + VAD + 音频分类

Test 1: 正常语音信号（无干扰）
Test 2: 超声波干扰信号频谱分析
Test 3: 麦克风非线性效应（单频超声→混叠噪声）
Test 4: 多频干扰效果（22+24+25kHz同时）
Test 5: VAD检测（有语音/无语音判断）
Test 6: 语音/音乐分类特征分析
Test 7: 干扰功率与距离关系
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_output')
os.makedirs(output_dir, exist_ok=True)

print("=== 2024-G Voice Recorder Jammer Simulator ===")
print("Technique: Ultrasonic Jamming + Microphone Nonlinearity + VAD\n")

# ============ 公共参数 ============
fs = 48000      # 录音设备采样率 (48kHz)
fs_ultra = 192000  # 超声波仿真采样率 (192kHz用于非线性仿真)
T = 2.0         # 信号时长 (s)
t = np.arange(0, T, 1/fs)
t_ultra = np.arange(0, T, 1/fs_ultra)

# ============ 信号生成函数 ============

def generate_voice(t, fs, f0=200, f1=800, harmonics=True):
    """生成模拟语音信号（基频扫描+谐波，模拟男声）"""
    # 基频从200Hz扫到800Hz，模拟语调变化
    f_inst = f0 + (f1 - f0) * t / T
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    voice = 0.5 * np.sin(phase)
    
    if harmonics:
        # 添加2次、3次谐波（模拟共振峰）
        voice += 0.25 * np.sin(2 * phase)
        voice += 0.15 * np.sin(3 * phase)
    
    # 添加浊音/清音交替（模拟音节）
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3Hz包络（音节率）
    voice *= envelope
    return voice * 0.3  # 语音幅度较小

def generate_ultrasonic(t, fs, freqs, amplitudes):
    """生成超声波信号"""
    ultra = np.zeros_like(t)
    for f, amp in zip(freqs, amplitudes):
        ultra += amp * np.sin(2 * np.pi * f * t)
    return ultra

def microphone_model(x, a1=1.0, a2=0.05, a3=0.01):
    """麦克风非线性模型（Volterra级数简化）"""
    # 二次项产生互调和谐波
    # 三次项产生更多互调产物
    return a1 * x + a2 * x**2 + a3 * x**3

def simulate_recording(voice, ultra, fs, fs_ultra):
    """模拟录音过程：超声+语音→麦克风非线性→降采样"""
    # 上采样语音到超声采样率
    voice_up = signal.resample(voice, len(ultra))
    
    # 混合信号
    mixed = voice_up + ultra
    
    # 麦克风非线性
    recorded = microphone_model(mixed)
    
    # 降采样回录音采样率（产生混叠）
    recorded_ds = signal.resample(recorded, len(voice))
    
    return recorded_ds

# ============ Test 1: 正常语音信号 ============
print("Test 1: Normal Voice Signal (No Jamming)")

voice = generate_voice(t, fs)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 时域
axes[0, 0].plot(t[:2000]*1000, voice[:2000], 'b-', linewidth=1)
axes[0, 0].set_title('(a) Voice Signal (Time Domain)')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

# 频谱
N_fft = 8192
freqs = np.fft.fftfreq(N_fft, 1/fs)[:N_fft//2]
voice_fft = np.abs(np.fft.fft(voice[:N_fft]))[:N_fft//2] * 2/N_fft
axes[0, 1].plot(freqs[:400], 20*np.log10(voice_fft[:400] + 1e-10), 'b-', linewidth=1.5)
axes[0, 1].set_title('(b) Voice Spectrum (0-4kHz)')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Magnitude (dB)')
axes[0, 1].grid(True, alpha=0.3)

# 语谱图
f_spec, t_spec, Sxx = signal.spectrogram(voice, fs, nperseg=512, noverlap=256)
axes[1, 0].pcolormesh(t_spec, f_spec[:100], 10*np.log10(Sxx[:100] + 1e-10), shading='gouraud', cmap='jet')
axes[1, 0].set_title('(c) Voice Spectrogram')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Frequency (Hz)')
axes[1, 0].set_ylim([0, 2000])

# 包络
envelope = np.abs(signal.hilbert(voice))
axes[1, 1].plot(t[:2000]*1000, voice[:2000], 'b-', alpha=0.5, linewidth=0.8, label='Waveform')
axes[1, 1].plot(t[:2000]*1000, envelope[:2000], 'r-', linewidth=1.5, label='Envelope')
axes[1, 1].set_title('(d) Voice Envelope')
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 1: Normal Voice Signal (Clean Recording)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test1_Normal_Voice.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test1_Normal_Voice.png")


# ============ Test 2: 超声波干扰信号频谱分析 ============
print("\nTest 2: Ultrasonic Jamming Signal Analysis")

ultra_freqs = [22000, 24000, 25000]
ultra_amps = [1.0, 0.8, 0.9]

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 生成超声波
ultra = generate_ultrasonic(t_ultra, fs_ultra, ultra_freqs, ultra_amps)

# 时域（局部放大）
N_show = int(0.002 * fs_ultra)  # 2ms
axes[0].plot(t_ultra[:N_show]*1000, ultra[:N_show], 'r-', linewidth=1)
axes[0].set_title(f'(a) Ultrasonic Signal (22+24+25kHz, Time Zoom 2ms)')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# 频谱
N_fft_u = 65536
freqs_u = np.fft.fftfreq(N_fft_u, 1/fs_ultra)[:N_fft_u//2]
ultra_fft = np.abs(np.fft.fft(ultra[:N_fft_u]))[:N_fft_u//2] * 2/N_fft_u
axes[1].plot(freqs_u/1000, 20*np.log10(ultra_fft + 1e-10), 'r-', linewidth=1.5)
for f in ultra_freqs:
    axes[1].axvline(x=f/1000, color='b', linestyle='--', alpha=0.5)
axes[1].set_title('(b) Ultrasonic Spectrum (0-30kHz)')
axes[1].set_xlabel('Frequency (kHz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_xlim([15, 30])
axes[1].grid(True, alpha=0.3)

fig.suptitle('Test 2: Ultrasonic Jamming Signal (22+24+25kHz)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test2_Ultrasonic_Signal.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test2_Ultrasonic_Signal.png")


# ============ Test 3: 麦克风非线性效应 ============
print("\nTest 3: Microphone Nonlinearity Effect")

# 生成语音+单频超声
voice = generate_voice(t, fs)
ultra_single = generate_ultrasonic(t_ultra, fs_ultra, [24000], [2.0])

# 模拟录音
recorded = simulate_recording(voice, ultra_single, fs, fs_ultra)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 原始语音频谱
voice_fft = np.abs(np.fft.fft(voice[:N_fft]))[:N_fft//2] * 2/N_fft
axes[0].plot(freqs[:400], 20*np.log10(voice_fft[:400] + 1e-10), 'b-', linewidth=1.5, label='Original Voice')
axes[0].set_title('(a) Original Voice Spectrum (Clean)')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_ylim([-80, 0])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 干扰后录音频谱
rec_fft = np.abs(np.fft.fft(recorded[:N_fft]))[:N_fft//2] * 2/N_fft
axes[1].plot(freqs[:400], 20*np.log10(rec_fft[:400] + 1e-10), 'r-', linewidth=1.5, label='Jammed Recording')
axes[1].set_title('(b) Recorded Signal Spectrum (After Jamming)')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude (dB)')
axes[1].set_ylim([-80, 0])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 噪声频谱（差值）
noise_fft = np.abs(rec_fft[:400] - voice_fft[:400])
axes[2].plot(freqs[:400], 20*np.log10(noise_fft + 1e-10), 'g-', linewidth=1.5, label='Noise (Jammed-Clean)')
axes[2].set_title('(c) Jamming Noise Spectrum (Baseband Aliasing)')
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Magnitude (dB)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.suptitle('Test 3: Microphone Nonlinearity → Baseband Noise', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test3_Nonlinearity_Effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test3_Nonlinearity_Effect.png")

# 计算SNR
signal_power = np.mean(voice**2)
noise_power = np.mean((recorded - voice)**2)
snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
print(f"  SNR after jamming: {snr_db:.1f} dB (Need <0dB for effective jamming)")


# ============ Test 4: 多频干扰效果对比 ============
print("\nTest 4: Multi-frequency Jamming Comparison")

# 对比单频 vs 三频干扰
voice = generate_voice(t, fs)

configs = [
    ("Single Freq (24kHz)", [24000], [2.0]),
    ("Dual Freq (22+25kHz)", [22000, 25000], [1.5, 1.5]),
    ("Triple Freq (22+24+25kHz)", [22000, 24000, 25000], [1.2, 1.2, 1.2]),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

results = []
for idx, (name, freqs_j, amps_j) in enumerate(configs):
    ultra_j = generate_ultrasonic(t_ultra, fs_ultra, freqs_j, amps_j)
    recorded_j = simulate_recording(voice, ultra_j, fs, fs_ultra)
    
    rec_fft_j = np.abs(np.fft.fft(recorded_j[:N_fft]))[:N_fft//2] * 2/N_fft
    
    # 计算SNR
    noise_j = recorded_j - voice
    snr_j = 10 * np.log10(np.mean(voice**2) / (np.mean(noise_j**2) + 1e-10))
    results.append((name, snr_j))
    
    axes[idx].plot(freqs[:400], 20*np.log10(rec_fft_j[:400] + 1e-10), 'r-', linewidth=1.5)
    axes[idx].plot(freqs[:400], 20*np.log10(voice_fft[:400] + 1e-10), 'b--', linewidth=1, alpha=0.7, label='Original')
    axes[idx].set_title(f'({chr(97+idx)}) {name}: SNR={snr_j:.1f}dB')
    axes[idx].set_xlabel('Frequency (Hz)')
    axes[idx].set_ylabel('Magnitude (dB)')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

fig.suptitle('Test 4: Multi-frequency Jamming Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test4_Multi_Freq_Comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test4_Multi_Freq_Comparison.png")
print("  Jamming results:")
for name, snr in results:
    status = "✓ Effective" if snr < 0 else "✗ Not enough"
    print(f"    {name}: SNR={snr:.1f}dB {status}")


# ============ Test 5: VAD检测 ============
print("\nTest 5: Voice Activity Detection (VAD)")

# 生成有语音段和静音段的信号
t_vad = np.arange(0, 3.0, 1/fs)
voice_segments = [
    (0.0, 0.8),   # 有语音
    (1.0, 1.8),   # 有语音
    (2.0, 2.8),   # 有语音
]

signal_vad = np.random.randn(len(t_vad)) * 0.01  # 背景噪声
for start, end in voice_segments:
    idx_start = int(start * fs)
    idx_end = int(end * fs)
    voice_seg = generate_voice(t_vad[idx_start:idx_end], fs)
    signal_vad[idx_start:idx_end] += voice_seg[:idx_end-idx_start]

# VAD算法：短时能量 + 过零率
frame_size = int(0.03 * fs)  # 30ms
hop_size = int(0.015 * fs)   # 15ms hop
n_frames = (len(signal_vad) - frame_size) // hop_size + 1

energy = []
zcr = []
for i in range(n_frames):
    frame = signal_vad[i*hop_size : i*hop_size + frame_size]
    energy.append(np.sum(frame**2))
    # 过零率
    zcr.append(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size))

energy = np.array(energy)
zcr = np.array(zcr)

# 归一化
energy_norm = energy / np.max(energy)
zcr_norm = zcr / np.max(zcr)

# 双门限判决
vad_decision = (energy_norm > 0.15) & (zcr_norm < 0.6)

# 绘制
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

t_frames = np.arange(n_frames) * hop_size / fs

axes[0].plot(t_vad, signal_vad, 'b-', linewidth=0.8)
for start, end in voice_segments:
    axes[0].axvspan(start, end, alpha=0.2, color='green', label='Voice' if start==0 else '')
axes[0].set_title('(a) Input Signal with Voice Segments')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_frames, energy_norm, 'r-', linewidth=1.5, label='Normalized Energy')
axes[1].plot(t_frames, zcr_norm, 'b-', linewidth=1.5, label='Normalized ZCR')
axes[1].axhline(y=0.15, color='r', linestyle='--', alpha=0.5, label='Energy Threshold')
axes[1].axhline(y=0.6, color='b', linestyle='--', alpha=0.5, label='ZCR Threshold')
axes[1].set_title('(b) VAD Features')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Normalized Value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_frames, vad_decision, 'g-', linewidth=2)
axes[2].set_title('(c) VAD Decision (1=Voice, 0=Silence)')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Decision')
axes[2].set_ylim([-0.1, 1.1])
axes[2].grid(True, alpha=0.3)

fig.suptitle('Test 5: Voice Activity Detection (VAD)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test5_VAD_Detection.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test5_VAD_Detection.png")


# ============ Test 6: 语音/音乐分类特征分析 ============
print("\nTest 6: Speech vs Music Classification Features")

# 生成模拟语音和音乐
voice = generate_voice(t, fs, f0=200, f1=600)

# 模拟音乐：多个稳定谐波 + 宽带
music = np.zeros_like(t)
np.random.seed(42)
for f in [262, 330, 392, 523]:  # C大调和弦
    music += 0.2 * np.sin(2 * np.pi * f * t + np.random.rand()*2*np.pi)
# 添加宽带成分
music += 0.1 * np.random.randn(len(t))
# 音乐包络相对连续
music *= 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t) ** 2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 时域对比
N_show = 3000
axes[0, 0].plot(t[:N_show]*1000, voice[:N_show], 'b-', linewidth=0.8, label='Speech')
axes[0, 0].set_title('(a) Speech Waveform')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t[:N_show]*1000, music[:N_show], 'r-', linewidth=0.8, label='Music')
axes[0, 1].set_title('(b) Music Waveform')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

# 频谱对比
voice_fft = np.abs(np.fft.fft(voice[:N_fft]))[:N_fft//2] * 2/N_fft
music_fft = np.abs(np.fft.fft(music[:N_fft]))[:N_fft//2] * 2/N_fft

axes[1, 0].plot(freqs[:300], 20*np.log10(voice_fft[:300] + 1e-10), 'b-', linewidth=1.5, label='Speech')
axes[1, 0].set_title('(c) Speech Spectrum')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Magnitude (dB)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(freqs[:300], 20*np.log10(music_fft[:300] + 1e-10), 'r-', linewidth=1.5, label='Music')
axes[1, 1].set_title('(d) Music Spectrum')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Magnitude (dB)')
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle('Test 6: Speech vs Music - Spectral Characteristics', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test6_Speech_Music_Classification.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test6_Speech_Music_Classification.png")

# 计算分类特征
# 频谱质心
spec_centroid_voice = np.sum(freqs[:300] * voice_fft[:300]) / np.sum(voice_fft[:300])
spec_centroid_music = np.sum(freqs[:300] * music_fft[:300]) / np.sum(music_fft[:300])
# 频谱平坦度
spec_flat_voice = np.exp(np.mean(np.log(voice_fft[:300] + 1e-10))) / np.mean(voice_fft[:300])
spec_flat_music = np.exp(np.mean(np.log(music_fft[:300] + 1e-10))) / np.mean(music_fft[:300])

print(f"  Classification features:")
print(f"    Spectral Centroid - Speech: {spec_centroid_voice:.0f}Hz, Music: {spec_centroid_music:.0f}Hz")
print(f"    Spectral Flatness - Speech: {spec_flat_voice:.3f}, Music: {spec_flat_music:.3f}")


# ============ Test 7: 干扰功率与距离关系 ============
print("\nTest 7: Jamming Power vs Distance")

# 声压级衰减模型：球面波扩散 + 空气吸收
# SPL(d) = SPL(1m) - 20*log10(d) - α*(d-1)
# α: 空气吸收系数 (约0.01 dB/m @ 24kHz, 常温)

distances = np.linspace(0.5, 3.0, 100)
SPL_at_1m = 100  # dB SPL at 1m (4W超声换能器)
alpha_absorption = 0.01  # dB/m

SPL = SPL_at_1m - 20*np.log10(distances) - alpha_absorption * (distances - 1)

# 录音设备麦克风灵敏度阈值（约40dB SPL可触发）
mic_threshold = 40

# 有效屏蔽距离：SPL > mic_threshold + 20dB margin
margin_needed = 20  # 需要比阈值高20dB才能有效干扰

fig, axes = plt.subplots(1, 1, figsize=(10, 6))

axes.plot(distances, SPL, 'b-', linewidth=2, label='Ultrasonic SPL')
axes.axhline(y=mic_threshold, color='r', linestyle='--', label=f'Microphone Threshold ({mic_threshold}dB)')
axes.axhline(y=mic_threshold + margin_needed, color='g', linestyle='--', label=f'Effective Jamming Level')
axes.axvline(x=1.0, color='purple', linestyle=':', alpha=0.7, label='Required Distance (1m)')

# 标注有效区域
effective_mask = SPL > (mic_threshold + margin_needed)
if np.any(effective_mask):
    max_effective_dist = distances[np.argmax(~effective_mask)] if not np.all(effective_mask) else distances[-1]
    axes.fill_between(distances, mic_threshold + margin_needed, 120, where=effective_mask, alpha=0.2, color='green')
    axes.text(0.6, 105, f'Effective Range: ~{max_effective_dist:.1f}m', fontsize=11, color='green', fontweight='bold')

axes.set_title('Ultrasonic Jamming: Sound Pressure Level vs Distance')
axes.set_xlabel('Distance (m)')
axes.set_ylabel('SPL (dB)')
axes.set_ylim([30, 110])
axes.legend()
axes.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Test7_Power_vs_Distance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: Test7_Power_vs_Distance.png")

# 计算满足1m要求的最低功率
# SPL = 10*log10(P) + SPL_ref
# 若4W对应100dB@1m, 则1W对应100-6=94dB@1m
SPL_1W = 94  # dB @ 1m
power_levels = [1, 2, 3, 4]  # W
SPL_at_1m_list = [SPL_1W + 10*np.log10(p) for p in power_levels]

print("  Power vs SPL at 1m:")
for p, spl in zip(power_levels, SPL_at_1m_list):
    status = "✓" if spl > (mic_threshold + margin_needed) else "✗"
    print(f"    {p}W → {spl:.1f}dB @1m {status}")

print(f"\n=== Simulation Complete ===")
print(f"Output: {output_dir}")
