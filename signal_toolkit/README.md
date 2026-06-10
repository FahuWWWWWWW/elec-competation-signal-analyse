# signal_toolkit — 电赛信号仪器类公共算法库

> **版本**: 1.0.0 | **依赖**: numpy, scipy, matplotlib  
> **覆盖**: 2021-2025 年全国大学生电子设计竞赛（信号及仪器仪表类）13题

---

## 一、安装

```bash
pip install numpy scipy matplotlib
# 将 signal_toolkit/ 放在你的项目目录下即可 import
```

或者安装到系统路径：

```bash
cd 信号及仪器仪表类
pip install -e .
```

## 二、模块概览

| 模块 | 文件 | 对应电赛题目 | 核心功能 |
|------|------|------------|---------|
| **fft_analysis** | `fft_analysis.py` | 2021-A, 2021-J, 2024-B | FFT频谱、THD计算、窗函数(Flat-top/Hanning/Hamming等) |
| **dds_synthesis** | `dds_synthesis.py` | 2022-F, 2024-C, 2025-G | DDS频率合成、AM/FM调制、扫频、脉冲生成 |
| **tdr_analysis** | `tdr_analysis.py` | 2023-B, 2025-D | TDR等效采样、反射检测、电缆故障建模 |
| **goertzel** | `goertzel.py` | 2023-H | Goertzel单频检测、频率扫描、多频识别 |
| **iq_demodulation** | `iq_demodulation.py` | 2022-F, 2023-D, 2023-H | I/Q正交解调、AM/FM/PM解调、相干分离 |
| **filters** | `filters.py` | 2021-A, 2025-G | 数字滤波器设计、Sallen-Key传递函数、阶数估算 |
| **utils** | `utils.py` | 通用 | dBm/Vpp转换、SNR计算、2的幂次 |

## 三、快速入门

### 3.1 THD 失真度分析（2021-A）

```python
from signal_toolkit import fft_analysis as fa
from signal_toolkit import dds_synthesis as dds

fs = 1_000_000
t, sig = dds.generate_sine(fs, 100_000, 0.01)
# 添加2~5次谐波
for h, a in zip([2,3,4,5], [0.03, 0.015, 0.008, 0.004]):
    _, hsig = dds.generate_sine(fs, 100_000*h, 0.01, a)
    sig += hsig

thd_pct, harmonics = fa.compute_thd(sig, fs, 100_000)
print(f"THD = {thd_pct:.2f}%")  # 预期 ~3.45%

freqs, mag, phase = fa.compute_spectrum(sig, fs, window='flat_top')
```

### 3.2 TDR 电缆故障定位（2023-B / 2025-D）

```python
from signal_toolkit import tdr_analysis as tdr

analyzer = tdr.TDR(fs=100e6, vf=0.67)
t, signal = analyzer.cable_model(length_m=5.0, fault_type='open')
idx, amp = analyzer.detect_reflection(signal, threshold=0.03)
if len(idx) >= 2:
    dist = analyzer.compute_distance(idx[1] - idx[0])
    print(f"故障距离: {dist:.1f}m")

# 反射系数
gamma = analyzer.reflection_coefficient(z_load=75)  # +0.2
gamma = analyzer.reflection_coefficient(z_load=1e6) # +1.0 (开路)
```

### 3.3 DDS 信号生成（2022-F / 2024-C / 2025-G）

```python
from signal_toolkit import dds_synthesis as dds

# 正弦波
t, sine = dds.generate_sine(fs=10e6, f_out=1e6, duration=0.001)

# AM调制
t, am = dds.generate_am(fs=10e6, f_carrier=2e6, f_mod=50e3, depth=0.5)

# FM调制
t, fm = dds.generate_fm(fs=10e6, f_carrier=2e6, f_dev=50e3, mod_freq=10e3)

# 扫频（频率特性测量）
t, sweep = dds.generate_sweep(fs=1e6, f_start=100, f_stop=1e6, duration=1.0)

# TDR脉冲
t, pulse = dds.generate_pulse(fs=100e6, duration=1e-6, tr=1e-9, pw=10e-9)

# DDS参数模型
dds_obj = dds.DDS(f_clk=125e6, phase_bits=32)
print(f"频率分辨率: {dds_obj.frequency_resolution:.4f}Hz")
ftw = dds_obj.frequency_to_ftw(1_000_000)
print(f"频率调谐字: {ftw}")
```

### 3.4 Goertzel 频率检测（2023-H）

```python
from signal_toolkit import goertzel as gz
import numpy as np

fs = 500_000
# 生成测试信号（含3个频率成分）
t = np.arange(0, 0.05, 1/fs)
s = np.sin(2*np.pi*23000*t) + 0.7*np.sin(2*np.pi*47000*t)

# 单频检测
mag, phase = gz.goertzel(s, target_freq=23000, fs=fs)

# 多频扫描
detected = gz.detect_frequencies(
    s, fs, f_min=10000, f_max=100000, step=500, threshold=0.1
)
for freq, mag in detected:
    print(f"检测到: {freq}Hz (幅值={mag:.3f})")
```

### 3.5 I/Q 解调与信号分离（2023-H / 2022-F）

```python
from signal_toolkit import iq_demodulation as iq
from signal_toolkit import dds_synthesis as dds

# AM解调
t, am = dds.generate_am(500_000, 100_000, 5_000, 0.5, 0.005)
envelope, m = iq.demodulate_am(am, 500_000, 100_000)
print(f"调制度估计: {m:.3f}")

# FM解调
t, fm = dds.generate_fm(500_000, 100_000, 20_000, 3_000, 0.005)
demod, dev = iq.demodulate_fm(fm, 500_000, 100_000)
print(f"频偏估计: {dev/1000:.1f}kHz")

# 信号分离（从混合信号中提取各频率成分）
freqs = [23000, 55000, 82000]
mixed = np.zeros(int(500000 * 0.02))
for f, a in zip(freqs, [1.0, 0.6, 0.3]):
    _, s = dds.generate_sine(500000, f, 0.02, a)
    mixed += s
components, amps = iq.separate_signals(mixed, freqs, 500000)
```

### 3.6 滤波器设计

```python
from signal_toolkit import filters as flt

# 低通数字滤波器
b, a = flt.design_lpf(f_cutoff=100_000, fs=1_000_000, order=4)
filtered = flt.apply_filter(signal, b, a)

# Sallen-Key LPF 传递函数
wn, zeta, Q = flt.sallen_key_transfer(R1=10e3, R2=10e3, C1=1e-9, C2=1e-9)
print(f"自然频率: {wn:.0f} rad/s, 阻尼: {zeta:.3f}")

# 计算抗混叠滤波器所需阶数
order = flt.butterworth_order(f_pass=500_000, f_stop=1_000_000, fs=2_000_000)
print(f"所需滤波器阶数: {order}")
```

### 3.7 工具函数

```python
from signal_toolkit import utils

# 功率/电压转换
vpp = utils.dbm_to_vpp(0)      # 0dBm → 0.447Vpp (50Ω)
dbm = utils.vpp_to_dbm(1.0)    # 1Vpp → 10dBm (50Ω)

# SNR 计算
snr_db = utils.snr(signal=1.0, noise=0.01)  # 40dB

# 2的幂次
n = utils.next_power_of_two(1000)  # 1024
```

## 四、题目 → 模块映射

| 题目 | 主要模块 | 次要模块 |
|------|---------|---------|
| 2021-A 信号失真度测量 | `fft_analysis` (THD/频谱) | `filters` (抗混叠) |
| 2021-J 波形识别 | `fft_analysis` (频谱) | `dds_synthesis` (参考信号) |
| 2022-F 调制度测量 | `iq_demodulation` (AM/FM解调) | `dds_synthesis` (DDS本振) |
| 2023-B 电缆TDR | `tdr_analysis` (全部) | `dds_synthesis` (脉冲) |
| 2023-C LCR测量 | `fft_analysis` (DFT) | `dds_synthesis` (测试信号) |
| 2023-D 调制识别 | `iq_demodulation` (I/Q特征) | `goertzel` (频率检测) |
| 2023-H 信号分离 | `goertzel` + `iq_demodulation` (联合) | — |
| 2024-B 功率分析 | `fft_analysis` (THD/谐波) | `filters` (抗混叠) |
| 2024-C 信道模拟 | `dds_synthesis` (DDS/AM) | — |
| 2024-G 录音屏蔽 | `utils` (SNR/VAD) | — |
| 2025-D 线缆测试 | `tdr_analysis` (全部) | `dds_synthesis` (脉冲/信号源) |
| 2025-F 自动接收 | `iq_demodulation` (解调) | `dds_synthesis` (本振) |
| 2025-G 电路探究 | `dds_synthesis` (扫频) + `filters` (IIR匹配) | `fft_analysis` (Bode) |

## 五、项目代码架构

```
signal_toolkit/
├── __init__.py            # 包初始化 + 顶层导出
├── fft_analysis.py        # FFT/窗函数/THD/谐波分析
├── dds_synthesis.py       # DDS类 + 正弦/AM/FM/扫频/脉冲生成
├── tdr_analysis.py        # TDR类 + 等效采样/反射检测/电缆模型
├── goertzel.py            # Goertzel单频检测 + 频率扫描
├── iq_demodulation.py     # I/Q解调 + AM/FM/PM + 相干分离
├── filters.py             # IIR滤波器 + Sallen-Key + 阶数计算
├── utils.py               # dBm⇔Vpp、SNR、2的幂次
└── examples/
    ├── example_fft.py       # THD计算+频谱图演示
    ├── example_dds.py       # DDS各种信号生成演示
    ├── example_tdr.py       # TDR电缆故障定位演示
    ├── example_goertzel.py  # Goertzel频率扫描演示
    └── example_iq.py        # I/Q解调+信号分离演示
```

## 六、代码设计思路

### 6.1 设计原则

1. **numpy-first**: 所有算法基于 numpy 向量化运算，避免 Python 循环
2. **物理模型驱动**: DDS 模拟硬件相位累加器，TDR 模拟脉冲传播物理过程
3. **竞赛问题映射**: 每个函数对应一个竞赛题的具体需求
4. **参数可验证**: 所有核心公式有参数验证数值可用

### 6.2 算法设计说明

#### `fft_analysis.compute_thd`
- 使用 Flat-top 窗解决非相干采样误差（2021-A 银弹技术）
- 在基频附近搜索实际峰值（容忍偏离），而非假设精确 bin
- 谐波搜索在 `k*f0 ± f0/2` 范围内找最大值

#### `goertzel.goertzel`
- 标准 2 阶 IIR 谐振器结构
- 计算量: N 个样本 × (2 乘 + 2 加) / 频率
- 比 FFT 快 10-100× 当只需要检测少量频率时

#### `iq_demodulation.demodulate_fm`
- 相位展开 + 数字差分实现鉴频
- 使用 `numpy.unwrap` 解决相位跳变
- f_dev 估计通过 RMS 测量频偏信号

#### `tdr_analysis.TDR.cable_model`
- 内部 10× 过采样保证脉冲上升沿平滑
- 反射幅度 = 入射幅度 × 反射系数
- 反射极性: 开路 +1.0, 短路 -1.0

#### `dds_synthesis.DDS`
- 模拟硬件 N-bit 相位累加器
- 频率分辨率 = f_clk / 2^N
- 相位截断效应建模（可选 spur 分析）
