# signal_toolkit — 电赛信号仪器类13题公共算法库
#
# 来源: 2021-2025全国大学生电子设计竞赛(信号及仪器仪表类)
# 覆盖: FFT/窗函数/DDS/TDR/Goertzel/IQ解调/滤波器/调制识别/功率分析/阻抗测量
# 依赖: numpy, scipy, matplotlib

__version__ = "1.1.0"

from . import utils
from . import fft_analysis
from . import filters
from . import goertzel
from . import dds_synthesis
from . import tdr_analysis
from . import iq_demodulation
from . import power_analysis
from . import impedance_analysis
from . import modulation_classification
from . import waveform_analysis

# ─── 方便顶层导入 ───────────────────────────────────────
# fft_analysis
from .fft_analysis import (
    flat_top_window,
    hanning_window,
    hamming_window,
    blackman_harris_window,
    compute_thd,
    compute_spectrum,
    find_fundamental,
    find_harmonics,
)

# filters
from .filters import (
    design_lpf,
    design_bpf,
    apply_filter,
    sallen_key_transfer,
    butterworth_order,
)

# goertzel (use goertzel_compute to avoid shadowing module name)
from .goertzel import (
    goertzel_bank,
    detect_frequencies,
)
from .goertzel import goertzel as goertzel_compute

# dds_synthesis
from .dds_synthesis import (
    DDS,
    generate_sine,
    generate_am,
    generate_fm,
    generate_sweep,
    generate_pulse,
)

# tdr_analysis
from .tdr_analysis import (
    TDR,
    equivalent_sampling,
    detect_reflection,
    compute_distance,
    cable_model,
)

# iq_demodulation
from .iq_demodulation import (
    iq_demodulate,
    demodulate_am,
    demodulate_fm,
    coherent_demodulate,
    separate_signals,
    demodulate_pm,
)

# power_analysis
from .power_analysis import (
    rms,
    active_power,
    apparent_power,
    reactive_power,
    power_factor,
    compute_power_parameters,
)

# impedance_analysis
from .impedance_analysis import (
    impedance_vi,
    quality_factor,
    lcr_from_impedance,
    series_resonance,
    parallel_resonance,
    detect_resonance,
    parallel_equivalent,
)

# modulation_classification
from .modulation_classification import (
    extract_features,
    classify_modulation,
)

# waveform_analysis
from .waveform_analysis import (
    zero_crossing_rate,
    count_peaks,
    duty_cycle,
    symmetry_ratio,
    estimate_frequency,
    classify_waveform,
)
