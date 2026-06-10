"""Unit tests for signal_toolkit package."""

import os
import sys
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from signal_toolkit import utils
from signal_toolkit import fft_analysis
from signal_toolkit import filters
from signal_toolkit import goertzel as goertzel_mod
from signal_toolkit import dds_synthesis as dds
from signal_toolkit import tdr_analysis as tdr
from signal_toolkit import iq_demodulation as iq


# ============================================================
# utils tests
# ============================================================

class TestUtils:
    def test_dbm_vpp_roundtrip(self):
        vpp = 2.0
        dbm = utils.vpp_to_dbm(vpp, 50)
        vpp2 = utils.dbm_to_vpp(dbm, 50)
        assert abs(vpp2 - vpp) < 1e-10

    def test_dbm_vpp_array(self):
        dbm = utils.vpp_to_dbm([1.0, 2.0], 50)
        assert len(dbm) == 2
        assert np.all(np.isfinite(dbm))

    def test_snr_finite(self):
        s = np.ones(100)
        n = np.random.randn(100) * 0.1
        r = utils.snr(s, n)
        assert np.isfinite(r)
        assert r > 10

    def test_snr_zero_noise(self):
        s = np.ones(10)
        n = np.zeros(10)
        assert utils.snr(s, n) == float('inf')

    def test_next_power_of_two(self):
        assert utils.next_power_of_two(1) == 1
        assert utils.next_power_of_two(3) == 4
        assert utils.next_power_of_two(1024) == 1024
        assert utils.next_power_of_two(1025) == 2048

    def test_moving_average(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = utils.moving_average(x, 3)
        assert len(y) == 5
        assert y[2] == pytest.approx(3.0, abs=1e-10)

    def test_moving_average_single(self):
        x = np.array([5.0])
        y = utils.moving_average(x, 1)
        assert y[0] == 5.0


# ============================================================
# fft_analysis tests
# ============================================================

class TestFftAnalysis:
    def test_flat_top_peak_one(self):
        w = fft_analysis.flat_top_window(128)
        assert abs(w[len(w)//2]) - 1.0 < 0.01

    def test_hanning_window(self):
        w = fft_analysis.hanning_window(100)
        assert abs(w[0]) < 1e-15
        assert abs(w[-1]) < 1e-15
        assert w[49] > 0

    def test_hamming_window(self):
        w = fft_analysis.hamming_window(100)
        assert w[0] > 0
        assert w[-1] > 0

    def test_blackman_harris_window(self):
        w = fft_analysis.blackman_harris_window(128)
        assert np.all(w >= 0)

    def test_spectrum_pure_sine(self):
        fs = 10000
        duration = 0.1
        t = np.arange(int(duration * fs)) / fs
        f0 = 1000
        sig = np.sin(2 * np.pi * f0 * t)
        freqs, mag, phase = fft_analysis.compute_spectrum(sig, fs, window='flat_top')
        idx = np.argmax(mag)
        assert abs(freqs[idx] - f0) < 50

    def test_find_fundamental_none(self):
        freqs = np.linspace(0, 1000, 100)
        mag = np.zeros(100)
        idx = fft_analysis.find_fundamental(freqs, mag, f_range=(100, 200))
        assert idx is None or mag[idx] == 0.0

    def test_find_harmonics_empty(self):
        freqs = np.linspace(0, 1000, 100)
        mag = np.zeros(100)
        h = fft_analysis.find_harmonics(freqs, mag, 100, 5)
        assert len(h) == 5
        assert all(v == 0.0 for v in h.values())

    def test_compute_thd_pure_sine(self):
        fs = 10000
        duration = 0.2
        t = np.arange(int(duration * fs)) / fs
        sig = np.sin(2 * np.pi * 1000 * t)
        thd, harmonics = fft_analysis.compute_thd(sig, fs, 1000)
        assert thd < 1.0

    def test_compute_thd_with_harmonics(self):
        fs = 10000
        duration = 0.2
        t = np.arange(int(duration * fs)) / fs
        sig = np.sin(2 * np.pi * 1000 * t) + 0.05 * np.sin(2 * np.pi * 2000 * t)
        thd, harmonics = fft_analysis.compute_thd(sig, fs, 1000)
        assert thd > 3.0
        assert thd < 7.0


# ============================================================
# filters tests
# ============================================================

class TestFilters:
    def test_design_lpf_stable(self):
        b, a = filters.design_lpf(1000, 10000, 4)
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(a))

    def test_design_bpf_stable(self):
        b, a = filters.design_bpf(300, 3000, 10000)
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(a))

    def test_apply_filter_shape(self):
        sig = np.random.randn(100)
        b, a = filters.design_lpf(1000, 10000)
        out = filters.apply_filter(sig, b, a)
        assert out.shape == sig.shape
        assert np.all(np.isfinite(out))

    def test_sallen_key(self):
        wn, zeta, Q = filters.sallen_key_transfer(10000, 10000, 1e-7, 1e-7)
        assert np.isfinite(wn)
        assert np.isfinite(zeta)
        assert np.isfinite(Q)
        assert zeta > 0

    def test_butterworth_order_known(self):
        n = filters.butterworth_order(1000, 5000, 10000, 3, 40)
        assert n >= 2

    def test_butterworth_order_no_attenuation(self):
        n = filters.butterworth_order(1000, 500, 10000)
        assert n == 0


# ============================================================
# goertzel tests
# ============================================================

class TestGoertzel:
    def test_goertzel_amplitude(self):
        fs = 10000
        duration = 0.05
        t = np.arange(int(duration * fs)) / fs
        f0 = 1000
        sig = np.sin(2 * np.pi * f0 * t)
        amp, phase = goertzel_mod.goertzel(sig, f0, fs)
        assert abs(amp - 1.0) < 0.05

    def test_goertzel_zero_samples(self):
        amp, phase = goertzel_mod.goertzel(np.array([]), 1000, 10000)
        assert amp == 0.0
        assert phase == 0.0

    def test_goertzel_bank(self):
        fs = 10000
        duration = 0.05
        t = np.arange(int(duration * fs)) / fs
        sig = np.sin(2 * np.pi * 1000 * t)
        results = goertzel_mod.goertzel_bank(sig, [1000, 2000, 3000], fs)
        assert len(results) == 3
        assert abs(results[0][1] - 1.0) < 0.05
        assert results[1][1] < 0.1
        assert results[2][1] < 0.1

    def test_detect_frequencies(self):
        fs = 10000
        duration = 0.5
        t = np.arange(int(duration * fs)) / fs
        sig = np.sin(2 * np.pi * 1234 * t)
        peaks = goertzel_mod.detect_frequencies(sig, fs, 1000, 1500, 5)
        max_peak = max(peaks, key=lambda p: p[1]) if peaks else (0, 0)
        assert len(peaks) >= 1
        assert abs(max_peak[0] - 1234) < 20

    def test_detect_frequencies_zero_signal(self):
        sig = np.zeros(100)
        peaks = goertzel_mod.detect_frequencies(sig, 10000, 100, 5000, 10)
        assert peaks == []


# ============================================================
# dds_synthesis tests
# ============================================================

class TestDdsSynthesis:
    def test_dds_frequency_resolution(self):
        d = dds.DDS(f_clk=125e6, phase_bits=32)
        assert d.frequency_resolution == pytest.approx(125e6 / 2 ** 32)

    def test_dds_ftw_roundtrip(self):
        d = dds.DDS()
        ftw = d.frequency_to_ftw(10e6)
        f = d.ftw_to_frequency(ftw)
        assert abs(f - 10e6) / 10e6 < 0.01

    def test_generate_sine_shape(self):
        t, y = dds.generate_sine(10000, 1000, 0.01)
        assert len(t) == 100
        assert len(y) == 100
        assert abs(np.max(y) - 1.0) < 0.1

    def test_generate_sine_phase(self):
        t, y = dds.generate_sine(10000, 1000, 0.01, phase=np.pi / 2)
        assert abs(y[0] - 1.0) < 0.05

    def test_generate_am_shape(self):
        t, y = dds.generate_am(50000, 10000, 1000, 0.5, 0.01)
        assert len(t) == 500
        assert abs(np.max(y)) <= 1.6

    def test_generate_fm_shape(self):
        t, y = dds.generate_fm(50000, 10000, 2000, 500, 0.01)
        assert len(t) == 500
        assert len(y) == 500

    def test_generate_sweep_linear(self):
        t, y = dds.generate_sweep(10000, 100, 1000, 0.01)
        assert len(t) == 100

    def test_generate_sweep_log(self):
        t, y = dds.generate_sweep(10000, 100, 1000, 0.01, method='log')
        assert len(t) == 100

    def test_generate_sweep_invalid_method(self):
        try:
            dds.generate_sweep(10000, 100, 1000, 0.01, method='invalid')
            assert False
        except ValueError:
            pass

    def test_generate_pulse_shape(self):
        t, y = dds.generate_pulse(100e6, 100e-9, tr=1e-9, pw=10e-9)
        assert len(t) == 10
        assert np.max(y) == 1.0

    def test_generate_pulse_short_tr(self):
        t, y = dds.generate_pulse(1e9, 50e-9, tr=0.1e-9, pw=5e-9)
        assert len(t) == 50
        assert np.max(y) == 1.0


# ============================================================
# tdr_analysis tests
# ============================================================

class TestTdrAnalysis:
    def test_tdr_init(self):
        t = tdr.TDR(fs=100e6, vf=0.67)
        assert t.fs == 100e6
        assert t.vf == 0.67

    def test_compute_distance(self):
        dist = tdr.compute_distance(1e-6, vf=0.67)
        expected = 299792458 * 0.67 * 1e-6 / 2
        assert abs(dist - expected) < 1e-6

    def test_reflection_coefficient_open(self):
        assert abs(tdr.TDR.reflection_coefficient(float('inf'), 50) - 1.0) < 1e-10

    def test_reflection_coefficient_short(self):
        assert abs(tdr.TDR.reflection_coefficient(0, 50) - (-1.0)) < 1e-10

    def test_reflection_coefficient_matched(self):
        assert abs(tdr.TDR.reflection_coefficient(50, 50)) < 1e-10

    def test_cable_model_all_types(self):
        for fault in ['open', 'short', 'load_75ohm', 'load_150ohm']:
            t = tdr.TDR(fs=1e9, vf=0.67)
            sig, time = t.cable_model(5.0, fault_type=fault)
            assert len(sig) == len(time)
            assert np.any(np.abs(sig) > 0.1)

    def test_detect_reflection_empty(self):
        t = tdr.TDR(fs=100e6)
        sig = np.zeros(100)
        idx, _ = t.detect_reflection(sig, threshold=0.05)
        assert len(idx) == 0

    def test_cable_model_distance_accuracy_open(self):
        t = tdr.TDR(fs=1e9, vf=0.67)
        sig, _ = t.cable_model(10.0, fault_type='open')
        idx, _ = t.detect_reflection(sig, threshold=0.05)
        assert len(idx) >= 2
        delay_samples = idx[1] - idx[0]
        dist = t.compute_distance(delay_samples)
        expected = 10.0
        assert abs(dist - expected) / expected < 0.02

    def test_cable_model_distance_accuracy_short(self):
        t = tdr.TDR(fs=1e9, vf=0.67)
        sig, _ = t.cable_model(5.0, fault_type='short')
        idx, _ = t.detect_reflection(sig, threshold=0.05)
        assert len(idx) >= 2
        delay_samples = idx[1] - idx[0]
        dist = t.compute_distance(delay_samples)
        assert abs(dist - 5.0) / 5.0 < 0.02

    def test_eq_sampling(self):
        wf = np.sin(np.linspace(0, 4*np.pi, 100))
        equiv, t_eff = tdr.equivalent_sampling(wf, m=10, trigger_period=1e-6)
        assert len(equiv) >= len(wf) * 10 - 10

    def test_detect_reflection_static(self):
        peaks = tdr.detect_reflection(np.array([0, 0, 1, 0, 0, 1, 0, 0]), 0.1)
        assert len(peaks) >= 2


# ============================================================
# iq_demodulation tests
# ============================================================

class TestIqDemodulation:
    def test_iq_demodulate_shape(self):
        fs = 500000
        t = np.arange(500) / fs
        sig = np.cos(2 * np.pi * 100000 * t)
        I, Q = iq.iq_demodulate(sig, 100000, fs)
        assert I.shape == sig.shape
        assert Q.shape == sig.shape

    def test_iq_demodulate_custom_cutoff(self):
        fs = 500000
        t = np.arange(500) / fs
        sig = np.cos(2 * np.pi * 100000 * t)
        I, Q = iq.iq_demodulate(sig, 100000, fs, cutoff=50000)
        assert np.all(np.isfinite(I))
        assert np.all(np.isfinite(Q))

    def test_demodulate_am_coherent(self):
        fs = 500000
        t, am = dds.generate_am(fs, 100000, 5000, 0.5, 0.005)
        env, m = iq.demodulate_am(am, fs, f_carrier=100000)
        assert abs(m - 0.5) < 0.05
        assert len(env) == len(am)

    def test_demodulate_am_hilbert(self):
        fs = 500000
        t, am = dds.generate_am(fs, 100000, 5000, 0.3, 0.005)
        env, m = iq.demodulate_am(am, fs)
        assert abs(m - 0.3) < 0.05

    def test_demodulate_am_deep(self):
        fs = 500000
        t, am = dds.generate_am(fs, 100000, 5000, 1.0, 0.005)
        env, m = iq.demodulate_am(am, fs, f_carrier=100000)
        assert abs(m - 1.0) < 0.05

    def test_demodulate_am_shallow(self):
        fs = 500000
        t, am = dds.generate_am(fs, 100000, 5000, 0.1, 0.005)
        env, m = iq.demodulate_am(am, fs, f_carrier=100000)
        assert abs(m - 0.1) < 0.02

    def test_demodulate_fm_coherent(self):
        fs = 500000
        t, fm = dds.generate_fm(fs, 100000, 20000, 3000, 0.005)
        dem, fd = iq.demodulate_fm(fm, fs, f_carrier=100000)
        assert abs(fd - 20000) / 20000 < 0.03

    def test_demodulate_fm_hilbert(self):
        fs = 500000
        t, fm = dds.generate_fm(fs, 100000, 10000, 2000, 0.005)
        dem, fd = iq.demodulate_fm(fm, fs)
        assert abs(fd - 10000) / 10000 < 0.03

    def test_demodulate_fm_with_fdev_hint(self):
        fs = 500000
        t, fm = dds.generate_fm(fs, 100000, 15000, 3000, 0.005)
        dem, fd = iq.demodulate_fm(fm, fs, f_carrier=100000, f_dev=15000)
        assert abs(fd - 15000) / 15000 < 0.03

    def test_demodulate_pm(self):
        fs = 500000
        t = np.arange(2500) / fs
        phase_dev = 2.0
        sig = np.cos(2 * np.pi * 100000 * t + phase_dev * np.sin(2 * np.pi * 3000 * t))
        inst_phase = iq.demodulate_pm(sig, fs, f_carrier=100000)
        assert len(inst_phase) == len(sig)
        phase_std = np.std(inst_phase[375:-375]) if len(inst_phase) > 750 else np.std(inst_phase)
        assert abs(phase_std - phase_dev / np.sqrt(2)) < 0.3

    def test_demodulate_pm_hilbert(self):
        fs = 500000
        t = np.arange(2500) / fs
        sig = np.cos(2 * np.pi * 100000 * t + np.sin(2 * np.pi * 3000 * t))
        inst_phase = iq.demodulate_pm(sig, fs)
        assert len(inst_phase) == len(sig)

    def test_coherent_demodulate(self):
        fs = 50000
        duration = 0.05
        t = np.arange(int(duration * fs)) / fs
        sig = 0.7 * np.cos(2 * np.pi * 3000 * t + 0.5)
        amp, phase, recon = iq.coherent_demodulate(sig, 3000, 3000, fs, 0)
        assert abs(amp - 0.7) < 0.05
        assert abs(phase - 0.5) < 0.1 or abs(phase - 0.5 + 2 * np.pi) < 0.1

    def test_coherent_demodulate_zero_cutoff(self):
        fs = 100
        t = np.arange(50) / fs
        sig = np.ones(50)
        amp, phase, recon = iq.coherent_demodulate(sig, 0, 0, fs, 0)
        assert np.isfinite(amp)

    def test_separate_signals(self):
        fs = 200000
        duration = 0.005
        t = np.arange(int(duration * fs)) / fs
        mixed = (1.0 * np.cos(2 * np.pi * 23000 * t)
                 + 0.6 * np.cos(2 * np.pi * 55000 * t)
                 + 0.3 * np.cos(2 * np.pi * 82000 * t))
        components, amps = iq.separate_signals(mixed, [23000, 55000, 82000], fs)
        assert len(amps) == 3
        assert abs(amps[0] - 1.0) < 0.05
        assert abs(amps[1] - 0.6) < 0.05
        assert abs(amps[2] - 0.3) < 0.05
        assert len(components) == 3
        for c in components:
            assert len(c) == len(t)

    def test_separate_signals_single(self):
        fs = 200000
        duration = 0.005
        t = np.arange(int(duration * fs)) / fs
        sig = 0.5 * np.cos(2 * np.pi * 10000 * t)
        components, amps = iq.separate_signals(sig, [10000], fs)
        assert len(amps) == 1
        assert abs(amps[0] - 0.5) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
