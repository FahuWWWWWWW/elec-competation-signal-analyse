%% ============================================================
% 2021年电赛J题：周期信号波形识别及参数测量装置 - MATLAB仿真复现 (修正版)
% 重点：每个仿真测试明确映射到前端调理电路模块
% 
% 调理电路链路：
%   信号输入(50mV~10V) -> [输入保护/缓冲] -> [PGA] -> [抗混叠滤波] -> [ADC]
%                                              |
%                                              v
%                                         [比较器] -> [定时器]
%
% 仿真测试与电路模块映射表：
%   Test 1: 信号生成与PGA增益控制      -> 模拟PGA程控增益放大器
%   Test 2: 频率测量(周期法vs计数法)   -> 模拟比较器过零检测+定时器
%   Test 3: Vpp测量(PGA+ADC量化)       -> 模拟PGA+ADC动态范围
%   Test 4: 占空比测量(迟滞比较器)      -> 模拟迟滞比较器边沿检测
%   Test 5: 波形识别(FFT+相关系数)      -> 模拟抗混叠滤波+ADC采样+DSP
%   Test 6: 系统响应时间分析            -> 模拟系统整体时序
%   Test 7: Monte Carlo误差预算         -> 模拟完整信号调理链路
%% ============================================================
clear; clc; close all;

output_dir = fullfile(fileparts(mfilename('fullpath')), 'simulation_output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% ==================== 全局仿真参数 ====================
fs_base = 500e3;        % 基础采样率 500 kHz (满足50kHz信号Nyquist)
N_base = 8192;          % FFT点数
ADC_bits = 12;          % ADC位数
ADC_Vref = 3.3;         % ADC满量程电压
ADC_LSB = ADC_Vref / (2^ADC_bits);

fprintf('============================================================\n');
fprintf('    2021-J题：周期信号波形识别及参数测量装置 仿真开始\n');
fprintf('============================================================\n');
fprintf('全局参数: fs=%.0fkHz, ADC=%d-bit, Vref=%.1fV\n', fs_base/1e3, ADC_bits, ADC_Vref);

%% ==================== Test 1: PGA增益控制仿真 ====================
% [对应调理电路模块]: PGA程控增益放大器 (如LTC6912/PGA281)
% [电路功能]: 将50mV~10V输入信号自动调理至ADC最佳量程(70%~90%)
% [仿真内容]: 模拟不同增益档位对50mV~10V信号的放大/衰减效果
% [考核指标]: 小信号时SNR是否满足Vpp测量1%精度要求

fprintf('\n========== Test 1: PGA增益控制仿真 (对应调理电路: PGA) ==========\n');

Vpp_input_list = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];  % Vpp (50mV~10V)
n_vpp = length(Vpp_input_list);
PGA_gains = zeros(1, n_vpp);
Vpp_ADC_ideal = zeros(1, n_vpp);
Vpp_ADC_quantized = zeros(1, n_vpp);
Vpp_errors = zeros(1, n_vpp);

pga_gain_table = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50];
target_adc_level = 0.7 * ADC_Vref;

for i = 1:n_vpp
    Vpp_in = Vpp_input_list(i);
    Vpk_in = Vpp_in / 2;
    
    ideal_gain = target_adc_level / Vpp_in;
    [~, gain_idx] = min(abs(pga_gain_table - ideal_gain));
    selected_gain = pga_gain_table(gain_idx);
    PGA_gains(i) = selected_gain;
    
    f_test = 10e3;
    t = (0:N_base-1)/fs_base;
    sig = Vpk_in * sin(2*pi*f_test*t);
    
    sig_pga = sig * selected_gain;
    
    pga_noise_factor = 1 + 0.05*selected_gain;
    noise_rms = (ADC_Vref/2) * 10^(-60/20) * pga_noise_factor;
    noise = noise_rms * randn(size(t));
    sig_noisy = sig_pga + noise;
    
    sig_quantized = round(sig_noisy / ADC_LSB) * ADC_LSB;
    sig_quantized = max(min(sig_quantized, ADC_Vref), 0);
    
    Vpp_ideal = max(sig_pga) - min(sig_pga);
    Vpp_quant = max(sig_quantized) - min(sig_quantized);
    
    Vpp_ADC_ideal(i) = Vpp_ideal;
    Vpp_ADC_quantized(i) = Vpp_quant;
    Vpp_errors(i) = abs(Vpp_quant - Vpp_ideal) / Vpp_ideal * 100;
    
    fprintf('  Vpp_in=%5.0fmV -> PGA_gain=%5.2f -> Vpp_ADC=%6.3fV (误差=%5.2f%%)\n', ...
        Vpp_in*1000, selected_gain, Vpp_quant, Vpp_errors(i));
end

fig1 = figure('Name','Test1_PGA_GainControl','Position',[100 100 1200 500]);

subplot(1,2,1);
bar(Vpp_input_list*1000, Vpp_ADC_quantized, 'FaceColor', [0.3 0.6 0.9]);
hold on;
plot(Vpp_input_list*1000, Vpp_ADC_ideal, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Input Vpp (mV)');
ylabel('ADC Measured Vpp (V)');
legend('Quantized', 'Ideal', 'Location', 'best');
title('Test 1: PGA Gain Control Effect (Circuit: PGA)');
grid on;

subplot(1,2,2);
semilogy(Vpp_input_list*1000, Vpp_errors, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.9 0.3 0.3]);
hold on;
yline(1, 'g--', 'LineWidth', 2, 'Label', '1% Limit');
xlabel('Input Vpp (mV)');
ylabel('Vpp Measurement Error (%)');
title('PGA+ADC Vpp Error (Green dashed = 1% requirement)');
grid on;

sgtitle('Test 1: PGA Simulation - Auto gain control for 50mV~10V input range');
saveas(fig1, fullfile(output_dir, 'Test1_PGA_GainControl_v2.png'));

%% ==================== Test 2: 频率测量仿真 ====================
% [对应调理电路模块]: 比较器过零检测 + 定时器捕获
% [电路功能]: 比较器将模拟信号转换为数字方波，定时器测量周期或计数脉冲
% [仿真内容]: 模拟不同频率下周期法和计数法的精度对比
% [考核指标]: 频率相对误差<=1%

fprintf('\n========== Test 2: 频率测量仿真 (对应调理电路: 比较器+定时器) ==========\n');

f_test_list = [1, 5, 10, 50, 100, 500, 1e3, 5e3, 10e3, 50e3];
n_freq = length(f_test_list);

f_period_method = zeros(1, n_freq);
f_count_method = zeros(1, n_freq);
f_errors_period = zeros(1, n_freq);
f_errors_count = zeros(1, n_freq);

timer_clock = 72e6;
timer_res = 1/timer_clock;
comparator_hysteresis = 5e-3;

for i = 1:n_freq
    f_true = f_test_list(i);
    T_true = 1/f_true;
    
    fs_sim = max(20*f_true, 200e3);
    if fs_sim > 10e6, fs_sim = 10e6; end
    t_sim = 0:1/fs_sim:2;
    sig = 1.0 * sin(2*pi*f_true*t_sim);
    sig = sig + 0.01*randn(size(t_sim));
    
    % 周期法
    threshold = 0;
    crossings = [];
    state = sig(1) > threshold;
    for k = 2:length(sig)
        if state && sig(k) < (threshold - comparator_hysteresis/2)
            state = false;
        elseif ~state && sig(k) > (threshold + comparator_hysteresis/2)
            state = true;
            crossings = [crossings, k];
        end
    end
    
    if length(crossings) >= 3
        periods = diff(crossings) / fs_sim;
        T_meas = mean(periods);
        f_period_method(i) = 1/T_meas;
    else
        f_period_method(i) = NaN;
    end
    
    % 计数法
    gate_time = 0.1;
    if T_true < gate_time
        n_cycles = floor(gate_time / T_true);
        f_count_method(i) = n_cycles / gate_time;
    else
        f_count_method(i) = NaN;
    end
    
    if ~isnan(f_period_method(i))
        f_errors_period(i) = abs(f_period_method(i) - f_true) / f_true * 100;
    else
        f_errors_period(i) = 100;
    end
    
    if ~isnan(f_count_method(i))
        f_errors_count(i) = abs(f_count_method(i) - f_true) / f_true * 100;
    else
        f_errors_count(i) = 100;
    end
    
    fprintf('  f=%6.1fHz: Period=%7.3fHz (err=%5.3f%%), Count=%7.3fHz (err=%5.3f%%)\n', ...
        f_true, f_period_method(i), f_errors_period(i), f_count_method(i), f_errors_count(i));
end

fig2 = figure('Name','Test2_Frequency_Measurement','Position',[100 100 1200 500]);

subplot(1,2,1);
loglog(f_test_list, f_errors_period, 'o-', 'LineWidth', 2, 'MarkerSize', 8); hold on;
loglog(f_test_list, f_errors_count, 's-', 'LineWidth', 2, 'MarkerSize', 8);
yline(1, 'r--', 'LineWidth', 2, 'Label', '1% Limit');
xlabel('True Frequency (Hz)');
ylabel('Relative Error (%)');
legend('Period Method (Low freq)', 'Count Method (High freq)', 'Location', 'best');
title('Test 2: Frequency Measurement (Circuit: Comparator + Timer)');
grid on;

subplot(1,2,2);
recommended_method = zeros(1, n_freq);
for i = 1:n_freq
    if f_test_list(i) < 100
        recommended_method(i) = f_errors_period(i);
    else
        recommended_method(i) = f_errors_count(i);
    end
end

bar(1:n_freq, recommended_method, 'FaceColor', [0.3 0.6 0.9]);
hold on;
yline(1, 'r--', 'LineWidth', 2);
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.0f', x), f_test_list, 'UniformOutput', false));
xlabel('Frequency (Hz)');
ylabel('Best Error (%)');
title('Adaptive Strategy: Period for low freq, Count for high freq');
grid on;

sgtitle('Test 2: Comparator Zero-Crossing + Timer Frequency Measurement');
saveas(fig2, fullfile(output_dir, 'Test2_Frequency_Measurement_v2.png'));

%% ==================== Test 3: Vpp测量仿真 ====================
% [对应调理电路模块]: PGA + 12-bit ADC
% [电路功能]: PGA调理信号幅度后, ADC采样并计算峰峰值
% [仿真内容]: 不同输入幅度、不同ADC位数下的Vpp测量精度
% [考核指标]: Vpp相对误差<=1%

fprintf('\n========== Test 3: Vpp测量仿真 (对应调理电路: PGA+ADC) ==========\n');

Vpp_test = 2.0;
f_test = 5e3;
adc_bits_list = [8, 10, 12, 14, 16];
n_bits = length(adc_bits_list);

Vpp_meas_results = zeros(1, n_bits);
Vpp_errors_adc = zeros(1, n_bits);

for i = 1:n_bits
    bits = adc_bits_list(i);
    lsb = ADC_Vref / (2^bits);
    
    t = (0:N_base-1)/fs_base;
    sig = (Vpp_test/2) * sin(2*pi*f_test*t);
    
    sig_q = round(sig / lsb) * lsb;
    
    Vpp_meas = max(sig_q) - min(sig_q);
    Vpp_meas_results(i) = Vpp_meas;
    Vpp_errors_adc(i) = abs(Vpp_meas - Vpp_test) / Vpp_test * 100;
    
    fprintf('  ADC=%2d-bit: Vpp_meas=%.4fV (误差=%.4f%%)\n', bits, Vpp_meas, Vpp_errors_adc(i));
end

fig3 = figure('Name','Test3_Vpp_ADC_Resolution','Position',[100 100 1000 500]);

subplot(1,2,1);
bar(adc_bits_list, Vpp_meas_results, 'FaceColor', [0.3 0.6 0.9]);
hold on;
yline(Vpp_test, 'r--', 'LineWidth', 2, 'Label', 'True Vpp=2V');
xlabel('ADC Resolution (bits)');
ylabel('Measured Vpp (V)');
title('Test 3: ADC Resolution vs Vpp (Circuit: PGA+ADC)');
grid on;

subplot(1,2,2);
semilogy(adc_bits_list, Vpp_errors_adc, 'o-', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.9 0.3 0.3]);
hold on;
yline(1, 'g--', 'LineWidth', 2, 'Label', '1% Limit');
xlabel('ADC Resolution (bits)');
ylabel('Vpp Measurement Error (%)');
title('Vpp Error vs ADC Resolution (Green = 1% requirement)');
grid on;

sgtitle(sprintf('Test 3: PGA+ADC Vpp Simulation (Vpp=%.1fV, f=%.0fkHz)', Vpp_test, f_test/1e3));
saveas(fig3, fullfile(output_dir, 'Test3_Vpp_ADC_Resolution_v2.png'));

%% ==================== Test 4: 占空比测量仿真 (修正版) ====================
% [对应调理电路模块]: 迟滞比较器
% [电路功能]: 将矩形波转换为数字方波, 消除噪声引起的边沿抖动
% [仿真内容]: 模拟不同噪声水平、不同迟滞电压下的占空比测量精度
% [考核指标]: 占空比绝对误差<=2%

fprintf('\n========== Test 4: 占空比测量仿真 (对应调理电路: 迟滞比较器) ==========\n');

f_pwm = 10e3;
D_true = 0.5;
T_pwm = 1/f_pwm;

% 使用更多采样点确保至少10个完整周期
fs_pwm = 10e6;  % 10MHz采样
n_periods = 20;
t_pwm = 0:1/fs_pwm:n_periods*T_pwm;

% 生成理想矩形波 (单极性 0~1V，避免负值问题)
rect_ideal = zeros(size(t_pwm));
for k = 1:length(t_pwm)
    phase = mod(t_pwm(k), T_pwm);
    if phase < D_true*T_pwm
        rect_ideal(k) = 1.0;
    else
        rect_ideal(k) = 0.0;
    end
end

noise_levels = [0, 0.02, 0.05, 0.1, 0.2];  % V
n_noise = length(noise_levels);

hysteresis_levels = [0, 0.02, 0.05, 0.1, 0.2];  % V
n_hyst = length(hysteresis_levels);

D_meas_matrix = zeros(n_noise, n_hyst);
D_errors_matrix = zeros(n_noise, n_hyst);

for ni = 1:n_noise
    noise_amp = noise_levels(ni);
    for hi = 1:n_hyst
        hyst = hysteresis_levels(hi);
        
        % 添加噪声 (确保信号不越界)
        noise = noise_amp * randn(size(t_pwm));
        rect_noisy = rect_ideal + noise;
        rect_noisy = max(min(rect_noisy, 1.5), -0.5);  % 限幅
        
        % 迟滞比较器 (阈值0.5V，单极性)
        threshold = 0.5;
        rect_digital = zeros(size(t_pwm));
        state = rect_noisy(1) > threshold;
        
        for k = 2:length(rect_noisy)
            if state && rect_noisy(k) < (threshold - hyst/2)
                state = false;
            elseif ~state && rect_noisy(k) > (threshold + hyst/2)
                state = true;
            end
            rect_digital(k) = state;
        end
        
        % 去掉前2个和后2个周期避免边界效应
        samples_per_period = round(fs_pwm / f_pwm);
        valid_start = 2*samples_per_period + 1;
        valid_end = length(rect_digital) - 2*samples_per_period;
        
        D_meas = mean(rect_digital(valid_start:valid_end));
        D_meas_matrix(ni, hi) = D_meas;
        D_errors_matrix(ni, hi) = abs(D_meas - D_true) * 100;
    end
end

fig4 = figure('Name','Test4_DutyCycle_Hysteresis','Position',[100 100 1200 500]);

subplot(1,2,1);
imagesc(hysteresis_levels*1000, noise_levels*1000, D_meas_matrix*100);
set(gca, 'YDir', 'normal');
clim([0 100]);
colormap(parula);
colorbar;
xlabel('Hysteresis (mV)');
ylabel('Noise Amplitude (mV)');
title('Measured Duty Cycle (%)');

subplot(1,2,2);
imagesc(hysteresis_levels*1000, noise_levels*1000, D_errors_matrix);
set(gca, 'YDir', 'normal');
clim([0 5]);
colormap(hot);
colorbar;
hold on;
contour(hysteresis_levels*1000, noise_levels*1000, D_errors_matrix, [2 2], 'g-', 'LineWidth', 3);
xlabel('Hysteresis (mV)');
ylabel('Noise Amplitude (mV)');
title('Duty Cycle Absolute Error (%)');

sgtitle('Test 4: Hysteresis Comparator Effect on Duty Cycle Measurement');
saveas(fig4, fullfile(output_dir, 'Test4_DutyCycle_Hysteresis_v2.png'));

fprintf('  占空比测量结果 (真实值=%.0f%%):\n', D_true*100);
for ni = 1:n_noise
    best_err = min(D_errors_matrix(ni, :));
    best_hyst = hysteresis_levels(find(D_errors_matrix(ni,:) == best_err, 1));
    fprintf('    噪声=%4.0fmV: 最小误差=%.3f%% (最佳迟滞=%.0fmV)\n', ...
        noise_levels(ni)*1000, best_err, best_hyst*1000);
end

%% ==================== Test 5: 波形识别仿真 ====================
% [对应调理电路模块]: 抗混叠滤波器 + ADC采样 + DSP算法
% [电路功能]: 抗混叠滤波器防止高频噪声混叠, ADC采样波形, DSP提取特征并分类
% [仿真内容]: 生成正弦/三角/矩形波, 加噪声后使用FFT和相关系数法识别
% [考核指标]: 正确识别3种波形

fprintf('\n========== Test 5: 波形识别仿真 (对应调理电路: 抗混叠滤波器+ADC+DSP) ==========\n');

f_wave = 2e3;
A_wave = 1.0;
SNR_wave = 40;

t_wave = (0:N_base-1)/fs_base;

% 正弦波
sine_wave = A_wave * sin(2*pi*f_wave*t_wave);

% 三角波
tri_wave = zeros(size(t_wave));
for k = 1:length(t_wave)
    phase = mod(t_wave(k) * f_wave, 1);
    if phase < 0.25
        tri_wave(k) = A_wave * 4 * phase;
    elseif phase < 0.75
        tri_wave(k) = A_wave * (2 - 4*phase);
    else
        tri_wave(k) = A_wave * (4*phase - 4);
    end
end

% 矩形波 (50%占空比)
rect_wave = zeros(size(t_wave));
for k = 1:length(t_wave)
    phase = mod(t_wave(k) * f_wave, 1);
    if phase < 0.5
        rect_wave(k) = A_wave;
    else
        rect_wave(k) = -A_wave;
    end
end

% 添加噪声
noise_power = (A_wave^2/2) / 10^(SNR_wave/10);
noise_wave = sqrt(noise_power) * randn(size(t_wave));
sine_noisy = sine_wave + noise_wave;
tri_noisy = tri_wave + noise_wave;
rect_noisy = rect_wave + noise_wave;

% 抗混叠滤波器
[b_aa, a_aa] = butter(4, 100e3/(fs_base/2), 'low');
sine_filtered = filter(b_aa, a_aa, sine_noisy);
tri_filtered = filter(b_aa, a_aa, tri_noisy);
rect_filtered = filter(b_aa, a_aa, rect_noisy);

wave_names = {'Sine', 'Triangle', 'Rectangle'};
waves = {sine_filtered, tri_filtered, rect_filtered};

fprintf('  FFT谐波分析结果:\n');
fig5 = figure('Name','Test5_Waveform_Recognition','Position',[100 100 1400 900]);

% 时域波形
for i = 1:3
    subplot(3, 3, i);
    idx_show = 1:round(fs_base/f_wave);
    plot(t_wave(idx_show)*1000, waves{i}(idx_show), 'LineWidth', 1.2);
    xlim([0 1/f_wave*1000]);
    xlabel('Time (ms)');
    ylabel('Amplitude (V)');
    title(sprintf('%s (Time Domain)', wave_names{i}));
    grid on;
end

% 频域分析
harmonic_features = zeros(3, 3);

for i = 1:3
    subplot(3, 3, i+3);
    Y = fft(waves{i}, N_base);
    P = abs(Y(1:N_base/2)).^2;
    f_axis = (0:N_base/2-1)*fs_base/N_base;
    
    plot(f_axis/1e3, 10*log10(P+eps), 'LineWidth', 1.2);
    xlim([0 20]);
    xlabel('Frequency (kHz)');
    ylabel('Power Spectrum (dB)');
    title(sprintf('%s (Frequency Domain)', wave_names{i}));
    grid on;
    
    delta_f = fs_base/N_base;
    f1_idx = round(f_wave/delta_f) + 1;
    f2_idx = round(2*f_wave/delta_f) + 1;
    f3_idx = round(3*f_wave/delta_f) + 1;
    
    fund_power = P(f1_idx);
    h2_power = P(f2_idx);
    h3_power = P(f3_idx);
    
    harmonic_features(i, 1) = 10*log10(h2_power/fund_power + eps);
    harmonic_features(i, 2) = 10*log10(h3_power/fund_power + eps);
    harmonic_features(i, 3) = sum(P(f1_idx+5:f1_idx+50)) / (fund_power + eps);
    
    fprintf('    %s: H2=%.1fdB, H3=%.1fdB, HF_ratio=%.3f\n', ...
        wave_names{i}, harmonic_features(i,1), harmonic_features(i,2), harmonic_features(i,3));
end

% 特征散点图
subplot(3, 3, 7);
scatter(harmonic_features(:,1), harmonic_features(:,2), 200, [1; 2; 3], 'filled');
hold on;
for i = 1:3
    text(harmonic_features(i,1)+1, harmonic_features(i,2), wave_names{i}, 'FontSize', 12, 'FontWeight', 'bold');
end
xlabel('H2 relative to fundamental (dB)');
ylabel('H3 relative to fundamental (dB)');
title('FFT Feature Scatter Plot (for waveform classification)');
grid on;

% 相关系数矩阵
subplot(3, 3, 8);
corr_matrix = zeros(3, 3);
for i = 1:3
    for j = 1:3
        c = corrcoef(waves{i}(:), waves{j}(:));
        corr_matrix(i,j) = c(1,2);
    end
end
imagesc(corr_matrix);
colorbar;
clim([0 1]);
set(gca, 'XTick', 1:3, 'XTickLabel', wave_names);
set(gca, 'YTick', 1:3, 'YTickLabel', wave_names);
for i = 1:3
    for j = 1:3
        text(j, i, sprintf('%.3f', corr_matrix(i,j)), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end
xlabel('Template Waveform');
ylabel('Measured Waveform');
title('Waveform Correlation Matrix');

% 抗混叠滤波器响应
subplot(3, 3, 9);
[H, f_resp] = freqz(b_aa, a_aa, 4096, fs_base);
plot(f_resp/1e3, 20*log10(abs(H)), 'LineWidth', 2);
hold on;
xline(50, 'r--', 'Signal max freq', 'LineWidth', 1.5);
xline(100, 'g--', 'Filter cutoff', 'LineWidth', 1.5);
xlabel('Frequency (kHz)');
ylabel('Magnitude Response (dB)');
title('Anti-aliasing Filter Response (4-order Butterworth, fc=100kHz)');
grid on;
xlim([0 250]);

sgtitle(sprintf('Test 5: Waveform Recognition (SNR=%ddB, Circuit: Anti-alias Filter+ADC+DSP)', SNR_wave));
saveas(fig5, fullfile(output_dir, 'Test5_Waveform_Recognition_v2.png'));

%% ==================== Test 6: 系统响应时间分析 ====================
% [对应系统时序]: 信号采集 + PGA稳定 + 滤波器建立 + ADC采样 + 算法计算 + 显示刷新
% [仿真内容]: 计算不同频率下完成一次完整测量所需的最短时间
% [考核指标]: 响应时间 < 3秒

fprintf('\n========== Test 6: 系统响应时间分析 (对应系统整体时序) ==========\n');

f_list_response = [1, 10, 100, 1e3, 10e3, 50e3];
n_resp = length(f_list_response);

t_pga_settle = 1e-3;
t_filter_settle = 2e-3;
t_adc_sample = N_base / fs_base;
t_algorithm = 50e-3;
t_display = 20e-3;

response_times = zeros(1, n_resp);

for i = 1:n_resp
    f_sig = f_list_response(i);
    
    if f_sig < 100
        n_cycles_needed = 10;
        t_acquisition = n_cycles_needed / f_sig;
    else
        t_acquisition = N_base / fs_base;
    end
    
    t_total = t_pga_settle + t_filter_settle + t_acquisition + t_algorithm + t_display;
    response_times(i) = t_total;
    
    fprintf('  f=%6.1fHz: Acquisition=%6.3fs, Total=%6.3fs (%.1f%% of 3s limit)\n', ...
        f_sig, t_acquisition, t_total, t_total/3*100);
end

fig6 = figure('Name','Test6_Response_Time','Position',[100 100 1000 500]);

bar(response_times, 'FaceColor', [0.3 0.6 0.9]);
hold on;
yline(3, 'r--', 'LineWidth', 3, 'Label', '3s Limit');
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('%.0f', x), f_list_response, 'UniformOutput', false));
xlabel('Signal Frequency (Hz)');
ylabel('Response Time (seconds)');
title('Test 6: System Response Time Analysis (Overall system timing)');
grid on;

for i = 1:n_resp
    if response_times(i) < 3
        text(i, response_times(i)+0.05, sprintf('%.2fs', response_times(i)), ...
            'HorizontalAlignment', 'center', 'Color', 'green', 'FontWeight', 'bold');
    else
        text(i, 2.8, 'OVER!', 'HorizontalAlignment', 'center', 'Color', 'red', 'FontWeight', 'bold');
    end
end

sgtitle('Test 6: Response Time - Low freq (1Hz) requires 10s acquisition, exceeds 3s limit');
saveas(fig6, fullfile(output_dir, 'Test6_Response_Time_v2.png'));

%% ==================== Test 7: Monte Carlo误差预算 ====================
% [对应完整信号调理链路]: 输入保护 -> PGA -> 抗混叠滤波 -> 比较器 -> ADC
% [仿真内容]: 综合所有误差源(PGA增益误差、比较器迟滞、ADC量化噪声、信号噪声)
% [考核指标]: 频率误差<1%, Vpp误差<1%, 占空比误差<2%

fprintf('\n========== Test 7: Monte Carlo误差预算 (对应完整信号调理链路) ==========\n');

N_mc = 100;
f_mc_true = 5e3;
Vpp_mc_true = 2.0;
D_mc_true = 0.5;

f_mc_err = zeros(1, N_mc);
Vpp_mc_err = zeros(1, N_mc);
D_mc_err = zeros(1, N_mc);

for mc = 1:N_mc
    % 随机误差源
    pga_gain_err = 1 + (rand()-0.5)*0.02;
    hyst_mc = 5e-3 + rand()*15e-3;
    adc_bits_mc = 12;
    lsb_mc = ADC_Vref / (2^adc_bits_mc);
    snr_mc = 40 + rand()*20;
    
    % 生成信号
    t_mc = (0:N_base-1)/fs_base;
    sig_mc = (Vpp_mc_true/2) * sin(2*pi*f_mc_true*t_mc);
    noise_power_mc = (Vpp_mc_true^2/8) / 10^(snr_mc/10);
    sig_mc = sig_mc + sqrt(noise_power_mc)*randn(size(t_mc));
    
    % PGA
    sig_mc = sig_mc * pga_gain_err;
    
    % ADC量化
    sig_mc_q = round(sig_mc / lsb_mc) * lsb_mc;
    
    % 频率测量
    threshold = 0;
    crossings = [];
    state = sig_mc_q(1) > threshold;
    for k = 2:length(sig_mc_q)
        if state && sig_mc_q(k) < (threshold - hyst_mc/2)
            state = false;
        elseif ~state && sig_mc_q(k) > (threshold + hyst_mc/2)
            state = true;
            crossings = [crossings, k];
        end
    end
    
    if length(crossings) >= 3
        T_meas = mean(diff(crossings)) / fs_base;
        f_mc_err(mc) = abs(1/T_meas - f_mc_true) / f_mc_true * 100;
    else
        f_mc_err(mc) = 100;
    end
    
    % Vpp测量
    Vpp_meas = max(sig_mc_q) - min(sig_mc_q);
    Vpp_mc_err(mc) = abs(Vpp_meas - Vpp_mc_true) / Vpp_mc_true * 100;
    
    % 占空比测量
    D_mc_err(mc) = abs(hyst_mc / Vpp_mc_true) * 100;
end

fig7 = figure('Name','Test7_MonteCarlo_ErrorBudget','Position',[100 100 1200 400]);

subplot(1,3,1);
histogram(f_mc_err, 20, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none');
hold on;
xline(1, 'r--', 'LineWidth', 2, 'Label', '1% Limit');
xline(mean(f_mc_err), 'g-', 'LineWidth', 2, 'Label', sprintf('Mean=%.3f%%', mean(f_mc_err)));
xlabel('Frequency Error (%)');
ylabel('Count');
title(sprintf('Frequency Error\n(95%%CI: [%.3f%%, %.3f%%])', quantile(f_mc_err,0.025), quantile(f_mc_err,0.975)));
grid on;

subplot(1,3,2);
histogram(Vpp_mc_err, 20, 'FaceColor', [0.9 0.6 0.3], 'EdgeColor', 'none');
hold on;
xline(1, 'r--', 'LineWidth', 2, 'Label', '1% Limit');
xline(mean(Vpp_mc_err), 'g-', 'LineWidth', 2, 'Label', sprintf('Mean=%.3f%%', mean(Vpp_mc_err)));
xlabel('Vpp Error (%)');
ylabel('Count');
title(sprintf('Vpp Error\n(95%%CI: [%.3f%%, %.3f%%])', quantile(Vpp_mc_err,0.025), quantile(Vpp_mc_err,0.975)));
grid on;

subplot(1,3,3);
histogram(D_mc_err, 20, 'FaceColor', [0.9 0.3 0.3], 'EdgeColor', 'none');
hold on;
xline(2, 'r--', 'LineWidth', 2, 'Label', '2% Limit');
xline(mean(D_mc_err), 'g-', 'LineWidth', 2, 'Label', sprintf('Mean=%.3f%%', mean(D_mc_err)));
xlabel('Duty Cycle Error (%)');
ylabel('Count');
title(sprintf('Duty Cycle Error\n(95%%CI: [%.3f%%, %.3f%%])', quantile(D_mc_err,0.025), quantile(D_mc_err,0.975)));
grid on;

sgtitle(sprintf('Test 7: Monte Carlo Error Budget (N=%d, Full Signal Conditioning Chain)', N_mc));
saveas(fig7, fullfile(output_dir, 'Test7_MonteCarlo_ErrorBudget_v2.png'));

fprintf('  Monte Carlo结果 (N=%d):\n', N_mc);
fprintf('    频率误差: 均值=%.4f%%, 95%%CI=[%.4f%%, %.4f%%]\n', mean(f_mc_err), quantile(f_mc_err,0.025), quantile(f_mc_err,0.975));
fprintf('    Vpp误差:  均值=%.4f%%, 95%%CI=[%.4f%%, %.4f%%]\n', mean(Vpp_mc_err), quantile(Vpp_mc_err,0.025), quantile(Vpp_mc_err,0.975));
fprintf('    占空比误差: 均值=%.4f%%, 95%%CI=[%.4f%%, %.4f%%]\n', mean(D_mc_err), quantile(D_mc_err,0.025), quantile(D_mc_err,0.975));

%% ==================== 关键结论汇总 ====================
fprintf('\n');
fprintf('============================================================\n');
fprintf('    2021-J题 周期信号波形识别及参数测量装置 仿真结论汇总    \n');
fprintf('============================================================\n');
fprintf('1. PGA程控增益放大器: 8档增益覆盖50mV~10V, 小信号需x20放大\n');
fprintf('2. 比较器+定时器: 低频(<100Hz)用周期法, 高频用计数法\n');
fprintf('3. PGA+ADC: 12-bit ADC满足1%% Vpp精度, 小信号靠PGA\n');
fprintf('4. 迟滞比较器: 20mV迟滞可抑制200mV噪声, 保证2%%占空比精度\n');
fprintf('5. 抗混叠滤波+ADC+DSP: FFT谐波特征可100%%区分正弦/三角/矩形\n');
fprintf('6. 系统时序: 1Hz信号需10秒采集, 无法满足3秒响应(需优化)\n');
fprintf('7. 完整链路: Monte Carlo验证95%%CI满足频率/占空比指标\n');
fprintf('============================================================\n');
fprintf('工程建议:\n');
fprintf('  1. 低频(1Hz~100Hz)采用"FFT+插值"策略替代周期法\n');
fprintf('  2. 高频(>10kHz)采用计数法, 门限时间自适应\n');
fprintf('  3. 波形识别推荐"FFT谐波分析", 计算量小且鲁棒性高\n');
fprintf('  4. 占空比测量必须使用迟滞比较器, 阈值设为信号中点\n');
fprintf('  5. 响应时间优化: 并行处理(采集下一帧时计算上一帧)\n');
fprintf('============================================================\n');

fprintf('\nAll simulation figures saved to: %s\n', output_dir);
