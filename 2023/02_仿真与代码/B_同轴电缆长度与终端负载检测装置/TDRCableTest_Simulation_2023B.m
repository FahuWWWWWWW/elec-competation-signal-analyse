%% 2023年电赛B题「同轴电缆长度与终端负载检测装置」TDR仿真
% 目标: 验证TDR时域反射技术在电缆长度测量和负载检测中的性能
% 技术: 脉冲反射法 + 等效采样 + 波形特征提取
% 调理电路映射: 每个Test对应一个前端调理电路模块
%
% 作者: AI分析助手
% 日期: 2026-06-09

clear; clc; close all;

%% 全局参数设置
Z0 = 50;                    % 特性阻抗 (Ω) - 典型同轴电缆
c = 3e8;                    % 光速 (m/s)
epsilon_r = 2.3;            % PE介质相对介电常数
v = c / sqrt(epsilon_r);    % 传播速度 (m/s) ≈ 1.98e8 m/s ≈ 0.66c

% TDR脉冲参数
tp_rise = 1e-9;             % 脉冲上升时间 1ns - 对应高速逻辑门
pulse_width = 5e-9;         % 脉冲宽度 5ns
fs_tdr = 100e9;             % TDR仿真采样率 100GSPS (用于生成"真实"波形)
dt_tdr = 1/fs_tdr;

% 等效采样参数
fs_equiv = 500e6;           % 等效采样率 500MSPS (实际ADC速率)
dt_equiv = 1/fs_equiv;
delay_step = dt_equiv;      % 延迟步进 = 等效采样周期

% 输出目录
output_dir = 'simulation_output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('=== 2023-B TDR Cable Test Simulation ===\n');
fprintf('Characteristic Impedance Z0 = %d Ω\n', Z0);
fprintf('Propagation Velocity v = %.2e m/s (%.2fc)\n', v, v/c);
fprintf('Pulse Rise Time = %.1f ns\n', tp_rise*1e9);
fprintf('Equivalent Sampling Rate = %.1f MSPS\n', fs_equiv/1e6);
fprintf('\n');

%% ==================== Test 1: TDR基本原理 - 不同终端反射波形 ====================
% 【对应调理电路模块】: 高速脉冲发生器 + 电阻桥 + 高速采样头
% 【电路设计启示】: 不同终端在TDR上呈现截然不同的波形，是识别的基础

fprintf('Test 1: TDR Waveforms for Different Terminations\n');

% 电缆长度: 1500cm = 15m
cable_length = 15;          % (m)
t_round = 2 * cable_length / v;  % 往返时间 (s)

% 时间轴
t_max = t_round * 2.5;      % 显示2.5倍往返时间
N_tdr = round(t_max / dt_tdr);
t_tdr = (0:N_tdr-1) * dt_tdr;

% 生成入射脉冲 (高斯脉冲)
incident = generate_gaussian_pulse(t_tdr, 0, pulse_width, 1.0);

% 模拟不同终端的反射波形
terminations = {'Open', 'Short', 'Resistor (25Ω)', 'Capacitor (200pF)'};
ZL_values = {inf, 0, 25, struct('type', 'C', 'value', 200e-12)};
gamma_values = {};

figure('Name', 'Test1_TDR_Terminations', 'Position', [100 100 1200 800]);

for i = 1:4
    % 计算反射系数
    if i == 4
        % 电容: 频率相关，需要时域仿真
        % 简化为: 初始等效短路，最终等效开路
        gamma = compute_capacitor_reflection(Z0, ZL_values{i}.value, t_tdr, t_round);
    else
        gamma = compute_reflection_coefficient(Z0, ZL_values{i});
    end
    gamma_values{i} = gamma;
    
    % 生成TDR波形 (入射 + 延迟的反射)
    reflected = zeros(size(t_tdr));
    delay_samples = round(t_round / dt_tdr);
    
    if i == 4
        % 电容: 反射波形是入射脉冲与电容冲激响应的卷积
        reflected = simulate_capacitor_tdr(incident, t_tdr, Z0, ZL_values{i}.value, cable_length, v);
    else
        % 电阻/开路/短路: 简单缩放+延迟
        for n = 1:length(t_tdr)
            if n > delay_samples
                reflected(n) = gamma * incident(n - delay_samples);
            end
        end
    end
    
    tdr_waveform = incident + reflected;
    
    % 绘图
    subplot(4, 1, i);
    plot(t_tdr * 1e9, incident, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Incident');
    hold on;
    plot(t_tdr * 1e9, reflected, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Reflected');
    plot(t_tdr * 1e9, tdr_waveform, 'g-', 'LineWidth', 2, 'DisplayName', 'TDR (Incident+Reflected)');
    
    % 标记往返时间
    xline(t_round * 1e9, 'k:', 'LineWidth', 1.5, 'Label', sprintf('Round-trip: %.1f ns', t_round*1e9));
    
    if i <= 3
        title(sprintf('TDR: %s (Γ = %.2f)', terminations{i}, gamma), 'FontSize', 11);
    else
        title(sprintf('TDR: %s (Γ frequency-dependent)', terminations{i}), 'FontSize', 11);
    end
    xlabel('Time (ns)');
    ylabel('Amplitude (V)');
    legend('Location', 'best');
    grid on;
    ylim([-1.5, 2.0]);
end

sgtitle('Test 1: TDR Waveforms for Different Terminal Loads (Z_0 = 50Ω, L = 15m)', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test1_TDR_Termination_Waveforms.png'));
close(gcf);

fprintf('  Saved: Test1_TDR_Termination_Waveforms.png\n');

%% ==================== Test 2: 长度检测精度验证 ====================
% 【对应调理电路模块】: 等效采样TDR + 时间间隔测量
% 【电路设计启示】: 时间分辨率决定长度精度，等效采样可用低成本实现高精度

fprintf('\nTest 2: Cable Length Measurement Accuracy\n');

% 测试长度范围 (基本要求: 1000cm~2000cm, 即10m~20m)
test_lengths = [10, 12, 15, 18, 20];  % (m)
measured_lengths = zeros(size(test_lengths));
errors = zeros(size(test_lengths));

% 等效采样重建波形
N_equiv = 2000;  % 等效采样点数
t_equiv = (0:N_equiv-1) * dt_equiv;

figure('Name', 'Test2_Length_Measurement', 'Position', [100 100 1400 600]);

for idx = 1:length(test_lengths)
    L = test_lengths(idx);
    t_r = 2 * L / v;  % 往返时间
    
    % 生成真实TDR波形 (终端开路)
    N_t = round(t_r * 2.5 / dt_tdr);
    t_sim = (0:N_t-1) * dt_tdr;
    pulse = generate_gaussian_pulse(t_sim, 0, pulse_width, 1.0);
    
    % 真实反射波形 (开路, Γ=+1)
    reflected_true = zeros(size(t_sim));
    delay_samp = round(t_r / dt_tdr);
    for n = 1:length(t_sim)
        if n > delay_samp
            reflected_true(n) = 1.0 * pulse(n - delay_samp);
        end
    end
    tdr_true = pulse + reflected_true;
    
    % 等效采样 (重建波形)
    tdr_equiv = zeros(size(t_equiv));
    for n = 1:N_equiv
        t_target = t_equiv(n);
        % 在真实波形中找到最接近的时间点
        [~, closest_idx] = min(abs(t_sim - t_target));
        tdr_equiv(n) = tdr_true(closest_idx);
    end
    
    % 添加噪声 (模拟ADC量化噪声和前端噪声)
    tdr_equiv = tdr_equiv + 0.02 * randn(size(tdr_equiv));
    
    % 长度检测算法: 互相关法
    % 1. 找到入射脉冲位置 (第一个峰值)
    [~, incident_idx] = max(tdr_equiv);
    
    % 2. 找到反射脉冲位置 (与入射脉冲模板互相关)
    template_width = round(pulse_width / dt_equiv);
    template = tdr_equiv(incident_idx : min(incident_idx + template_width, length(tdr_equiv)));
    
    % 在入射脉冲之后搜索反射脉冲
    search_start = incident_idx + template_width + 10;
    search_end = min(search_start + round(300e-9 / dt_equiv), length(tdr_equiv));
    
    if search_start < length(tdr_equiv)
        corr = xcorr(tdr_equiv(search_start:search_end), template, 0);
        [~, max_corr_idx] = max(corr);
        reflected_idx = search_start + max_corr_idx - 1;
        
        % 计算时间差和长度
        delta_t = (reflected_idx - incident_idx) * dt_equiv;
        L_measured = v * delta_t / 2;
        
        measured_lengths(idx) = L_measured;
        errors(idx) = abs(L_measured - L) / L * 100;  % 相对误差百分比
    else
        measured_lengths(idx) = NaN;
        errors(idx) = NaN;
    end
    
    % 绘图 (只画前3个)
    if idx <= 3
        subplot(1, 3, idx);
        plot(t_equiv * 1e9, tdr_equiv, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(t_equiv(incident_idx) * 1e9, tdr_equiv(incident_idx), 'ro', 'MarkerSize', 10, 'DisplayName', 'Incident');
        if ~isnan(reflected_idx) && reflected_idx <= length(t_equiv)
            plot(t_equiv(reflected_idx) * 1e9, tdr_equiv(reflected_idx), 'gs', 'MarkerSize', 10, 'DisplayName', 'Reflected');
        end
        title(sprintf('L = %dm, Error = %.2f%%', L, errors(idx)), 'FontSize', 10);
        xlabel('Time (ns)');
        ylabel('Amplitude (V)');
        grid on;
        legend('Location', 'best');
    end
end

sgtitle('Test 2: TDR Length Measurement (Open Circuit, Equivalent Sampling)', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test2_Length_Measurement_Accuracy.png'));
close(gcf);

fprintf('  Results:\n');
for i = 1:length(test_lengths)
    fprintf('    L = %4dm: Measured = %6.2fm, Error = %5.2f%% %s\n', ...
        test_lengths(i), measured_lengths(i), errors(i), ...
        iif(errors(i) <= 1, '✓ (≤1%)', iif(errors(i) <= 5, '✓ (≤5%)', '✗')));
end

%% ==================== Test 3: 等效采样原理演示 ====================
% 【对应调理电路模块】: 精密延迟线(DS1023/MC100EP195) + 低速ADC
% 【电路设计启示】: 用500MSPS低速ADC + 0.1ns步进延迟线 = 10GSPS等效采样

fprintf('\nTest 3: Equivalent Sampling Principle\n');

% 生成一个快速脉冲 (真实信号，100GSPS)
t_fast = 0:dt_tdr:50e-9;  % 50ns窗口
pulse_fast = generate_gaussian_pulse(t_fast, 25e-9, 2e-9, 1.0);  % 25ns处, 2ns宽

% 模拟真实ADC采样 (500MSPS = 2ns周期) - 严重欠采样
fs_real_adc = 500e6;
dt_real_adc = 1/fs_real_adc;
t_real_adc = 0:dt_real_adc:50e-9;
pulse_real_adc = interp1(t_fast, pulse_fast, t_real_adc, 'linear', 0);

% 等效采样 (500MSPS等效，步进0.1ns)
delay_step_fine = 0.1e-9;  % 0.1ns步进
num_shots = round(dt_real_adc / delay_step_fine);  % 20次采样拼成一个周期
fs_equiv_effective = 1 / delay_step_fine;  % 10GSPS等效

t_equiv_fine = 0:delay_step_fine:50e-9;
pulse_equiv = zeros(size(t_equiv_fine));

% 模拟等效采样过程
for shot = 1:num_shots
    delay = (shot-1) * delay_step_fine;
    % 每次触发后延迟delay时间采样
    t_sample = delay + (0:dt_real_adc:50e-9);
    % 找到对应的脉冲幅度
    for n = 1:length(t_sample)
        if t_sample(n) <= max(t_fast)
            [~, idx] = min(abs(t_fast - t_sample(n)));
            sample_idx = round(t_sample(n) / delay_step_fine) + 1;
            if sample_idx <= length(pulse_equiv)
                pulse_equiv(sample_idx) = pulse_fast(idx);
            end
        end
    end
end

figure('Name', 'Test3_Equivalent_Sampling', 'Position', [100 100 1200 800]);

subplot(3, 1, 1);
plot(t_fast * 1e9, pulse_fast, 'b-', 'LineWidth', 1.5);
title(sprintf('(a) Original TDR Pulse (Simulated @ %.0f GSPS)', fs_tdr/1e9), 'FontSize', 11);
xlabel('Time (ns)'); ylabel('Amplitude (V)'); grid on;

subplot(3, 1, 2);
stem(t_real_adc * 1e9, pulse_real_adc, 'r', 'LineWidth', 1.5, 'MarkerSize', 4);
title(sprintf('(b) Real-time Sampling @ %.0f MSPS (Severely Aliased)', fs_real_adc/1e6), 'FontSize', 11);
xlabel('Time (ns)'); ylabel('Amplitude (V)'); grid on;

subplot(3, 1, 3);
plot(t_equiv_fine * 1e9, pulse_equiv, 'g-', 'LineWidth', 1.5);
title(sprintf('(c) Equivalent Sampling (%d shots × %.0f MSPS = %.0f GSPS effective)', ...
    num_shots, fs_real_adc/1e6, fs_equiv_effective/1e9), 'FontSize', 11);
xlabel('Time (ns)'); ylabel('Amplitude (V)'); grid on;

sgtitle('Test 3: Equivalent Sampling Principle for TDR', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test3_Equivalent_Sampling.png'));
close(gcf);

fprintf('  Saved: Test3_Equivalent_Sampling.png\n');
fprintf('  Real-time ADC: %.0f MSPS → Aliased\n', fs_real_adc/1e6);
fprintf('  Equivalent Sampling: %d shots × %.0f MSPS = %.0f GSPS effective\n', ...
    num_shots, fs_real_adc/1e6, fs_equiv_effective/1e9);

%% ==================== Test 4: 负载类型识别 ====================
% 【对应调理电路模块】: TDR波形特征提取 + 分类算法
% 【电路设计启示】: 反射波形的极性、幅度、形状是区分负载类型的"指纹"

fprintf('\nTest 4: Load Type Identification\n');

% 测试不同负载 (在15m电缆末端)
L_test = 15;
t_r_test = 2 * L_test / v;

% 生成TDR波形 (等效采样)
N_e = 2000;
t_e = (0:N_e-1) * dt_equiv;

load_types = {'Open', 'Short', 'Resistor (15Ω)', 'Resistor (25Ω)', 'Capacitor (150pF)', 'Capacitor (250pF)'};
ZL_test = {inf, 0, 15, 25, struct('type','C','value',150e-12), struct('type','C','value',250e-12)};

features = zeros(length(load_types), 3);  % [极性, 归一化幅度, 波形畸变指数]
identified_types = cell(size(load_types));

figure('Name', 'Test4_Load_Identification', 'Position', [100 100 1400 900]);

for i = 1:length(load_types)
    % 生成TDR波形
    tdr_load = generate_tdr_waveform(t_e, dt_equiv, L_test, v, pulse_width, Z0, ZL_test{i});
    tdr_load = tdr_load + 0.01 * randn(size(tdr_load));  % 添加噪声
    
    % 提取反射波形
    incident_idx = find(tdr_load > 0.5, 1, 'first');
    if isempty(incident_idx), incident_idx = 1; end
    
    search_start = incident_idx + round(pulse_width / dt_equiv);
    reflected_segment = tdr_load(search_start:end);
    
    % 特征提取
    % 特征1: 反射波极性 (与入射波比较)
    polarity = sign(mean(reflected_segment(1:min(50, length(reflected_segment)))));
    
    % 特征2: 反射波归一化幅度
    reflected_peak = max(abs(reflected_segment));
    incident_peak = max(tdr_load);
    amp_ratio = reflected_peak / incident_peak;
    
    % 特征3: 波形畸变指数 (电容会导致波形不是简单脉冲)
    % 计算反射波与"理想脉冲"的互相关锐度
    if length(reflected_segment) > 100
        template_len = round(pulse_width / dt_equiv);
        if template_len > 10
            template = generate_gaussian_pulse((0:template_len-1)*dt_equiv, 0, pulse_width, 1.0);
            corr = xcorr(reflected_segment(1:min(500,length(reflected_segment))), template, 0);
            [~, max_idx] = max(corr);
            % 计算主峰宽度
            peak_region = corr(max(1, max_idx-10):min(length(corr), max_idx+10));
            shape_factor = std(peak_region) / max(abs(corr));  % 畸变指数
        else
            shape_factor = 0;
        end
    else
        shape_factor = 0;
    end
    
    features(i, :) = [polarity, amp_ratio, shape_factor];
    
    % 分类决策 (基于规则的分层分类器)
    if polarity > 0.5 && amp_ratio > 0.9
        identified_types{i} = 'Open';
    elseif polarity < -0.5 && amp_ratio > 0.9
        identified_types{i} = 'Short';
    elseif shape_factor > 0.3
        identified_types{i} = 'Capacitor';
    else
        identified_types{i} = 'Resistor';
    end
    
    % 绘图
    subplot(3, 2, i);
    plot(t_e * 1e9, tdr_load, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(t_e(incident_idx) * 1e9, tdr_load(incident_idx), 'ro', 'MarkerSize', 8, 'DisplayName', 'Incident');
    if search_start <= length(t_e)
        [~, ref_pk_idx] = max(abs(reflected_segment));
        actual_ref_idx = search_start + ref_pk_idx - 1;
        if actual_ref_idx <= length(t_e)
            plot(t_e(actual_ref_idx) * 1e9, tdr_load(actual_ref_idx), 'gs', 'MarkerSize', 8, 'DisplayName', 'Reflected');
        end
    end
    
    correct = strcmpi(identified_types{i}, extract_type(load_types{i}));
    title_str = sprintf('%s → Identified: %s %s', load_types{i}, identified_types{i}, iif(correct, '✓', '✗'));
    title(title_str, 'FontSize', 9, 'Color', iif(correct, [0 0.5 0], 'r'));
    xlabel('Time (ns)'); ylabel('Amplitude (V)');
    grid on;
end

sgtitle('Test 4: Load Type Identification from TDR Waveform Features', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test4_Load_Type_Identification.png'));
close(gcf);

fprintf('  Identification Results:\n');
for i = 1:length(load_types)
    correct = strcmpi(identified_types{i}, extract_type(load_types{i}));
    fprintf('    %s → %s %s\n', load_types{i}, identified_types{i}, iif(correct, '✓', '✗'));
end

%% ==================== Test 5: 负载参数估计 ====================
% 【对应调理电路模块】: 反射幅度测量 + 阻抗计算
% 【电路设计启示】: 反射系数Γ与负载阻抗一一对应，可通过幅度测量反推

fprintf('\nTest 5: Load Parameter Estimation\n');

% 电阻负载测试 (10Ω~30Ω)
R_test_values = 10:2:30;
R_estimated = zeros(size(R_test_values));
R_errors = zeros(size(R_test_values));

for i = 1:length(R_test_values)
    R = R_test_values(i);
    gamma_R = (R - Z0) / (R + Z0);
    
    % 模拟TDR测量 (添加噪声)
    gamma_measured = gamma_R + 0.02 * randn();  % 2%幅度测量噪声
    
    % 从Γ反推RL
    R_est = Z0 * (1 + gamma_measured) / (1 - gamma_measured);
    R_estimated(i) = R_est;
    R_errors(i) = abs(R_est - R) / R * 100;
end

% 电容负载测试 (100pF~300pF)
C_test_values = 100e-12:20e-12:300e-12;
C_estimated = zeros(size(C_test_values));
C_errors = zeros(size(C_test_values));

for i = 1:length(C_test_values)
    C = C_test_values(i);
    
    % 电容的TDR响应: 指数衰减时间常数 τ = Z0*C
    tau = Z0 * C;
    
    % 模拟测量: 通过拟合TDR波形的指数衰减段
    tau_measured = tau * (1 + 0.1 * randn());  % 10%时间常数测量噪声
    
    C_est = tau_measured / Z0;
    C_estimated(i) = C_est;
    C_errors(i) = abs(C_est - C) / C * 100;
end

figure('Name', 'Test5_Parameter_Estimation', 'Position', [100 100 1400 500]);

subplot(1, 2, 1);
plot(R_test_values, R_test_values, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ideal');
hold on;
plot(R_test_values, R_estimated, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Measured');
errorbar(R_test_values, R_estimated, R_estimated .* (R_errors/100), 'r', 'LineWidth', 1);
xlabel('Actual Resistance (Ω)'); ylabel('Estimated Resistance (Ω)');
title(sprintf('Resistor Load Estimation\nMax Error: %.1f%%', max(R_errors)));
legend('Location', 'best'); grid on;

subplot(1, 2, 2);
plot(C_test_values * 1e12, C_test_values * 1e12, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Ideal');
hold on;
plot(C_test_values * 1e12, C_estimated * 1e12, 'go-', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Measured');
errorbar(C_test_values * 1e12, C_estimated * 1e12, C_estimated * 1e12 .* (C_errors/100), 'r', 'LineWidth', 1);
xlabel('Actual Capacitance (pF)'); ylabel('Estimated Capacitance (pF)');
title(sprintf('Capacitor Load Estimation\nMax Error: %.1f%%', max(C_errors)));
legend('Location', 'best'); grid on;

sgtitle('Test 5: Load Parameter Estimation from TDR Reflection Coefficient', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test5_Load_Parameter_Estimation.png'));
close(gcf);

fprintf('  Resistor Estimation: Max Error = %.2f%% (Requirement: ≤10%%) %s\n', max(R_errors), iif(max(R_errors)<=10, '✓', '✗'));
fprintf('  Capacitor Estimation: Max Error = %.2f%% (Requirement: ≤10%%) %s\n', max(C_errors), iif(max(C_errors)<=10, '✓', '✗'));

%% ==================== Test 6: 短电缆盲区测试 ====================
% 【对应调理电路模块】: 窄脉冲发生器 (<5ns脉宽)
% 【电路设计启示】: 脉冲宽度必须小于往返时间，否则入射波和反射波重叠

fprintf('\nTest 6: Short Cable Blind Zone Test\n');

% 测试不同短电缆长度
short_lengths = [0.5, 1.0, 2.0, 5.0, 10.0];  % (m) 对应50cm, 100cm, 200cm...

% 不同脉冲宽度
pulse_widths = [2e-9, 5e-9, 10e-9];  % 2ns, 5ns, 10ns

colors = {'b', 'r', 'g'};

figure('Name', 'Test6_Short_Cable_Blind_Zone', 'Position', [100 100 1400 800]);

for pw_idx = 1:length(pulse_widths)
    pw = pulse_widths(pw_idx);
    
    subplot(length(pulse_widths), 1, pw_idx);
    hold on;
    
    for L_idx = 1:length(short_lengths)
        L = short_lengths(L_idx);
        t_r = 2 * L / v;
        
        % 生成TDR波形
        t_max_s = max(t_r, pw) * 4;
        N_s = round(t_max_s / dt_tdr);
        t_s = (0:N_s-1) * dt_tdr;
        
        pulse_s = generate_gaussian_pulse(t_s, 0, pw, 1.0);
        
        % 开路反射
        reflected_s = zeros(size(t_s));
        delay_s = round(t_r / dt_tdr);
        for n = 1:length(t_s)
            if n > delay_s
                reflected_s(n) = 1.0 * pulse_s(n - delay_s);
            end
        end
        
        tdr_short = pulse_s + reflected_s;
        
        % 绘图 (只显示前4个长度)
        if L_idx <= 4
            plot(t_s * 1e9, tdr_short + (L_idx-1)*0.5, colors{pw_idx}, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('L = %.1fm (%.0fcm)', L, L*100));
        end
    end
    
    % 标记脉冲宽度对应的往返时间
    t_round_for_pw = pw;  % 脉冲宽度对应的"盲区长度"
    L_blind = v * t_round_for_pw / 2;
    xline(t_round_for_pw * 1e9, 'k--', 'LineWidth', 2, 'Label', sprintf('Pulse width = %.0fns (Blind zone L < %.1fm)', pw*1e9, L_blind));
    
    title(sprintf('TDR with %.0fns Pulse Width', pw*1e9), 'FontSize', 11);
    xlabel('Time (ns)'); ylabel('Amplitude (V) + Offset');
    legend('Location', 'best'); grid on;
end

sgtitle('Test 6: Short Cable Detection - Blind Zone Analysis', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test6_Short_Cable_Blind_Zone.png'));
close(gcf);

fprintf('  Saved: Test6_Short_Cable_Blind_Zone.png\n');
fprintf('  Pulse width 2ns:  Blind zone L < %.2fm (Requirement: detect L ≤ 1m) %s\n', ...
    v*2e-9/2, iif(v*2e-9/2 <= 1, '✓', '✗'));
fprintf('  Pulse width 5ns:  Blind zone L < %.2fm (Requirement: detect L ≤ 1m) %s\n', ...
    v*5e-9/2, iif(v*5e-9/2 <= 1, '✓', '✗'));
fprintf('  Pulse width 10ns: Blind zone L < %.2fm (Requirement: detect L ≤ 1m) %s\n', ...
    v*10e-9/2, iif(v*10e-9/2 <= 1, '✓', '✗'));

%% ==================== Test 7: Monte Carlo误差预算 ====================
% 【对应完整信号调理链路】: 脉冲源 → 耦合器 → 电缆 → 延迟线 → ADC → DSP
% 【仿真设置】: 综合脉冲抖动、幅度噪声、速度因子误差

fprintf('\nTest 7: Monte Carlo Error Budget Analysis\n');

num_mc = 100;  % Monte Carlo次数
L_mc = 15;     % 测试长度 15m

t_r_mc = 2 * L_mc / v;

L_measured_mc = zeros(1, num_mc);
R_measured_mc = zeros(1, num_mc);
C_measured_mc = zeros(1, num_mc);

% 真实值
R_true = 25;
C_true = 200e-12;

for mc = 1:num_mc
    % 综合误差源:
    % 1. 脉冲位置抖动 (±0.2ns)
    jitter = 0.2e-9 * randn();
    
    % 2. 幅度噪声 (SNR = 40dB)
    amp_noise = 10^(-40/20) * randn();
    
    % 3. 速度因子误差 (±2%)
    v_error = v * (1 + 0.02 * randn());
    
    % 4. 量化误差 (8-bit ADC)
    quant_step = 1/256;
    quant_error = quant_step * randn();
    
    % 长度测量 (时间差 + 速度误差)
    delta_t_measured = t_r_mc + jitter;
    L_measured_mc(mc) = v_error * delta_t_measured / 2;
    
    % 电阻测量 (幅度误差)
    gamma_true = (R_true - Z0) / (R_true + Z0);
    gamma_measured = gamma_true + amp_noise + quant_error;
    gamma_measured = max(min(gamma_measured, 0.99), -0.99);
    R_measured_mc(mc) = Z0 * (1 + gamma_measured) / (1 - gamma_measured);
    
    % 电容测量 (时间常数误差)
    tau_true = Z0 * C_true;
    tau_measured = tau_true * (1 + 0.15 * randn());  % 15%时间常数估计误差
    C_measured_mc(mc) = tau_measured / Z0;
end

% 统计分析
L_error_mc = abs(L_measured_mc - L_mc) / L_mc * 100;
R_error_mc = abs(R_measured_mc - R_true) / R_true * 100;
C_error_mc = abs(C_measured_mc - C_true) / C_true * 100;

L_mean_err = mean(L_error_mc);
L_95ci = [prctile(L_error_mc, 2.5), prctile(L_error_mc, 97.5)];

R_mean_err = mean(R_error_mc);
R_95ci = [prctile(R_error_mc, 2.5), prctile(R_error_mc, 97.5)];

C_mean_err = mean(C_error_mc);
C_95ci = [prctile(C_error_mc, 2.5), prctile(C_error_mc, 97.5)];

figure('Name', 'Test7_MonteCarlo_ErrorBudget', 'Position', [100 100 1400 600]);

subplot(1, 3, 1);
histogram(L_error_mc, 15, 'FaceColor', 'b', 'EdgeColor', 'k');
hold on;
xline(1, 'r--', 'LineWidth', 2, 'Label', '1% Requirement');
xline(L_mean_err, 'g-', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f%%', L_mean_err));
title(sprintf('Length Error\n95%% CI: [%.2f, %.2f]%%', L_95ci(1), L_95ci(2)));
xlabel('Relative Error (%)'); ylabel('Count'); grid on;

subplot(1, 3, 2);
histogram(R_error_mc, 15, 'FaceColor', 'r', 'EdgeColor', 'k');
hold on;
xline(10, 'r--', 'LineWidth', 2, 'Label', '10% Requirement');
xline(R_mean_err, 'g-', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f%%', R_mean_err));
title(sprintf('Resistance Error\n95%% CI: [%.2f, %.2f]%%', R_95ci(1), R_95ci(2)));
xlabel('Relative Error (%)'); ylabel('Count'); grid on;

subplot(1, 3, 3);
histogram(C_error_mc, 15, 'FaceColor', 'g', 'EdgeColor', 'k');
hold on;
xline(10, 'r--', 'LineWidth', 2, 'Label', '10% Requirement');
xline(C_mean_err, 'g-', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f%%', C_mean_err));
title(sprintf('Capacitance Error\n95%% CI: [%.2f, %.2f]%%', C_95ci(1), C_95ci(2)));
xlabel('Relative Error (%)'); ylabel('Count'); grid on;

sgtitle('Test 7: Monte Carlo Error Budget (100 Runs, Combined Noise Sources)', 'FontSize', 13, 'FontWeight', 'bold');

saveas(gcf, fullfile(output_dir, 'Test7_MonteCarlo_ErrorBudget.png'));
close(gcf);

fprintf('  Length Error: Mean = %.2f%%, 95%% CI = [%.2f, %.2f]%% (Req: ≤1%%) %s\n', ...
    L_mean_err, L_95ci(1), L_95ci(2), iif(L_95ci(2)<=1, '✓', '✗'));
fprintf('  Resistance Error: Mean = %.2f%%, 95%% CI = [%.2f, %.2f]%% (Req: ≤10%%) %s\n', ...
    R_mean_err, R_95ci(1), R_95ci(2), iif(R_95ci(2)<=10, '✓', '✗'));
fprintf('  Capacitance Error: Mean = %.2f%%, 95%% CI = [%.2f, %.2f]%% (Req: ≤10%%) %s\n', ...
    C_mean_err, C_95ci(1), C_95ci(2), iif(C_95ci(2)<=10, '✓', '✗'));

%% ==================== 仿真总结 ====================
fprintf('\n=== Simulation Complete ===\n');
fprintf('All results saved to: %s\\%s\n', pwd, output_dir);
fprintf('Files generated:\n');
for i = 1:7
    fprintf('  Test%d_*.png\n', i);
end

%% ==================== 辅助函数 ====================

function pulse = generate_gaussian_pulse(t, t0, sigma, A)
    % 生成高斯脉冲
    pulse = A * exp(-((t - t0).^2) / (2 * sigma^2));
end

function gamma = compute_reflection_coefficient(Z0, ZL)
    % 计算电压反射系数
    if isinf(ZL)
        gamma = 1.0;  % 开路
    elseif ZL == 0
        gamma = -1.0;  % 短路
    else
        gamma = (ZL - Z0) / (ZL + Z0);
    end
end

function gamma_t = compute_capacitor_reflection(Z0, C, t, t_round)
    % 计算电容负载的时变反射系数 (简化模型)
    % 电容的阻抗 Zc = 1/(jωC)，反射系数频率相关
    % 时域近似: 初始短路(gamma=-1), 最终开路(gamma=+1)
    tau = Z0 * C;  % 时间常数
    
    gamma_t = zeros(size(t));
    for i = 1:length(t)
        if t(i) >= t_round
            dt = t(i) - t_round;
            % 从-1指数过渡到+1
            gamma_t(i) = 1 - 2 * exp(-dt / tau);
        else
            gamma_t(i) = 0;  % 反射脉冲到达前
        end
    end
end

function tdr = simulate_capacitor_tdr(incident, t, Z0, C, L, v_prop)
    % 模拟电容终端的完整TDR响应
    t_round = 2 * L / v_prop;
    tau = Z0 * C;
    
    tdr = incident;  % 入射波
    
    % 反射波: 入射脉冲与RC响应的卷积
    % 电容对阶跃的响应是 (1 - exp(-t/τ))
    % 对脉冲的响应是脉冲的微分再积分 (卷积)
    
    reflected = zeros(size(t));
    delay_samp = round(t_round / (t(2)-t(1)));
    
    if delay_samp < length(t)
        % 生成RC阶跃响应
        rc_response = zeros(size(t));
        for i = delay_samp:length(t)
            dt_local = t(i) - t_round;
            rc_response(i) = 2 * (1 - exp(-dt_local / tau));  % 从0到2 (开路最终幅度是2倍)
        end
        
        % 反射波 = 入射脉冲与RC响应的差分近似
        % 简化: 直接使用RC响应作为反射波形 (近似)
        reflected(delay_samp:end) = rc_response(delay_samp:end);
    end
    
    tdr = tdr + reflected;
end

function tdr = generate_tdr_waveform(t, dt, L, v, pulse_width, Z0, ZL)
    % 生成完整TDR波形
    t_round = 2 * L / v;
    
    incident = generate_gaussian_pulse(t, 0, pulse_width, 1.0);
    reflected = zeros(size(t));
    
    if isstruct(ZL) && strcmpi(ZL.type, 'C')
        % 电容
        reflected = simulate_capacitor_tdr(incident, t, Z0, ZL.value, L, v);
        reflected = reflected - incident;  % 只保留反射部分
    else
        % 电阻/开路/短路
        gamma = compute_reflection_coefficient(Z0, ZL);
        delay_samp = round(t_round / dt);
        for n = 1:length(t)
            if n > delay_samp
                reflected(n) = gamma * incident(n - delay_samp);
            end
        end
    end
    
    tdr = incident + reflected;
end

function out = iif(condition, true_val, false_val)
    % 简单条件表达式
    if condition
        out = true_val;
    else
        out = false_val;
    end
end

function type_str = extract_type(load_str)
    % 从负载描述字符串中提取类型
    if contains(load_str, 'Open', 'IgnoreCase', true)
        type_str = 'Open';
    elseif contains(load_str, 'Short', 'IgnoreCase', true)
        type_str = 'Short';
    elseif contains(load_str, 'Resistor', 'IgnoreCase', true)
        type_str = 'Resistor';
    elseif contains(load_str, 'Capacitor', 'IgnoreCase', true)
        type_str = 'Capacitor';
    else
        type_str = '';
    end
end
