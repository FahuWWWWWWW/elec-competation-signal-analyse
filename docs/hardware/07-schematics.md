# 2023-D 信号调制方式识别与参数估计装置 — 模块原理图

- **日期：** 2026-06-10
- **输出方式：** ASCII 框图 + 结构化连接表
- **目标 EDA：** 立创 EDA (EasyEDA)

---

## 模块 1：射频前端 (BNC → AD8331 VGA)

### ASCII 框图

```
  BNC           ESD          DC-BLOCK        50Ω Term
  J1 ───── PESD5V0S1UB ─── 100nF/50V ──┬── 50Ω ──┬── GND
                                        │         │
                                     AD8331.INH  AD8331.INL
                                        │
                                   AD8331.LNA
                                     增益 19dB
                                        │
                                   ┌────┴────┐
                                   │  X-AMP  │  VGA 48dB range
                                   │  VGA    │  Gain = 25mV/dB
                                   └────┬────┘
                                        │
                                   AD8331.VOH, VOL (差分输出)
                                        │
                                   到 混频器
```

### 连接表

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| RF_IN | BNC J1 中心针 | PESD5V0S1UB Pin1 | 50Ω 输入 |
| RF_IN | PESD5V0S1UB Pin2 | C1 (100nF/50V) | DC-block, X7R |
| RF_IN | C1 | R1 (49.9Ω 1% 0603) → GND | 50Ω 终端匹配 |
| RF_IN | C1 | AD8331 INH (Pin10) | LNA 正输入端 |
| GND_ANA | BNC J1 外壳 | GND 平面 | 接地 |
| GND_ANA | PESD5V0S1UB Pin3 | GND 平面 | ESD 泄放地 |
| LNA_INL | AD8331 INL (Pin11) | R2 (49.9Ω 1%) → GND | LNA 负输入端阻抗匹配 |
| LNA_BIAS | AD8331 RBIAS (Pin19) | R3 (10kΩ 1%) → GND | 偏置电流设置，参考手册 |
| VGA_GAIN | AD8331 GPOS (Pin7) | STM32 DAC1_OUT | AGC 控制电压 0~1.4V, 25mV/dB |
| VGA_GAIN_REF | AD8331 GNEG (Pin6) | GND_ANA | 单端增益控制 |
| VGA_OUTP | AD8331 VOH (Pin15) | 到 混频器 I/Q RF 输入 (SA612A-I Pin1) | VGA 差分正输出 |
| VGA_OUTN | AD8331 VOL (Pin14) | GND_ANA (单端驱动 SA612A) | VGA 差分负输出，接地 |
| VGA_5V | AD8331 VPOS (Pin17) | 5V_RAW (电源轨) | VGA 供电 |
| VGA_GND | AD8331 COMM (Pin12,13,18,20) | GND_ANA | 所有 GND 引脚共地 |
| VGA_BYP | AD8331 VPSI (Pin21/裸露焊盘) | GND_ANA | 芯片底部焊盘接地 |

### 关键参数
- **C1**: 100nF/50V X7R, 隔直电容, 高通截止 f_c = 1/(2π*50Ω*100nF) ≈ 32kHz < 100kHz ✓
- **R1**: 49.9Ω ±1%, 50Ω 终端匹配
- **R3**: 10kΩ ±1%, AD8331 LNA 偏置电流设置
- **增益控制电压**: STM32 DAC 输出 0~1.4V → AD8331 增益 -4.5~+55.5dB

---

## 模块 2：IQ 混频器 (SA612A × 2 + Si5351A LO)

### ASCII 框图

```
                    Si5351A
                 ┌────────────┐
                 │ OUT0 (CLK0)│─── LO_I (0°)  ─→ SA612A-I Pin6
                 │ OUT1 (CLK1)│─── LO_Q (90°) ─→ SA612A-Q Pin6
                 │ OUT2 (CLK2)│── (未用)
                 └────────────┘
                    I²C: SCL/SDA ─→ STM32

  VGA_OUTP ──┬── C2 ─┬── SA612A-I Pin1 (RF+)
             │       └── R4 ─→ GND
             │
             └── C3 ─┬── SA612A-Q Pin1 (RF+)
                     └── R5 ─→ GND

  SA612A-I Pin4 (OUT+) ─── R6 ─┬── C4 ──→ LPF_I (到模块3)
                               └── R7 ─→ GND

  SA612A-Q Pin4 (OUT+) ─── R8 ─┬── C5 ──→ LPF_Q (到模块3)
                               └── R9 ─→ GND
```

### I 路混频器连接表 (SA612A-I)

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| RF_I | SA612A-I Pin1 (RF+) | C2 (100pF) → VGA_OUTP | RF 输入, 隔直 |
| RF_BIAS_I | SA612A-I Pin2 (GND) | GND_ANA | RF 地 |
| OUTPUT_I | SA612A-I Pin4 (OUT+) | C4 → R6, R7→GND | 混频输出, 负载 1.5kΩ |
| GND_I | SA612A-I Pin3 (GND) | GND_ANA | 地 |
| LO_I | SA612A-I Pin6 (OSC) | C6 (100pF) → Si5351A CLK0 | LO 输入, 隔直 |
| VCC_MIXER | SA612A-I Pin5 (VCC) | 5V_RAW (通过 R10/10Ω 隔离) | 5V 供电, 2.4mA |
| NC_I | SA612A-I Pin7,8 | 悬空 | 内部振荡器未用 |

### Q 路混频器连接表 (SA612A-Q)

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| RF_Q | SA612A-Q Pin1 (RF+) | C3 (100pF) → VGA_OUTP | RF 输入, 隔直 |
| OUTPUT_Q | SA612A-Q Pin4 (OUT+) | C5 → R8, R9→GND | 混频输出, 负载 1.5kΩ |
| LO_Q | SA612A-Q Pin6 (OSC) | C7 (100pF) → Si5351A CLK1 | LO 输入, 90° 移相 |
| VCC_MIXER | SA612A-Q Pin5 (VCC) | 5V_RAW (通过 R10/10Ω 隔离) | 5V 供电 |

### 关键参数
- **C2/C3**: 100pF NPO/COG, RF 隔直电容
- **C6/C7**: 100pF NPO/COG, LO 隔直电容
- **R4/R5**: 1kΩ → GND, RF 输入直流偏置
- **R6/R8**: 1.5kΩ, 混频器输出负载电阻
- **R7/R9**: 1.5kΩ → GND, 输出下拉
- **R10**: 10Ω/0603, VCC 隔离电阻, 去耦 0.1µF+10µF to GND

---

## 模块 3：基带低通滤波器 + ADC 驱动 (OPA2365)

### ASCII 框图

```
  OUTPUT_I ──┬── R11 ──┬── C8 ──┬── OPA2365A (Unity Gain)
             │         │       │    ┌──────────┐
             │         C9      │    │ Pin3 (+)←── 偏置
             │         │       │    │ Pin2 (-)←── R12─┬── OUT
             │         GND     │    └─┬────────┘      │
             │                  │     │               C10
             │                  │     │               │
             │                  └─────┤   ←──R13 ←───┘
             │                         │
             └── R14 ──┬── C11 ──┬────┘  4th Order Sallen-Key LPF
                       │         │           f_c ≈ 200kHz
                       C12       │
                       │         │
                       GND       OPA2365B → ADC_IN1

                       (第二级 LPF, 结构同上)

  ADC_IN1 ─── R15 (51Ω) ──┬── STM32 ADC1_INP (PA6)
                           │
                          C13 (10pF) ─→ GND_ANA
```

### 连接表 (I 路基带 LPF, 4 阶 Sallen-Key)

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| IF_I | R11 (1kΩ 1%) | C8 (330pF NPO/COG) | 一阶 RC 滤波 |
| IF_I | C8 | OPA2365A Pin3 (+) | Sallen-Key 正输入 |
| LPF_GND | C9 (330pF NPO/COG) | GND_ANA | 滤波电容到地 |
| LPF_FB | OPA2365A Pin1 (OUT) | R12 (1kΩ 1%) → OPA2365A Pin2 (-) | 反馈电阻 |
| LPF_FB | OPA2365A Pin2 (-) | C10 (330pF) → OPA2365A Pin1 | 反馈电容 |
| LPF_BIAS | OPA2365A Pin3 (+) | R13 → ADC_IN_BIAS (VCC/2) | 偏置到 1.65V (用于单电源) |
| LPF_OUT1 | OPA2365A Pin1 | R14 (1kΩ 1%) | 第二级输入 |
| LPF_OUT1 | R14 | C11 (330pF) → OPA2365B Pin5 (+) | 第二级 Sallen-Key 输入 |
| ADC_IN | OPA2365B Pin7 (OUT) | R15 (51Ω 0603) | ADC 驱动输出, 串联限流 |
| ADC_IN | R15 | C13 (10pF NPO) → GND | ADC 输入采样电容 |
| ADC_IN | R15 | STM32 PA6 (ADC1_INP) | 到 MCU ADC 通道 |

### 关键参数
- **LPF 截止频率**: f_c = 1/(2π*1kΩ*330pF) ≈ 482kHz (略有偏高, 可调整至 200kHz)
- **修正参数**: R11=R12=R14=3.3kΩ, C8=C9=C10=C11=330pF → f_c ≈ 144kHz (更合适)
- **偏置电压**: 使用电阻分压 (R16/R17=10kΩ) 从 VCC_3V3_ANA 获取 1.65V, 经 OPA2365 缓冲后供所有 LPF 使用
- **Q 路完全相同**, 使用 OPA2365 的另一颗双运放

---

## 模块 4：对数检波器 (AD8307)

### ASCII 框图

```
  VGA_OUTP ──┬── C14 (100nF) ─┬── AD8307 Pin1 (INP)
             │                │
             └── C15 (10pF) ──┘
                               │
                          AD8307 Pin8 (OUT) ──┬── R18 (51Ω) ──→ STM32 ADC_IN3 (PA7)
                                              │
                                             C16 (10nF) → GND

  AD8307 Pin2 (INN) ──── C17 (100nF) ──→ GND

  AD8307 Pin3 (COMM) ──── GND_ANA
  AD8307 Pin4 (ENB) ───── STM32 GPIO (高电平=禁用)
  AD8307 Pin5 (POS) ───── 5V_RAW (通过 R19/10Ω)
  AD8307 Pin6 (OUT) ───── (输出)
  AD8307 Pin7 (VPS) ───── 5V_RAW
  AD8307 Pin8 (OUT) ───── (见上)
```

### 连接表

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| DET_IN | AD8307 Pin1 (INP) | C14 (100nF X7R) → VGA_OUTP | AC 耦合输入 |
| DET_IN | AD8307 Pin1 | C15 (10pF NPO) → GND | 低通滤波, 抗混叠 |
| DET_GND | AD8307 Pin2 (INN) | C17 (100nF) → GND_ANA | 负输入端 AC 耦合 |
| DET_GND | AD8307 Pin3 (COMM) | GND_ANA | 公共地 |
| DET_EN | AD8307 Pin4 (ENB) | STM32 GPIO (PB0) | 使能, 高电平=禁用, 低=启用 |
| DET_5V | AD8307 Pin5,7 (POS,VPS) | 5V_RAW, 0.1µF+10µF → GND | 供电 |
| DET_OUT | AD8307 Pin6,8 (OUT) | R18 (51Ω) + C16 (10nF) → STM32 PA7 | 检波输出, 25mV/dB |

### 关键参数
- **输出斜率**: 25mV/dB
- **输入范围**: -75dBm ~ +17dBm (50Ω 系统)
- **输出范围**: 0.25V ~ 2.5V (对应 -75dBm ~ +17dBm)
- **响应时间**: <500ns, 满足扫频需求

---

## 模块 5：DDS/本振 (Si5351A)

### ASCII 框图

```
  25MHz XTAL
  Y1 ──┬── XTAL_IN ───┬── Si5351A Pin1 (XTAL)
       │               │
      C18             C19
       │               │
      GND             GND

  Si5351A
  Pin2  GND
  Pin3  VDD ── 3V3_DIG (0.1µF+10µF → GND)
  Pin4  VDD
  Pin5  SDA ──┬── STM32 PB7 (I2C1_SDA)
              └── R20 (4.7kΩ) → 3V3_DIG
  Pin6  SCL ──┬── STM32 PB6 (I2C1_SCL)
              └── R21 (4.7kΩ) → 3V3_DIG
  Pin7  CLK0 ── C6 (100pF) ──→ SA612A-I Pin6 (LO_I, 0°)
  Pin8  CLK1 ── C7 (100pF) ──→ SA612A-Q Pin6 (LO_Q, 90°)
  Pin9  CLK2 ── 悬空 (未用)
  Pin10 GND
```

### 连接表

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| XTAL | Y1 (25MHz) Pin1 | Si5351A Pin1 (XTAL) | 并联谐振晶振 |
| XTAL | Y1 Pin2 | Si5351A Pin1 | 负载电容内部集成 |
| XTAL_GND | Y1 Pin3,4 | GND 平面 | 晶振外壳接地 |
| DDS_VDD | Si5351A Pin3,4 (VDD) | VCC_3V3_DIG | 3.3V 供电 |
| DDS_GND | Si5351A Pin2,10 (GND) | GND_平面 | 地 |
| I2C_SCL | Si5351A Pin6 (SCL) | STM32 PB6 + R21 (4.7kΩ)→3V3 | I²C 时钟, 100kHz |
| I2C_SDA | Si5351A Pin5 (SDA) | STM32 PB7 + R20 (4.7kΩ)→3V3 | I²C 数据 |
| LO_I | Si5351A Pin7 (CLK0) | C6 (100pF) → SA612A-I Pin6 | I 路 LO, 0° 相位 |
| LO_Q | Si5351A Pin8 (CLK1) | C7 (100pF) → SA612A-Q Pin6 | Q 路 LO, 90° 相位 |

### Si5351A 寄存器配置要点
- **晶体频率**: 25MHz
- **PLL 配置**: PLLA 用作 CLK0/CLK1 源, PLLB 悬空
- **CLK0 相位**: 0° (寄存器 MSNA_PHASE_OFFSET = 0)
- **CLK1 相位**: 90° (寄存器 MSNA_PHASE_OFFSET = 1/4 分频比)
- **输出幅度**: 2mA 驱动 (寄存器 CLKx_DRV_STRENGTH = 2)
- **输出格式**: 方波 (寄存器 CLKx_INV = 0)

---

## 模块 6：MCU 模块 (STM32H750VBT6)

### ASCII 框图

```
  ┌──────────────────────────────────────────────────┐
  │                  STM32H750VBT6                   │
  │  LQFP-100                                        │
  │                                                  │
  │  ADC1_INP(PA6) ←── 基带 I (模块3)                │
  │  ADC1_INP(PA7) ←── 基带 Q                       │
  │  ADC1_INP(PC5) ←── AD8307 检波 (模块4)          │
  │                                                  │
  │  DAC1_OUT(PA4) ──→ AD8331 GPOS (AGC)            │
  │                                                  │
  │  UART7_TX(PF7) ──→ TJC 屏 RX                    │
  │  UART7_RX(PF6) ←── TJC 屏 TX                    │
  │                                                  │
  │  I2C1_SCL(PB6) ──→ Si5351A SCL                  │
  │  I2C1_SDA(PB7) ──→ Si5351A SDA                  │
  │                                                  │
  │  QSPI: P F8/F9/F10/F11 ──→ W25Q64JV (外部 Flash)│
  │                                                  │
  │  SWD: PA13(SWDIO) + PA14(SWCLK)                 │
  │                                                  │
  │  USB_OTG_FS: PA11(D-) + PA12(D+)                │
  │                                                  │
  │  RCC: PC14(OSC32_IN)+PC15(OSC32_OUT) (32.768kHz)│
  │       PH0(OSC_IN)+PH1(OSC_OUT)  (8MHz)          │
  └──────────────────────────────────────────────────┘
```

### 关键引脚连接表

| 引脚 | 信号 | 方向 | 连接目标 | 备注 |
|------|------|------|----------|------|
| PA6 (ADC1_INP) | ADC_I | 输入 | OPA2365B OUT (基带 I) | ADC 通道 6 |
| PA7 (ADC1_INP) | ADC_Q | 输入 | OPA2365D OUT (基带 Q) | ADC 通道 7 |
| PC5 (ADC1_INP) | ADC_DET | 输入 | AD8307 OUT | ADC 通道 15 |
| PA4 (DAC1_OUT) | AGC_VCTRL | 输出 | AD8331 GPOS | DAC 输出 0~1.4V |
| PB0 | DET_EN | 输出 | AD8307 ENB | 高=禁用检波器 |
| PB6 (I2C1_SCL) | I2C_SCL | 输出 | Si5351A SCL + 4.7kΩ 上拉 | I²C 时钟 |
| PB7 (I2C1_SDA) | I2C_SDA | 双向 | Si5351A SDA + 4.7kΩ 上拉 | I²C 数据 |
| PF6 (UART7_RX) | TJC_TX | 输入 | TJC 屏 TX | 串口接收 |
| PF7 (UART7_TX) | TJC_RX | 输出 | TJC 屏 RX | 串口发送 |
| PA13 (SWDIO) | SWDIO | 双向 | SWD 调试口 10-pin | 编程调试 |
| PA14 (SWCLK) | SWCLK | 输出 | SWD 调试口 10-pin | 编程调试 |
| PA11 (USB_DM) | USB_D- | 双向 | USB-B 座 DM | USB OTG FS |
| PA12 (USB_DP) | USB_D+ | 双向 | USB-B 座 DP | USB OTG FS |
| PC11-14 (QSPI) | QSPI_D0-3 | 双向 | W25Q64JV | 外部 Flash |
| PF8 (QSPI_CLK) | QSPI_CLK | 输出 | W25Q64JV CLK | QSPI 时钟 |
| PF9 (QSPI_CS) | QSPI_CS | 输出 | W25Q64JV CS | QSPI 片选 |
| BOOT0 | BOOT0 | 输入 | 10kΩ → GND (默认 User Flash) | 启动选择 |

### MCU 基本电路

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| MCU_VDD | STM32 Pin19,32,48,64,75,100 | VCC_3V3_DIG | 数字供电, 每组 0.1µF+10µF → GND |
| MCU_VDDA | STM32 Pin13,14 | VCC_3V3_ANA | 模拟供电, 0.1µF+10µF → GND_ANA |
| MCU_VREF+ | STM32 Pin12 | REF3033 OUT | 3.3V ADC 参考 |
| MCU_VBAT | STM32 Pin6 | VCC_3V3_DIG (或 CR1220 电池) | VBAT 备用电源 |
| MCU_NRST | STM32 Pin7 | R22 (10kΩ) → 3V3 + C20 (100nF) → GND | 外部复位 |
| OSC_IN | STM32 PH0 | Y2 (8MHz) + C21/C22 (15pF) | 主时钟 8MHz |
| OSC_OUT | STM32 PH1 | Y2 (8MHz) | 主时钟 |
| OSC32_IN | STM32 PC14 | Y3 (32.768kHz) + C23/C24 (12.5pF) | RTC 时钟 |
| OSC32_OUT | STM32 PC15 | Y3 (32.768kHz) | RTC 时钟 |

---

## 模块 7：电源管理

### ASCII 框图

```
  USB_5V ───┬── SS34 (D1) ─┬── 5V_RAW ──┬── AMS1117-3.3 ── VCC_3V3_DIG
            │              │            │
            │              │            └── ADP150-3.3 ── VCC_3V3_ANA
  3S LiPo ──┴── LM2596 ───┘            │
     11.1V         5V/1A                └── (直供 TJC 屏 5V)
```

### 连接表

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| USB_5V | USB-B 座 Pin1 | D1 SS34 (阳极) | USB 5V 输入, 反接保护 |
| BAT_RAW | 3S LiPo JST-XH | D2 SS34 (阳极) | 电池输入 9.6-12.6V |
| BUCK_IN | D1/D2 阴极 (合并) | LM2596 VIN + 10µF+100µF → GND | 理想二极管 OR, 自动切换 |
| BUCK_OUT | LM2596 OUT | FB 反馈 (R23/R24 分压) → 5V_RAW | 5.0V ±2% |
| 5V_RAW | BUCK_OUT | 各负载 (见电源树表) | 5V 总线 |
| 5V_DIG | 5V_RAW | AMS1117-3.3 VIN + 10µF → GND | 数字 LDO 输入 |
| 3V3_DIG | AMS1117-3.3 OUT | + 10µF + 0.1µF → GND | 3.3V 数字电源 |
| 5V_ANA | 5V_RAW | R25 (10Ω/0805) + C25 (47µF) → GND | 模拟电源去耦 |
| 3V3_ANA | ADP150-3.3 OUT | + 10µF + 0.1µF → GND_ANA | 3.3V 低噪声模拟电源 |
| VREF | REF3033 IN | 5V_RAW + 1µF → GND | 参考电压输入 |
| VREF_OUT | REF3033 OUT | C26 (1µF) → GND + STM32 VREF+ | 3.3V 0.2% 基准 |

### 电源树参数

| 电源轨 | 来源 | 电压 | 最大电流 | 去耦 |
|--------|------|------|----------|------|
| 5V_RAW | LM2596 | 5.0V ±2% | 1A | 100µF 电解 + 10µF MLCC |
| VCC_3V3_DIG | AMS1117-3.3 | 3.3V ±1% | 800mA | 10µF + 0.1µF × N (每 IC) |
| VCC_3V3_ANA | ADP150-3.3 | 3.3V ±1% | 150mA | 10µF + 0.1µF (低 ESR) |
| 5V_ANA | 5V_RAW via R25 | 5.0V | 100mA | 47µF + 0.1µF |
| VREF_OUT | REF3033 | 3.3V ±0.2% | 10mA | 1µF × 2 |

---

## 模块 8：TJC 串口屏接口

### 连接表

| 网络名 | 源端 | 目标端 | 参数 |
|--------|------|--------|------|
| TJC_5V | 5V_RAW | TJC 屏 VCC | 5V ±5%, ~100mA |
| TJC_GND | GND | TJC 屏 GND | 共地 |
| UART_TX | STM32 PF7 (UART7_TX) | TJC 屏 RX | MCU→TJC 指令, 3.3V TTL |
| UART_RX | STM32 PF6 (UART7_RX) | TJC 屏 TX | TJC→MCU 触摸回传, 3.3V TTL |
| TJC_BUSY | TJC 屏 BUSY | (可选) STM32 GPIO | 屏忙检测, 也可省略 |

> **注意**: 陶晶弛 TJC 屏 UART 为 3.3V TTL 电平, 与 STM32 直连, 无需电平转换。波特率建议 115200 或 921600。

---

## 模块连接汇总

```
  BNC ──→ ESD ──→ AD8331(VGA) ──┬── [I] SA612A ──→ LPF ──→ OPA2365 ──→ STM32 ADC1_CH6 (PA6)
                                  │
                                  ├── [Q] SA612A ──→ LPF ──→ OPA2365 ──→ STM32 ADC1_CH7 (PA7)
                                  │
                                  └── AD8307 ──→ STM32 ADC1_CH15 (PC5)
                                            ↑
                                    Si5351A ── I²C ──→ STM32 I2C1 (PB6/PB7)
                                                        │
                                                    TJC 屏 ← UART7 (PF6/PF7)
                                                        │
                                                    W25Q64 ← QSPI (PF8-11/PC11-14)

  电源: USB 5V ──┬── SS34 ──┬── LM2596 ── 5V_RAW ──┬── AMS1117-3.3 ── VCC_3V3_DIG
                  │          │                      │
                  │          │                      └── ADP150-3.3 ── VCC_3V3_ANA
                  │          │
  3S LiPo 11.1V ─┴── SS34 ──┘
```

以上全部连接表和参数可直接在立创 EDA 中进行原理图绘制。
