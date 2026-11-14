# ASTC 纹理压缩回归报告（500 样本）

## 1. 测试对象与环境

| 项目 | 内容 |
| --- | --- |
| 被测程序 | `astcenc` v6.0.0，native 构建（avx512+vbmi+popcnt+f16c） |
| 源码版本 | `ada5a3441a1134bfa065ca8041bbd79cb916207d` |
| 构建 | `cmake -B build-native -DCMAKE_BUILD_TYPE=Release -DASTCENC_ISA_NATIVE=ON` |
| 机器 | Ubuntu 26.04，16 核，g++ 15.2 |
| 编码配置 | `-cl`（LDR linear），块大小 `6x6`（3.56 bpp），质量 `-medium`，线程固定 `-j 4` |
| 法线贴图 | 额外使用 `-normal`（XY 双分量法线模式） |
| 语料 | 500 张 PNG，6 类，尺寸 16x16 至 512x512，共 20,282,112 纹素 |

线程数固定、质量预设固定，且构建开启 `ASTCENC_INVARIANCE=ON`（默认），保证确定性。

## 2. 语料构成

生成器 `gen_textures.py` 使用固定种子的 LCG，逐类确定性生成；已验证重新生成的
500 张 PNG 与原语料 **SHA-256 完全一致**（字节级可重现）。

| 类别 | 数量 | 生成方式 |
| --- | --- | --- |
| solid 纯色 | 83 | 恒定 RGB，少量带 1-6 级微条带 |
| gradient 渐变 | 83 | 线性/径向/双停渐变，随机角度与端点色 |
| hardedge 硬边 | 84 | 棋盘、对角带、竖条、块状锐边界，2-6 色调色板 |
| noise 噪声 | 84 | 白噪声、相关随机游走、量化块噪声 |
| alpha 透明 | 83 | RGBA，渐变 alpha、硬裁切、抖动、棋盘镂空 |
| normal 法线 | 83 | 高斯高度场求导归一化，单位长度 XYZ 法线 |

## 3. 验收标准与结果

| 要求 | 结论 | 证据 |
| --- | --- | --- |
| 质量指标不低于基线 | **通过** | `check_baseline.py`：run2 全部聚合指标（均值/最小值/P5）及逐样本指标均 ≥ run1 基线，0 项失败 |
| 两次运行字节一致 | **通过** | 每样本压缩两次：1000/1000 次内部哈希一致；跨运行 500 个 ASTC 文件 SHA-256 全部一致（0 差异）；两次 PSNR/SSIM 数值也完全相同 |
| 零崩溃 | **通过** | 每样本 3 次调用（压缩 x2 + test 模式），共 3000 次进程调用，返回码全部为 0 |

独立校验：用纯 Python 重新计算的 RGBA PSNR 与编码器自报 PSNR 最大偏差
`5.07e-05 dB`（仅浮点取整差异），指标可信。

## 4. 质量指标汇总（run1，作为冻结基线）

PSNR 单位 dB；999 dB 为编码器的无损哨兵值，统计有限值最小值时已排除。

| 指标 | 均值 | 有限值最小值 | P5 | 中位数 |
| --- | --- | --- | --- | --- |
| PSNR RGBA | 243.10 | 15.09 | 24.14 | 50.71 |
| PSNR RGB | 242.37 | 13.84 | 22.89 | 49.46 |
| SSIM RGBA | 0.9719 | 0.5654 | 0.7628 | 0.9966 |
| SSIM RGB | 0.9637 | 0.4255 | 0.6861 | 0.9955 |

其中 104/500 张（纯色、硬边类为主）达到无损（PSNR=999，SSIM=1.0）。

### 分类别质量

| 类别 | 数量 | PSNR 均值 | PSNR 最低 | SSIM 均值 | SSIM 最低 |
| --- | --- | --- | --- | --- | --- |
| solid | 83 | 758.57 | 45.94 | 0.9989 | 0.9876 |
| gradient | 83 | 51.25 | 41.34 | 0.9957 | 0.9924 |
| hardedge | 84 | 525.43 | 39.98 | 0.9995 | 0.9964 |
| noise | 84 | 25.51 | 15.09 | 0.9337 | 0.7375 |
| alpha | 83 | 39.41 | 28.69 | 0.9068 | 0.5654 |
| normal | 83 | 57.65 | 40.54 | 0.9970 | 0.9854 |

压缩体积：输入 PNG 合计 9.90 MB，ASTC 合计 9.18 MB（6x6 固定 16 字节/块）。

## 5. 最差样本

### PSNR 最差 Top 10（全部为白噪声，符合块压缩对高频随机信号的预期）

| 排名 | id | 类别 | 尺寸 | PSNR RGBA (dB) | PSNR RGB | SSIM RGBA | 文件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 250 | noise | 288x224 | 15.09 | 13.84 | 0.7449 | `corpus/noise/noise_000_288x224.png` |
| 2 | 321 | noise | 400x304 | 15.12 | 13.87 | 0.7416 | `corpus/noise/noise_071_400x304.png` |
| 3 | 297 | noise | 120x88 | 15.12 | 13.87 | 0.7394 | `corpus/noise/noise_047_120x88.png` |
| 4 | 251 | noise | 120x88 | 15.13 | 13.88 | 0.7409 | `corpus/noise/noise_001_120x88.png` |
| 5 | 281 | noise | 160x160 | 15.14 | 13.89 | 0.7490 | `corpus/noise/noise_031_160x160.png` |
| 6 | 330 | noise | 160x160 | 15.14 | 13.89 | 0.7484 | `corpus/noise/noise_080_160x160.png` |
| 7 | 304 | noise | 80x80 | 15.18 | 13.93 | 0.7422 | `corpus/noise/noise_054_80x80.png` |
| 8 | 333 | noise | 32x16 | 15.19 | 13.94 | 0.7519 | `corpus/noise/noise_083_32x16.png` |
| 9 | 323 | noise | 72x72 | 15.19 | 13.94 | 0.7385 | `corpus/noise/noise_073_72x72.png` |
| 10 | 295 | noise | 40x24 | 15.24 | 13.99 | 0.7375 | `corpus/noise/noise_045_40x24.png` |

### 各类别自身最差样本

| 类别 | id | PSNR (dB) | SSIM | 文件 |
| --- | --- | --- | --- | --- |
| solid | 24 | 45.94 | 0.9876 | `corpus/solid/solid_024_48x48.png` |
| gradient | 161 | 41.34 | 0.9928 | `corpus/gradient/gradient_078_16x16.png` |
| hardedge | 185 | 39.98 | 0.9964 | `corpus/hardedge/hardedge_019_24x24.png` |
| noise | 250 | 15.09 | 0.7449 | `corpus/noise/noise_000_288x224.png` |
| alpha | 370 | 28.69 | 0.7508 | `corpus/alpha/alpha_036_32x32.png` |
| normal | 429 | 40.54 | 0.9869 | `corpus/normal/normal_012_16x32.png` |

SSIM 最差样本为 id=378（alpha，512x512，SSIM=0.5654，PSNR=29.32 dB），源于锐利
alpha 裁切边界在 6x6 块内的权重量化；属已知的块压缩权衡，非缺陷。

最差样本的源图/解码图对比见 `results/worst_samples/`（含 `contact_sheet.png` 拼图，
左为源图、右为 ASTC 解码图）。

## 6. 结论

- 500 张六类纹理全部通过：质量不低于基线、两次运行字节一致、零崩溃。
- 最差质量集中在白噪声类（PSNR 下限 15.09 dB），这是 3.56 bpp 固定码率下对
  不可压缩高频随机信号的理论性表现；纯色/硬边 50% 样本无损，法线类 SSIM 不低于 0.985。
- 未发现编码器缺陷；后续若引入代码变更，可直接运行 `bash run_all.sh` 复跑并自动比对基线。

## 7. 产物与复现

| 路径 | 说明 |
| --- | --- |
| `gen_textures.py` | 确定性语料生成器（500 张 + `manifest.csv`） |
| `run_regression.py` | 双次压缩、哈希、指标采集、独立 PSNR 复核 |
| `analyze.py` | 跨运行字节比对、统计、最差样本分析 -> `results/analysis.json` |
| `check_baseline.py` | 基线门禁 -> `results/baseline_check.json` |
| `run_all.sh` | 一键端到端复跑 |
| `results/run1`, `results/run2` | 每次运行的 ASTC、解码 PNG、逐样本 CSV |
| `results/baseline.json` | 冻结基线（聚合指标 + 编码配置） |
| `results/worst_samples/` | 最差 6 个样本对比图与拼图 |
