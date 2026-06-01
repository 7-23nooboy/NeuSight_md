# 基于 NeuSight 的 DeepMD-kit 推理性能预测器项目报告

**项目名称：** 面向 DeepMD-kit 的端到端推理性能预测器研究与实现  
**项目目录：** `NeuSight_MD`  
**报告日期：** 2026 年 5 月 30 日  
**报告版本：** Final Report v3  
**当前有效模型：** v6 混合 wall-time 模型（kernel-count fixed overhead + roofline + transition bubble + 修法 A opgraph）  
**适用范围：** 课题组内部汇报、导师审阅、阶段性项目总结

---

## 摘要

本项目围绕 DeepMD-kit 分子动力学神经势模型的推理性能预测问题，基于 NeuSight GPU 性能预测框架进行了面向 DeepMD-kit 的系统性改造。NeuSight 原始版本主要服务 Transformer 类深度学习模型，依赖 HuggingFace / FX tracing 自动抽取计算图，并通过 `MLP_WAVE` 模型预测 Linear、BMM、elementwise 等标准算子的 GPU 执行时间。DeepMD-kit 的 PyTorch 推理路径与 Transformer 有显著差异，包含 neighbor list 构建、environment matrix 生成、按原子类型展开的 embedding network、descriptor 矩阵运算、fitting network 以及 force backward 等过程，无法直接套用原始 NeuSight 前端。

本项目的核心工作是：保留 NeuSight 已训练的算子级预测器，重写 DeepMD-kit 专用解析前端，并构建面向 DeepMD 推理特点的端到端开销模型。最终系统能够从 DeepMD 配置文件出发，静态生成 NeuSight 兼容的算子图，预测可见神经网络算子延迟，并通过 fixed overhead、未建模 GPU roofline 项和 transition bubble 项补齐 NeuSight 无法覆盖的系统开销。

在 NVIDIA H100 NVL 单卡环境下，当前版本针对 4 个 `se_e2_a` 体系（copper、water、LiAlOCl、he6）和 9 个原子数规模（N=32 到 8192）进行了 36 个测量点验证。以最终保守口径统计，端到端 wall-time 预测达到 **MAE 7.1%、RMSE 8.3%**；其中 N≤4096 的 32 个测量点全部落在 ±20% 误差范围内。结果表明，当前系统已经能够作为 H100 上 `se_e2_a` DeepMD 模型的端到端推理延迟估计工具，用于实验规模预估、模型配置比较和资源规划。

同时，本项目也明确识别出当前模型的主要不足：`cmp` 部分仍受 NeuSight 原始 `MLP_WAVE` 训练分布限制，对 DeepMD 特有的极端 Linear/BMM shape 存在系统性低估。当前 wall-time 结果可用，但阶段级拆分，尤其是 `cmp` 与 `roof` 的独立解释，还不能直接作为 kernel 级优化建议。这个问题是后续继续改进本项目的核心方向。

**关键词：** DeepMD-kit；NeuSight；GPU 性能预测；分子动力学；推理延迟；kernel launch；roofline；pipeline bubble；模型校准

---

## 1. 项目背景

### 1.1 DeepMD-kit 推理性能预测的需求

DeepMD-kit 是分子动力学和科学机器学习中常用的神经势模型框架。在实际模拟任务中，模型结构、原子数规模、邻居数设置、描述子类型以及硬件平台都会显著影响推理延迟。对于大规模模拟任务而言，推理性能直接决定生产任务吞吐和资源消耗。因此，在正式运行昂贵 benchmark 或长时间分子动力学模拟之前，若能对 DeepMD-kit 模型的端到端推理延迟进行快速预测，将具有实际工程价值。

DeepMD-kit 的推理耗时并不只由神经网络层的矩阵乘决定。对于小原子数，数百个 CUDA kernel launch、Python/PyTorch 调度和 autograd 元数据开销会形成明显平台期；对于较大原子数，neighbor list、environment matrix、descriptor 构造和随机访存逐渐成为主要开销；在中等原子数区域，CPU launch chain 与 GPU compute pipeline 接近同一量级，会出现 transition bubble。这些现象使得简单的 FLOPs 估算、单纯神经网络算子预测或经验幂律拟合都难以稳定覆盖完整范围。

### 1.2 NeuSight 的基础能力与局限

NeuSight 是一个面向深度学习训练和推理的 GPU 性能预测框架。其核心能力是利用已训练的 `MLP_WAVE` 算子级模型预测标准算子的 GPU 执行时间，例如 Linear、BMM、激活函数和 elementwise 操作。原始 NeuSight 的输入流程主要围绕 Transformer 模型设计：通过 HuggingFace / FX tracing 获取计算图，再对图中算子逐项预测并聚合。

这一设计无法直接应用于 DeepMD-kit，原因包括：

1. DeepMD-kit 不是 Transformer 结构，无法自然走 HuggingFace tracing 路径。
2. DeepMD 推理中包含 neighbor list、topk、gather、scatter、environment matrix 等大量非标准神经网络操作。
3. DeepMD 的 embedding network 与原子类型、邻居类型密切相关，尤其 `type_one_side=False` 时会出现 `ntypes^2` 规模的子网络展开。
4. Force 计算依赖 autograd backward，其 kernel 组成和 forward MLP 不完全一致。
5. 小 N 下端到端延迟主要由 CPU/kernel launch 固定开销决定，而 NeuSight 原始模型只预测 GPU 算子执行时间。

因此，本项目不是简单调用 NeuSight，而是对 NeuSight 的前端和端到端开销建模进行面向 DeepMD-kit 的专门改造。

---

## 2. 项目目标与工作范围

### 2.1 项目目标

本项目的总体目标是构建一个 DeepMD-kit 推理性能预测器，使其能够在不实际运行完整 DeepMD 推理 benchmark 的情况下，基于模型配置、原子数和 GPU 信息预测端到端 wall-time。

具体目标包括：

1. 分析 DeepMD-kit PyTorch 后端推理路径，建立可静态解析的计算阶段划分。
2. 将 DeepMD 推理过程转换为 NeuSight 可识别的算子图，使 `MLP_WAVE` 能参与预测。
3. 建立 NeuSight 无法覆盖部分的 overhead model，包括固定调度开销、未建模 GPU 访存项和 transition bubble。
4. 在 H100 GPU 上完成多体系、多原子数的实测验证，明确模型精度和适用边界。
5. 整理关键迭代思路、实验结果、局限性和后续研究方向，形成可用于导师汇报的阶段性项目报告。

### 2.2 当前工作范围

当前最终报告的结论主要覆盖以下范围：

| 维度 | 当前范围 |
|---|---|
| 硬件 | NVIDIA H100 NVL，单卡评估 |
| 软件后端 | DeepMD-kit 3.x PyTorch 后端 |
| 主要描述子 | `se_e2_a` |
| 推理路径 | `compute_force=True`，即 energy forward + force backward |
| 验证体系 | copper、water、LiAlOCl、he6 |
| 原子数范围 | N=32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| 主评价指标 | 端到端 wall-time 误差 |

以下范围不作为当前最终精度承诺：

- DPA-1 / DPA-2 / attention descriptor；
- 未经重新校准的跨 GPU 预测；
- N>8192 的真实测量与外推；
- 压缩 `.pb` lookup-table 推理路径；
- kernel 级或阶段级精确瓶颈诊断。

---

## 3. 技术路线

### 3.1 总体方案

项目采用“NeuSight 后端复用 + DeepMD 前端重写 + 端到端 overhead 补偿”的技术路线。

原始 NeuSight 流程为：

```text
HuggingFace 模型 -> FX tracing -> Transformer op graph -> MLP_WAVE -> latency aggregation
```

本项目改造后的流程为：

```text
DeepMD config / input.json
    -> DeepMD 配置解析
    -> DeepMD 静态算子图构造
    -> NeuSight MLP_WAVE 逐算子预测
    -> DeepMD latency aggregation
    -> fixed / roof / bubble overhead model
    -> wall-time prediction + regime + confidence
```

这种方案的优点是边界清晰：NeuSight 已有算子预测能力被保留，DeepMD 专有结构由新前端解析，系统级开销由独立模型处理。这样既避免了重新训练完整预测器，也避免把 DeepMD 特有的 neighbor-list 和 kernel-launch 现象强行映射成普通深度学习算子。

### 3.2 DeepMD 推理阶段划分

根据 DeepMD-kit PyTorch 后端源码，`se_e2_a` 推理路径被划分为七个阶段：

| 阶段 | 主要内容 | 当前处理方式 |
|---|---|---|
| 1. Neighbor List | ghost atom 扩展、距离计算、topk 邻居选择 | 大部分进入 roofline；最终 gather 用 MEM 近似 |
| 2. Environment Matrix | gather 邻居坐标，计算 `1/r`、方向项和平滑函数 | MEM + VEC 近似，未覆盖部分进入 roofline |
| 3. Embedding Network | 对邻居距离特征执行 per-type 或 per-pair MLP | Linear + VECtanh + residual，按 `type_one_side` 展开 |
| 4. Descriptor BMM | embedding 输出与环境矩阵做 descriptor 矩阵运算 | BMM + MEM |
| 5. Fitting Network | 对每个原子的 descriptor 做 MLP | Linear + activation + residual |
| 6. Output | 输出 per-atom energy 并求和 | Linear + reduction |
| 7. Force Backward | autograd 反向传播得到 force | backward Linear/BMM/MEM 近似 |

项目中特别处理了三个关键结构问题。

第一，embedding network 必须按原子类型或类型对展开。对于 water，`sel=[46, 92]`，两个邻居类型对应不同 batch 规模的 embedding 子网络；对于 `type_one_side=False` 的多元素体系，则需要按中心类型和邻居类型组合展开，规模接近 `ntypes^2`。这一点直接影响 kernel 数和 fixed overhead。

第二，force backward 不能只用一个反向 Linear 表示。修法 A 中，每个 backward Linear 被拆成 `grad_input`、`grad_weight` 和 `grad_bias` 三个子操作，使 opgraph 更贴近 PyTorch autograd 的实际 kernel 结构。

第三，neighbor list 和 environment matrix 的主体开销没有被 NeuSight 原始算子模型覆盖，必须通过独立的 overhead model 进行补偿。

### 3.3 核心代码模块

| 模块 | 路径 | 作用 |
|---|---|---|
| DeepMD 配置解析 | `neusight/Tracing/parse_deepmd_input.py` | 解析 `type_map`、`sel`、descriptor、network 等字段 |
| DeepMD 算子图构造 | `neusight/Tracing/trace_deepmd.py` | 将七阶段推理路径转换为 NeuSight opgraph |
| DeepMD 预测入口 | `neusight/Prediction/predictor_deepmd.py` | 串联解析、opgraph、MLP_WAVE、aggregation 和 overhead model |
| 开销模型 | `neusight/Prediction/overhead_model.py` | 实现 v6 fixed overhead、roofline 和 transition bubble |
| DeepMD 聚合 | `neusight/Prediction/aggregator.py` | 对 DeepMD opgraph 所有节点直接求和 |
| 主校准脚本 | `scripts/calibrate_fixed_overhead.py` | 拟合 fixed 和 roofline 参数 |
| 主验证脚本 | `scripts/benchmark_cross_system.py` | 多体系、多 N wall-time 预测验证 |
| 阶段拆分脚本 | `scripts/measure_real_breakdown_v3.py` | profiler v3 桶分类，获取阶段级实测参考 |

---

## 4. 模型设计

### 4.1 端到端模型结构

当前有效模型将 DeepMD 推理 wall-time 拆成四个部分：

```text
wall = max(fix, cmp + roof) + bubble
```

其中：

- `cmp`：NeuSight MLP_WAVE 对可见神经网络算子的预测总和；
- `fix`：CPU 调度、CUDA kernel launch chain、autograd 元数据等固定开销；
- `roof`：neighbor list、environment matrix、descriptor 相关的未建模 GPU 访存补偿项；
- `bubble`：transition 区 CPU launch 与 GPU compute 交错导致的额外空泡开销。

### 4.2 NeuSight 可见计算项 `cmp`

DeepMD opgraph 中所有 NeuSight 可识别算子的 latency 逐项预测并求和：

```text
cmp = sum_i MLP_WAVE(op_i)
```

与 Transformer 不同，DeepMD 没有“层数复制”的聚合方式，因此所有 opgraph 节点直接求和。

### 4.3 固定开销项 `fix`

v6 版本将 fixed overhead 从早期查表升级为 kernel-count driven 模型：

```text
fix = alpha + beta * K_modeled + delta * ntypes^p + gamma * is_force
```

当前 H100 v6 参数为：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `alpha_fixed_ms` | 2.000822 | 固定开销基项 |
| `beta_per_kernel_ms` | 0.02944444 | 每个 modeled kernel 的开销贡献 |
| `delta_ntypes_sq_ms` | 1.0315 | 多元素 type-dispatch 结构项 |
| `gamma_force_ms` | 0.15 | force backward 附加开销 |

这里的 `p` 根据 descriptor 结构自适应：

```text
type_one_side=True  -> p = 1
type_one_side=False -> p = 2
```

该设计是本项目的重要改进之一。早期版本只对 1-type、2-type fixed overhead 查表，无法外推到 LiAlOCl 等 4 元素体系。v6 公式通过 kernel count 和 `ntypes^p` 结构项捕捉了多元素体系中 fixed overhead 的增长。

### 4.4 未建模 GPU 项 `roof`

NeuSight 无法直接覆盖 neighbor list、topk、gather、scatter 和 environment matrix 等 memory-bound 操作。当前模型用 roofline 形式补偿：

```text
n_all = 27 * N
n_nei = sum(sel)
roof = (C_quad * N * n_all + C_linear * N * n_nei) * 8 / Mem_BW
```

当前 H100 v6 参数为：

| 参数 | 数值 |
|---|---:|
| `C_quad` | 27.899466 |
| `C_linear` | 1433.678957 |

需要注意，当前 `roof` 更适合作为 wall-time 补偿项，而不是独立的 descriptor kernel 时间解释。后续 profiler v3 结果表明，真实 DeepMD-PyTorch 路径更接近 cell-list + `sel` 截断机制，当前 roof 的一部分作用是在补偿 `cmp` 的系统性低估。

### 4.5 Transition Bubble

如果只使用：

```text
wall = max(fix, cmp + roof)
```

模型会在 transition 区出现系统性低估。原因是实际执行更接近：

```text
real ~= sum_i max(launch_i, compute_i)
```

而简化模型使用的是：

```text
model ~= max(sum_i launch_i, sum_i compute_i)
```

两者差值就是 pipeline bubble。

当前版本以 `ratio=(cmp+roof)/fix` 定义 transition 状态，并在 `ratio` 接近 1 时加入平滑 `sin^2` 修正：

```text
wall = max(fix, cmp + roof) + bubble_peak_fraction * fix * bubble_factor(ratio)
```

其中 `bubble_peak_fraction=0.35`，`transition_lo=0.4`，`transition_hi=2.0`。该项能够缓解 transition 区低估，但不同体系的真实 bubble 强度并不完全一致，后续仍需 per-system 或 feature-aware 校准。

### 4.6 Regime 与 Confidence

模型根据 `ratio=(cmp+roof)/fix` 输出性能状态：

| Regime | 条件 | 解释 | 置信度 |
|---|---|---|---|
| overhead-bound | `ratio < 0.4` | CPU 调度和 kernel launch 主导，小 N 平台区 | high |
| transition | `0.4 <= ratio <= 2.0` | CPU launch 与 GPU compute 同量级，bubble 明显 | low |
| compute-bound | `ratio > 2.0` | GPU 计算和访存主导，大 N 区间 | high |

这一输出对于实际使用很重要。系统不仅给出延迟点估计，也告诉用户当前预测处于哪种性能状态，哪些点更需要实测或重新校准。

---

## 5. 关键迭代节点与模型认识

项目中间存在较多实验版本，但最终报告不逐一罗列所有编号。这里仅保留对建模思路有实质影响的关键节点，用于说明本项目从“直接套用 NeuSight”逐步发展到当前混合模型的推理过程。

| 关键节点 | 当时观察到的问题 | 模型或实现修改 | 得到的认识 |
|---|---|---|---|
| 1. 纯 `MLP_WAVE` 基线 | 仅预测 Linear/BMM/VEC 等可见算子时，DeepMD wall-time 被严重低估 | 保留 NeuSight 算子预测器，但不再把它视为完整端到端模型 | DeepMD 推理存在大量 NeuSight 不可见开销，必须单独建模 fixed overhead 和 descriptor/neighbor-list 工作 |
| 2. fixed plateau 与未建模 GPU 工作分离 | 小 N 下真实延迟几乎不随 N 增长，大 N 下又快速增长 | 将延迟拆为 `fix` 与 `cmp+roof` 两条路径，并用 `max(fix, cmp+roof)` 表示主导瓶颈切换 | DeepMD 性能不是单一随 N 增长的曲线，而是存在 overhead-bound 与 compute-bound 两种机制 |
| 3. 多元素体系 fixed overhead 修正 | LiAlOCl 等多元素体系中，早期按 1-type/2-type 查表的 fixed model 明显低估 | 引入 `fix=alpha+beta*K+delta*ntypes^p+gamma`，并根据 `type_one_side` 区分 `ntypes` 与 `ntypes^2` | fixed overhead 与 kernel count、原子类型结构强相关，多元素体系不能用简单线性外推处理 |
| 4. transition bubble 修正 | N=1024 到 2048 附近，`max(fix, cmp+roof)` 仍会系统性低估 | 在 `ratio=(cmp+roof)/fix` 接近 1 的区间加入平滑 `sin^2` bubble 项 | CPU launch chain 和 GPU compute 不是完美重叠，transition 区需要显式建模 pipeline bubble |
| 5. backward 与 profiler 修正 | backward Linear 表达过粗，且早期 profiler 桶把 topk/norm/cat 等 descriptor kernel 误归类 | 修法 A 将 backward Linear 拆为 input/weight/bias；profiler v3 修正真实阶段拆分口径 | 当前 `cmp` 表达更合理，但仍受 `MLP_WAVE` OOD 限制；`cmp/roof` 单独解释不可靠，wall 准确性部分来自误差抵消 |

这些关键节点说明，本项目的核心进展不在于某一个单独版本号，而在于逐步识别并分离了 DeepMD 推理延迟中的几类机制：小 N 的固定调度平台、大 N 的 GPU 计算与访存增长、多元素体系的 kernel 结构扩展，以及 transition 区的 pipeline bubble。当前模型正是这些认识累积后的结果。

---

## 6. 实验设计

### 6.1 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA H100 NVL |
| 评估方式 | 单卡评估 |
| DeepMD-kit | 3.1.2，PyTorch 后端 |
| PyTorch | 2.8.0 + CUDA 12.8 |
| 校准文件 | `results/calibration/h100_nvl_v6.json` |
| 推理路径 | `compute_force=True` |

端到端 wall-time 采用同步计时方式：

```python
torch.cuda.synchronize()
t0 = time.perf_counter()
_ = model(coord, atype, box)
torch.cuda.synchronize()
latency = (time.perf_counter() - t0) * 1000
```

### 6.2 验证体系

| 体系 | 原子类型数 | 描述子 | 特点 |
|---|---:|---|---|
| copper | 1 | `se_e2_a` | 单元素基线 |
| water | 2 | `se_e2_a` | 典型双元素体系，验证 per-type embedding |
| LiAlOCl | 4 | `se_e2_a` | 多元素、高邻居数压力测试 |
| he6 | 6 | `se_e2_a` | 原子类型数扩展压力测试 |

原子数取：

```text
N = 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
```

合计 36 个主验证点。

DPA-1 / attention 描述子作为分布外样本分析，不纳入主精度指标。

### 6.3 评价指标

主指标为端到端 wall-time 相对误差：

```text
err% = (pred - real) / real * 100%
```

报告同时关注 MAE、RMSE、最差单点误差、不同 N 区间覆盖情况以及 leave-one-system-out 预测带。

---

## 7. 实验结果与可视化分析

### 7.1 总体结果

当前最终口径为 v6 校准 + 修法 A opgraph。36 个主验证点上的 wall-time 指标为：

| 指标 | 数值 |
|---|---:|
| mean error | +4.3% |
| MAE | 7.1% |
| RMSE | 8.3% |
| 最差单点 | +21.6%（LiAlOCl, N=8192） |

按体系聚合后的误差分布如下：

| 体系 | n | mean error | MAE | RMSE | worst |
|---|---:|---:|---:|---:|---:|
| copper | 9 | +6.0% | 8.4% | 9.1% | +13.9% |
| water | 9 | +5.1% | 6.5% | 7.9% | +15.2% |
| LiAlOCl | 9 | +5.2% | 7.4% | 9.3% | +21.6% |
| he6 | 9 | +0.9% | 6.1% | 6.6% | -11.8% |

从覆盖范围看：

| 范围 | 覆盖情况 | 结论 |
|---|---|---|
| N≤4096 | 32/32 个点在 ±20% 内 | 当前推荐使用范围 |
| N=8192 | 3/4 个点在 ±20% 内 | 可用但应谨慎，LiAlOCl 越界 |
| 新体系预测带 | LOSO 约 ±14% | 适用于 in-distribution `se_e2_a` |

如果采用修法 A 之前的旧 opgraph，wall MAE 约 5.0%、RMSE 约 7.0%。但旧版本对 backward Linear 的表达不够完整，因此最终报告采用 7.1% MAE 作为更保守、更可信的汇报口径。

### 7.2 可视化结果

**图 1** 在对数坐标下绘制 36 个测量点的预测 wall-time 对实测 wall-time。点的形状区分体系，颜色区分 overhead-bound / transition / compute-bound 三种 regime，灰色带为 ±20% 误差带。**核心观察：** 36 个点几乎全部落在 ±20% 带内，且按 regime 自然聚成三段，说明端到端模型在主验证范围内对延迟数量级和增长趋势的拟合是稳定的；唯一系统性越界的 LiAlOCl, N=8192 已在图中显式标注，作为当前模型的外推边界。

![Figure 1. Predicted vs measured wall time, colour-coded by regime; the only out-of-band point (LiAlOCl, N=8192) is annotated.](figures/wall_pred_vs_real.svg)

**图 2** 是 4 个体系 × 9 个原子数的 signed error 热力图，每个格子同时标注误差百分比和 regime 标签（OH / T / CB），最右下角虚线方框标出全局最差点。**核心观察：** 误差在小 N 多为正、中段为负、大 N 又翻正，说明误差并非单调累积，而是与 regime 紧密相关，验证了引入 transition bubble 和大 N 行为分析的必要性。

![Figure 2. Signed wall-time error per (system, N), annotated with regime tag; the worst-case cell (LiAlOCl, N=8192, +21.6%) is highlighted with a dashed box.](figures/wall_error_heatmap.svg)

**图 3** 在对数 N 轴上叠加每个体系的 signed error 曲线与四体系的 mean |error| 折线，阴影区标出 transition 区（N=1024–2048），背景灰带为 ±20% 误差带。**核心观察：** 几乎所有体系都在 transition 区出现明显下凹（pipeline bubble 残差），并在 N=8192 出现回升（大 N descriptor 外推），共同形成 "浴缸" 形误差曲线。这直接支持 §4 中对 transition bubble 与未建模 GPU 访存项的拆分式建模。

![Figure 3. Bathtub-shaped error trend: transition undershoot near N=1024–2048 and large-N overshoot at N=8192.](figures/wall_error_trend.svg)

### 7.3 典型大 N 结果

N=8192 是当前可测范围中最能体现大规模外推能力的点。修法 A 口径下：

| 体系 | real wall | pred wall | error |
|---|---:|---:|---:|
| copper | 123.08 ms | 129.11 ms | +4.9% |
| water | 127.16 ms | 130.97 ms | +3.0% |
| LiAlOCl | 201.00 ms | 244.43 ms | +21.6% |
| he6 | 158.67 ms | 151.40 ms | -4.6% |

这些结果说明模型总体抓住了大 N wall-time 增长趋势；LiAlOCl 的偏差则说明当前 roofline 对高 `sel`、高邻居数体系还不够稳，应作为未来 descriptor / neighbor-list 访存模型优化方向。

### 7.4 分区行为详细分析

为了让 overhead-bound 、transition 、compute-bound 三个区间的表现可合判、可诊断，本小节按 regime 逐一给出定量统计、机制解释与当前模型响应。三个区间的 ·误差 · 占比总览如下（根据 `ratio = (cmp + roof) / fix` 划分，lo=0.4，hi=2.0）：

| Regime | n | 占比 | mean error | MAE | 代表区间 |
|---|---:|---:|---:|---:|---|
| overhead-bound | 21 / 36 | 58% | +7.3% | 8.0% | 多体系中 N ≤1024 的平台区 |
| transition | 8 / 36 | 22% | -4.2% | 5.4% | copper/water N=1024–2048；LiAlOCl N=512–2048；he6 N=4096 |
| compute-bound | 7 / 36 | 19% | +5.0% | 6.4% | 多体系中 N≥4096 |

三个 regime 呈现明显不同的误差符号：**overhead-bound 为正，transition 为负，compute-bound 轻微为正**。这三个偏差方向是后续分析的出发点。

#### 7.4.1 Overhead-bound 区：CPU 调度主导的平台区

**范围与机制。** OH 区是 `cmp + roof ≪ fix` 的区间，模型输出近似等于 `fix`。在实测中表现为：N 从 32 变化到 ≈1024 的过程中，实测 wall-time 几乎不随 N 增长（copper 4.99–6.57 ms，water 8.54–10.17 ms，LiAlOCl 22.08–23.14 ms，he6 43.39–45.84 ms）。该平台的物理源是 Python/PyTorch dispatch、CUDA driver 的 kernel launch 链以及 autograd 元数据开销，与 GPU 计算量无关，仅与模型结构（kernel 数 K 、原子类型数 ntypes、是否 force）相关，这正是 §4.3 中 `fix = α + βK + δ ntypes^p + γ` 公式要拟合的量。

**表现。** OH 区 21 个点的 mean error 为 +7.3%，MAE 8.0%，总体稳定；未出现负偏。偏高的主要原因是：修法 A 把 backward Linear 拆为三个子算子，造成 opgraph 中 modeled kernel 数增加了约 14 个，使 `βK` 项随之上抬，但本轮报告未重新拟合 `β`。该偏偏是参数拟合问题，不是公式结构问题。

**含义。** OH 区预测可作为“DeepMD 推理在小体系上的下限”，用于预估 throughput 玩具、筛选模型配置或设计低延迟推理管线。在该区间增加 N 不会给端到端延迟带来线性增长，这个物理事实对使用者判断“什么时候加原子才会变慢”十分重要。

#### 7.4.2 Transition 区：pipeline bubble 为主要误差来源

**为什么这个区间会出现负偏。** 在 `cmp + roof` 与 `fix` 同量级的区间（`ratio` 接近 1），`max(fix, cmp+roof)` 隐含“两条关键路径完全重叠”的假设，但 PyTorch 推理中 CPU 的 launch chain 和 GPU 的 kernel 执行是交替发生的，其真实延迟更接近于逐 kernel 取 max 后求和：

```text
real  ≈  Σ_i max(launch_i, compute_i)
model ≈  max(Σ_i launch_i, Σ_i compute_i)
```

由 Jensen 不等式，前者总是不小于后者，差值在 `launch_i` 与 `compute_i` 另体量级接近、且大小 kernel 交错明显时达到最大。这个差值就是 **pipeline bubble**。实测数据中，它表现为 transition 区 8 个点中有6 个出现负偏，mean error − 4.2%，最差点 − 11.8%（he6, N=4096）。图 3 中几乎所有体系都在该区出现明显下凹，这是 “浴缸形” 误差曲线的最主要成因。

**当前模型的响应。** §4.5 中的平滑 `sin²` bubble 项（`bubble_peak_fraction = 0.35`，区间 `[0.4, 2.0]`）会在 `ratio ≈ 1` 附近加上一个嵌入项，部分弥补 max 近似带来的偏差。从 transition 区 8 个点看，mean error 从未加 bubble 时的负位收敛到 − 4.2%，MAE 从两位数压缩到 5.4%，其贡献是明显的。

**但 bubble 项仍不完全足够。** 该修正采用跨体系常数，而真实 bubble 强度依赖体系结构（kernel 大小分布、ntypes、sel）。以在 `wall_P` 中代入后仍需补足的 `(wall_R - wall_P) / fix_P` 估计每点需要的额外 bubble 占比为例：

| 体系 | N | `ratio` | 需补足额外 bubble | 误差 |
|---|---:|---:|---:|---:|
| copper | 1024 | 0.70 | − 0.02 · fix | +1.8% |
| copper | 2048 | 1.88 | +0.23 · fix | − 10.7% |
| water  | 1024 | 0.47 | +0.02 · fix | − 2.2% |
| water  | 2048 | 1.17 | +0.06 · fix | − 3.8% |
| LiAlOCl | 512 | 0.43 | +0.01 · fix | − 1.3% |
| LiAlOCl | 1024 | 0.80 | − 0.03 · fix | +2.7% |
| LiAlOCl | 2048 | 1.66 | +0.16 · fix | − 8.5% |
| he6 | 4096 | 1.02 | +0.18 · fix | − 11.8% |

从该表可以读出三个现象。

1. **偏差越接近 `ratio = 1` 越大。** he6 N=4096 是唯一出现在 `ratio ≈ 1.0` 的点，也是 transition 区误差最大的点。这符合 bubble 在 `ratio = 1` 达到峰值的假设。
2. **靠近上边界 `ratio ≈ 2.0` 仍有较大偏差。** copper N=2048（ratio = 1.88）、LiAlOCl N=2048（ratio = 1.66）都出现十余个百分点的负偏。当前 `sin²` 右侧衰减过快，在 ratio 接近 2.0 时 bubble factor 迅速趋 0，导致这些点几乎拿不到修正。该现象提示 `transition_hi` 可能需要适当上推，或者右侧使用更宽的衰减函数。
3. **不同体系需要的额外 bubble 占比不同。** 以 `ratio ≈ 2` 附近为例，copper 需 ≈23% · fix 额外，water 仅需 ≈6%。这说明跨体系统一的 `bubble_peak_fraction = 0.35` 并不能完全拟合不同 kernel mix 下的 bubble 强度。

**小结。** transition 区是当前三段中唯一带系统性偏差的区间，负偏与同量级 CPU–GPU pipeline 交错直接相关；当前 `sin²` bubble 项能把偏差从两位数压缩到 ~ ± 10%，但进一步收敛需要 per-system 或 structure-aware 的 bubble 强度，以及更宽的右侧衰减（详见 §8.3）。作为阶性报告中的实际使用建议，**该区间预测应一律标为 low confidence**，并依赖 §4.6 中的 `e2e_lower_ms / e2e_upper_ms` 区间估计。

#### 7.4.3 Compute-bound 区：GPU 计算与 descriptor 访存主导

**范围与机制。** CB 区是 `cmp + roof ≫ fix` 的区间，预测近似为 `cmp + roof + bubble ≈0`。实测中该区出现在多体系的 N≥4096：随原子数几乎按 N · sel 量级增长（copper 33.27 → 123.08 ms，water 36.65 → 127.16 ms，LiAlOCl 89.93 → 201.00 ms，he6 N=8192=158.67 ms），同时 CPU launch / dispatch 被 GPU 计算隐藏。

**表现。** CB 区 7 个点 mean error +5.0%，MAE 6.4%，多数点在 ±10% 以内。则是两个点进入边界区：

- LiAlOCl, N=8192：+21.6%，全局最差。该点 `sum(sel) · N ≈ 1.68×10^7`，是谨慎区边界 `1.5×10^7` 以上。
- he6, N=8192：− 4.6%，误差仍在负偊，反映 he6 体系中 profiler v3 阶段拆分本身偏差较大、`cmp+roof` 估计偏低。

**误差抵消现象。** CB 区是 §8.2 和图 4 中 “`cmp` 低估、`roof` 高估、二者叠加抵消” 现象最明显的区域。以 LiAlOCl N=8192 为例，`cmp_P` 相对 profiler `cmp_R` 偏低约 35%，`roof_P` 相对 `roof_R` 偏高约 47%，二者总额仅偏高 17%，wall 误差仅为 +21.6%，其中包含了该体系高 `sel` 带来的附加 descriptor 访存偏差。这是 “仅靠 wall 不能反推阶段” 的典型例证。

**含义。** CB 区预测是当前对大体系估算的主要依据，适合用于实验师估算 GPU·时间需求。但应遵守 §7.4.4 中的使用边界，高 `sel` 体系（如 LiAlOCl）需另行检验。

#### 7.4.4 推荐使用边界

```text
可信范围：N ≤ 4096 的 se_e2_a 体系
谨慎范围：N = 8192 且 sum(sel) * N ≤ 1.5e7
不保证：DPA-1 / N > 8192 / ntypes > 6 / sel 超出当前采样范围
```

在可信范围内，32 个测量点全部落在 ±20% 误差带内，且所有 OH 区及多数 CB 区可标为 high confidence；transition 区不论体系一律标为 low confidence，并用 §4.6 中的 `[lower, upper]` 范围代替点估计。

### 7.5 大于 N=8192 的限制

当前 PyTorch 后端在 N=16384 以上会遇到单张量内存分配瓶颈。DeepMD-PyTorch 参考实现可能构造完整 fp64 pairwise 张量，N=16384 时单块分配约 162 GB，N=32768 时约 648 GB。本机 2×80GB H100 无法稳定验证这些点。若要继续扩展，需要切换 LAMMPS C++ 后端或实现更适合大规模 neighbor list 的分块路径。

---

## 8. 误差分析与当前问题

当前版本已经具备 wall-time 预测能力，但仍存在几个必须说明的问题。这些问题不建议作为“失败”处理，而应作为项目后续研究路线。

### 8.1 `cmp` 仍然系统性低估

修法 A 后，cmp 阶段平均低估约 49%。主要原因是 NeuSight 原始 `MLP_WAVE` 的训练分布更偏向 Transformer / LLM 形状，而 DeepMD 的矩阵形状更特殊，例如：

```text
Linear: batch = N * nnei, hidden = 25/50/100 或 240
BMM:    B = N, M = 4, K = sel_i, N = ng
```

这些 shape 往往是大 batch、小 hidden 或极瘦长矩阵，对原始 MLP_WAVE 来说属于 OOD 输入。因此，opgraph 表达更完整之后，剩余误差主要来自算子级模型本身。

### 8.2 `roof` 对 wall 有效，但阶段物理意义不足

当前 `roof` 在 wall 级别有效，但 profiler v3 显示它不应被直接解释为真实 descriptor kernel 时间。小 N 时 roof 可能低估 70%-99%，大 N 时又可能高估 20%-46%。这是因为当前校准目标只优化 wall，导致 roof 部分承担了补偿 `cmp` 低估的作用。

换句话说：

```text
cmp 单独不准
roof 单独不准
cmp + roof 的总额接近真实 GPU 总时间
```

因此，当前系统可以作为 wall-time predictor，但不能作为 kernel 级瓶颈定位工具。要让阶段拆分独立可信，必须同时完成 MLP_WAVE 重训和 roof 模型重写。

**图 4** 在 water 与 LiAlOCl 两个代表体系上，对 4 个具有代表性的 N 同时绘出预测分量（P，左实色柱）和 profiler 真实分量（R，右斜线柱）。同色对应同一阶段（fix / cmp / roof），黑点为实测 wall。**核心观察：** 在 N=4096 和 N=8192，`cmp(pred)` 明显矮于 `cmp(real)`，而 `roof(pred)` 同时高于 `roof(real)` —— 两条蓝色与橙色双向箭头直接标注了这两类反向偏差。它们叠加后，预测柱与实测柱的总高度几乎相等，因此 wall-time 仍能落在 ±5–10% 的误差带内。这张图是本报告关于“`cmp` 与 `roof` 单独不可解释、wall 准确性部分来自误差抵消”这一结论最直接的实验证据。

![Figure 4. Predicted (P) vs profiler-measured (R) stage decomposition; cmp is systematically under-predicted while roof is over-predicted, and their cancellation keeps wall accurate.](figures/component_decomposition.svg)

### 8.3 Transition bubble 的残留问题

§7.4.2 已从机制、表现和拟合估计三个角度详细说明了 transition 区的偏差来源。总结下来，当前统一的 `bubble_peak_fraction = 0.35` 能把该区间的 MAE 从两位数压缩到 ~ 5%，但仍有两项未完全解决：（1）不同体系需要的 bubble 峰值在 ~ 2% 到 ~ 23% · fix 之间，不能用跨体系常数完全拟合；（2）靠近 `ratio ≈ 2.0` 上边界时，当前 `sin²` 衰减过快，导致 bubble 修正接近 0 但实际偏差仍明显。后续可考虑不依赖均一 0.35，而是用 `ntypes`、`sel`、kernel 数分布等特征训练 per-system bubble fraction，同时适当上推 `transition_hi`。

### 8.4 DPA-1 尚未支持

water_dpa1 的已有测试表明，DPA-1 attention descriptor 误差可达约 -33%。这说明当前 `se_e2_a` 模型不能直接推广到 attention descriptor。DPA-1 需要单独建模 attention BMM、softmax、注意力邻居聚合和额外 fixed overhead。

---

## 9. 项目成果

### 9.1 工程成果

本项目已经形成以下工程产出：

1. DeepMD 配置解析器：支持从 DeepMD 输入配置中抽取预测所需结构信息。
2. DeepMD 静态算子图构造器：可生成 NeuSight 兼容 opgraph。
3. DeepMD 专用预测入口：复用 NeuSight `OperatorPredictor` 后端完成逐算子预测。
4. v6 overhead model：实现 kernel-count fixed overhead、roofline 和 transition bubble。
5. 多体系 benchmark 脚本：支持 copper、water、LiAlOCl、he6 等体系验证。
6. profiler v3 阶段拆分脚本：修正 topk/norm/cat 等 DeepMD 特有 kernel 分类。
7. P1 数据采集 scaffold：已准备 DeepMD-shape Linear/BMM 重训数据入口。

### 9.2 实验成果

主要实验成果包括：

- 完成 H100 NVL 上 4 个 `se_e2_a` 体系、36 个主测量点验证。
- 建立 N≤4096 的推荐使用范围。
- 识别 N=8192 高 `sel` 体系的谨慎边界。
- 验证 fixed overhead 与 `ntypes^p`、kernel count 的强相关性。
- 定位 transition bubble 是 `max(fix, cmp+roof)` 模型的主要误差来源之一。
- 通过 profiler v3 确认 `cmp/roof` 误差抵消现象，避免过度解读阶段拆分。

### 9.3 文档与数据沉淀

项目中已沉淀以下资料：

| 类型 | 路径 | 内容 |
|---|---|---|
| 当前最终报告 | `docs/FINAL_REPORT.md` | 本文档 |
| 当前验证报告 | `docs/VALIDATION_REPORT.md` | 36 点验证、阶段拆分和统计分析 |
| 实验过程报告 | `DeepMD_Experiment_Report.md` | 从早期阶段到 v6/P0/P1 的完整记录 |
| 代码变更清单 | `CODE_CHANGES.md` | 新增和修改文件总览 |
| 当前校准文件 | `results/calibration/h100_nvl_v6.json` | H100 v6 参数 |
| 主验证结果 | `results/cross_system/v3opgraph_summary.json` | 修法 A 后 36 点结果 |
| 阶段拆分结果 | `results/cross_system/real_breakdown_v3.json` | profiler v3 桶分类结果 |

---

## 10. 后续工作计划

当前项目最主要的不足集中在 `cmp` 部分，也就是 NeuSight 原始 `MLP_WAVE` 对 DeepMD 特有算子 shape 的适配不足。因此，后续工作建议聚焦于算子级预测器的训练分布扩展，而不在本报告中展开其他方向。

具体来说，下一步应重点补充 DeepMD 场景下的 Linear 和 BMM 训练样本。DeepMD 的 embedding network 和 descriptor 计算会产生大量与 Transformer 不同的矩阵形状，例如大 batch、小 hidden 的 Linear，以及 `B=N, M=4, K=sel_i, N=ng` 这类极瘦长 BMM。当前 `MLP_WAVE` 对这些 shape 存在明显 OOD 问题，导致 `cmp` 阶段平均低估约 49%。

因此，后续计划可以聚焦为三项：

1. **补充 DeepMD-shape 算子 profile 数据。** 继续使用已有 `collect_he6_dpa1_op_samples.py` scaffold，扩大 Linear/BMM 样本覆盖范围，重点覆盖不同 `N`、`sel`、`ntypes` 和 embedding/fitting 网络宽度下的真实算子延迟。
2. **重训或微调 NeuSight 的 `MLP_WAVE`。** 将 DeepMD-shape 样本加入 NeuSight 训练集，分别优化 Linear 和 BMM predictor，使 `cmp` 不再系统性低估。
3. **重新校准端到端模型。** 在 `cmp` 预测改善后，重新拟合 fixed/roof 相关参数，检查 wall-time 精度是否保持稳定，并确认 `cmp` 与 `roof` 的阶段拆分是否比当前版本更可信。

这部分既是当前项目的主要不足，也是最直接的改进路线。若 `MLP_WAVE` 对 DeepMD shape 的预测能力得到提升，当前模型中依赖 `roof` 补偿 `cmp` 误差的现象会减弱，端到端预测将更具物理可解释性。

---

## 11. 复现方式

### 11.1 主验证命令

```bash
cd NeuSight_MD
export NEUSIGHT_DEEPMD_CALIBRATION=results/calibration/h100_nvl_v6.json

python scripts/benchmark_large_atoms_v6.py \
  --atoms 32 64 128 256 512 1024 2048 4096 8192 \
  --systems copper water LiAlOCl he6 \
  --warmup 10 --runs 30
```

### 11.2 阶段拆分验证命令

```bash
python scripts/measure_real_breakdown_v3.py \
  --atoms 32 64 128 256 512 1024 2048 4096 8192 \
  --systems copper water LiAlOCl he6 \
  --warmup 10 --runs 30
```

### 11.3 关键文件

| 文件 | 作用 |
|---|---|
| `results/calibration/h100_nvl_v6.json` | 当前有效校准参数 |
| `results/cross_system/v3opgraph_summary.json` | 最终保守口径的 36 点结果 |
| `results/cross_system/v3_summary.json` | 旧 opgraph 对照结果 |
| `results/cross_system/real_breakdown_v3.json` | profiler v3 阶段拆分结果 |
| `docs/VALIDATION_REPORT.md` | 当前最详细验证报告 |
| `CODE_CHANGES.md` | 项目代码改动清单 |

---

## 12. 结论

本项目完成了从 NeuSight 到 DeepMD-kit 推理性能预测器的核心改造，形成了一套可复现、可解释、在主验证范围内可用的端到端 wall-time 预测方法。项目的主要贡献包括：

1. 构建了 DeepMD-kit 专用静态算子图生成前端，使 NeuSight 的算子级预测能力能够应用于 DeepMD 推理。
2. 提出了面向 DeepMD 的混合端到端模型，将 `cmp`、`fix`、`roof` 和 `bubble` 组合为 wall-time 预测。
3. 通过 kernel-count 和 `ntypes^p` 结构项解决了多元素体系 fixed overhead 外推问题。
4. 在 H100 NVL 上完成 4 个 `se_e2_a` 体系、36 个测量点验证，最终保守口径达到 7.1% MAE 和 8.3% RMSE。
5. 通过 profiler v3 明确识别了当前 `cmp/roof` 阶段拆分的误差抵消问题，为后续模型改进提供了清晰方向。

当前版本的定位应当是：**H100 上 `se_e2_a` DeepMD-kit 模型的端到端 wall-time 预测器**。在 N≤4096 范围内，其预测结果已经具有较好的工程参考价值；在 N=8192、高 `sel`、DPA-1、跨 GPU 和更大规模场景下，需要继续校准和扩展。

下一阶段建议优先推进 DeepMD-shape `MLP_WAVE` 数据补充与重训，重点解决当前 `cmp` 部分对 DeepMD 特有 Linear/BMM shape 的系统性低估问题。只有这一部分改善后，当前端到端预测中的阶段拆分才有可能从“wall-time 可用”进一步提升到“内部归因可信”。