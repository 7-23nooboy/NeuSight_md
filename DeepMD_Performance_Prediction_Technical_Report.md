# DeepMD-kit 推理性能预测：源码分析、图分解建模与实验验证

## 目录

1. [问题定义](#1-问题定义)
2. [DeepMD-kit 源码分析与计算图分解](#2-deepmd-kit-源码分析与计算图分解)
3. [算子到 NeuSight 的映射](#3-算子到-neusight-的映射)
4. [Overhead 模型：未建模计算的解析建模](#4-overhead-模型未建模计算的解析建模)
5. [完整预测流水线](#5-完整预测流水线)
6. [实验数据与精度分析](#6-实验数据与精度分析)
7. [转换区误差分析：Pipeline Bubble](#7-转换区误差分析pipeline-bubble)
8. [解决方案：Confidence-Aware Prediction](#8-解决方案confidence-aware-prediction)
9. [总结](#9-总结)

---

## 1. 问题定义

**目标**: 给定一个 DeepMD-kit 模型配置（descriptor 类型、embedding/fitting 网络结构）、原子数 N 和 GPU 型号，**不运行实际推理**，预测其端到端推理延迟。

**难点**:
- NeuSight 原本为 Transformer 模型设计（HuggingFace FX tracing），无法直接用于 DeepMD-kit
- DeepMD-kit 的推理过程包含大量**未被 NeuSight 算子覆盖**的计算（neighbor list、env_mat 等）
- 固定开销（CUDA kernel launch chain）在小原子数时主导延迟，而 NeuSight 的 MLP_WAVE predictor 只预测纯 GPU 计算

**方案**: 保留 NeuSight 后端的 MLP_WAVE predictor（已训练好的算子级 GPU 性能预测器），重写前端——通过分析 DeepMD-kit 源码，将推理过程**解析式**地分解为 NeuSight 兼容的算子序列。

---

## 2. DeepMD-kit 源码分析与计算图分解

### 2.1 推理过程的 7 个阶段

通过阅读 DeepMD-kit PyTorch 后端源码，将推理过程分解为 7 个阶段：

```
阶段 1: Neighbor List 构建        ← 未建模，归入 overhead model
阶段 2: Environment Matrix 构造   ← 部分建模
阶段 3: Embedding Network (per-type) ← 核心建模
阶段 4: Descriptor 矩阵运算       ← 核心建模
阶段 5: Fitting Network           ← 核心建模
阶段 6: Output (energy sum)       ← 建模
阶段 7: Force backward (autograd) ← 建模 (可选)
```

### 2.2 阶段 1: Neighbor List — 源码分析

**源码位置**: `deepmd/pt/model/descriptor/nlist.py` `build_neighbor_list()` L47-135

```python
# 核心计算 — O(N²) broadcast distance
diff = coord.unsqueeze(1) - coord_local.unsqueeze(2)  # [nframes, N, nall, 3]
# nall = ns × N, ns ≈ 27 (ghost cell replicas)
# 数据量 = N × nall × 3 × 8 bytes (float64)
distance = torch.linalg.norm(diff, dim=-1)             # [N, nall]
topk_result = torch.topk(distance, sel, largest=False)  # per-type 筛选
```

**关键发现**: 这一步的计算量为 O(N²)，其中 `nall = ns × N`（ns 为 ghost cell 复制因子，通常为 27）。这是一个纯 memory-bandwidth-bound 操作（大张量的逐元素广播 + norm + topk），NeuSight 的 MLP_WAVE 没有对应的算子类型。

**处理方式**: 不在算子图中建模，改用解析 roofline 模型估算（见第 4 节）。

### 2.3 阶段 2: Environment Matrix — 源码分析

**源码位置**: `deepmd/pt/model/descriptor/env_mat.py` `_make_env_mat()` L11-48

```python
coord_r = torch.gather(coord_pad, 1, index)    # gather 邻居坐标 [N, M, 3]
diff = coord_r - coord_l                         # 坐标差
length = torch.linalg.norm(diff, dim=-1)         # 距离 |r|
t0 = 1 / length                                  # 1/r
t1 = diff / length.unsqueeze(-1)**2              # x/r², y/r², z/r²
weight = compute_smooth_weight(length, ...)       # 5阶多项式平滑
env_mat = torch.cat([t0.unsqueeze(-1), t1], dim=-1) * weight  # [N, M, 4]
```

**算子分解**: 这一步涉及 gather (MEM)、逐元素运算 (VECadd/VECmul)、归一化 (VECadd)。全部用 NeuSight 已有的算子类型表达。

在 `trace_deepmd.py` 中的实现：

```python
def _build_env_matrix_ops(N, M):
    ops = []
    ops.append(_op("env_gather",       "MEM",    [("MEM", [(N, M, 3)])], ...))
    ops.append(_op("env_diff_norm",    "VECadd", [("VECadd", (NM, 3))], ...))
    ops.append(_op("env_inv_r",        "VECmul", [("VECmul", (NM, 4))], ...))
    ops.append(_op("env_smooth_weight","VECmul", [("VECmul", (NM, 1))], ...))
    ops.append(_op("env_mat_mul",      "VECmul", [("VECmul", (NM, 4))], ...))
    ops.append(_op("env_normalize",    "VECadd", [("VECadd", (NM, 4))], ...))
    return ops
```

### 2.4 阶段 3: Embedding Network — 源码分析（最关键）

**源码位置**: `deepmd/pt/model/descriptor/se_a.py` `DescrptBlockSeA.forward()` L789-836

```python
# type_one_side=True 时：ntypes 个独立 embedding 网络
for ii in range(self.ntypes):
    rr = dmatrix[:, sec[ii]:sec[ii+1], :]   # 取第 ii 种类型的 sel[ii] 个邻居
    ss = rr[:, :, :1]                         # 标量输入 s(r) = 1/r
    gg = self.filter_layers[ii].forward(ss)   # 独立 embedding net
    gr = torch.matmul(rr.permute(0,2,1), gg)  # [N, 4, sel[ii]] × [N, sel[ii], ng] → [N, 4, ng]
    xyz_scatter += gr                          # 累加
```

**v2 修复的关键问题**: 初版 (v1) 将所有类型的 embedding 合并为一个网络（batch = N × M，M = sum(sel)），这与源码不符。v2 修正为按类型独立构建：

```python
def _build_embedding_net_ops(N, sel, neuron, ...):
    for ti in range(ntypes):
        ni = sel[ti]          # 该类型邻居数, e.g. sel=[46, 92]
        batch = N * ni        # 独立的 batch size

        # 每种类型独立的 embedding MLP
        ops.extend(_build_single_embedding_net_ops(f"emb_t{ti}", batch, neuron))

        # per-type 的 matmul: rr.T @ gg
        ops.append(_op(f"emb_t{ti}_matmul", "BMM",
                       [("BMM", (N, 4, ni, ng))], ...))
```

**影响**: 对于 Water (2 types, sel=[46, 92])：
- v1 错误做法: 1 个 embedding net, batch = N × 138
- v2 正确做法: 2 个独立 net, batch₀ = N × 46, batch₁ = N × 92

这影响了 MLP_WAVE predictor 的 tile 选择和 wave 计算，因为小 batch 和大 batch 的 GPU 利用率差别很大。

### 2.5 阶段 3 内部: Embedding MLP 的 ResNet 结构

**源码位置**: `deepmd/pt/model/network/mlp.py` `MLPLayer.forward()` L188-219

```python
yy = F.linear(xx, weight, bias)     # Linear
yy = torch.tanh(yy)                  # Activation
if self.resnet:
    if xx.shape[-1] == yy.shape[-1]:
        yy = yy + xx                 # same dim: 直接 add
    elif 2 * xx.shape[-1] == yy.shape[-1]:
        yy = yy + torch.cat([xx, xx], dim=-1)  # double dim: concat + add
```

对于典型的 embedding net [25, 50, 100]：
- Layer 0: Linear(1→25), tanh — 无 skip（第一层）
- Layer 1: Linear(25→50), tanh, concat([25,25])→50, add — double dim skip
- Layer 2: Linear(50→100), tanh, concat([50,50])→100, add — double dim skip

在算子图中，每个 double dim skip 会生成一个 `MEM`（concat）+ 一个 `VECadd`（add）。

### 2.6 阶段 4: Descriptor 矩阵运算

**源码位置**: `se_a.py` L838-844

```python
xyz_scatter /= nnei                                    # 归一化
xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)           # [N, ng, 4]
xyz_scatter_2 = xyz_scatter[:, :, :self.axis_neuron]   # [N, 4, axis_neuron]
result = torch.matmul(xyz_scatter_1, xyz_scatter_2)    # [N, ng, axis_neuron]
# → reshape to [N, ng × axis_neuron] 作为 descriptor
```

建模为: `VECmul`（div by nnei）+ `BMM`（[N, ng, 4] × [N, 4, axis_neuron]）+ `MEM`（reshape）

### 2.7 阶段 5: Fitting Network

**源码位置**: `fitting.py` `GeneralFitting._forward_common()` L847-857

与 embedding net 类似的 MLP 结构，但输入维度为 descriptor dim（ng × axis_neuron = 100 × 16 = 1600），网络为 [240, 240, 240]。

### 2.8 阶段 7: Force backward

**源码位置**: `transform_output.py` `task_deriv_one()` L65-96

```python
extended_force = torch.autograd.grad(
    [energy], [extended_coord],
    grad_outputs=[grad_output], retain_graph=True, create_graph=False
)[0]
```

Autograd 会反向遍历: output → fitting → descriptor → embedding。每个 Linear(B, I, O) 的 backward 对应一个 Linear(B, O, I)（grad_input 计算）。

---

## 3. 算子到 NeuSight 的映射

### 3.1 映射表

| DeepMD 源码操作 | NeuSight 算子 | 形状 | 说明 |
|---|---|---|---|
| `torch.gather(coord, index)` | `MEM` | (N, M, 3) | 纯内存访问 |
| `coord_r - coord_l` | `VECadd` | (NM, 3) | 逐元素减法 |
| `1/r, x/r²` | `VECmul` | (NM, 4) | 逐元素除法 |
| `smooth_weight(r)` | `VECmul` | (NM, 1) | 5 阶多项式 |
| `F.linear(x, W, b)` | `Linear` | (batch, in, out) | GEMM |
| `torch.tanh(x)` | `VECtanh` | (batch, dim) | 激活函数 |
| `y + x` (ResNet skip) | `VECadd` | (batch, dim) | 逐元素加 |
| `cat([x, x])` (double dim) | `MEM` | (batch, 2×dim) | concat |
| `matmul(rr.T, gg)` | `BMM` | (N, 4, sel_i, ng) | batch matmul |
| `matmul(G1, G2)` (descriptor) | `BMM` | (N, ng, 4, axis_n) | batch matmul |
| `sum(energy)` | `VECadd` | (N, 1) | reduction |
| `autograd.grad(Linear)` | `Linear` | (B, O, I) | 反向 GEMM |

### 3.2 完整算子图示例

以 Water 模型 (sel=[46,92], emb=[25,50,100], fit=[240,240,240], N=32, force=True) 为例，v2 opgraph 生成 **65 个算子节点**：

```
[Neighbor List]    1 MEM
[Env Matrix]       6 ops: MEM + VECadd + VECmul×3 + VECadd
[Embedding t0]     9 ops: (Linear + VECtanh + MEM/VECadd)×3 + BMM
[Embedding t1]     10 ops: 同上 + VECadd (累加)
[Descriptor]       3 ops: VECmul + BMM + MEM
[Fitting Net]      9 ops: (Linear + VECtanh + VECadd)×3
[Output]           2 ops: Linear + VECadd
[Force backward]   ~25 ops: 反向 MLP + 反向 descriptor + 反向 embedding
```

### 3.3 聚合方式

DeepMD 推理不像 Transformer 有层复制结构，所有算子的 latency **简单求和**即可：

```python
def aggregate_deepmd(trace):
    fw = trace["fw_latency"].sum()
    bw = trace["bw_latency"].sum()
    acc = trace["acc_latency"].sum()
    e2e = fw + bw + acc
    return e2e, fw, bw, bwall, acc
```

---

## 4. Overhead 模型：未建模计算的解析建模

### 4.1 问题

NeuSight 的 MLP_WAVE predictor 只预测了**建模算子**（Linear/BMM/VEC/MEM）的 GPU 计算时间。实际推理中还有两类额外开销：

1. **固定开销 (fixed overhead)**: ~350 个 CUDA kernel 的 launch chain，约 5-6ms，不随原子数变化
2. **未建模 GPU 计算**: neighbor list 的 O(N²) broadcast distance + env_mat 的 random gather 等

### 4.2 固定开销分析

通过 Nsight Systems profiling，发现 N=32 到 N=1024 的实测延迟几乎恒定（~5.9ms for Water），而 MLP_WAVE 预测的纯计算时间只有 ~1.1ms。差值 ~4.8ms 来自约 350 个小 CUDA kernel 的 launch overhead：

```
fixed_overhead ≈ n_kernels × per_launch_us
               ≈ 350 × 16μs = 5.6ms  (H100 NVL)
```

不同模型结构的固定开销不同（kernel 数量不同）：
- Water (2 types): 5.715ms
- Copper (1 type): 4.850ms

### 4.3 未建模 GPU 计算的解析 Roofline 模型

通过分析 DeepMD-kit 源码，识别出两个主要的未建模计算分量：

**(a) O(N²) 分量 — neighbor list broadcast distance:**

```python
# nlist.py:
diff = coord.unsqueeze(1) - coord_local.unsqueeze(2)  # [N, nall, 3]
# 数据量 = N × nall × 8 bytes, nall = ns × N
```

延迟由内存带宽决定:
```
quad_ms = C_QUAD × N × nall × 8 / GPU_MEM_BW × 1000
```

**(b) O(N) 分量 — env_mat gather + type sort:**

```python
# env_mat.py:
torch.gather(coord_pad, 1, index)  # random access [N, nnei, 4]
```

延迟:
```
linear_ms = C_LINEAR × N × nnei × 8 / GPU_MEM_BW × 1000
```

**(c) 系数拟合:**

使用 Water + Copper 的 ground truth 数据，用 `scipy.optimize.minimize (L-BFGS-B)` 拟合两个系数：

```python
C_QUAD = 28.3284     # broadcast distance 的有效乘数 (多遍访问 + partial sort)
C_LINEAR = 2053.2321 # random gather 的有效乘数 (cache miss penalty)
```

### 4.4 两区间模型

```python
# 完整公式
gpu_overhead_ms = C_QUAD × N × ns × N × 8 / BW + C_LINEAR × N × nnei × 8 / BW
adjusted_compute_ms = mlp_compute_ms + gpu_overhead_ms
e2e_total_ms = max(fixed_overhead_ms, adjusted_compute_ms)
```

物理含义：CPU 提交 kernel launch chain（固定时间）与 GPU 执行计算（随 N 增长）形成**流水线 (pipeline)**。总时间取两者的 max：

- 小 N: 计算快于 launch → e2e ≈ fixed_overhead
- 大 N: 计算慢于 launch → e2e ≈ mlp_compute + gpu_overhead

---

## 5. 完整预测流水线

```
输入: GPU 配置 + DeepMD 模型配置 + 原子数 N
  │
  ├──→ parse_deepmd_input()          解析 DeepMD input.json
  │      提取 sel, neuron, type_map, rcut 等
  │
  ├──→ build_deepmd_opgraph()        解析式生成算子图
  │      7 个阶段 → ~65 个 NeuSight 算子节点
  │
  ├──→ OperatorPredictor.predict()   逐算子预测
  │      Linear → MLP_WAVE_MM
  │      BMM    → MLP_WAVE_MM
  │      VEC*   → MLP_WAVE_VEC
  │      MEM    → BW 公式估算
  │
  ├──→ aggregate_deepmd()            简单求和
  │      compute_latency = Σ fw_latency
  │
  ├──→ DeepMDOverheadModel.estimate() 估算 overhead
  │      fixed_overhead + analytical_gpu_overhead
  │      e2e = max(fixed, compute + gpu_oh)
  │
  └──→ 输出: e2e_latency, confidence, bounds
```

---

## 6. 实验数据与精度分析

### 6.1 实验环境

- **GPU**: NVIDIA H100 80GB NVL (Mem_BW = 3430 GB/s)
- **DeepMD-kit**: v3.x, PyTorch backend
- **测试模型**:
  - Water: se_e2_a, 2 types (O, H), sel=[46, 92], emb=[25,50,100], fit=[240,240,240]
  - Copper: se_e2_a, 1 type (Cu), sel=[120], emb=[25,50,100], fit=[240,240,240]
- **Profiling**: warmup 30 次, 测量 100 次, 取 mean, 确认 std < 1% (clean GPU)

### 6.2 Water 模型精度数据 (N=32 ~ 2048)

| N | 实测 (ms) | std | 预测 (ms) | compute | overhead | 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 5.890 | 0.043 | 5.865 | 1.136 | 4.729 | -0.4% |
| 64 | 5.893 | 0.293 | 5.865 | 1.072 | 4.793 | -0.5% |
| 128 | 5.902 | 0.550 | 5.865 | 1.083 | 4.782 | -0.6% |
| 192 | 5.995 | 0.354 | 5.865 | 1.040 | 4.825 | -2.2% |
| 256 | 5.939 | 0.148 | 5.865 | 1.077 | 4.788 | -1.2% |
| 512 | 5.941 | 0.127 | 5.865 | 1.369 | 4.496 | -1.3% |
| 1024 | 5.873 | 0.035 | 5.865 | 1.683 | 4.182 | -0.1% |
| 2048 | 11.823 | 0.030 | 11.268 | 2.432 | 8.836 | -4.7% |

**误差统计**: MAE = 1.4%, 最大误差 = -4.7% (N=2048)

### 6.3 大原子数精度数据 (N=4096 ~ 8192)

| 模型 | N | 实测 (ms) | std | 预测 (ms) | 误差 |
|---:|---:|---:|---:|---:|---:|
| Water | 4096 | 35.532 | 0.266 | 36.597 | +3.0% |
| Water | 8192 | 129.113 | 4.973 | 132.327 | +2.5% |
| Copper | 2048 | 11.137 | 0.023 | 11.091 | -0.4% |
| Copper | 4096 | 33.763 | 0.282 | 36.244 | +7.3% |
| Copper | 8192 | 125.034 | 3.302 | 131.620 | +5.3% |

**误差统计**: MAE = 3.7%, 最大误差 = +7.3%

### 6.4 精度分区总结

| 区间 | N 范围 (Water) | MAE | 最大误差 | 评价 |
|---:|---:|---:|---:|---:|
| Overhead-bound | 32 ~ 1024 | 1.0% | -2.2% | 非常好 |
| 转换区 | ~1280 ~ 1920 | ~10% | ~15% | 有系统性误差 |
| Compute-bound | 2048 ~ 8192 | 3.7% | 7.3% | 可接受 |

---

## 7. 转换区误差分析：Pipeline Bubble

### 7.1 现象

在 overhead-bound 到 compute-bound 的过渡区间（Water 约 N=1280~1920），预测误差显著增大，最高可达 ~15%。

密集 profiling 数据（N=768~2048, step=128, 每点 100 次）：

| N | 实测 (ms) | max(fixed, adj) | 误差 |
|---:|---:|---:|---:|
| 1024 | 5.873 | 5.865 | -0.1% |
| 1152 | 5.942 | 5.865 | -1.3% |
| 1280 | 6.214 | 5.865 | -5.6% |
| 1408 | 6.640 | 6.467 | -2.6% |
| 1536 | 6.950 | 7.324 | +5.4% |
| 1664 | 7.525 | 8.219 | +9.2% |
| 1792 | 8.275 | 9.213 | +11.3% |
| 1920 | 9.378 | 10.225 | +9.0% |
| 2048 | 11.823 | 11.266 | -4.7% |

### 7.2 根因分析

**Pipeline Bubble** — 硬 max() 模型的数学缺陷。

模型假设 CPU launch chain 和 GPU 计算是完美流水线：

```
e2e = max(Σ launch_i, Σ compute_i)
```

但实际 GPU 执行中，大小 kernel 交替调度：

```
实际 e2e = Σ max(launch_i, compute_i)  ≠  max(Σ launch_i, Σ compute_i)
```

**具体示例**: 假设有 3 个 kernel，每个 launch overhead = 15μs：
- Kernel A (大): compute = 50μs → max(15, 50) = 50μs
- Kernel B (小): compute = 5μs  → max(15, 5) = 15μs
- Kernel C (大): compute = 50μs → max(15, 50) = 50μs

```
实际 e2e = 50 + 15 + 50 = 115μs
max() 预测 = max(45, 105) = 105μs  (-8.7%)
```

当处于转换区（总 compute ≈ 总 launch）时，这种"大小 kernel 交替的 idle gap"效应最为显著。

### 7.3 Bubble 的定量特征

通过 Water 和 Copper 的密集 profiling 数据分析：

**Crossover 公式** — 转换区中心（ratio = 1.0）对应的原子数：

```
N_cross ≈ √(fixed_overhead × BW / (C_QUAD × ns × 8))
```

**Bubble 幅度**: 约为 fixed_overhead 的 15-25%
- Water (2 types): peak bubble ≈ 25.1% of fixed
- Copper (1 type): peak bubble ≈ 15.6% of fixed

**不泛化原因**: Bubble 大小取决于 kernel 计算量的**分布方差**：
- Water 有 2 个不同大小的 embedding net（batch = N×46 vs N×92），方差更大 → bubble 更大
- Copper 只有 1 个 embedding net，kernel 大小更均匀 → bubble 更小

### 7.4 尝试过的修正方案

| 方案 | 结果 | 失败原因 |
|---|---|---|
| 全局 softmax 平滑 | 破坏两端精度 | 影响范围过大 |
| 局部 Gaussian 修正 | Water OK, Copper 不泛化 | bubble 高度依赖模型结构 |
| 非对称 Gaussian | 同上 | peak 差 60% (Water 0.25 vs Copper 0.16) |
| Per-kernel 流水线模拟 | MAE 反而更差 (13.0%) | framework kernel 分布未知 |

---

## 8. 解决方案：Confidence-Aware Prediction

### 8.1 设计思路

既然转换区的精确值无法用统一参数校正，采用**诚实标注**策略：

```
当预测处于转换区时，不给一个不准确的点预测，
而是标注 confidence="low" 并给出参考性 bounds。
```

### 8.2 实现

**转换区检测** — 基于物理定义，跨 GPU 通用：

```python
ratio = adjusted_compute_ms / fixed_overhead_ms
# ratio = 1.0 永远是 crossover point (物理定义)

if ratio < 0.8:     regime = "overhead-bound",  confidence = "high"
elif ratio > 2.0:   regime = "compute-bound",   confidence = "high"
else:                regime = "transition",      confidence = "low"
```

**参考性 bounds** — 对称区间，Gaussian 衰减：

```python
uncertainty = BUBBLE_PEAK_FRACTION × fixed × exp(-0.5 × ((ratio-1)/σ)²)
# 非对称 σ: σ_left=0.2 (陡峭), σ_right=1.0 (缓慢衰减)
lower = point_estimate - uncertainty
upper = point_estimate + uncertainty
```

### 8.3 验证

| 数据集 | 转换区点数 | Bounds 覆盖率 | 说明 |
|---:|---:|---:|---:|
| Water | 8 | 100% (8/8) | 校准数据 |
| Copper | 7 | 100% (7/7) | 泛化验证 |

### 8.4 输出格式

```json
{
  "e2e_latency": 7.338,
  "confidence": {
    "level": "low",
    "regime": "transition",
    "transition_ratio": 1.251,
    "e2e_lower_ms": 6.20,
    "e2e_upper_ms": 8.47
  }
}
```

终端输出示例：
```
DeepMD E2E latency for deepmd_se_e2_a_n1536_force on H100: 7.338 ms ⚠️ transition zone [6.20, 8.47]ms
```

### 8.5 泛化性说明

- **confidence 标注** (high/low): 基于 ratio 的物理定义，跨 GPU、跨模型通用
- **bounds 宽度**: 基于 H100 NVL 实测校准，在其他 GPU 上可能偏大或偏小，仅作参考

---

## 9. 总结

### 9.1 技术路线

```
DeepMD-kit 源码分析
    → 7 阶段推理过程分解
    → 算子映射到 NeuSight (Linear/BMM/VEC/MEM)
    → MLP_WAVE 预测纯计算时间
    → 解析 Roofline 补偿未建模 GPU 计算 (O(N²) + O(N))
    → max(fixed, compute) 两区间模型
    → Confidence-Aware 转换区标注
```

### 9.2 精度

| 区间 | MAE | 最大误差 | Confidence |
|---:|---:|---:|---:|
| Overhead-bound (N≤1024) | 1.0% | 2.2% | high |
| Transition (~1280-1920) | ~10% | ~15% | low (标注) |
| Compute-bound (N≥2048) | 3.7% | 7.3% | high |

### 9.3 代码清单

| 文件 | 职责 |
|---|---|
| `neusight/Tracing/trace_deepmd.py` | 解析式算子图构建 (7 阶段分解) |
| `neusight/Tracing/parse_deepmd_input.py` | DeepMD input.json 解析 |
| `neusight/Prediction/predictor_deepmd.py` | DeepMD 预测器入口 |
| `neusight/Prediction/overhead_model.py` | 两区间 overhead 模型 + confidence |
| `neusight/Prediction/aggregator.py` | aggregate_deepmd() 简单求和 |
| `scripts/pred_deepmd.py` | CLI 入口 |
| `scripts/full_accuracy_test.py` | N=32~2048 精度验证 |
| `scripts/test_large_atoms.py` | N=4096~8192 精度验证 |
| `scripts/verify_confidence_aware.py` | Confidence-Aware 功能验证 |

### 9.4 已识别的局限性

1. **转换区 (~15% 误差)**: Pipeline bubble 无法用统一参数精确校正，已用 confidence 标注处理
2. **固定开销的 GPU 依赖**: fixed_overhead 需要 per-GPU 校准（不同 GPU 的 kernel launch latency 不同）
3. **大原子数 (~7% 误差)**: O(N²) 的 roofline 系数 C_QUAD 在极端大 N 时可能需要修正（cache 效应）
4. **DPA-1 (se_atten) 未充分验证**: 当前主要在 se_e2_a 上验证，attention-based descriptor 需要更多实验
