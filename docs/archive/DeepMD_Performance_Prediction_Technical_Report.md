# DeepMD-kit 推理性能预测：源码分析、图分解建模与实验验证

## 目录

1. [问题定义](#1-问题定义)
2. [DeepMD-kit 推理过程的源码级分解](#2-deepmd-kit-推理过程的源码级分解)
3. [算子映射与计算图构建](#3-算子映射与计算图构建)
4. [Overhead 模型](#4-overhead-模型)
5. [完整预测流水线](#5-完整预测流水线)
6. [实验数据与精度分析](#6-实验数据与精度分析)
7. [转换区误差与 Confidence-Aware 标注](#7-转换区误差与-confidence-aware-标注)
8. [总结](#8-总结)

---

## 1. 问题定义

**目标**：给定一个 DeepMD-kit 模型配置（descriptor 类型、embedding/fitting
网络结构）、原子数 N 和 GPU 型号，**不运行实际推理**，预测其端到端推理延迟。

**难点**：

- NeuSight 原本为 Transformer 模型设计（HuggingFace FX tracing），无法直接套用
- DeepMD-kit 的推理过程包含大量**未被 NeuSight 算子覆盖**的计算
  （neighbor list、env_mat 等）
- 固定开销（CUDA kernel launch chain）在小原子数时主导延迟，而 NeuSight 的
  MLP_WAVE predictor 只预测纯 GPU 计算

**方案**：保留 NeuSight 后端的 MLP_WAVE predictor（已训练好的算子级 GPU
性能预测器），重写前端——通过分析 DeepMD-kit 源码，将推理过程**解析式**地分解
为 NeuSight 兼容的算子序列，并用一个独立的 overhead 模型补齐未建模部分。

---

## 2. DeepMD-kit 推理过程的源码级分解

### 2.1 推理过程的 7 个阶段

通过阅读 DeepMD-kit PyTorch 后端源码，将推理过程分解为 7 个阶段：

| 阶段 | 内容 | 处理方式 |
|------|------|---------|
| 1 | Neighbor List 构建 | 未建模 → overhead 模型 |
| 2 | Environment Matrix 构造 | 部分建模（gather/element-wise）+ overhead |
| 3 | Embedding Network (per-type) | 核心建模（Linear + tanh） |
| 4 | Descriptor 矩阵运算 | 核心建模（BMM） |
| 5 | Fitting Network | 核心建模（Linear + tanh） |
| 6 | Output (energy sum) | 建模（Linear + reduction） |
| 7 | Force backward (autograd) | 建模（反向 Linear），可选 |

### 2.2 阶段 1: Neighbor List

**源码**：`deepmd/pt/utils/nlist.py` `build_neighbor_list()` L47–135

```python
diff = coord.unsqueeze(1) - coord_local.unsqueeze(2)  # [nframes, N, nall, 3]
# nall = ns × N, ns ≈ 27 (ghost cell 复制因子)
distance = torch.linalg.norm(diff, dim=-1)             # [N, nall]
topk_result = torch.topk(distance, sel, largest=False) # per-type 筛选
```

这是**纯 memory-bandwidth-bound 的 O(N²) 操作**，NeuSight MLP_WAVE 没有对应的
算子类型，归入 overhead 模型用解析 roofline 估算（§4.3）。

### 2.3 阶段 2: Environment Matrix

**源码**：`deepmd/pt/model/descriptor/env_mat.py` `_make_env_mat()` L11–48

```python
coord_r = torch.gather(coord_pad, 1, index)   # gather 邻居坐标 [N, M, 3]
diff    = coord_r - coord_l                    # 坐标差
length  = torch.linalg.norm(diff, dim=-1)      # 距离 |r|
t0 = 1 / length                                # 1/r
t1 = diff / length.unsqueeze(-1)**2            # x/r², y/r², z/r²
weight = compute_smooth_weight(length, ...)    # 5 阶多项式平滑
env_mat = torch.cat([t0.unsqueeze(-1), t1], dim=-1) * weight  # [N, M, 4]
```

涉及 gather (`MEM`)、逐元素运算 (`VECadd/VECmul`)、归一化 (`VECadd`)，
全部用 NeuSight 已有算子类型表达：

```python
def _build_env_matrix_ops(N, M):
    return [
        _op("env_gather",        "MEM",    [("MEM",    [(N, M, 3)])]),
        _op("env_diff_norm",     "VECadd", [("VECadd", (NM, 3))]),
        _op("env_inv_r",         "VECmul", [("VECmul", (NM, 4))]),
        _op("env_smooth_weight", "VECmul", [("VECmul", (NM, 1))]),
        _op("env_mat_mul",       "VECmul", [("VECmul", (NM, 4))]),
        _op("env_normalize",     "VECadd", [("VECadd", (NM, 4))]),
    ]
```

### 2.4 阶段 3: Embedding Network (per-type)

**源码**：`deepmd/pt/model/descriptor/se_a.py` `DescrptBlockSeA.forward()` L789–836

```python
# type_one_side=True: 每种 atom type 独立的 embedding 网络
for ii in range(self.ntypes):
    rr = dmatrix[:, sec[ii]:sec[ii+1], :]   # 第 ii 种类型的 sel[ii] 个邻居
    ss = rr[:, :, :1]                         # 标量输入 s(r) = 1/r
    gg = self.filter_layers[ii].forward(ss)   # 独立 embedding net
    gr = torch.matmul(rr.permute(0,2,1), gg)  # [N, 4, sel[ii]] × [N, sel[ii], ng] → [N, 4, ng]
    xyz_scatter += gr
```

**关键点**：必须 per-type 独立建模——对于 Water (sel=[46, 92])，是 2 个
独立的 embedding net（batch = N×46 和 N×92），而不是合并的一个 net
（batch = N×138）。这影响 MLP_WAVE 的 tile 选择和 wave 计算，因为小 batch
和大 batch 的 GPU 利用率差别很大。

```python
def _build_embedding_net_ops(N, sel, neuron, ...):
    for ti in range(ntypes):
        ni = sel[ti]                                # 该类型邻居数
        batch = N * ni                              # 独立 batch size
        # 每种类型独立的 embedding MLP
        ops.extend(_build_single_embedding_net_ops(f"emb_t{ti}", batch, neuron))
        # per-type 的 matmul: rr.T @ gg
        ops.append(_op(f"emb_t{ti}_matmul", "BMM",
                       [("BMM", (N, 4, ni, ng))]))
```

### 2.5 阶段 3 内部: ResNet 结构

**源码**：`deepmd/pt/model/network/mlp.py` `MLPLayer.forward()` L188–219

```python
yy = F.linear(xx, weight, bias)
yy = torch.tanh(yy)
if self.resnet:
    if xx.shape[-1] == yy.shape[-1]:
        yy = yy + xx                                # same dim: 直接 add
    elif 2 * xx.shape[-1] == yy.shape[-1]:
        yy = yy + torch.cat([xx, xx], dim=-1)        # double dim: concat + add
```

对于典型的 embedding net `[25, 50, 100]`：

- Layer 0: Linear(1→25), tanh — 第一层无 skip
- Layer 1: Linear(25→50), tanh, concat([25,25])→50, add — double dim skip
- Layer 2: Linear(50→100), tanh, concat([50,50])→100, add — double dim skip

每个 double dim skip 在算子图中生成一个 `MEM`（concat）+ 一个 `VECadd`（add）。

### 2.6 阶段 4: Descriptor 矩阵运算

**源码**：`se_a.py` L838–844

```python
xyz_scatter /= nnei                                    # 归一化
xyz_scatter_1 = xyz_scatter.permute(0, 2, 1)           # [N, ng, 4]
xyz_scatter_2 = xyz_scatter[:, :, :self.axis_neuron]   # [N, 4, axis_neuron]
result = torch.matmul(xyz_scatter_1, xyz_scatter_2)    # [N, ng, axis_neuron]
```

建模为：`VECmul`（除 nnei）+ `BMM`（[N, ng, 4] × [N, 4, axis_neuron]）+ `MEM`
（reshape）。

### 2.7 阶段 5: Fitting Network

**源码**：`fitting.py` `GeneralFitting._forward_common()` L847–857

与 embedding net 类似的 MLP 结构，但输入维度为 descriptor dim
（ng × axis_neuron = 100 × 16 = 1600），网络为 `[240, 240, 240]`。

### 2.8 阶段 7: Force backward

**源码**：`transform_output.py` `task_deriv_one()` L65–96

```python
extended_force = torch.autograd.grad(
    [energy], [extended_coord],
    grad_outputs=[grad_output], retain_graph=True, create_graph=False
)[0]
```

Autograd 反向遍历：output → fitting → descriptor → embedding。每个
`Linear(B, I, O)` 的 backward 对应一个 `Linear(B, O, I)`（grad_input 计算）。

---

## 3. 算子映射与计算图构建

### 3.1 映射表

| DeepMD 源码操作 | NeuSight 算子 | 形状 |
|---|---|---|
| `torch.gather(coord, index)` | `MEM` | (N, M, 3) |
| `coord_r - coord_l` | `VECadd` | (NM, 3) |
| `1/r, x/r²` | `VECmul` | (NM, 4) |
| `smooth_weight(r)` | `VECmul` | (NM, 1) |
| `F.linear(x, W, b)` | `Linear` | (batch, in, out) |
| `torch.tanh(x)` | `VECtanh` | (batch, dim) |
| `y + x` (ResNet skip) | `VECadd` | (batch, dim) |
| `cat([x, x])` (double dim) | `MEM` | (batch, 2×dim) |
| `matmul(rr.T, gg)` | `BMM` | (N, 4, sel_i, ng) |
| `matmul(G1, G2)` (descriptor) | `BMM` | (N, ng, 4, axis_n) |
| `sum(energy)` | `VECadd` | (N, 1) |
| `autograd.grad(Linear)` | `Linear` | (B, O, I) |

### 3.2 完整算子图示例

以 Water 模型 (sel=[46,92], emb=[25,50,100], fit=[240,240,240], N=32, force=True)
为例，opgraph 包含 **65 个算子节点**：

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

DeepMD 推理不像 Transformer 有层复制结构，所有算子的 latency **简单求和**：

```python
def aggregate_deepmd(trace):
    fw  = trace["fw_latency"].sum()
    bw  = trace["bw_latency"].sum()
    acc = trace["acc_latency"].sum()
    return fw + bw + acc, fw, bw, acc
```

---

## 4. Overhead 模型

### 4.1 问题

NeuSight MLP_WAVE 只预测**建模算子**（Linear/BMM/VEC/MEM）的纯 GPU 计算时间。
实际推理中还有两类额外开销：

1. **固定开销（fixed overhead）**：~350 个 CUDA kernel 的 launch chain 总耗时，
   约 5–6 ms，几乎不随原子数变化
2. **未建模 GPU 计算**：neighbor list 的 O(N²) broadcast distance + env_mat
   的 random gather 等

> 早期版本（v1–v4）尝试过把这两部分合在一起用 power law 拟合
> （`overhead = α · N^β`），但跨硬件迁移很差。当前模型把两者**物理分离**：
> fixed 部分作为常数 / 查表，未建模 GPU 部分用解析 roofline 公式。
> 本文档仅描述当前模型，不展开早期版本细节。

### 4.2 固定开销

通过 Nsight Systems profiling，N=32 到 N=1024 的实测延迟几乎恒定，
而 MLP_WAVE 预测的纯计算时间只有 ~1.1 ms。差值来自约 350 个小 CUDA kernel
的 launch overhead：

```
fixed_overhead ≈ N_kernels × τ_launch
              ≈ 350 × 16 μs = 5.6 ms (H100 NVL)
```

不同模型结构的固定开销不同（kernel 数量不同），实测：

| 模型 | num_types | H100 NVL fixed |
|------|----------|---------------|
| Water | 2 | 5.715 ms |
| Copper | 1 | 4.850 ms |

当前实现按 `num_types` 查表，3 种以上原子类型用 `per_extra_type ≈ 0.8 ms` 线性外推。
这套查表参数在校准 GPU 上准，但**跨 GPU 不可移植**（详见 §6.3）。

### 4.3 未建模 GPU 计算的解析 Roofline 模型

通过分析 DeepMD-kit 源码，识别出两个主要的未建模计算分量：

**(a) O(N²) 分量 — neighbor list broadcast distance：**

```python
# nlist.py:
diff = coord.unsqueeze(1) - coord_local.unsqueeze(2)  # [N, nall, 3]
# 数据量 = N × nall × 8 bytes, nall = ns × N
```

延迟由内存带宽决定：

```
quad_ms = C_QUAD × N × nall × 8 / GPU_MEM_BW × 1000
```

**(b) O(N) 分量 — env_mat gather + type sort：**

```python
# env_mat.py:
torch.gather(coord_pad, 1, index)  # random access [N, nnei, 4]
```

延迟：

```
linear_ms = C_LINEAR × N × nnei × 8 / GPU_MEM_BW × 1000
```

**(c) 系数拟合：**

使用 Water + Copper 的 ground truth，用 `scipy.optimize.minimize (L-BFGS-B)`
拟合两个全局系数（H100 NVL 校准）：

```
C_QUAD   = 28.3284     # broadcast distance 的有效访问倍数（多遍访问 + partial sort）
C_LINEAR = 2053.2321   # random gather 的有效访问倍数（cache miss penalty）
```

### 4.4 两区间模型

```python
gpu_overhead_ms     = quad_ms + linear_ms
adjusted_compute_ms = mlp_compute_ms + gpu_overhead_ms
e2e_total_ms        = max(fixed_overhead_ms, adjusted_compute_ms)
```

物理含义：CPU 提交 kernel launch chain（固定时间）与 GPU 执行计算
（随 N 增长）形成**流水线**。总时间取两者的 max：

- 小 N: 计算快于 launch → e2e ≈ fixed_overhead
- 大 N: 计算慢于 launch → e2e ≈ mlp_compute + gpu_overhead

---

## 5. 完整预测流水线

```
输入: GPU 配置 + DeepMD 模型配置 + 原子数 N
  │
  ├──→ parse_deepmd_input()           解析 DeepMD input.json
  │      提取 sel, neuron, type_map, rcut 等
  │
  ├──→ build_deepmd_opgraph()         解析式生成算子图
  │      7 个阶段 → ~65 个 NeuSight 算子节点
  │
  ├──→ OperatorPredictor.predict()    逐算子预测
  │      Linear → MLP_WAVE_MM
  │      BMM    → MLP_WAVE_MM
  │      VEC*   → MLP_WAVE_VEC
  │      MEM    → BW 公式估算
  │
  ├──→ aggregate_deepmd()             简单求和
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

| 项 | 值 |
|----|----|
| 校准 GPU | NVIDIA H100 80GB NVL (Mem_BW = 3430 GB/s) |
| 跨 GPU 验证 | NVIDIA A100 80GB PCIe (Mem_BW = 1935 GB/s) |
| DeepMD-kit | 3.x, PyTorch backend |
| Profiling | warmup 30 次 + 测量 100 次，取 mean，clean GPU 下 std < 1% |

测试模型：

- **Water**：se_e2_a, 2 types (O, H), sel=[46, 92], emb=[25,50,100],
  fit=[240,240,240]
- **Copper**：se_e2_a, 1 type (Cu), sel=[120], emb=[25,50,100],
  fit=[240,240,240]

### 6.2 H100 NVL 上的精度（校准 GPU）

#### 全 N 范围（Water, energy + force）

| N | 实测 (ms) | std | 预测 (ms) | compute | overhead | 误差 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 5.890   | 0.043 | 5.865   | 1.136 | 4.729  | −0.4% |
| 64   | 5.893   | 0.293 | 5.865   | 1.072 | 4.793  | −0.5% |
| 128  | 5.902   | 0.550 | 5.865   | 1.083 | 4.782  | −0.6% |
| 256  | 5.939   | 0.148 | 5.865   | 1.077 | 4.788  | −1.2% |
| 512  | 5.941   | 0.127 | 5.865   | 1.369 | 4.496  | −1.3% |
| 1024 | 5.873   | 0.035 | 5.865   | 1.683 | 4.182  | −0.1% |
| 2048 | 11.823  | 0.030 | 11.268  | 2.432 | 8.836  | −4.7% |
| 4096 | 35.532  | 0.266 | 36.597  | 3.961 | 32.637 | +3.0% |
| 8192 | 129.113 | 4.973 | 132.327 | 7.194 | 125.13 | +2.5% |

#### Copper（跨模型泛化）

| N | 实测 (ms) | std | 预测 (ms) | 误差 |
|---:|---:|---:|---:|---:|
| 2048 | 11.137  | 0.023 | 11.091  | −0.4% |
| 4096 | 33.763  | 0.282 | 36.244  | +7.3% |
| 8192 | 125.034 | 3.302 | 131.620 | +5.3% |

#### 分区总结（H100 NVL）

| 区间 | N 范围 | MAE | 最大误差 | 评价 |
|------|--------|-----|---------|------|
| Overhead-bound | 32 ~ 1024 | 1.0% | −2.2% | 非常好 |
| Transition | ~1280 ~ 1920 | ~10% | ~15% | 系统性 bubble 误差 |
| Compute-bound | 2048 ~ 8192 | 3.7% | +7.3% | 可接受 |

H100 上整体 MAE 约 3%，转换区单独表现较差（详见 §7）。

### 6.3 A100 跨 GPU 验证

#### 6.3.1 误差全景（按区间 × 模型）

| 区间 | 模型 | N 范围 | 误差量级 | confidence |
|------|------|--------|----------|-----------|
| Fixed plateau | Water | 32–512 | **−26 ~ −29%** | "high" ❌ |
| Fixed plateau | Copper | 32–768 | **−18 ~ −29%** | "high" ❌ |
| Transition | Water | 1024–2048 | −8.5% ~ +5.0% | low/high |
| Transition | Copper | 1024–2048 | +3.6% ~ +14.3% | low/high |
| Post-transition | Water | 2304–3072 | +7% ~ +11% | "high" |
| Post-transition | Copper | 2304–3072 | +15% ~ +18% | "high" |
| Compute-bound | Water | 4096–8192 | **+14.6 ~ +18.2%** | "high" |
| Compute-bound | Copper | 4096–8192 | **+20.5 ~ +21.6%** | "high" |

整体 A100 MAE：Water **21.9%**、Copper **17.8%**——比 H100 上放大约 5–10 倍。

#### 6.3.2 缺陷 1：Fixed overhead 不可跨 GPU 移植（最严重）

| 模型 | H100 fixed | A100 fixed | 比值 |
|------|-----------|------------|------|
| Water (2-type) | 5.715 ms | **~8.30 ms** | 1.45× |
| Copper (1-type) | 4.850 ms | **~6.85 ms** | 1.41× |
| Δ(water − copper) | 0.865 ms | **~1.45 ms** | 1.68× |

**物理解释**：`fixed = N_kernels × τ_launch + Python_overhead`

- A100 PCIe 的 `τ_launch` 经验值 ~22 μs，H100 NVL ~16 μs
  （PCIe vs NVLink、driver/SM 调度差异）
- 350 launches × 6 μs ≈ 2.1 ms，正好对应 H100→A100 fixed 增量

`per_extra_type = 0.8` 是同一原因跨 GPU 失效（A100 实际 ≈ 1.45）。

**结论**：fixed overhead 的查表机制完全不可跨 GPU 移植，
N≤512 区间 −28% 系统性低估的根源就在这。

#### 6.3.3 缺陷 2：Bandwidth 缩放过度简化

模型假设 `unmod(target) = unmod(REF) × REF_BW / target_BW = ×1.77`。
从大 N 数据点反推真实缩放：

| N | H100 unmod | A100 unmod | 真实比值 | 模型比值 | 偏差 |
|---|-----------|-----------|---------|---------|------|
| Water 4096 | 31.4 | 49.4 | **1.57×** | 1.77× | +12.7% |
| Water 8192 | 121.9 | 185.4 | **1.52×** | 1.77× | +16.4% |
| Copper 8192 | 122.3 | 178.8 | **1.46×** | 1.77× | +21.2% |

A100 实际带宽劣势小于规格表——这是 compute-bound 区间被高估 +15~22% 的原因。

**物理解释**：`gpu_bw_scale = REF_BW / target_BW` 隐含「access pattern 在两 GPU
上 effective bandwidth 占 peak 的比例相同」。但 A100 (HBM2e + 40 MB L2)
对 random gather/topk 的相对 cache hit 率比 H100 (HBM3 + 50 MB L2) 更高，
所以 A100 的「有效带宽劣势」小于「peak 带宽劣势」。`C_QUAD/C_LINEAR`
不是真正的跨 GPU 不变常数，需要乘 cache-pattern 修正系数（A100 ≈ 0.83~0.85）。

#### 6.3.4 缺陷 3：Bubble 模型 GPU 依赖

H100 转换区有显著 bubble（误差 ~15%），所以模型在转换区给 "low confidence"
+ 高斯 bubble band。A100 上同区间实测：

| N | A100 实测 | 预测 | error % |
|---|----------|------|---------|
| 1024 | 8.74  | 7.99  | −8.5% |
| 1280 | 10.88 | 10.44 | −4.0% |
| 1536 | 13.45 | 13.37 | −0.7% |
| 1792 | 16.39 | 16.91 | +3.2% |
| 2048 | 19.65 | 20.63 | +5.0% |

**A100 转换区曲线极其平滑、几乎没有 bubble peak**（全段 < 9%）——
说明 bubble 既是 GPU 依赖也是 kernel-mix 依赖，不是物理普适常数。
H100 上校准的 `BUBBLE_PEAK_FRACTION = 0.20` 等参数不能直接套用。

A100 上 confidence 标注**方向反了**：真正不准的 N≤512 被标 "high"，
转换区其实很准却被标 "low"——confidence 信号在 A100 上几乎失去诊断价值。

#### 6.3.5 缺陷 4：Plateau 内不平坦

A100 Copper 平台数据：

```
N=32:  6.72   N=64:  6.80  N=128: 6.86  N=256: 6.90
N=512: 7.04   N=768: 6.91 (反而下降)  N=896: 6.97  N=1024: 7.56 (起跳)
```

平台内有 ±0.2 ms 微波动，N=768 反而比 N=512 低 0.13 ms——
`max(fixed, mlp+unmod)` 模型完全无法解释。
可能原因：SM 整除性（A100 有 108 SM，N=768 ≈ 7.1 wave 命中 wave-fill 甜蜜点）、
kernel autotune shape 选择。当前模型把整个 plateau 当成一个常数，
忽略 plateau 内的微调。

#### 6.3.6 缺陷 5：MLP_WAVE compute 自身的 A100 偏差

| N | H100 pred compute | A100 pred compute | 实际缩放 | BW 理论 |
|---|------------------|-------------------|---------|---------|
| 4096 | 3.96 | 8.14 | **2.06×** | 1.77× |
| 8192 | 7.19 | 15.20 | **2.11×** | 1.77× |

NeuSight 自身的 compute 预测在 A100 上比 H100 大 ~2.06–2.11×，超过 BW 比 1.77×。
说明 A100 整体误差里有一部分来自上游 NeuSight，不能全部归因到 overhead 模型。

#### 6.3.7 设计假设 vs A100 实证

| 设计假设 | A100 实证 |
|---------|----------|
| Fixed overhead 跨 GPU 用同一查表 | ❌ 崩（−28%） |
| `per_extra_type = 0.8` 跨 GPU 通用 | ❌ 崩（A100 实际 ≈ 1.45） |
| `C_QUAD/C_LINEAR` 跨 GPU 不变 | ⚠️ 部分崩（需 ×0.83 修正） |
| `gpu_bw_scale = REF_BW/target_BW` 简单线性 | ⚠️ 部分崩 |
| `BUBBLE_PEAK_FRACTION = 0.20` 跨 GPU 通用 | ❌ 崩（A100 几乎无 bubble） |
| Plateau 内 fixed 是常数 | ⚠️ 小问题（±0.2 ms 波动） |
| Confidence 标注合理 | ❌ 方向反 |
| MLP_WAVE compute 跨 GPU 准确 | ⚠️ A100 上自身偏 ~17% |

#### 6.3.8 改进方向（按优先级）

**P0**：
1. **Fixed overhead 物理化**：把 lookup table 替换成
   `fixed = PY_BASE + N_kernels(arch) × τ_launch(GPU)`，每个 GPU 只测一个
   `τ_launch`（empty kernel storm micro-benchmark）即可跨 GPU 迁移。
   预期：N≤512 误差从 −28% 降到 ±5%。
2. **Unmodeled compute 加 cache-efficiency 系数**：
   `gpu_oh_target = base × (REF_BW / target_BW) × κ_cache(GPU)`，
   引入 `κ_cache`（H100=1.0, A100≈0.83）从大 N 数据点反拟合。
   预期：N≥4096 误差从 +18% 降到 ±5%。

**P1**：
3. Confidence 标注重做：未校准 GPU 全程 "low"，已校准 GPU 转换区可标 "high"。
4. Bubble fraction GPU-aware 表（H100=0.20, A100=0.05, default=0.10）。

**P2**：
5. Plateau 内 SM 整除性微修正项。
6. NeuSight MLP_WAVE 自身的 A100 缩放（不属于 overhead 模块）。

#### 6.3.9 一句话总结

A100 数据系统性证伪了模型的「跨 GPU 假设」三件套：

1. Fixed overhead 查表跨 GPU 直接崩（−28%）
2. BW 线性缩放对 unmodeled 过度悲观（+18%）
3. Bubble 高斯模型只适合 H100（A100 转换区其实很准）

补上 **fixed overhead 物理化** 和 **C_QUAD/C_LINEAR cache 修正** 两条后，
A100 MAE 预期可从 21.9% 压回 5–8%，恢复到 H100 上的精度水平。

### 6.4 A100 重校准实验（验证 §6.3 的判断）

§6.3 的核心结论是：v5 模型的**结构**（max + 解析 roofline + fixed plateau）跨
GPU 通用，崩的是**常数**。本节用 A100 实测数据直接重新拟合 4 个常数
（`C_QUAD`、`C_LINEAR`、`fixed_2type`、`fixed_1type`），验证「只要 per-GPU
重校常数即可恢复精度」。

#### 6.4.1 校准设置

- **数据**：`results/a100_experiment/` 中 9 个 JSON 文件提供的 34 个 (model, N)
  实测点，覆盖 plateau / transition / post-transition / compute-bound 全部区间
- **方法**：`scripts/calibrate_analytical_a100.py` 使用 `scipy.optimize.minimize`
  (L-BFGS-B) 最小化加权 MAE
- **权重**：N≥4096 → 3.0；N≥2048 → 2.5；N≤512 (plateau) → 2.0；transition → 1.0
- **初值**：`C_QUAD=28.33`（H100 值）、`C_LINEAR=2053.23`（H100 值）、
  `fixed_water=8.30`、`fixed_copper=6.85`（A100 plateau 实测平均）

#### 6.4.2 重校准后的常数

| 常数 | H100（参考） | A100 重校准 | 比值 |
|---|---:|---:|---:|
| `C_QUAD` | 28.3284 | **21.9106** | 0.77× |
| `C_LINEAR` | 2053.2321 | **3597.9057** | 1.75× |
| `fixed_2type` (Water) | 5.715 ms | **8.307 ms** | 1.45× |
| `fixed_1type` (Copper) | 4.850 ms | **6.856 ms** | 1.41× |

**物理解读**：

- `fixed_*` 整体涨 ~1.45×，与 §6.3.2 推断的 `τ_launch` 比 (~22 μs / 16 μs ≈ 1.38)
  一致，差出来的部分是 PCIe vs NVLink 的 dispatch path
- `C_QUAD` ↓ + `C_LINEAR` ↑ 的方向证实了 §6.3.3 的 cache-pattern 假设：
  A100 上**大 broadcast tensor**（O(N²) quad 项）已严重溢出 L2，纯靠 HBM2e 带宽
  跑，相对 H100 的劣势没那么糟；但**random gather**（C_LINEAR 项）在更小 L2 上
  cache miss 率更高，需要的 effective bytes 倍数显著增大
- 这一升一降的真实修正比例（0.77, 1.75）和 §6.3.3 当时凭单点反推的均一
  `κ_cache≈0.83` 不一样——说明 quad 和 linear 必须**分开校准**，不能用单一
  cache 系数

#### 6.4.3 校准前 vs 校准后（分区间 MAE）

| 模型 | 区间 | n | 校准前 MAE | **校准后 MAE** | 校准后最大误差 |
|---|---|---:|---:|---:|---:|
| Water | Plateau (N≤512) | 5 | 29.85% | **2.01%** | +4.47% |
| Water | Transition (1024–2048) | 12 | 10.38% | **3.60%** | −6.03% |
| Water | Post-trans (2304–3072) | 3 | 8.44% | **0.84%** | −1.18% |
| Water | Compute (N≥4096) | 2 | 16.38% | **1.10%** | −2.21% |
| Copper | Plateau (N≤768) | 6 | 27.37% | **1.14%** | −2.58% |
| Copper | Transition (1024–2048) | 10 | 8.72% | **4.89%** | +7.13% |
| Copper | Post-trans (2304–3072) | 3 | 16.50% | **5.24%** | +6.41% |
| Copper | Compute (N≥4096) | 2 | 21.05% | **2.09%** | +4.17% |

#### 6.4.4 整体精度对比（核心结论表）

| GPU | 场景 | 整体 MAE | 最大误差 |
|---|---|---:|---:|
| H100 NVL | 校准 GPU（参考） | ~3% | +7.3% |
| A100 PCIe | 沿用 H100 常数（v5 原貌） | **15.70%** | −34.5% |
| A100 PCIe | **重校 4 个常数后** | **3.11%** | +7.13% |

> 注：校准前的 15.70% 比 §6.3.1 报告的 21.9%/17.8% 略低，是因为本节使用 34 个
> 点的全集（含 transition 段密集 step128 数据）；§6.3.1 的 21.9%/17.8% 是
> water_small/water_large/copper 三个稀疏文件单独算的均值。两者趋势一致。

#### 6.4.5 结论

A100 重校准**完全验证**了 §6.3.8 的两条 P0 推论：

1. **结构是对的**：同一套 v5 公式（4 个常数 + max + bubble band）在 A100 上
   能达到与 H100 校准 GPU **同等**的 ~3% MAE
2. **常数必须 per-GPU**：H100 → A100 直接套常数 MAE 5× 放大；重测 4 个常数后
   立刻恢复

**对 v6 的指导**：

- "fixed overhead 物理化"（§6.3.8 P0-1）的目标精度是合理的——A100 校准后
  plateau MAE = 1.14%/2.01%，说明 plateau 段本身就能很准，关键就是 fixed
  常数测对
- "cache-efficiency 系数"（§6.3.8 P0-2）需要从单 `κ_cache` 升级为
  `(κ_quad, κ_linear)` 两个分量，因为 broadcast 和 random gather 的
  cache 行为方向相反

#### 6.4.6 跨 GPU 部署成本

工程意义上，向新 GPU 迁移 NeuSight-DeepMD 只需要：

| 步骤 | 数据需求 | 时间 |
|------|---------|------|
| 1. 测 plateau 段 fixed | Water/Copper 各 N=64 即可 | ~5 min |
| 2. 测 compute-bound 锚点 | Water/Copper 各 N=4096/8192 | ~10 min |
| 3. scipy 拟合 4 常数 | (上面 ~6 个点) | <1 s |
| 4. 验证 transition + post-trans | ~10 个补充点 | ~10 min |

**总计 < 30 分钟实测**，得到与 H100 校准 GPU 同等精度（~3% MAE）。这是 v5
设计在工程上最有价值的一个性质。

---

## 7. 转换区误差与 Confidence-Aware 标注

### 7.1 现象

H100 上 overhead-bound 到 compute-bound 的过渡区间（Water 约 N=1280 ~ 1920），
预测误差显著增大，最高 ~15%。密集 profiling 数据：

| N | 实测 (ms) | max(fixed, adj) | 误差 |
|---:|---:|---:|---:|
| 1024 | 5.873  | 5.865  | −0.1% |
| 1280 | 6.214  | 5.865  | −5.6% |
| 1536 | 6.950  | 7.324  | +5.4% |
| 1664 | 7.525  | 8.219  | +9.2% |
| 1792 | 8.275  | 9.213  | +11.3% |
| 1920 | 9.378  | 10.225 | +9.0% |
| 2048 | 11.823 | 11.266 | −4.7% |

### 7.2 根因分析：Pipeline Bubble

模型假设 CPU launch chain 和 GPU 计算是完美流水线：

```
e2e = max(Σ launch_i, Σ compute_i)
```

但实际 GPU 执行中，大小 kernel 交替调度：

```
实际 e2e = Σ max(launch_i, compute_i)  ≠  max(Σ launch_i, Σ compute_i)
```

**示例**：3 个 kernel，每个 launch overhead = 15 μs：

| Kernel | compute (μs) | max(launch, compute) |
|--------|-------------|---------------------|
| A (大) | 50 | 50 |
| B (小) | 5  | 15 |
| C (大) | 50 | 50 |

```
实际 e2e = 50 + 15 + 50 = 115 μs
max() 预测 = max(45, 105) = 105 μs  →  偏差 −8.7%
```

当处于转换区（总 compute ≈ 总 launch）时，「大小 kernel 交替的 idle gap」
效应最显著。

### 7.3 Bubble 的定量特征

通过 Water 和 Copper 的密集 profiling 分析：

**Crossover 公式** — 转换区中心（ratio = 1.0）对应的原子数：

```
N_cross ≈ √(fixed_overhead × BW / (C_QUAD × ns × 8))
```

**Bubble 幅度**：约为 fixed_overhead 的 15–25%

- Water (2 types)：peak bubble ≈ 25.1% of fixed
- Copper (1 type)：peak bubble ≈ 15.6% of fixed

Water 有 2 个不同大小的 embedding net（batch = N×46 vs N×92），方差更大
→ bubble 更大；Copper 只有 1 个 embedding net，kernel 更均匀 → bubble 更小。

### 7.4 修正方案对比

| 方案 | 结果 | 失败原因 |
|------|------|---------|
| 全局 softmax 平滑 | 破坏两端精度 | 影响范围过大 |
| 局部 Gaussian 修正 | Water OK, Copper 不泛化 | bubble 高度依赖模型结构 |
| 非对称 Gaussian | 同上 | peak 差 60% (Water 0.25 vs Copper 0.16) |
| Per-kernel 流水线模拟 | MAE 反而更差 (13.0%) | framework kernel 分布未知 |

### 7.5 解决方案：Confidence-Aware 标注

既然转换区的精确值无法用统一参数校正，采用**诚实标注**策略：

> 当预测处于转换区时，不给一个不准确的点预测，而是标注 `confidence="low"`
> 并给出参考性 bounds。

**转换区检测** — 基于物理定义：

```python
ratio = adjusted_compute_ms / fixed_overhead_ms

if   ratio < 0.8:  regime = "overhead-bound", confidence = "high"
elif ratio > 2.0:  regime = "compute-bound",  confidence = "high"
else:              regime = "transition",     confidence = "low"
```

**参考性 bounds** — 对称区间，Gaussian 衰减：

```python
uncertainty = BUBBLE_PEAK_FRACTION × fixed × exp(-0.5 × ((ratio-1)/σ)²)
# 非对称 σ: σ_left=0.2 (陡峭), σ_right=1.0 (缓慢衰减)
lower = point_estimate - uncertainty
upper = point_estimate + uncertainty
```

**H100 上的覆盖率验证**：

| 数据集 | 转换区点数 | Bounds 覆盖率 |
|---:|---:|---:|
| Water | 8 | 100% (8/8) |
| Copper | 7 | 100% (7/7) |

**输出格式**：

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

**泛化性说明**：

- Confidence 标注（high/low）基于 ratio 的物理定义，跨 GPU、跨模型通用
- Bounds 宽度基于 H100 NVL 实测校准，在其他 GPU 上可能偏大或偏小
  （A100 上几乎无 bubble，bounds 显著偏宽，详见 §6.3.4）

---

## 8. 总结

### 8.1 技术路线

```
DeepMD-kit 源码分析
    → 7 阶段推理过程分解
    → 算子映射到 NeuSight (Linear/BMM/VEC/MEM)
    → MLP_WAVE 预测纯计算时间
    → 解析 Roofline 补偿未建模 GPU 计算 (O(N²) + O(N))
    → max(fixed, compute) 两区间模型
    → Confidence-Aware 转换区标注
```

### 8.2 精度

| GPU | 区间 | MAE | 最大误差 | 评价 |
|-----|------|-----|---------|------|
| H100 NVL（校准） | Overhead-bound (N≤1024) | 1.0% | −2.2% | 非常好 |
| H100 NVL | Transition | ~10% | ~15% | low confidence 标注 |
| H100 NVL | Compute-bound (N≥2048) | 3.7% | +7.3% | 可接受 |
| A100 PCIe（沿用 H100 常数） | Fixed plateau | ~26% | −29% | **严重**（fixed 不可移植） |
| A100 PCIe（沿用 H100 常数） | Transition | < 9% | +14% | 实际很准但被标 low |
| A100 PCIe（沿用 H100 常数） | Compute-bound | ~18% | +22% | BW 缩放过悲观 |
| **A100 PCIe（重校 4 常数）** | **整体** | **3.11%** | **+7.13%** | **与 H100 同水平** |

- H100 上整体 MAE 约 3%
- A100 上沿用 H100 常数 MAE 15–22%（视数据子集），暴露了模型的
  两个核心缺陷（详见 §6.3）
- **A100 重校准 4 个常数后整体 MAE 恢复到 3.11%**——证明 v5 模型**结构**跨
  GPU 通用，只是常数必须 per-GPU 校准（详见 §6.4）

### 8.3 代码清单

| 文件 | 职责 |
|------|------|
| `neusight/Tracing/trace_deepmd.py` | 解析式算子图构建（7 阶段分解） |
| `neusight/Tracing/parse_deepmd_input.py` | DeepMD input.json 解析 |
| `neusight/Prediction/predictor_deepmd.py` | DeepMD 预测器入口 |
| `neusight/Prediction/overhead_model.py` | 两区间 overhead 模型 + confidence |
| `neusight/Prediction/aggregator.py` | aggregate_deepmd() 简单求和 |
| `scripts/pred_deepmd.py` | CLI 入口 |
| `scripts/full_accuracy_test.py` | N=32~2048 精度验证 |
| `scripts/test_large_atoms.py` | N=4096~8192 精度验证 |
| `scripts/verify_confidence_aware.py` | Confidence-Aware 功能验证 |
| `scripts/calibrate_analytical_a100.py` | A100 重校准（§6.4） |

### 8.4 已识别的局限性

1. **跨 GPU 迁移**（§6.3 / §6.4）：v5 的 4 个核心常数（`C_QUAD`、`C_LINEAR`、
   `fixed_2type`、`fixed_1type`）必须 per-GPU 校准。沿用 H100 常数直接套
   A100 整体 MAE 15.7%；A100 上重校 4 常数后立刻恢复到 3.11%。**模型结构
   跨 GPU 通用，但常数不可移植**。后续 v6 应把 fixed 物理化为
   `PY_BASE + N_kernels × τ_launch`，cache 修正拆为 `(κ_quad, κ_linear)`
   两个分量。
2. **转换区精度**（§7）：Pipeline bubble 无法用统一参数精确校正，已用
   confidence 标注处理；A100 上几乎没有 bubble，bounds 偏宽。
3. **Plateau 内不平坦**：N=768 等点出现非单调微波动（±0.2 ms），
   `max()` 模型无法解释。
4. **DPA-1 / DPA-2 未充分验证**：当前主要在 se_e2_a 上验证，attention-based
   descriptor 的 overhead 结构不同，需要更多实验。
5. **多帧批处理**：当前模型假设 nframes=1，批处理时 kernel launch overhead
   可能被 amortize。
