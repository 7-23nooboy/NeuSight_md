# Overhead 建模深度解析

> NeuSight-DeepMD Predictor v5 的 overhead 模块拆解、源码对照、误差归因
>
> 本文档不是设计文档，而是「问题文档」——目的是把当前 v5 模型里
> **哪些是物理推导、哪些是查表、哪些是欠定的拟合、哪些区间的误差从何而来**
> 一次性讲清楚，作为后续 v6 升级的依据。

---

## 0. TL;DR

NeuSight 的 MLP_WAVE 只能预测 fitting net 里**有 nn.Module hook 的 op**
（Linear / BMM / 激活），其余开销 v5 模型用两个分量近似：

```
e2e = max( fixed_overhead, mlp_compute + unmodeled_gpu_compute )
       │                                  │
       │                                  └─ "未建模 GPU 计算"，源码可读，
       │                                     来自 nlist + env_mat
       │                                     → 用 roofline 公式拟合 2 个全局系数
       │
       └─ "固定 overhead"，CUDA kernel launch chain + Python dispatcher
          → 当前查表（按 num_types / model_type 分档）
```

两部分各自的问题：

| 部分 | 当前形式 | 物理性 | 主要问题 |
|------|----------|--------|----------|
| **Fixed overhead** | 查表 (`{1_type: 4.85, 2_type: 5.715}`) | ❌ 没有 | 不能跨架构、跨 descriptor、跨 GPU 外推 |
| **Unmodeled compute** | `C_QUAD·N·nall·8/BW + C_LINEAR·N·nnei·8/BW` | ✅ 半物理 | 在校准点 MAE 2.24%；**转换区**因 max() 假设失效误差 +25% |
| **转换区** | `max()` 取大 + 高斯 uncertainty band | ⚠️ 几何近似 | bubble (大小 kernel 交错) 无法用单一公式描述 |

下面分章节展开。

---

## 1. NeuSight 看不到的部分：DeepMD-kit 真实推理 pipeline

DeepMD-kit `se_e2_a` descriptor 在 PyTorch backend 下的一次推理 forward 大致如下
（基于 `deepmd-kit/deepmd/pt/utils/nlist.py` 和
`deepmd-kit/deepmd/pt/model/descriptor/env_mat.py` 源码）：

```
forward(coord, atype, box):
    ┌─────────────────────────────────────────────────┐
    │ Step A. extend_coord_with_ghosts                │  ← nlist.py:407
    │   - Periodic boundary → ghost cell 复制          │
    │   - 输出 extended_coord [nf, nall*3], nall=ns·N  │
    │   - ns = (2·ceil(rcut/box_face)+1)^3 ≈ 27       │
    ├─────────────────────────────────────────────────┤
    │ Step B. build_neighbor_list                     │  ← nlist.py:47
    │   - diff = coord.unsqueeze(1) - coord.unsqueeze(2)│  ← O(N·nall) tensor
    │   - rr = ||diff||                               │  ← norm
    │   - rr, nlist = topk(rr, nsel+1)                │  ← partial sort
    │   - mask + distinguish types                    │  ← per-type loop
    ├─────────────────────────────────────────────────┤
    │ Step C. prod_env_mat (env_mat.py:51)            │
    │   - gather coord_r from nlist                   │  ← random gather
    │   - diff = coord_r - coord_l                    │
    │   - length, t0, t1, weight                      │  ← element-wise math
    │   - normalize: (env_mat - mean) / std           │
    ├─────────────────────────────────────────────────┤
    │ Step D. embedding_net(env_mat)                  │  ← NeuSight 看得到（部分）
    │   - per type pair → MLP → [N, nnei, M]          │
    │   - num_types² 个 MLP 调用                       │
    ├─────────────────────────────────────────────────┤
    │ Step E. descriptor: BMM(g, r, gT)               │  ← NeuSight 看得到
    ├─────────────────────────────────────────────────┤
    │ Step F. fitting_net(descriptor)                 │  ← NeuSight 看得到
    │   - L_fit 层 Linear + tanh                       │
    ├─────────────────────────────────────────────────┤
    │ Step G. force = -∂E/∂coord (autograd backward)  │  ← NeuSight 看不到
    │   - reverse pass + scatter_reduce               │
    └─────────────────────────────────────────────────┘
```

**NeuSight 的 op_df 只能抓到 D / E / F 中的 nn.Module 层**——其余全部缺失。

| Pipeline 阶段 | NeuSight 是否建模 | 落入 v5 的哪个分量 |
|---------------|------------------|-------------------|
| A. ghost cell extend | ❌ | unmodeled |
| B. nlist (broadcast + topk) | ❌ | **unmodeled (主导项)** |
| C. env_mat (gather + math) | ❌ | unmodeled |
| D. embedding net | ✅ (Linear/激活) | mlp_compute |
| E. descriptor BMM | ✅ | mlp_compute |
| F. fitting net | ✅ | mlp_compute |
| G. autograd backward | ❌ | fixed (+0.15ms) |
| Python dispatcher / kernel launches | ❌ | **fixed (主导项)** |

---

## 2. Unmodeled GPU compute 的拟合原理

### 2.1 源码到公式的映射

观察 `nlist.py:113-127` 的核心几行：

```python
# build_neighbor_list (nlist.py)
diff = vcoord_xyz.unsqueeze(1) - vcoord_local_xyz.unsqueeze(2)
assert diff.shape == (batch_size, nloc, nall, 3)        # ① broadcast → [N, nall, 3]

rr = torch.linalg.norm(diff, dim=-1)                     # ② reduce → [N, nall]
rr[:, idx, idx] -= 1.0                                   #   self-distance fix

top_k = min(nsel + 1, nnei)
rr, nlist = torch.topk(rr, top_k, largest=False)         # ③ partial sort
```

和 `env_mat.py:26-33`：

```python
# _make_env_mat
nlist = torch.where(mask, nlist, nall)                   # ④ mask
coord_l = coord[:, :natoms].view(bsz, -1, 1, 3)
index = nlist.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
coord_pad = torch.concat([coord, coord[:, -1:, :] + rcut], dim=1)
coord_r = torch.gather(coord_pad, 1, index)              # ⑤ random gather → [N, nnei, 3]
coord_r = coord_r.view(bsz, natoms, nnei, 3)
diff = coord_r - coord_l                                 # ⑥ element-wise
length = torch.linalg.norm(diff, dim=-1, keepdim=True)
```

加上 `nlist_distinguish_types` (nlist.py:287-320) 里按 type 循环的：

```python
for ii, ss in enumerate(sel):
    pick_mask = (tnlist == ii).to(torch.int32)
    pick_mask, imap = torch.sort(pick_mask, dim=-1, descending=True, stable=True)
    inlist = torch.gather(nlist, 2, imap)                # ⑦ per-type sort + gather
```

### 2.2 算术强度分析

把上面 7 个核心算子按算术强度分类：

| Op | Tensor 形状 | FLOPs | Bytes | FLOPs/Byte | bound |
|----|------------|-------|-------|------------|-------|
| ① broadcast diff | `[N, nall, 3]` | N·nall·3 | 2·N·nall·3·8 | 0.06 | **memory** |
| ② norm + sum | `[N, nall]` | 4·N·nall | 3·N·nall·8 | 0.17 | **memory** |
| ③ topk | `[N, nall]→[N, nnei]` | ~N·nall·log(nsel) | (N·nall+N·nnei)·8 | <1 | **memory** |
| ④ mask / where | `[N, nnei]` | N·nnei | 2·N·nnei·8 | 0.06 | **memory** |
| ⑤ gather | `[N, nnei, 3]` | 0 | 2·N·nnei·3·8 | 0 | **memory** |
| ⑥ env_mat math | `[N, nnei, 4]` | ~10·N·nnei | 3·N·nnei·4·8 | 0.10 | **memory** |
| ⑦ per-type sort × T | `[N, nnei]` × T | T·N·nnei·log(nnei) | T·N·nnei·8 | <1 | **memory** |

**结论：全部 memory-bound**。这意味着 latency 完全由
`总访问字节数 / GPU memory bandwidth` 决定，FLOPs 几乎不重要。
这正是 roofline 模型最简单的情况。

### 2.3 把 7 个算子合并成两项

按数据规模归类：

```
O(N²) 类（数据量 ∝ N × nall = N × 27N）:
  ① broadcast diff
  ② norm
  ③ topk 输入扫描
   → 主导带宽消耗在 [N, nall, 3] 这个张量

O(N) 类（数据量 ∝ N × nnei，nnei = sum(sel) ≈ 138）:
  ④ mask
  ⑤ gather
  ⑥ env_mat math
  ⑦ per-type sort × T types
   → 主导带宽消耗在 [N, nnei, 3 or 4] 这些张量
```

合并后只留两项：

```
T_quad   = C_QUAD   × (N · nall · 8) / BW         ← O(N²) 项
T_linear = C_LINEAR × (N · nnei · 8) / BW         ← O(N)  项
T_unmod  = T_quad + T_linear
```

注意：

- `× 8` 是 float64 字节数（DeepMD-kit 默认）
- `/ BW` 是 memory-bound roofline ceiling
- `nall = ns · N`，`ns = 27` 来自周期边界条件下的 ghost cell 复制
  （`extend_coord_with_ghosts` at nlist.py:407-496）
- `nnei = sum(sel)` 直接从模型 config 读

### 2.4 C_QUAD 和 C_LINEAR 的物理含义

公式里只有这两个**抽象常数**需要拟合：

```python
# overhead_model.py:78-79
C_QUAD = 28.3284
C_LINEAR = 2053.2321
```

它们的物理含义是 **"有效访问遍数"**——一个数据元素被这一组 kernel 平均
touched 多少次。比如 broadcast diff 的实际 bytes 流：

```
实际访问:                                                          倍数
  read coord_xyz (N·3·8)                                          ×1
  read vcoord_local_xyz (N·3·8)                                   ×1
  write diff [N, nall, 3] (N·nall·3·8)                            ×1
  read diff for norm (N·nall·3·8)                                 ×1
  write rr [N, nall] (N·nall·8)                                   ×1
  read rr for topk (多遍 partial sort, log nsel 次)               ×~5
  write topk output                                               ×1
  ────────────────────────────────────────────────────────────────
  对 [N, nall] 这个量级的访问累计 ≈ 8–10 倍裸数据量
  再加上 PyTorch 的 intermediate buffer 和 cache miss penalty
  → effective C_QUAD ≈ 28
```

`C_LINEAR ≈ 2053` 看起来夸张，但其实合理：

- gather (⑤) 是 random access pattern，effective bandwidth 通常只有
  peak BW 的 1–5% → 折合 20–100 倍系数
- per-type sort (⑦) 要循环 T 次，每次都要读写 `[N, nnei]`
- env_mat 的 division / smooth weight 计算又是几遍读写
- 加起来 effective 倍数轻松上千

这两个常数**抽象掉了** PyTorch 算子实现的所有细节
（cache 行为、kernel fusion、launch pattern），只暴露**数据规模**这个唯一的输入。

### 2.5 拟合过程

只有 2 个未知数，目前有 5 个数据点：

| 模型 | N | real_ms | mlp_ms | unmod_ms = real - mlp |
|------|---|---------|--------|----------------------|
| Water | 2048 | 11.742 | 2.172 | 9.570 |
| Water | 4096 | 35.236 | 3.834 | 31.402 |
| Copper | 2048 | 11.332 | 2.172 | 9.160 |
| Copper | 4096 | 35.128 | 3.834 | 31.294 |
| Copper | 8192 | 129.528 | 7.228 | 122.300 |

写成超定线性系统：

```
    | 9.570  |     | quad_water_2k    linear_water_2k |   | C_QUAD   |
    | 31.402 |  =  | quad_water_4k    linear_water_4k | × | C_LINEAR |
    | 9.160  |     | quad_cu_2k       linear_cu_2k    |
    | 31.294 |     | quad_cu_4k       linear_cu_4k    |
    | 122.30 |     | quad_cu_8k       linear_cu_8k    |
```

scipy 最小二乘解出 `C_QUAD = 28.33, C_LINEAR = 2053.23`，
拟合 MAE = 2.24%, max_error = 6.2%。

### 2.6 为什么这是「半物理」而不是「纯回归」

| 元素 | 来源 |
|------|------|
| `N × nall × 8` | **物理**：直接对应 `coord.unsqueeze(1) - coord.unsqueeze(2)` 输出张量大小 |
| `N × nnei × 8` | **物理**：直接对应 `gather` 输出张量大小 |
| `nall = ns × N`, `ns=27` | **物理**：来自 `extend_coord_with_ghosts` |
| `/ BW` | **物理**：memory-bound roofline |
| `× 8` | **物理**：dtype 字节数 |
| `C_QUAD` | **拟合**：抽象 effective access 次数 |
| `C_LINEAR` | **拟合**：抽象 random gather + per-type loop |

**只有 2 个标量被拟合**，其余结构全部从源码推出。换 GPU 时 `BW` 自动调整；
换 sel/nnei 时 `nnei` 自动调整；换 box_size 时 `ns` 自动调整。
`C_QUAD/C_LINEAR` 是 PyTorch 算子实现层面的常数，跨 GPU 几乎不变
（PyTorch 的 broadcast/gather/topk kernel 实现不依赖具体 GPU 型号）。

这就是 v5 unmodeled 部分**跨 GPU 几乎不会崩**的根本原因。

---

## 3. Fixed overhead：v5 的"老大难"

### 3.1 物理来源

固定开销在 N≤1024 区间主导，物理上由三部分组成：

```
fixed = T_python + Σ launch_latency + Σ small_kernel_exec
        │           │                    │
        │           │                    └─ 跑不满 SM 的 kernel
        │           │                       (mask 生成、scatter、small reduce)
        │           │
        │           └─ ~350 launches × ~16 μs/launch
        │              （H100 NVL，由 CUDA driver / PCIe 决定）
        │
        └─ Python dispatcher、py-bind11、autograd graph build
           ≈ 0.5–2 ms（与 model.forward 的复杂度相关）
```

### 3.2 v5 当前实现：纯查表

```python
# overhead_model.py:92-103
FIXED_OVERHEAD_MS = {
    "se_e2_a": {
        "1_type": 4.850,        # ← Copper 实测平均
        "2_type": 5.715,        # ← Water 实测平均
        "per_extra_type": 0.8,  # ← 纯猜
    },
    "se_atten": {
        "1_type": 5.2,          # ← 纯猜（没实测）
        "2_type": 6.1,          # ← 纯猜
        "per_extra_type": 0.8,
    },
}
```

`_get_fixed_overhead` (overhead_model.py:227) 直接按 `num_types` 查表，
3-type 以上线性外推：

```python
if num_types <= 1:
    fixed = overhead_cfg["1_type"]
elif num_types == 2:
    fixed = overhead_cfg["2_type"]
else:
    fixed = overhead_cfg["2_type"] + (num_types - 2) * 0.8

if compute_force:
    fixed += 0.15  # autograd backward
```

### 3.3 这个查表在哪些场景会失效

| 维度 | 当前能否处理 | 失效程度 |
|------|------------|----------|
| 同 model_type, 改 N | ✅ 不依赖 N | — |
| 同 model_type, 改 num_types (1↔2) | ✅ 有实测 | 准 |
| 同 model_type, num_types ≥ 3 | ⚠️ 用 0.8/type 外推 | 10–15% 误差 |
| 改 embedding_net 深度 / 宽度 | ❌ 表里没这维度 | **完全失效** |
| 改 fitting_net 深度 / 宽度 | ❌ 表里没这维度 | **完全失效** |
| descriptor: se_atten / DPA-1 | ❌ 表项是猜的 | 30–50% 误差 |
| descriptor: DPA-2 / NequIP / Allegro | ❌ 表里没有 | **不支持** |
| 跨 GPU (H100→A100) | ❌ 表是 H100 实测 | 必须重测整张表 |
| 改 dtype (fp64→fp32) | ❌ 表里没这维度 | 未知 |

### 3.4 `_count_framework_kernels` 是个未利用的金矿

`overhead_model.py:209-225` 已经在数 kernel 数量，但**只用于 info 输出**，
没有反哺 fixed overhead 的计算：

```python
# overhead_model.py:135-146
FRAMEWORK_KERNELS = {
    "se_e2_a": {
        "base": 250,
        "per_type": 8,
        "force_extra": 30,
    },
    ...
}

def _count_framework_kernels(self, deepmd_config, compute_force):
    framework_cfg = self.FRAMEWORK_KERNELS.get(model_type, ...)
    num_types = len(deepmd_config.get("type_map", ["X", "Y"]))
    n_framework = framework_cfg["base"] + num_types * framework_cfg["per_type"]
    if compute_force:
        n_framework += framework_cfg["force_extra"]
    return n_framework
```

如果把它接到 fixed：

```python
# 假想的 v6 公式
fixed = PY_BASE + n_framework × τ_launch
```

只需要标定两个 GPU/PyTorch 全局常数 (`PY_BASE`, `τ_launch`)，就能：

- 跨 GPU 外推（重测 τ_launch 一次）
- 跨架构外推（n_framework 自动随结构变）
- 跨 descriptor 外推（base / per_type 各 descriptor 一组常数）

**这是 v5 → v6 单点收益最大的升级。**

### 3.5 为什么 v5 没这么做

可见的原因是**校准数据不够**：当前只有 Water + Copper 两个数据点，
做不出 4 参数物理模型 (PY_BASE, τ_launch, per_type, per_emb_layer) 的标定。
两个点拟合 4 参数 → 欠定 → 必崩。

需要补的最小数据集（每个 N=64 单点即可）：

| 实验 | 变化 | 解决的参数 |
|------|------|-----------|
| Water 默认 (已有) | baseline | — |
| Copper 默认 (已有) | num_types=1 | per_type 初值 |
| 三元体系 (e.g. NaCl) | num_types=3 | per_type 二次项 |
| Water 改 embedding_net 深度 | L_emb=4 vs 3 | per_emb_layer |
| Water 改 fitting_net 深度 | L_fit=2 vs 3 | per_fit_layer |
| Empty kernel storm micro-benchmark | — | τ_launch 直接测 |

约 6 组实验 → 物理模型可以达到 5–8% 全场景误差，
并获得跨 GPU / 跨架构 / 跨 descriptor 的外推能力。

---

## 4. 转换区误差：为什么 max() 模型在 N=1024–2048 区间崩

### 4.1 现象

实测 H100 NVL Water：

| N | fixed_pred | (mlp+unmod)_pred | v5 max() 预测 | 实测 | 误差 |
|---|-----------|------------------|---------------|------|------|
| 64 | 5.7 | 0.3 | 5.7 | 5.69 | 0.2% |
| 256 | 5.7 | 0.5 | 5.7 | 5.71 | 0.2% |
| 1024 | 5.7 | 1.8 | 5.7 | 6.84 | **+20%** |
| 2048 | 5.7 | 6.7 | 6.7 | 11.74 | **+75%** |
| 4096 | 5.7 | 35.2 | 35.2 | 35.24 | 0.1% |

转换区（N=1024–2048）误差最大可达 +75%（point prediction 偏低）。
N≤512 和 N≥4096 都很准。

### 4.2 物理图像：大小 kernel 交错

#### Overhead-bound 区间 (N≤512)

所有 kernel 都很小，时间线长这样：

```
GPU:  [k1] idle [k2] idle [k3] idle [k4] idle ... [k350]
       ↑     ↑16μs↑
       每段 idle ≈ τ_launch = 16 μs

总时间 ≈ N_kernels × τ_launch = 350 × 16μs = 5.6 ms ✓
GPU 利用率 ≈ Σ kernel_exec / total ≈ 5%
```

`fixed_overhead = 5.7 ms` 完美对应。

#### Compute-bound 区间 (N≥4096)

所有 kernel 都很大：

```
GPU:  [——————— k1 ———————][——————— k2 ———————][———— k3 ————]...
       ↑
       launch latency 完全被 kernel 自己的执行时间 hide

总时间 ≈ Σ kernel_exec ≈ mlp + unmod = 35 ms ✓
GPU 利用率 ≈ 95%
```

#### 转换区 (N=1024–2048)：大小 kernel 交错

```
GPU:  [k1_small] idle [—— k2_big ——] [k3_small] idle [—— k4_big ——]
      ^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^
      被 launch 限制                   被 launch 限制
                       ^^^^^^^^^^^^^               ^^^^^^^^^^^^^
                       被 compute 限制               被 compute 限制
```

每个 kernel 的瓶颈来源不同：

```
真实情况:    Σ_i max(launch_i, compute_i)
v5 假设:    max(Σ_i launch_i, Σ_i compute_i)
            ↑ 两个累加再取 max——只有当所有 kernel 同质时才相等
```

这两个不等式之间的差就是 **pipeline bubble**。

### 4.3 Bubble 的代数估计

设 N_big 个大 kernel 各自执行时间 t_big，N_small 个小 kernel 各自 launch
时间 τ。两种极端：

**模型 A（v5 用的）：max() — 假设两条独立关键路径**
```
T_A = max(N_small·τ + N_big·τ,  N_big·t_big)
```

**模型 B（真实）：交错 → 串行求和**
```
T_B = N_big·t_big + N_small·τ
```

差值（bubble）：
```
Δ = T_B − T_A
  = N_small·τ              当 N_big·t_big ≥ N_small·τ + N_big·τ 时
  = N_big·τ                当 N_small·τ + N_big·τ ≥ N_big·t_big 时
```

- N_small=300, τ=16μs → 5 ms 的"被吞掉"开销
- 在 N=2048 时，N_big·t_big ≈ 5 ms，N_small·τ ≈ 5 ms → 几乎相等
- max() 取一个，但实际是两者**串行**累加 → 差 5 ms ≈ +75%

完美对应实测的转换区误差。

### 4.4 为什么不同模型的 bubble 大小不同

实测 Water 转换区 bubble ≈ 25%，Copper ≈ 16%。原因：

| 因素 | Water (2-type) | Copper (1-type) |
|------|---------------|-----------------|
| Type pair 数 | 4 | 1 |
| Per-pair embedding | 4 次小 kernel | 1 次小 kernel |
| `nlist_distinguish_types` 循环次数 | 2 | 1 |
| 大小 kernel 比例 | 小 kernel 多 | 小 kernel 少 |
| Bubble 强度 | **大** | **小** |

也就是 **type 越多 → 小 kernel 越多 → 大小交错越频繁 → bubble 越大**。
v5 没建模这个维度。

### 4.5 v5 的应对：confidence band

```python
# overhead_model.py:411-453
if fixed_overhead_ms > 0:
    transition_ratio = adjusted_compute_ms / fixed_overhead_ms

if transition_ratio < 0.8:
    regime = "overhead-bound"; confidence = "high"
    # bounds collapse to point
elif transition_ratio > 2.0:
    regime = "compute-bound"; confidence = "high"
else:
    regime = "transition"; confidence = "low"
    bubble_peak_ms = 0.20 * fixed_overhead_ms
    sigma_left, sigma_right = 0.2, 1.0
    distance = transition_ratio - 1.0
    sigma = sigma_left if distance <= 0 else sigma_right
    decay = exp(-0.5 * (distance / sigma) ** 2)
    uncertainty_ms = bubble_peak_ms * decay
    e2e_lower = e2e - uncertainty_ms
    e2e_upper = e2e + uncertainty_ms
```

物理含义：

- `transition_ratio = (mlp + unmod) / fixed`，1.0 是物理 crossover point
- ratio = 1.0 时 bubble 最严重（两条 ceiling 撞在一起）
- 偏离 1.0 越远，某一边的瓶颈占主导，bubble 衰减
- 左侧（overhead-bound 那侧）衰减快（σ=0.2）：
  GPU 没饱和，bubble 一旦消失就消失
- 右侧（compute-bound 那侧）衰减慢（σ=1.0）：
  大 kernel 占满 GPU 之后小 kernel 还要慢慢"扫尾"，残留 bubble

但**这只给 confidence interval，不修 point prediction**。
所以转换区的点预测仍然偏 15–25%。

### 4.6 为什么 bubble 难精确建模

要把转换区误差降下来，需要知道：

1. **每个 kernel 的 size class**——是 launch-bound 还是 compute-bound？
2. **kernel 之间的依赖图**——哪些可以重叠 launch？
3. **CUDA stream 调度细节**——PyTorch 默认单 stream 还是多 stream？
4. **CPU launch queue 深度**——driver 能 prefetch 几个 kernel？

这些信息在 NeuSight 的 op_df 里**完全没有**——op_df 只知道 op 的输入输出
shape，不知道它会被编译成多少个 CUDA kernel、每个 kernel 占多少 SM、
调度顺序如何。要拿到这些得 hook 到 CUDA driver 层（CUPTI）。

所以 v5 选择**承认转换区不准、用 confidence band 标注 + 不强行精确化**。
这是工程取舍，不是理论极限。

---

## 5. 误差归因总表

| 区间 | 主导分量 | 误差来源 | 当前 MAE | 上限策略 |
|------|---------|---------|---------|---------|
| N≤512 (overhead-bound) | fixed | 查表精度 | <2% | 已饱和 |
| N=1024–2048 (transition) | max() | bubble (大小 kernel 交错) | 15–25% | 需要 per-kernel 调度 / NN 残差 |
| N≥4096 (compute-bound) | mlp + unmod | C_QUAD/C_LINEAR 拟合 | 2–5% | 增加校准点 |
| 跨 GPU | 全部 | fixed 必须重测 | 未知 | 物理化 fixed (kernel-counting) |
| 跨 descriptor (DPA-1+) | fixed (主) | 表项是猜的 | 30–50% | 物理化 fixed |
| 跨架构 (改深度) | fixed | 表里没这维度 | **完全失效** | 物理化 fixed |

---

## 6. 物理推导 vs 拟合 vs 查表 — 元素级清单

| 元素 | 当前形式 | 来源 | 文件位置 |
|------|---------|------|---------|
| `nall = ns × N` | 物理 | nlist.py:483 | overhead_model.py:281 |
| `ns = 27` | 物理 (PBC, box >> rcut) | nlist.py:461 | overhead_model.py:84 |
| `nnei = sum(sel)` | 物理 (config 直读) | nlist.py:124 | overhead_model.py:356 |
| `× 8 bytes` | 物理 (float64) | env.GLOBAL_PT_FLOAT_PRECISION | overhead_model.py:282 |
| `/ BW` | 物理 (memory-bound roofline) | — | overhead_model.py:288 |
| `× (REF_BW / target_BW)` | 物理 (跨 GPU 缩放) | — | overhead_model.py:374 |
| `C_QUAD = 28.33` | 拟合 (5 个数据点) | scipy lstsq | overhead_model.py:78 |
| `C_LINEAR = 2053.23` | 拟合 (5 个数据点) | scipy lstsq | overhead_model.py:79 |
| `fixed_2type = 5.715` | **查表** | Water 实测平均 | overhead_model.py:95 |
| `fixed_1type = 4.850` | **查表** | Copper 实测平均 | overhead_model.py:94 |
| `per_extra_type = 0.8` | **猜测** | — | overhead_model.py:96 |
| `se_atten 1_type = 5.2` | **猜测** | 没实测过 | overhead_model.py:99 |
| `force_extra = 0.15ms` | **猜测** | autograd 经验值 | overhead_model.py:249 |
| `bubble_peak_fraction = 0.20` | **拟合** | Water/Copper 转换区 | overhead_model.py:132 |
| `sigma_left = 0.2` | **几何近似** | 经验 | overhead_model.py:444 |
| `sigma_right = 1.0` | **几何近似** | 经验 | overhead_model.py:445 |
| `transition_lo = 0.8` | **几何近似** | 经验 | overhead_model.py:130 |
| `transition_hi = 2.0` | **几何近似** | 经验 | overhead_model.py:131 |
| `n_framework = base + num_types*per_type` | 半物理 (源码 walk) | nlist.py + env_mat.py | overhead_model.py:222 |
| `kernel_count` | 半物理 | nn.Module hook | overhead_model.py:201 |

> 共 20 个关键参数：
> - **物理推导：6 个**（nall, ns, nnei, dtype, BW, gpu_scale）
> - **半物理（源码 walk）：2 个**（n_framework, kernel_count）
> - **少量拟合：3 个**（C_QUAD, C_LINEAR, bubble_peak）
> - **查表：2 个**（fixed_2type, fixed_1type）
> - **猜测/几何近似：7 个**（per_extra_type, se_atten 系列, force_extra,
>   sigma_left/right, transition_lo/hi）

> "猜测/查表" 加起来 9 个——**这是 v6 的攻击面**。

---

## 7. v6 升级路线（仅作参考）

按 ROI 排序：

### Phase A — fixed overhead 物理化（最大杠杆）

```
fixed = PY_BASE + n_framework(arch) × τ_launch(GPU)

需要标定:
  PY_BASE       ≈ 1.5 ms          （Python/dispatcher 常数）
  τ_launch      ≈ 16 μs (H100)    （micro-benchmark 直接测）
  n_framework   = base + L_emb·g(T) + L_fit·k1 + force·k2
```

需补的实验：6 组小标定（见 §3.5）。
预期收益：
- 跨 GPU：自动迁移
- 跨架构：自动外推
- 跨 descriptor：每个 descriptor 标定一次 base
- 校准范围内误差：5–8%（vs 当前 2–5%，略退化但换来跨场景能力）

### Phase B — bubble 残差 NN（少数 NN 真正合适的场景）

转换区 bubble 是调度器的"经验性偏差"，没有解析公式可推。
训一个小 MLP（输入: ratio + num_types + descriptor + N_big/N_small
比例，输出: bubble fraction），代替当前的高斯 decay 公式。

预期收益：
- 转换区误差从 15–25% 降到 5–10%
- 模型仍主体物理，NN 只学残差

### Phase C — 跨 dtype / 跨 batch_size 扩展

当前公式 hard-code `× 8 bytes`，未来切到 fp32 / mixed precision 时
需要 `dtype_bytes` 参数化。batch_size > 1 时整个公式只需在头部加 `× nframes`。

---

## 8. 一句话总结

**v5 的 overhead 模型在精度上够用、在物理性上一半到位**：

- **Unmodeled compute** 已经是漂亮的源码可推 roofline，只 2 个拟合系数
- **Fixed overhead** 仍然是查表 + 猜测，跨场景外推能力差
- **转换区** 用几何 bubble 近似 + confidence band 兜底，
  不修值但标注不可靠

最值得做的 v6 升级是 **把 fixed overhead 也变成 kernel-counting 物理公式**，
让整个模型从「半物理 + 查表」彻底变成「全物理 + 少量残差 NN」。

---

## 附录 A. 关键源码片段索引

| 代码片段 | 文件:行 | 在 v5 公式中的角色 |
|---------|---------|-------------------|
| `extend_coord_with_ghosts` | `pt/utils/nlist.py:407-496` | 决定 `ns`, `nall` |
| `nbuff = ceil(rcut / face)` | `pt/utils/nlist.py:461` | `ns = (2·nbuff+1)^3` |
| `diff = coord.unsqueeze(1) - coord.unsqueeze(2)` | `pt/utils/nlist.py:113` | C_QUAD 项主导 |
| `rr = norm(diff, dim=-1)` | `pt/utils/nlist.py:116` | C_QUAD 项 |
| `rr, nlist = torch.topk(rr, top_k)` | `pt/utils/nlist.py:127` | C_QUAD 项（partial sort） |
| `nlist_distinguish_types` 循环 | `pt/utils/nlist.py:287-320` | C_LINEAR 项（× num_types） |
| `coord_r = torch.gather(coord_pad, 1, index)` | `pt/model/descriptor/env_mat.py:30` | C_LINEAR 项（random gather） |
| `length = norm(diff, dim=-1)` | `pt/model/descriptor/env_mat.py:33` | C_LINEAR 项 |
| `(env_mat - mean) / std` | `pt/model/descriptor/env_mat.py:91` | C_LINEAR 项 |
| `embedding_net (per type pair)` | `pt/model/descriptor/se_a.py` | mlp_compute（NeuSight 建模） |
| `fitting_net (Linear×L_fit)` | `pt/model/task/ener.py` | mlp_compute（NeuSight 建模） |
| `force = -∂E/∂coord (autograd)` | `pt/model/atomic_model/...` | fixed +0.15ms（autograd backward） |

---

*文档版本：1.0 — 基于 NeuSight-DeepMD predictor v5 (2025-Q4) + DeepMD-kit
v3.x se_e2_a descriptor PyTorch backend。*
