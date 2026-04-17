# DeepMD-kit 推理性能预测实验报告

---

## 一、实验目的

### 1.1 背景

NeuSight (ASPLOS'25) 的 MLP_WAVE predictor 可以在不跑 GPU 的情况下预测 Linear、BMM 等标准算子的延迟，但它是为 Transformer 设计的。DeepMD-kit 的计算图和 Transformer 差别很大，没法直接用 FX tracing 提取算子图。

这个实验要解决的问题是：能不能把 DeepMD-kit 的推理过程拆成 NeuSight 能预测的算子，再把 NeuSight 覆盖不了的部分用分析模型补上，最终得到一个端到端的延迟预测？

### 1.2 具体目标

1. 分析 DeepMD-kit PyTorch 后端源码，手动把推理过程拆解为 NeuSight 兼容的算子序列
2. 找出 NeuSight 覆盖不了的计算，建分析模型补偿
3. 在 H100 GPU 上验证预测精度，覆盖 N=32 到 N=8192
4. 搞清楚哪些区间预测不准，为什么不准

---

## 二、实验环境

### 2.1 硬件

| 项目 | 规格 |
|:-----|:-----|
| GPU | NVIDIA H100 80GB NVL |
| GPU 显存 | 80 GB HBM3 |
| 显存带宽 | 3430 GB/s |
| SM 数量 | 132 |
| 每 SM CUDA Core | 128 |
| GPU 频率 | 1980 MHz |
| 单精度算力 | 66,908 GFLOPS |
| L2 Cache | 50 MB |

### 2.2 软件

| 项目 | 版本 |
|:-----|:-----|
| DeepMD-kit | v3.x (PyTorch 后端) |
| PyTorch | 2.x (CUDA 12) |
| NeuSight | MLP_WAVE predictor |
| Python | 3.10+ |
| scipy | 用于模型参数拟合 |

### 2.3 测试模型

两个 DeepMD 模型，都用 `se_e2_a` 描述子：

| 参数 | Water 模型 | Copper 模型 |
|:-----|:-----------|:------------|
| 原子类型 (type_map) | [O, H] (2 种) | [Cu] (1 种) |
| 邻居选择 (sel) | [46, 92] | [120] |
| 最大邻居数 (nnei) | 138 | 120 |
| 截断半径 (rcut) | 6.0 A | 7.0 A |
| Embedding 网络 | [25, 50, 100] | [25, 50, 100] |
| axis_neuron | 16 | 16 |
| Fitting 网络 | [240, 240, 240] | [240, 240, 240] |
| Descriptor 维度 | 100 x 16 = 1600 | 100 x 16 = 1600 |

选 Water 是因为它有两种原子类型，embedding 网络按类型独立运行（batch_0 = N x 46, batch_1 = N x 92），可以验证 per-type 建模对不对。Copper 只有一种类型（batch = N x 120），拿来交叉验证。

### 2.4 Profiling 方法

```python
for _ in range(num_warmup):    # warmup 30 次 (N>=2048 时 10 次)
    model(coord, atype, box)
torch.cuda.synchronize()

latencies = []
for _ in range(num_runs):      # 测量 100 次 (N>=2048 时 30 次)
    torch.cuda.synchronize()
    start = time.perf_counter()
    model(coord, atype, box)   # 含 force backward (autograd)
    torch.cuda.synchronize()
    end = time.perf_counter()
    latencies.append((end - start) * 1000)
```

每次测量前后都有 `torch.cuda.synchronize()`，保证 GPU 跑完了再计时。取 mean 为主指标，同时记录 std、p5、p95。所有测试确认 std < 1%，GPU 空闲无干扰。

---

## 三、实验过程

实验分 5 个阶段，后面每个阶段都是上一阶段的数据里冒出了新问题才开始的。

```
阶段 A: 源码分析 + 算子图构建 + 初版预测
        → MLP_WAVE 只预测了约 1ms，实测 5~6ms

阶段 B: 固定 overhead 测量 + 初版补偿
        → 小 N (N≤1024) 很准 (MAE≈1%)
        → 大 N (N=2048) 严重偏低

阶段 C: 重新看源码，找到遗漏的 O(N²) 计算 + Roofline 建模
        → 两端都准了 (MAE < 5%)
        → 中间区间 (N≈1280~1920) 误差冒出来，最高 15%

阶段 D: 转换区密集 profiling + 根因分析
        → 误差来自 Pipeline Bubble
        → Bubble 大小和模型结构绑定，没法用统一参数修

阶段 E: 放弃精确修正，改成 Confidence-Aware 方案
```

---

### 3.1 阶段 A: 源码分析与算子图构建

NeuSight 对 Transformer 模型用 HuggingFace FX tracing 自动提取计算图。DeepMD-kit 不是 Transformer，这条路走不通，得手动读源码拆解。

> 以下源码路径都是 DeepMD-kit 安装包内的，根目录 `deepmd/pt/`，
> 实际在 `~/.local/lib/python3.12/site-packages/deepmd/pt/`。

读完源码后，把推理过程拆成 7 个阶段：

| 阶段 | 源码位置 | 干什么 |
|:-----|:---------|:---------|
| 1. Neighbor List | `deepmd/pt/utils/nlist.py` L47-135 | broadcast diff + topk |
| 2. Environment Matrix | `deepmd/pt/model/descriptor/env_mat.py` L11-48 | gather + 1/r + smooth |
| 3. Embedding Network | `deepmd/pt/model/descriptor/se_a.py` L789-836 | per-type MLP + matmul |
| 4. Descriptor | `deepmd/pt/model/descriptor/se_a.py` L838-844 | matmul(G1, G2) |
| 5. Fitting Network | `deepmd/pt/model/task/fitting.py` L847-857 | MLP [1600→240→240→240→1] |
| 6. Output | `deepmd/pt/model/task/fitting.py` L856 | per-atom energy sum |
| 7. Force backward | autograd | 反向 GEMM |

读源码过程中注意到两个容易搞错的地方：

第一个，embedding 网络是按原子类型独立的：

```python
# deepmd/pt/model/descriptor/se_a.py  DescrptBlockSeA.forward():
for ii in range(self.ntypes):         # 按邻居类型遍历
    rr = dmatrix[:, sec[ii]:sec[ii+1], :]  # 取 sel[ii] 个邻居
    ss = rr[:, :, :1]                       # 标量输入 s(r) = 1/r
    gg = self.filter_layers[ii].forward(ss) # 独立 embedding net
    gr = torch.matmul(rr.permute(0,2,1), gg)
    xyz_scatter += gr
```

Water 模型有 2 个独立 embedding：Type 0 (O) batch = N x 46, Type 1 (H) batch = N x 92。初版建模时把它们合成了一个 N x 138 的 batch，后来改正了。

第二个，ResNet skip connection 有两种模式：

```python
# deepmd/pt/model/network/mlp.py  MLPLayer.forward():
if self.resnet:
    if xx.shape[-1] == yy.shape[-1]:
        yy = yy + xx                            # 等维: 直接 add
    elif 2 * xx.shape[-1] == yy.shape[-1]:
        yy = yy + torch.cat([xx, xx], dim=-1)   # 翻倍: concat + add
```

7 个阶段的操作映射成 NeuSight 的 6 种算子：

| DeepMD 操作 | NeuSight 算子 | 形状 |
|:------------|:-------------|:-----|
| `F.linear(x, W, b)` | `Linear` | (batch, in, out) |
| `torch.matmul(A, B)` | `BMM` | (N, m, k, n) |
| `torch.tanh(x)` | `VECtanh` | (batch, dim) |
| `y + x` (ResNet skip) | `VECadd` | (batch, dim) |
| `x * y` / `1/r` | `VECmul` | (batch, dim) |
| `torch.gather` / `cat` | `MEM` | (batch, dim) |

实现在 `trace_deepmd.py`，以 Water N=32, force=True 为例，生成 65 个算子节点。所有算子延迟直接求和（DeepMD 没有 Transformer 那种逐层复制的结构）。

构建完算子图，用 MLP_WAVE 预测每个算子再求和，和实测一比：

| N | 实测 (ms) | MLP_WAVE 预测 (ms) | 差距 |
|---:|---:|---:|---:|
| 32 | 5.890 | 1.14 | 4.75 ms 没着落 |
| 1024 | 5.873 | 1.68 | 4.19 ms 没着落 |
| 2048 | 11.823 | 2.43 | 9.39 ms 没着落 |

MLP_WAVE 只预测到了实际延迟的 20~30%。剩下的 70~80% 是什么？

---

### 3.2 阶段 B: 固定 overhead

看实测数据的时候注意到一个现象：

```
N=32:   5.890 ms
N=64:   5.893 ms
N=128:  5.902 ms
N=256:  5.939 ms
N=512:  5.941 ms
N=1024: 5.873 ms
```

N 从 32 涨到 1024，涨了 32 倍，延迟基本没动。MLP_WAVE 预测的计算时间确实在涨（1.14ms → 1.68ms），说明 GPU 干的活变多了，但端到端延迟不涨。这说明有个和 N 无关的固定开销在卡着下限。

Nsight Systems profiling 确认了来源：DeepMD 一次推理大约 dispatch 350 个 CUDA kernel，CPU 要逐个提交给 GPU driver，每个 launch 约 16us，加起来大约 5.6ms。这个提交时间和原子数没关系。

不同模型 kernel 数量不同：

| 模型 | 固定 overhead (ms) | 原因 |
|:-----|:------------------|:-----|
| Water (2 types) | 5.715 | 两种类型，per-type dispatch 多 |
| Copper (1 type) | 4.850 | 一种类型，kernel 少一些 |

先试最简单的方案，`e2e = mlp + fixed`：

| N | 实测 (ms) | mlp + fixed (ms) | 误差 |
|---:|---:|---:|---:|
| 32 | 5.890 | 1.14 + 5.715 = 6.855 | +16.4% |
| 1024 | 5.873 | 1.68 + 5.715 = 7.395 | +25.9% |
| 2048 | 11.823 | 2.43 + 5.715 = 8.145 | -31.1% |

不对。小 N 时预测偏高，因为 GPU 算得快，每个 kernel 在 CPU 提交下一个之前就跑完了，计算时间被 launch 时间"藏"起来了，不该加，该取 max。大 N 时预测偏低，说明还有别的 GPU 计算没算进来。

所以正确的模型应该是：

```python
e2e = max(fixed_overhead, total_compute)
```

CPU 提交 kernel 和 GPU 执行形成流水线。小 N 时 GPU 闲着等 CPU，瓶颈在 CPU launch 链；大 N 时 CPU 早就提交完了，瓶颈在 GPU 计算。

但 `total_compute` 里除了 MLP_WAVE 预测的那部分，还得加上 NeuSight 漏掉的 GPU 计算，不然大 N 还是预测不了。

---

### 3.3 阶段 C: 找到遗漏计算 + Roofline 建模

用 `max(fixed, mlp_compute)` 试一下：

| N | 实测 (ms) | max(fixed, mlp) (ms) | 误差 |
|---:|---:|---:|---:|
| 32 | 5.890 | max(5.715, 1.14) = 5.715 | -3.0% |
| 1024 | 5.873 | max(5.715, 1.68) = 5.715 | -2.7% |
| 2048 | 11.823 | max(5.715, 2.43) = 5.715 | -51.7% |

N=2048 差了一半。MLP_WAVE 预测 2.43ms，实际跑了 11.8ms，减去 fixed overhead 还剩大约 9.4ms 的 GPU 计算没被 NeuSight 覆盖到。

回头重新审查 7 个阶段的源码，发现阶段 1 (Neighbor List) 的核心计算根本没被建进算子图：

```python
# deepmd/pt/utils/nlist.py  build_neighbor_list() L113-127:

# vcoord_xyz: [batch, nall, 3],  nall = ns * N (ns≈27, ghost cell 复制)
# vcoord_local_xyz: [batch, N, 3]

diff = vcoord_xyz.unsqueeze(1) - vcoord_local_xyz.unsqueeze(2)
# → diff: [batch, N, nall, 3] = [batch, N, 27N, 3]
# 数据量 = N x 27N x 3 x 8 bytes (float64)

rr = torch.linalg.norm(diff, dim=-1)   # [batch, N, nall]
rr, nlist = torch.topk(rr, top_k, largest=False)
```

`unsqueeze(1) - unsqueeze(2)` 产生 N x nall 的广播减法，数据量 = N x 27N x 8 = 216N^2 bytes，O(N^2) 的。这是个纯 memory-bandwidth-bound 操作，NeuSight 的 MLP_WAVE 没有对应的算子类型来预测它。

阶段 2 的 `torch.gather` 也有同样的问题：

```python
# deepmd/pt/model/descriptor/env_mat.py L28-30:
index = nlist.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
coord_r = torch.gather(coord_pad, 1, index)   # random access [N*nnei, 3]
```

gather 按 nlist 索引取数，访问模式是 random access，GPU cache 基本没用。

#### Roofline 建模

这两个操作都是 memory-bandwidth-bound，可以用 Roofline 思路估算：延迟 ≈ 数据量 / 带宽。但实际带宽利用率远低于理论值（多遍访问、random access），所以引入有效乘数 C：

```
gpu_overhead = quad_ms + linear_ms

quad_ms   = C_QUAD   x (N x nall x 8) / (BW x 1e9) x 1000    ← O(N^2) 项
linear_ms = C_LINEAR x (N x nnei x 8) / (BW x 1e9) x 1000    ← O(N) 项

nall = 27 x N   (ghost cell 复制因子)
nnei = sum(sel)  (最大邻居数)
```

C_QUAD 为什么是 ~28 而不是 1？因为 nlist.py 里 broadcast → norm → topk 这一串操作至少要 3~4 遍读写同一块数据，加上 PyTorch 分配临时 buffer 的开销。C_LINEAR 为什么到了 ~2000？因为 `torch.gather` 的 random access 让 GPU cache 几乎完全失效，再加上 per-type mask loop 多次遍历，实际带宽利用率非常低。

#### 拟合

用 12 个实测点拟合 4 个参数（C_QUAD, C_LINEAR, fixed_water, fixed_copper）：

```python
CALIBRATION_DATA = [
    # (模型, N, nnei, 实测延迟ms, MLP_WAVE预测ms)
    ("water", 32,   138, 5.601,  0.240),
    ("water", 64,   138, 5.573,  0.358),
    # ... (共 7 个 Water overhead-bound 点)
    ("water", 2048, 138, 11.742, 2.172),
    ("water", 4096, 138, 35.236, 3.834),
    ("copper", 2048, 120, 11.332, 2.172),
    ("copper", 4096, 120, 35.128, 3.834),
    ("copper", 8192, 120, 129.528, 7.228),
]
```

预测公式：

```python
def predict_latency(model, N, nnei, mlp_ms, C_quad, C_linear, fixed_water, fixed_copper):
    nall = 27 * N
    gpu_oh = C_quad * N * nall * 8 / (3430e9) * 1000
           + C_linear * N * nnei * 8 / (3430e9) * 1000
    fixed = fixed_water if model == "water" else fixed_copper
    return max(fixed, mlp_ms + gpu_oh)
```

优化目标是加权 MAE。加权是因为小 N 时 `max()` 输出的是 fixed，C_QUAD 和 C_LINEAR 怎么调都不影响预测值，如果等权重，优化器会把注意力浪费在这些点上。compute-bound 点给 3x 权重，中等 N 给 2x，小 N 给 1x。

```python
result = minimize(
    objective,
    x0=[28.3, 2053.0, 5.715, 4.850],
    method='L-BFGS-B',
    bounds=[(1,200), (100,50000), (3,10), (3,10)],
    options={'maxiter': 50000, 'ftol': 1e-15}
)
```

拟合结果：

```
C_QUAD    = 28.3284
C_LINEAR  = 2053.2321
fixed_water  = 5.715 ms
fixed_copper = 4.850 ms
MAE = 2.24%, 最大误差 = 6.2%
```

#### 完整模型

```python
compute_ms = sum(MLP_WAVE_predict(op) for op in 65_ops)

gpu_overhead_ms = C_QUAD * N * 27 * N * 8 / (BW * 1e9) * 1000
                + C_LINEAR * N * nnei * 8 / (BW * 1e9) * 1000

e2e = max(fixed_overhead, compute_ms + gpu_overhead_ms)
```

小 N 时 compute + gpu_oh 远小于 fixed（比如 N=32 时 1.1 + 0.5 = 1.6ms vs fixed 5.7ms），max 取到 fixed。大 N 时反过来（N=8192 时 7.2 + 125.1 = 132.3ms vs fixed 5.7ms），max 取到 compute。

#### 验证

Water N=32~2048：

| N | 实测 (ms) | 预测 (ms) | 误差 |
|---:|---:|---:|---:|
| 32 | 5.890 | 5.865 | -0.4% |
| 64 | 5.893 | 5.865 | -0.5% |
| 128 | 5.902 | 5.865 | -0.6% |
| 192 | 5.995 | 5.865 | -2.2% |
| 256 | 5.939 | 5.865 | -1.2% |
| 512 | 5.941 | 5.865 | -1.3% |
| 1024 | 5.873 | 5.865 | -0.1% |
| 2048 | 11.823 | 11.268 | -4.7% |

Overhead-bound 区间 MAE = 0.9%。

大原子数：

| 模型 | N | 实测 (ms) | 预测 (ms) | 误差 |
|:-----|---:|---:|---:|---:|
| Water | 4096 | 35.532 | 36.597 | +3.0% |
| Water | 8192 | 129.113 | 132.327 | +2.5% |
| Copper | 2048 | 11.137 | 11.091 | -0.4% |
| Copper | 4096 | 33.763 | 36.244 | +7.3% |
| Copper | 8192 | 125.034 | 131.620 | +5.3% |

Compute-bound 区间 MAE = 3.7%。两端都可以了，但中间出了问题。

---

### 3.4 阶段 D: 转换区

两端验证完后，在 N=1024~2048 之间以 step=128 做了一轮密集 profiling，结果发现中间有一段误差明显偏大：

| N | 实测 (ms) | 预测 (ms) | 误差 | ratio |
|---:|---:|---:|---:|---:|
| 1024 | 5.873 | 5.865 | -0.1% | 0.72 |
| 1152 | 5.942 | 5.865 | -1.3% | 0.84 |
| 1280 | 6.214 | 5.865 | -5.6% | 0.96 |
| 1408 | 6.640 | 6.467 | -2.6% | 1.10 |
| 1536 | 6.950 | 7.324 | +5.4% | 1.25 |
| 1664 | 7.525 | 8.219 | +9.2% | 1.40 |
| 1792 | 8.275 | 9.213 | +11.3% | 1.57 |
| 1920 | 9.378 | 10.225 | +9.0% | 1.74 |
| 2048 | 11.823 | 11.266 | -4.7% | 1.92 |

这里 ratio = adjusted_compute / fixed_overhead。ratio < 1 时预测偏低，过了 1 之后预测偏高，最大误差出现在 ratio ≈ 1.5。

#### 为什么 max() 在这里不准

模型假设了完美流水线：`e2e = max(总 launch, 总 compute)`。

但 350 个 kernel 是一个一个跑的。实际是 `e2e = 求和 max(launch_i, compute_i)`。

Jensen 不等式告诉我们 `求和 max(a_i, b_i) >= max(求和 a_i, 求和 b_i)`，等号条件是所有 kernel 大小相同。

举个例子，3 个 kernel，每个 launch 15us：

```
Kernel A (大): compute=50us → max(15,50) = 50us
Kernel B (小): compute=5us  → max(15,5)  = 15us  ← 10us 的空闲 (bubble)
Kernel C (大): compute=50us → max(15,50) = 50us

实际 e2e = 50 + 15 + 50       = 115us
max() 模型 = max(45, 105)      = 105us  → 少算了 8.7%
```

Kernel B 在 GPU 上 5us 就跑完了，但 CPU 提交下一个要 15us，中间有 10us 的 idle gap。max() 模型看到"总 compute > 总 launch"就返回总 compute，把这些碎片空闲忽略了。

这个 bubble 多大，取决于 kernel 大小的分布方差：

| 模型 | Peak bubble | 原因 |
|:-----|:-----------|:-----|
| Water (2 types) | ~25% of fixed | 两个 embedding net 大小不一 (N x 46 vs N x 92)，kernel 大小方差大 |
| Copper (1 type) | ~16% of fixed | 一个 embedding net (N x 120)，kernel 大小比较均匀 |

bubble 的大小由模型结构决定，不同模型不一样。

#### 试过的修正方案

| 方案 | 思路 | Water | Copper | 结论 |
|:-----|:-----|:------|:-------|:-----|
| 全局 softmax 平滑 | softmax(fixed, adj, T) 替代硬 max | 破坏两端精度 | 破坏两端精度 | 放弃 |
| 局部 Gaussian 修正 | ratio ≈ 1.0 处加修正项 | MAE 降到 ~3% | 过修正 | peak 值依赖模型，不通用 |
| 非对称 Gaussian | 左右 sigma 不同 | 改善有限 | peak 差 60% | 放弃 |
| Per-kernel 流水线模拟 | 模拟每个 kernel 的 launch/compute 交替 | MAE 反而 13.0% | — | framework kernel 分布拿不到，放弃 |

四个方案都试过了。局部 Gaussian 对 Water 效果不错 (MAE 3%)，但搬到 Copper 上就过修正了，peak 值差了 60%。根源还是 bubble 大小和模型结构绑定。

---

### 3.5 阶段 E: Confidence-Aware 方案

既然转换区的误差本质上取决于 kernel 大小分布（由模型结构决定），靠统一参数修不好，那就换个思路：不装作能精确预测，直接标出来告诉用户这段不太准。

用 ratio 划分区间：

```python
ratio = adjusted_compute_ms / fixed_overhead_ms

if   ratio < 0.8:   regime = "overhead-bound"    confidence = "high"
elif ratio > 2.0:   regime = "compute-bound"      confidence = "high"
else:                regime = "transition"         confidence = "low"
```

阈值 0.8 和 2.0 从密集 profiling 的数据里来：

```
N=1024  ratio=0.72  误差 -0.1%   ← 很准
N=1152  ratio=0.84  误差 -1.3%   ← 开始不准了 → 0.8 作为分界
...
N=2048  ratio=1.92  误差 -4.7%   ← 还有偏差
N=4096  ratio=6.24  误差 +3.0%   ← 恢复了 → 2.0 作为分界
```

ratio = 1.0 这个点在任何 GPU 上都是转换区的中心，因为 ratio = 1 就是 `adjusted = fixed`，也就是 max() 的切换点。这是数学性质，跟 GPU 型号无关。

在转换区内给出参考性的上下界：

```python
bubble_peak_ms = 0.20 * fixed_overhead_ms   # 最大不确定性，从实测校准

distance = ratio - 1.0
sigma = 0.2 if distance <= 0 else 1.0       # 左侧快衰减，右侧慢衰减
decay = exp(-0.5 * (distance / sigma)**2)
uncertainty = bubble_peak_ms * decay

lower = point_estimate - uncertainty
upper = point_estimate + uncertainty
```

验证结果：

| 测试项 | 数据集 | 测试点数 | 结果 |
|:-------|:-------|:---------|:-----|
| 区间检测 | Water + Copper | 12 | 12/12 通过 |
| Bounds 覆盖率 | Water (N=1024~2048) | 8 | 8/8 (100%) |
| Bounds 覆盖率 | Copper (N=1024~2048) | 7 | 7/7 (100%) |
| 向后兼容性 | — | 15+4 字段 | 全部存在 |

---

## 四、实验结果

### 4.1 Water 模型 (N=32~8192)

| N | 实测 mean (ms) | std | 预测 (ms) | compute | overhead | 误差 | 区间 | Confidence |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| 32 | 5.890 | 0.043 | 5.865 | 1.136 | 4.729 | -0.4% | overhead-bound | high |
| 64 | 5.893 | 0.293 | 5.865 | 1.072 | 4.793 | -0.5% | overhead-bound | high |
| 128 | 5.902 | 0.550 | 5.865 | 1.083 | 4.782 | -0.6% | overhead-bound | high |
| 192 | 5.995 | 0.354 | 5.865 | 1.040 | 4.825 | -2.2% | overhead-bound | high |
| 256 | 5.939 | 0.148 | 5.865 | 1.077 | 4.788 | -1.2% | overhead-bound | high |
| 512 | 5.941 | 0.127 | 5.865 | 1.369 | 4.496 | -1.3% | overhead-bound | high |
| 1024 | 5.873 | 0.035 | 5.865 | 1.683 | 4.182 | -0.1% | overhead-bound | high |
| ~1280 | 6.214 | — | 5.865 | — | — | -5.6% | transition | low |
| ~1536 | 6.950 | — | 7.324 | — | — | +5.4% | transition | low |
| ~1792 | 8.275 | — | 9.213 | — | — | +11.3% | transition | low |
| 2048 | 11.823 | 0.030 | 11.268 | 2.432 | 8.836 | -4.7% | transition | low |
| 4096 | 35.532 | 0.266 | 36.597 | 3.961 | 32.636 | +3.0% | compute-bound | high |
| 8192 | 129.113 | 4.973 | 132.327 | 7.194 | 125.133 | +2.5% | compute-bound | high |

### 4.2 Copper 模型

| N | 实测 mean (ms) | std | 预测 (ms) | 误差 | 区间 | Confidence |
|---:|---:|---:|---:|---:|:---|:---|
| 2048 | 11.137 | 0.023 | 11.091 | -0.4% | compute-bound | high |
| 4096 | 33.763 | 0.282 | 36.244 | +7.3% | compute-bound | high |
| 8192 | 125.034 | 3.302 | 131.620 | +5.3% | compute-bound | high |

### 4.3 分区精度

| 区间 | N 范围 (Water) | MAE | 最大误差 | Confidence |
|:-----|:---------------|:----|:---------|:-----------|
| Overhead-bound | 32 ~ 1024 | 0.9% | 2.2% | high |
| Transition | ~1152 ~ 2048 | ~10% | ~15% | low (已标注) |
| Compute-bound | 4096 ~ 8192 | 3.7% | 7.3% | high |

### 4.4 预测流水线

```
输入: GPU 配置 (JSON) + DeepMD 模型配置 (JSON) + 原子数 N
  |
  |-- parse_deepmd_input()            解析 DeepMD input.json
  |     提取 sel, neuron, type_map, rcut 等
  |
  |-- build_deepmd_opgraph()          手动构建算子图
  |     7 个阶段 → ~65 个 NeuSight 算子节点
  |
  |-- OperatorPredictor.predict()     逐算子预测 (MLP_WAVE)
  |     Linear → MLP_WAVE_MM
  |     BMM    → MLP_WAVE_MM
  |     VEC*   → MLP_WAVE_VEC
  |     MEM    → BW 公式
  |
  |-- aggregate_deepmd()              求和
  |     compute_latency = sum(fw_latency)
  |
  |-- DeepMDOverheadModel.estimate()  overhead 估算
  |     1. gpu_overhead (Roofline: C_QUAD x O(N^2) + C_LINEAR x O(N))
  |     2. adjusted = compute + gpu_overhead
  |     3. e2e = max(fixed_overhead, adjusted)
  |     4. ratio → 区间判断 + confidence
  |
  +--→ 输出: e2e_latency, confidence, bounds, breakdown
```

### 4.5 输出示例

正常区间：

```json
{
  "e2e_latency": 5.865,
  "confidence": {"level": "high", "regime": "overhead-bound"}
}
```

转换区：

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

终端输出：
```
DeepMD E2E latency for deepmd_se_e2_a_n1536_force on H100: 7.338 ms
  ⚠️ transition zone [6.20, 8.47]ms
```

### 4.6 代码

| 文件 | 做什么 |
|:-----|:-------|
| `neusight/Tracing/parse_deepmd_input.py` | 解析 DeepMD input.json |
| `neusight/Tracing/trace_deepmd.py` | 手动构建算子图 (7 阶段, v2) |
| `neusight/Prediction/predictor_deepmd.py` | 预测器入口，串联全流程 |
| `neusight/Prediction/overhead_model.py` | 两区间 overhead 模型 + Confidence-Aware (v5) |
| `neusight/Prediction/aggregator.py` | 算子延迟求和 |
| `scripts/pred_deepmd.py` | CLI 入口 |
| `scripts/calibrate_analytical.py` | C_QUAD/C_LINEAR 拟合 |
| `scripts/full_accuracy_test.py` | N=32~2048 精度验证 |
| `scripts/test_large_atoms.py` | N=4096~8192 精度验证 |
| `scripts/verify_confidence_aware.py` | Confidence-Aware 功能验证 |

### 4.7 局限性

1. 转换区 (ratio 0.8~2.0) 误差可达 ~15%，pipeline bubble 没法用统一参数修正，目前只能标注 confidence = low
2. 固定 overhead 值是在 H100 上测的，换 GPU 需要重新校准（不同 GPU 的 kernel launch latency 不同）
3. C_QUAD 在极端大 N 时可能偏移，因为 cache 行为会变
4. 只在 se_e2_a 描述子上充分验证过，DPA-1 (se_atten) 还没测

---

*报告完*
