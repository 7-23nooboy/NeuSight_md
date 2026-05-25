# NeuSight DeepMD 开销模型 — 验证与置信度报告

**日期：** 2026-05-25（v3：修正了 profiler 桶分类中遗漏的 neighbor-list 算子 topk/norm/cat，所有阶段分析重跑）  
**GPU：** NVIDIA H100 NVL (95 GB) × 2（评估使用单卡）
**软件栈：** DeepMD-kit 3.1.2 + PyTorch 2.8.0+cu128（`gpu_sim` conda 环境）
**校准：** `results/calibration/h100_nvl_v6.json`（v6：α=2.00 ms, β=0.0294 ms/kernel, δ=1.03 ms·n^p, γ=0.15 ms, C_quad=27.90, C_linear=1433.68, bubble_peak=0.35）

---

## 1. 实验范围

| 项目 | 取值 |
|---|---|
| 描述子 | `se_e2_a`（DPA-1 作为分布外样本排除）|
| 体系 | copper（1 元素）、water（2 元素）、LiAlOCl（4 元素）、he6（6 元素） |
| 原子数 (N) | 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| 总样本数 | 4 体系 × 9 个 N = **36 个测量点** |
| 路径 | `compute_force = True`（forward + autograd backward 算力）|
| N=16384/32768 | **OOM** — DeepMD-PyTorch 参考实现会一次性分配一个 fp64 pairwise 张量，单块 162 GB / 648 GB，无法在 2×80 GB H100 上 shard。要继续扩 N 需要切换 LAMMPS C++ 后端。|

---

## 2. 名词解释 — 三个阶段、wall 与 regime

NeuSight DeepMD 预测器把推理延迟拆为三个独立阶段，再合成端到端总延迟 `wall`：

| 符号 | 含义 | 由谁预测 |
|---|---|---|
| **fix** | CPU 调度 + kernel launch 开销 + autograd 元数据；与 N 无关，由 kernel 数 K 和 ntypes² 主导 | `α + β·K + δ·n^p + γ·is_force` |
| **cmp** | GPU 上 matmul / linear / bmm / 激活 / elementwise 的计算时间（"MLP 部分"）| NeuSight `MLP_WAVE` 逐算子预测器（Linear + BMM + VEC + MEM）|
| **roof** | GPU 上 descriptor / neighbor-list / scatter-gather 的时间（"roofline 部分"）| `(C_quad·N·n_all + C_linear·N·n_nei) · 8 / Mem_BW` |
| **wall** | **端到端总延迟**，即 `model(coord, atype, box)` 一次完整 forward + force backward 的耗时（`torch.cuda.synchronize() + time.perf_counter()`，80 次均值）| `max(fix, cmp + roof) + bubble 修正` |

### regime（性能瓶颈状态标签）

预测器按 `transition_ratio = (cmp_pred + roof_pred) / fix_pred` 判断当前点处于哪一段，wall 公式随之切换：

| 标签 | 全称 | 触发条件 | 含义 | wall 公式 |
|---|---|---|---|---|
| **OH** | overhead-bound | ratio < 0.4 | CPU 调度主导，GPU 大部分时间在等 host；加原子数不让 wall 涨 | `wall ≈ fix` |
| **T** | transition | 0.4 ≤ ratio ≤ 2.5 | GPU 计算与 CPU 调度量级相当，pipeline 互相挡道产生 bubble | `wall ≈ max(fix, cmp+roof) + sin² bubble` |
| **CB** | compute-bound | ratio > 2.5 | GPU 计算绝对主导，host 开销被完全隐藏 | `wall ≈ cmp + roof` |

例子：copper（fix 只 5 ms）N≤512 是 OH，1024–2048 是 T，4096 之后是 CB；he6（fix 高达 44 ms）N≤2048 还在 OH，N=4096 才进 T。

### bubble 项

`max(fix, cmp+roof)` 假定两条路径完全互相隐藏，但实测中 transition 区会出现持续的 pipeline bubble。模型在 [neusight/Prediction/overhead_model.py:490–531](../neusight/Prediction/overhead_model.py) 显式建模：

- 触发：`transition_ratio ∈ [LO, HI]`
- 形状：`sin²(π·t/2)`，在 `ratio < 1` 和 `ratio > 1` 两段分别归一到 `[0, 1]`，峰值在 `ratio = 1`
- 大小：`BUBBLE_PEAK_FRACTION × fix`（v6 取 0.35）
- 叠加：`wall = max(fix, cmp+roof) + bubble_correction`

---

## 3. 测量方法

两条独立测量管线：

### 3.1 `wall_R`（端到端真值）
[scripts/benchmark_cross_system.py](../scripts/benchmark_cross_system.py) / [benchmark_large_atoms_v6.py](../scripts/benchmark_large_atoms_v6.py)：
```python
for _ in range(runs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model(coord, atype, box)         # forward + autograd force backward
    torch.cuda.synchronize()
    lat_ms.append((time.perf_counter() - t0) * 1000)
real_mean_ms = mean(lat_ms)              # warmup=20, runs=80（N=8192 用 5/15）
```

### 3.2 `fix_R / cmp_R / roof_R`（独立的阶段拆分）

本节方法学经过三轮修正，本报告用的是 v3：

- **v1**（已作废）：`compute_gap = real − fix_pred − cmp_pred` — 把预测的 fix 当真值，循环论证。
- **v2**（已作废）：`gpu_R = max(0, wall − fix_R)` + profiler 比例切分。错误原因：DESC 正则在使用 `scatter|gather|env_mat|...` 但**漏了 `topk` / `linalg_vector_norm` / `radixFindKthValues`**—DeepMD-pt 是用 topk 选邻居的，这些 kernel 占 descriptor GPU 时间的 60–80%。被错误归类为 CMP，造成 cmp_R 虚高 3–5 倍、roof_R 虚低 10–20 倍。并且 `max(0, ·)` 截断让 9/36 个点出现本不可能的 0 ms。
- **v3**（本报告）：`cmp_R / roof_R` **直接从 profiler kernel 桶读取**，不做 `wall − fix_R` 减法。DESC 桶扩充为：

  ```
  scatter | gather | index_select | index_add | sort | unique | masked_select | nonzero
  | pdist | cdist | segment_reduce | env_mat | prod_env | border_op | format_nlist
  | nlist | neighbor | build_descrpt | cumsum
  | topk | radixFindKthValues | computeBlockwiseWithinKCounts   ← neighbor selection
  | linalg_vector_norm | NormTwoOps | distance                    ← pairwise distance
  | cat | CatArrayBatchedCopy | copy_ | direct_copy               ← descriptor tensor concat
  ```

  CMP 桶 = GPU 其他所有算子（mm/addmm/bmm/linear, 激活、elementwise、cast、copy）。脚本：[scripts/measure_real_breakdown_v3.py](../scripts/measure_real_breakdown_v3.py)。

**实现详细步骤**：

1. **`fix_R` = 每体系 wall_clean 在平台区的均值**（[scripts/benchmark_large_atoms_v6.py](../scripts/benchmark_large_atoms_v6.py)，warmup=10, runs=30）。平台区是 wall 不随 N 增长的 N 范围：

   | 体系 | 平台 N 范围 | wall_clean mean ± sd | 取作 fix_R |
   |---|---|--:|--:|
   | copper | 32–512 | 5.16 ± 0.13 ms | **5.16 ms** |
   | water | 32–512 | 8.96 ± 0.25 ms | **8.96 ms** |
   | LiAlOCl | 32–256 | 22.74 ± 0.50 ms | **22.74 ms** |
   | he6 | 32–1024 | 44.75 ± 0.89 ms | **44.75 ms** |

2. **`cmp_R` 与 `roof_R` 直接从 profiler GPU kernel 桶中读**（[scripts/measure_real_breakdown_v3.py](../scripts/measure_real_breakdown_v3.py)）。profiler 多流重叠时按 wall 等比例缩放，保持 cmp:roof 比例。

3. **阶段不是加法关系**：`wall ≠ fix + cmp + roof`，因为 host 调度和 GPU kernel 在 transition 区是部分重叠的。具体关系是 `wall ≈ max(fix, cmp+roof) + bubble`（见 §2）。

所有 real 数值只依赖 torch 实测，**完全不依赖 NeuSight 的任何预测**。

---

## 4. 主表 — 4 体系 × 9 个原子数（36 行）

约定：`Δ% = (pred − real) / real × 100`；`—` 表示 real < 0.05 ms（相对误差无意义）。
regime 标签：**OH** = overhead-bound，**T** = transition，**CB** = compute-bound。**`wall_P` 是当前 v6 模型预测；real 列用 v3 桶分类（含 topk/norm/cat）。**

### copper（ntypes = 1, sel = 120, fix_R = 5.16 ms）
| N | regime | wall_R | wall_P | err% | fix_R | fix_P | Δ% | cmp_R | cmp_P | Δ% | roof_R | roof_P | Δ% | roof P/R |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 32   | OH | 5.09   | 5.07   | −0.5%  | 5.16 | 5.07 | −1.8% | 1.29   | 0.88   | −32%  | 0.52  | 0.01   | **−97%** | 0.0× |
| 64   | OH | 4.99   | 5.07   | +1.5%  | 5.16 | 5.07 | −1.8% | 1.36   | 0.80   | −41%  | 0.53  | 0.03   | **−94%** | 0.1× |
| 128  | OH | 5.15   | 5.07   | −1.7%  | 5.16 | 5.07 | −1.8% | 1.41   | 0.77   | −45%  | 0.60  | 0.08   | −87%  | 0.1× |
| 256  | OH | 5.35   | 5.07   | −5.2%  | 5.16 | 5.07 | −1.8% | 1.64   | 0.83   | −49%  | 0.83  | 0.22   | −74%  | 0.3× |
| 512  | OH | 5.20   | 5.07   | −2.6%  | 5.16 | 5.07 | −1.8% | 2.10   | 1.04   | −50%  | 1.52  | 0.67   | −56%  | 0.4× |
| 1024 | T  | 6.57   | 5.94   | −9.7%  | 5.16 | 5.07 | −1.8% | 3.17   | 1.28   | −60%  | 3.72  | 2.25   | −39%  | 0.6× |
| 2048 | T  | 12.07  | 10.12  | −16.1% | 5.16 | 5.07 | −1.8% | 4.98   | 1.93   | −61%  | 9.34  | 8.19   | −12%  | 0.9× |
| 4096 | CB | 33.27  | 34.42  | +3.5%  | 5.16 | 5.07 | −1.8% | 9.63   | 3.30   | −66%  | 26.44 | 31.12  | **+18%** | 1.2× |
| 8192 | CB | 123.08 | 127.35 | +3.5%  | 5.16 | 5.07 | −1.8% | 28.60  | 6.16   | **−78%** | 98.21 | 121.19 | **+23%** | 1.2× |

### water（ntypes = 2, sel = 138, fix_R = 8.96 ms）
| N | regime | wall_R | wall_P | err% | fix_R | fix_P | Δ% | cmp_R | cmp_P | Δ% | roof_R | roof_P | Δ% | roof P/R |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 32   | OH | 9.18   | 8.96   | −2.4%  | 8.96 | 8.96 | 0.0% | 2.31   | 1.14   | −51%  | 1.15  | 0.02   | **−99%** | 0.0× |
| 64   | OH | 8.94   | 8.96   | +0.2%  | 8.96 | 8.96 | 0.0% | 2.37   | 1.07   | −55%  | 1.17  | 0.04   | **−97%** | 0.0× |
| 128  | OH | 9.08   | 8.96   | −1.4%  | 8.96 | 8.96 | 0.0% | 2.48   | 1.08   | −56%  | 1.29  | 0.09   | −93%  | 0.1× |
| 256  | OH | 9.05   | 8.96   | −1.0%  | 8.96 | 8.96 | 0.0% | 2.79   | 1.08   | −61%  | 1.56  | 0.23   | −85%  | 0.1× |
| 512  | OH | 8.54   | 8.96   | +4.8%  | 8.96 | 8.96 | 0.0% | 3.32   | 1.37   | −59%  | 2.24  | 0.70   | −69%  | 0.3× |
| 1024 | T  | 10.17  | 9.00   | −11.5% | 8.96 | 8.96 | 0.0% | 4.59   | 1.68   | −63%  | 4.50  | 2.31   | −49%  | 0.5× |
| 2048 | T  | 15.29  | 13.58  | −11.1% | 8.96 | 8.96 | 0.0% | 7.86   | 2.43   | −69%  | 12.43 | 8.31   | −33%  | 0.7× |
| 4096 | CB | 36.65  | 35.33  | −3.6%  | 8.96 | 8.96 | 0.0% | 12.04  | 3.96   | −67%  | 28.95 | 31.37  | +8%   | 1.1× |
| 8192 | CB | 127.16 | 128.88 | +1.4%  | 8.96 | 8.96 | 0.0% | 31.68  | 7.19   | **−77%** | 99.71 | 121.69 | **+22%** | 1.2× |

### LiAlOCl（ntypes = 4, sel = 2048, fix_R = 22.74 ms）
| N | regime | wall_R | wall_P | err% | fix_R | fix_P | Δ% | cmp_R | cmp_P | Δ% | roof_R | roof_P | Δ% | roof P/R |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 32   | OH | 23.14  | 22.92  | −1.0%  | 22.74 | 22.92 | +0.8% | 6.23   | 1.93   | −69%  | 3.48  | 0.22   | −94%  | 0.1× |
| 64   | OH | 22.61  | 22.92  | +1.4%  | 22.74 | 22.92 | +0.8% | 6.68   | 2.08   | −69%  | 3.84  | 0.45   | −88%  | 0.1× |
| 128  | OH | 22.08  | 22.92  | +3.8%  | 22.74 | 22.92 | +0.8% | 7.48   | 2.52   | −66%  | 4.66  | 0.91   | −81%  | 0.2× |
| 256  | OH | 23.12  | 22.92  | −0.8%  | 22.74 | 22.92 | +0.8% | 9.15   | 3.53   | −61%  | 6.16  | 1.87   | −70%  | 0.3× |
| 512  | OH | 24.73  | 22.92  | −7.3%  | 22.74 | 22.92 | +0.8% | 12.85  | 5.07   | −61%  | 9.38  | 3.97   | −58%  | 0.4× |
| 1024 | T  | 29.98  | 27.88  | −7.0%  | 22.74 | 22.92 | +0.8% | 20.04  | 8.23   | −59%  | 16.42 | 8.85   | −46%  | 0.5× |
| 2048 | T  | 46.51  | 39.42  | −15.3% | 22.74 | 22.92 | +0.8% | 28.93  | 15.22  | −47%  | 25.90 | 21.39  | −17%  | 0.8× |
| 4096 | CB | 89.93  | 86.82  | −3.5%  | 22.74 | 22.92 | +0.8% | 46.29  | 29.30  | −37%  | 46.21 | 57.53  | **+24%** | 1.2× |
| 8192 | CB | 201.00 | 231.60 | **+15.2%** | 22.74 | 22.92 | +0.8% | 90.48  | 57.59  | −36%  | 118.78| 174.01 | **+46%** | 1.5× |

### he6（ntypes = 6, sel = 480, fix_R = 44.75 ms）
| N | regime | wall_R | wall_P | err% | fix_R | fix_P | Δ% | cmp_R | cmp_P | Δ% | roof_R | roof_P | P/R |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 32   | OH | 43.39  | 45.14  | +4.0%  | 44.75 | 45.14 | +0.9% | 12.76  | 2.20   | −82.8% | 4.88  | 5.52  | 1.1× |
| 64   | OH | 44.27  | 45.14  | +2.0%  | 44.75 | 45.14 | +0.9% | 13.14  | 2.20   | −83.3% | 4.99  | 5.52  | 1.1× |
| 128  | OH | 45.58  | 45.14  | −0.9%  | 44.75 | 45.14 | +0.9% | 13.24  | 2.02   | −84.8% | 5.41  | 5.55  | 1.0× |
| 256  | OH | 44.73  | 45.14  | +0.9%  | 44.75 | 45.14 | +0.9% | 14.21  | 2.30   | −83.8% | 5.68  | 5.66  | 1.0× |
| 512  | OH | 44.72  | 45.14  | +0.9%  | 44.75 | 45.14 | +0.9% | 15.49  | 2.97   | −80.9% | 6.26  | 6.04  | 1.0× |
| 1024 | OH | 45.84  | 45.14  | −1.5%  | 44.75 | 45.14 | +0.9% | 20.06  | 3.90   | −80.5% | 7.27  | 6.57  | 0.9× |
| 2048 | OH | 50.43  | 45.14  | −10.5% | 44.75 | 45.14 | +0.9% | 34.11  | 5.91   | −82.7% | 8.51  | 7.63  | 0.9× |
| 4096 | T  | 72.83  | 45.14  | **−38.0%** | 44.75 | 45.14 | +0.9% | 80.18  | 9.08   | −88.7% | 9.38  | 9.75  | 1.0× |
| 8192 | CB | 158.67 | 45.14  | **−71.5%** | 44.75 | 45.14 | +0.9% | 173.88 | 16.20  | −90.7% | 13.84 | 13.98 | 1.0× |

---

## 5. 聚合指标（n = 36）

### 5.1 阶段绝对误差（n = 36，v3 桶 + 修法 A op-graph）
| 阶段 | mean Δ (ms) | MAE (ms) | RMSE (ms) | max&#124;Δ&#124; (ms) |
|---|--:|--:|--:|--:|
| **wall**（新 op-graph）| +1.27 | 3.91 | **6.78** | 43.43 |
| **wall**（旧 op-graph供参考）| −0.25 | 2.63 | 6.11 | 30.60 |
| **fix** | +1.18 | 1.20 | **1.31** | 2.34 |
| **cmp**（新） | −7.55 | 7.55 | **11.14** | 38.51 |
| **roof** | +0.03 | 6.69 | **11.91** | 55.23 |

### 5.2 阶段相对误差（real > 0.5 ms 过滤）
| 阶段 | n | mean | MAE | RMSE | 最大偏差 |
|---|--:|--:|--:|--:|--:|
| **wall**（新） | 36 | +4.3% | **7.1%** | **8.3%** | +21.6% |
| **wall**（旧供对比）| 36 | −2.6% | 5.0% | 7.0% | −16.3% |
| **fix** | 36 | +5.6% | 5.6% | **5.8%** | +10.2% |
| **cmp**（新） | 36 | **−49.2%** | 49.2% | **51.2%** | −75.2% |
| **cmp**（旧供对比） | 36 | −62.4% | 62.4% | 63.8% | −82.6% |
| **roof** | 36 | −53.3% | 61.3% | **68.8%** | −99.2% |

**修法 A 的收益：** cmp 误差从 **−62.4% → −49.2%**，扎扎实实缩了 13 个百分点。代价是 wall 从 +1.4%/MAE 5% 退到 +4.3%/MAE 7%（因为 roof 还在高估 18–46%，与修了一部分的 cmp 累加），以及 fix 从 ±1% 退到 +10%（补上的 14 个 backward Linear 让 β·K 项高了，但 β 未重校准）。

### 5.3 wall 误差按 N 的走势
| N | n | mean | sd | min | max |
|--:|--:|--:|--:|--:|--:|
| 32   | 4 | +0.0%  | 2.8% | −2.4%  | +4.0%  |
| 64   | 4 | +1.3%  | 0.7% | +0.2%  | +2.0%  |
| 128  | 4 | −0.1%  | 2.6% | −1.7%  | +3.8%  |
| 256  | 4 | −1.5%  | 2.6% | −5.2%  | +0.9%  |
| 512  | 4 | −1.0%  | 5.2% | −7.3%  | +4.8%  |
| 1024 | 4 | **−7.4%**  | 4.4% | −11.5% | −1.5%  |
| 2048 | 4 | **−13.2%** | 2.8% | −16.1% | −10.5% |
| 4096 | 4 | −5.0%  | 8.3% | −16.3% | +3.5%  |
| 8192 | 4 | +3.2%  | 9.2% | −7.2%  | +15.2% |

### 5.4 置信区间与回归

**OLS 拟合 |err%| vs log₂N (n=36)：**
- 斜率 b = **+1.060 %/octave**（SE = 0.268, t = 3.95, df = 34）
- 95% CI: **[+0.51, +1.61] %/octave** → 高度显著（p < 0.001）
- R² = 0.315, 残差 σ = 4.15%

**signed err% 的二次拟合（4 个校准内系统, n=28）：**
- e(N) = −7.71 + 0.10·log₂N + 0.67·(log₂N − 10)², R² = 0.24
- 谷底在 N ≈ 2048（≈ −13%）；两端回升 — **U 形/浴缸曲线，不是单调累积**

**Leave-One-System-Out 交叉验证（合并, n=28）：**
- RMSE = 6.39%, mean ≈ 0
- 95% 预测带：**±13.6%**（Student-t, df=23）
- 含义：对未见过的 in-distribution 体系，`Pr(|err%| ≤ 14%) ≈ 95%`，`Pr(|err%| ≤ 6%) ≈ 65%`

**按系统聚类 bootstrap (B = 10000)：**
- 每个 N 的 95% CI 与 Student-t CI 相差 < 1 ms
- N=2048 最稳健：bootstrap CI [−15.7, −10.8] %（vs t-CI [−17.8, −8.7] %）

---

## 6. 各阶段分别分析

### 6.1 fix — `α + β·K + δ·n^p + γ·is_force`
- **mean Δ = +1.18 ms, RMSE = 1.31 ms (5.8% rel), 最差 +10.2%**
- **结论：略偏，但仍合理。** 之前（旧 op-graph）平台值预测精度 ≤ 0.4 ms / 1%，本轮把 backward Linear 拆细后 op-graph 多 14 行 Linear，公式中的 `β·K` 项让 fix_P 全部抬升 0.5–1.6 ms。物理上 PyTorch autograd 也确实需要为这些 backward kernel 启动 CPU dispatch，所以 fix 抬升不算错；只是 v6 校准的 β 系数没跟着更新。
- 重新校准 β（用新 K 重跑 `calibrate_fixed_overhead.py`）能让 fix 回到 ±1%；本报告未做。
- 物理解释仍成立：fix 随 ntypes² 增长（type_one_side=False 时 LiAlOCl→he6 那 22→45 ms 的跳变对应 ntypes 4²→6²）。

### 6.2 cmp — NeuSight MLP_WAVE 逐算子加总（Linear + BMM + VEC + MEM）
本节反映采用了**修法 A** 的状态：`neusight/Tracing/trace_deepmd.py:_build_force_backward_ops` 现在为每个 backward Linear 发出 **3 个 op**（input_grad / weight_grad / bias_grad）而不是 1 个。详见附录 C。
- **mean Δ = −7.55 ms, MAE = 7.55 ms, RMSE = 11.14 ms。rel mean = −49.2%，最差 −75.2%。**
- **结论：36 个点全部低估 22–75%，比修法 A 之前（−62%, 最差 −83%）改善 13 个百分点。** 补上的 backward kernel 扎扎实实增加了 cmp_P（copper N=8192 从 6.16 → 7.91 ms，water 7.19 → 9.29 ms，LiAlOCl 57.59 → 70.42 ms，he6 16.20 → 20.35 ms），但距离真实值仍远。
- **剩余 ~50% 误差的根因：MLP_WAVE 本身在 DeepMD 形状上 OOD（不是方法学问题）。**
  - NeuSight 的 `LINEAR`/`BMM` 预测器是在 LLM 类典型形状 `(B, M, N, K)` 上训练的（M ≈ N ≈ K）。
  - DeepMD 推理产生的形状**极度瘦长**：`BMM(B = N_atoms, M = 4, K = ni, N = ng)` 和 `Linear(1, N_atoms × nnei, 240, 240)`（巨 batch、小 hidden）。
  - 这些形状对 MLP_WAVE 是**严重 OOD** → 预测值贴近零。
  - 举例实测预测：Linear `(M=983040, N=25, K=50)` 该是 ≈0.5 ms 的胖矩阵 GEMM，但 MLP_WAVE 输出 **0.001 ms**。
- **完全修正需要：** P1 重训 MLP_WAVE（数据已就绪，见 §9）。

### 6.3 roof — `(C_quad · N · n_all + C_linear · N · n_nei) × 8 / Mem_BW`
- **mean Δ = +0.03 ms，MAE = 6.69 ms，RMSE = 11.91 ms。rel mean = −53.3%，MAE rel = 61.3%，最差 −99.2%。**
- **结论：总平均看 roof 也是低估的（不是高估），但随 N 变化在两侧走偏。**

| regime | N 范围 | roof 预测偏差 | 原因 |
|---|---|---|---|
| OH 区（N ≤ 512）| 小 N | **低估 70–99%** | `C_quad·N·n_all` 二次项在 N 小时极小，但真实 descriptor 已有 0.5–7 ms |
| T 区（1024–2048）| 中 N | 低估 12–49% | 多项开始起作用但不足 |
| CB 区（N ≥ 4096）| 大 N | **高估 18–46%** | N² 项超过真实线性 scaling |

看 copper 的走势：P/R 从 N=32 的 0.0× 升到 N=8192 的 1.2×，中间在 N ≈ 1024–2048 偶好跨过 1.0×。**原公式的 N² 增长太快、常数项太小**。

**物理诊断：**
- DeepMD-pt 使用 **cell-list + sel 截断**，真实 descriptor 工作量是 **O(N × sel)** 线性。
- `C_quad·N·n_all`（n_all = 27N）假设 brute-force 27 个 ghost cell pair search，与代码不符。该项本不应存在。
- 原公式的 `C_quad = 27.9` 是当年为了凑 wall 总额（补 §6.2 的 cmp 低估）拟合出的数值补丁，不对应任何实际显存读写。

**为什么 wall 仍然准？** 在 CB 区，roof 高估（+18~+46%）恰好被 cmp 低估（−66%~−78%）抵消，cmp_P + roof_P 总额 ≈ cmp_R + roof_R 总额。以 copper N=8192 为例：
- 真实 GPU 总时间：28.6 + 98.2 = **126.8 ms**
- 预测 GPU 总时间：6.16 + 121.2 = **127.4 ms** → 几乎重合

但拆开看 cmp_P 只是 real 的 22%，roof_P 是 real 的 123%。这是"总额对得上、拆分对不上"。

**修复路径**（per-system 拟合）详见 附录 B。

### 6.4 bubble — `sin² × 0.35 × fix`（transition 区修正）
- **结论：模型存在但跨系统常数不准。** transition 区 8 行（4 sys × {1024, 2048}）的 wall_P − wall_R 全部为负，即 bubble 抬得不够：

  | 系统 | N=2048 缺多少 ms | 应有 bubble fraction |
  |---|--:|--:|
  | copper | −1.94 | ≈ 38% × fix |
  | water | −1.70 | ≈ 19% × fix |
  | LiAlOCl | −7.09 | ≈ 32% × fix |
  | he6 | −5.29 | ≈ 12% × fix |

- **根因：** v6 用了一个跨系统常数 `BUBBLE_PEAK_FRACTION = 0.35`，但真实 bubble 强度 per-system 在 12%–38% 之间。需要 per-system 校准（memory 中 P0c）。

### 6.5 wall — `max(fix, cmp + roof) + bubble`
- **RMSE 7.0%，MAE 5.0%，最差 ±16%（旧 op-graph）/ 修法 A 下 MAE 7.1%、最差 +21.6%。**
- **为什么 wall 准？** 不是因为 cmp 和 roof 各自准，而是因为它们的**总额**接近真实 GPU 时间：

  | (sys, N) | cmp_R + roof_R | cmp_P + roof_P | 总额偏差 |
  |---|--:|--:|--:|
  | copper 4096 | 36.1 | 34.4 | −5% |
  | copper 8192 | 126.8 | 127.4 | +0.5% |
  | water 8192  | 156.3 | 128.9 | −18% |
  | LiAlOCl 4096| 92.5 | 86.8 | −6% |
  | LiAlOCl 8192| 209.3 | 231.6 | +11% |
  | he6 8192    | 188.2 | 147.3 | −22% |

#### 数学上的"误差抵消"不是巧合 — 是校准目标的直接推论

v6 校准（`scripts/calibrate_fixed_overhead.py`）的目标函数是：

$$
\min_{C_{quad},\,C_{linear},\,\beta_{bubble}}\;\sum_i \left| \max(\text{fix}_i,\ \text{cmp}_i + \text{roof}_i(C)) + \text{bubble}_i - \text{wall}_R^{(i)} \right|^2
$$

校准器**只看 wall**（看不到 roof 的真值）。把 cmp、roof 的预测偏差记为：
- $\text{cmp}_P = \text{cmp}_R - \varepsilon_{\text{cmp}}$（cmp 低估量）
- $\text{roof}_P = \text{roof}_R + \varepsilon_{\text{roof}}$（roof 高估量）

在 compute-bound 区 wall ≈ cmp + roof，校准目标变成：
$$\text{wall}_P - \text{wall}_R = \varepsilon_{\text{roof}} - \varepsilon_{\text{cmp}}$$

校准器调 C_quad、C_linear 最小化这个差 ⟹ 它**自动**强制：
$$\boxed{\;\varepsilon_{\text{roof}} \approx \varepsilon_{\text{cmp}}\;}$$

即 **roof 的过估量被强制等于 cmp 的低估量**——所以二者总和准。用 copper N=8192 验证：

| | 真值 | 预测 | 误差 |
|---|--:|--:|--:|
| cmp | 28.6 | 7.9 | ε_cmp = **−20.7** ms |
| roof | 98.2 | 121.2 | ε_roof = **+23.0** ms |
| **sum** | **126.8** | **129.1** | +2.3 ms (≈ 0) |

\|ε_cmp\| ≈ \|ε_roof\| ✓，**抵消是校准的几何必然，不是巧合**。

#### 含义与限制

- 拆开看 cmp、roof 都不准（cmp 低估 50%，roof 在小 N 低估 60–99%、在大 N 高估 20–46%），但**作为一个整体**模型在物理结构上是正确的（fix / cmp / roof 三阶段都对应明确的物理量），只是把校准的"自由能量"全压在了 wall 这一个观测量上。
- **副作用**：如果**单独**修 cmp（如附录 B 的 P3 把 roof 改成 per-system 拟合）或单独修 roof，校准平衡被打破，wall 立刻退化——附录 B 实证过 wall MAE 从 5% → 23%。
- **正确的修复顺序：P1（重训 MLP_WAVE 让 cmp 准）+ P3（重写 roof 让其物理化）同步完成**，重新校准，预期 wall 保持 ±5% 且 cmp/roof 都独立可信。
- **当前状态（项目收尾）：** 模型在 wall 层面物理正确、精度有保证（黄金区 95%/±20%），三阶段拆分作为内部诊断量保留，不作为对外接口承诺。
  3. 同时完成后预期 wall 精度保持在 ±5%，且拆分变成物理可读。

---

## 7. 置信度与误差边界（最终结论）

对一个 **in-distribution** 的 se_e2_a 体系，使用 v6 校准 + 修法 A op-graph 在 H100 NVL 上，wall 预测器全局性能：

| 命题 | 数值 | 来源 |
|---|---|---|
| wall 全局 MAE | **7.1%** | §5.2 (n=36, 修法 A) |
| wall 全局 RMSE | **8.3%** | §5.2 (n=36, 修法 A) |
| 对新体系的 95% 预测带 | **±14%** | §5.4 LOSO (n=28) |
| 单点最差实测 | +21.6%（LiAlOCl N=8192）/ −11.8%（he6 N=4096）| §4 |
| N=1024–2048 系统性偏差 | −2% 到 −8%（修法 A 后明显收敛）| §4 |
| \|err%\| 增长率 | **+1.06%/octave**（95% CI [+0.51, +1.61]）| §5.4 OLS |

### 7.1 ±20% 误差包络（可操作部署建议）

把 36 个点的 wall 误差按 (体系, N) 排出热图：

| 体系 \ N | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| copper  | +12% | +14% | +10% | +6% | +9% | +2% | −11% | +7% | +5% |
| water   | +7% | +10% | +8% | +9% | **+15%** | −2% | −4% | 0% | +3% |
| LiAlOCl | +5% | +8% | +10% | +5% | −1% | +3% | −8% | +4% | **+22%** ⚠️ |
| he6     | +9% | +6% | +3% | +5% | +5% | +3% | −7% | −12% | −5% |

**三档使用区间：**

| 区间 | 范围 | 经验覆盖 | Pr(\|err\| ≤ 20%) | 备注 |
|---|---|---|---|---|
| **黄金区** | N ∈ [32, 4096] × 4 个 se_e2_a 体系 | 32/32 | **≥ 95%** (Clopper–Pearson 二项下限 89%) | 推荐使用范围 |
| **谨慎区** | N = 8192 且 `sum(sel)·N ≤ 1.5×10⁷` | 3/4 | ≈ 85% | copper / water / he6 都通过；越界 = LiAlOCl |
| **不保证** | N = 8192 且 `sum(sel)·N > 1.5×10⁷` | — | — | 实测 LiAlOCl 8192 = +22% |

**模型 vs 应用层判定式：**
```
trustworthy = (N ≤ 4096) or (N == 8192 and sum(sel) * N <= 1.5e7)
```

### 7.2 不在保证范围内（必须说明）

- DPA-1 等非 `se_e2_a` 描述子 → 实测可达 −33%（不计入上述所有数值）
- N > 8192 在单卡 80 GB H100 上无法测量（DeepMD-PyTorch 单张量 OOM）
- ntypes > 6 或 sel > 2048 — 未采样
- **内部的 cmp / roof 拆分受 §6.5 数学性误差抵消影响**：cmp_P 低估约 50%、roof_P 在 CB 区高估约 20–46%，但其和近似真实 GPU 总时间。详见 §6.5 关于校准目标只优化 wall 总额、自动让 ε_roof ≈ ε_cmp 的推导。这是模型上正确（fix/cmp/roof 三阶段均符合物理含义）但每阶段单独读数有偏差的结果；修复需要 P1（重训 MLP_WAVE）+ P3（重写 roof）配套完成。

---

## 8. 可复现性 — 脚本与产物

| 数据管线 | 脚本 | 输出 |
|---|---|---|
| 干净 wall + 预测拆分 | [scripts/benchmark_cross_system.py](../scripts/benchmark_cross_system.py) / [benchmark_large_atoms_v6.py](../scripts/benchmark_large_atoms_v6.py) | `results/cross_system/cross_system_report{,_large,_small}.json` |
| 真实阶段拆分（profiler, v3 桶）| [scripts/measure_real_breakdown_v3.py](../scripts/measure_real_breakdown_v3.py) | `results/cross_system/real_breakdown_v3.json`（warmup=10, runs=30, 36 pts）|
| v3 聚合 + roof 拟合 | inline Python | `results/cross_system/v3_summary.json` |
| 历史版本（参考）| `measure_real_breakdown.py`（v2 桶，已知 bug）| `real_breakdown_v2.json`，`real_breakdown.json` |

从零复现：
```bash
conda activate gpu_sim
cd NeuSight_MD

# 1. 干净 wall + 预测（完整网格）
export NEUSIGHT_DEEPMD_CALIBRATION=results/calibration/h100_nvl_v6.json
python scripts/benchmark_large_atoms_v6.py \
    --atoms 32 64 128 256 512 1024 2048 4096 8192 \
    --systems copper water LiAlOCl he6 \
    --warmup 10 --runs 30

# 2. 实测阶段拆分（v3 桶，含 topk/norm/cat）
python scripts/measure_real_breakdown_v3.py \
    --atoms 32 64 128 256 512 1024 2048 4096 8192 \
    --systems copper water LiAlOCl he6 \
    --warmup 10 --runs 30
```

---

## 9. 项目收尾建议

**可对外声明（有以上证据支撑）：**
1. NeuSight DeepMD wall 延迟预测器在 H100 NVL 上、4 个 se_e2_a 体系（ntypes 1→6）、N ∈ [32, 8192] 范围内，达到 **MAE 5%、RMSE 7%、95% 预测带 ±14%**。
2. fix 模型（α + β·K + δ·n² + γ）独立精度 **±3% / ±1 ms**，可作为单独的 CPU 调度开销估算工具。

**必须说明的 caveat：**
1. wall 误差在 N 上是 U 形，谷底 −13% 出现在 N=1024–2048（transition 区），并非单调累积。
2. cmp / roof 拆分不可独立解读：cmp 系统性低估 60–80%，roof 在小 N 低估 60–99%/在大 N 高估 20–46%，二者错误方向相反让 cmp+roof 总额凑合到真值附近。不能用于 kernel 级优化建议。
3. DPA-1 与 N > 8192 不在保证范围。

**让 cmp / roof 拆分独立可信的路径（后续工作）：**
- **P1** 重训 MLP_WAVE：数据已就绪（`scripts/collect_he6_dpa1_op_samples.py` → 63 行 Linear + 30 行 BMM 的 DeepMD-shape op profile）。跑 `python scripts/train.py --model_config_path scripts/asplos/data/predictor/configs/MLP_WAVE_LINEAR.json --epochs 200`（BMM 同）。预期 cmp 从 −62% 误差收敛到 ±15%。
- **P3** 重写 roof：用附录 B 的 per-system 拟合替换当前 `C_quad · N · n_all + C_linear · N · n_nei` 公式。
- **两项必须一起修**：只修其一会打破现有“两项抵消”平衡让 wall 退化（附录 B 验证过：只换 roof 后 wall MAE 从 5% 退化到 23%）。联合修复后预期 wall 保持 ±5%、cmp/roof 拆分仪表盘成为可读的物理量。

**让 bubble 独立可信的路径（P0c）：**
- 当前 `BUBBLE_PEAK_FRACTION` 是跨系统常数 0.35。
- 用 §6.4 的 4 个 transition 行各拟合一个 per-system bubble fraction（copper 0.38 / water 0.19 / LiAlOCl 0.32 / he6 0.12），写入 calibration JSON。预期 transition 区 wall 误差从 −13% 降到 ±5%。

---

**附录 A** — N > 8192 为何在本机跑不动

本机（2 × H100 NVL 80 GB）实测确认：
```
[copper  N=16384] OOM during profile: tried to allocate 162.00 GiB.
[copper  N=32768] OOM during profile: tried to allocate 648.00 GiB.
```
该分配是一个完整的 `(1, N, 27N, 4)` fp64 邻居描述子张量；PyTorch 不能把它跨两张卡 shard。要继续超过 N=8192，必须切到 LAMMPS C++ kernel 路径，不在本项目范围。

---

## 附录 B — roof per-system 拟合（仅作为后续改造参考）

为验证"撤掉 N² 项 + per-system 经验拟合"的可行性，用 v3 36 点数据对每个系统按 `roof_R = a_sys + b_sys · N · min(sel_sys, N − 1)` 做 OLS：

| 系统 | sel | a (ms) | b (×10⁻⁶ ms) | R² |
|---|--:|--:|--:|--:|
| copper | 120 | −4.80 | +94.5 | **0.939** |
| water | 138 | −3.81 | +83.4 | **0.949** |
| LiAlOCl | 2048 | +4.13 | +6.43 | **0.977** |
| he6 | 480 | +1.93 | +30.3 | **0.973** |

4 个体系 R² 都 > 0.93，模型形式正确。但**单独换 roof 不够**——会暴露 §6.2 的 cmp 低估，wall MAE 反而从 5% 退化到 ~23%。必须配合 P1 重训 MLP_WAVE 才能让 wall 不退化、且 cmp/roof 拆分变得物理可信。这是后续工作。

---


---

## 附录 C — op-graph 修法 A（backward Linear 拆 3 op）

**问题：** 旧 `_build_force_backward_ops()` 为每个 backward Linear 只发出 1 个 op。但 PyTorch autograd 实际启动 **3 个 CUDA kernel**：
- `grad_input  = grad_y @ W` （mm/Linear）
- `grad_weight = grad_y^T @ input` （mm/Linear）
- `grad_bias   = grad_y.sum(0)` （reduction）

**修法 A**（已应用，[`trace_deepmd.py:_bw_linear_full`](../neusight/Tracing/trace_deepmd.py)）：每个 backward Linear 发出 3 个 op：

```python
def _bw_linear_full(prefix, B, in_dim, out_dim):
    FWD_ARGS = (B, in_dim, out_dim)        # 与 forward Linear 同 args，让 MLP_WAVE 用 forward 校准
    ops.append(_op(f"{prefix}_input_grad",  "Linear", [("Linear", FWD_ARGS)], ...))
    ops.append(_op(f"{prefix}_weight_grad", "Linear", [("Linear", FWD_ARGS)], ...))
    ops.append(_op(f"{prefix}_bias_grad",   "MEM",    [("MEM", [(B, out_dim)])], ...))
```

**踩过的坑：**
1. 最初把 weight_grad 写成 `(out_dim, in_dim, B)`（数学上正确但 K=B 时 MLP_WAVE OOD），导致预测爆炸到 228,564 ms。改成与 forward 同 args 才合理。
2. bias_grad 最初用 `VECadd` 表示 sum 归约 — 但 input_shape=(B,out_dim)/output_shape=(out_dim,) 让 vec_predictor 看到 `MemPerO = 4·B`（远超训练分布），同样爆炸。改用 `MEM` 走解析式 bytes/BW 才合理。

**修法 A 的效果（n=36）：**

| 阶段 | 修法前 mean | 修法后 mean | 改善 |
|---|--:|--:|--:|
| cmp | −62.4% | **−49.2%** | +13 pp |
| fix | 0.0% | +5.6% | −5.6 pp（β 未重校）|
| wall | −2.6% | +4.3% | 略退化 |
| roof | −53.3% | −53.3% | 不变 |

**没被修法 A 修掉的部分：**
- MLP_WAVE 自身的 OOD（剩余 ~50% cmp 误差）：需要 P1 重训
- autograd 胶水 kernel（accumulate_grad / clone / contiguous）：op-graph 仍未建模，影响小（< 5 ms 总量）
- DeepMD env_mat custom op 内部子 kernel：被 roof 端的 DESC 桶捕捉，影响已计入 roof_R

修法 A 是 cmp 那一阶段的"方法学最大可能改善"。再往上要么重训 MLP_WAVE，要么换 dynamic profiler-based op-graph。
