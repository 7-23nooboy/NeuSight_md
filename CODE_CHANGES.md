# NeuSight → DeepMD 改造代码清单

> 范围：从上游 NeuSight (commit `6945927`) 到当前 v6 + P0a/P0b/P2/P1 版本，本仓库新增/修改的全部代码与配置。
> 上游基线：[NeuSight (ASPLOS '25)](https://github.com/SamsungLabs/NeuSight)，原本只支持 Transformer + HuggingFace FX tracing。
> 改造原则：**保留 NeuSight 后端 (`MLP_WAVE_*` 算子级 predictor)，重写前端**（DeepMD-kit input.json → 算子图 → overhead model）。

---

## 一、核心库 — `neusight/` (4 新增 + 2 改动)

| 文件 | 状态 | 行数级别 | 作用 |
|:---|:---:|:---:|:---|
| [`neusight/Tracing/parse_deepmd_input.py`](neusight/Tracing/parse_deepmd_input.py) | **新增** | ~150 | 解析 DeepMD-kit 的 `input.json`：descriptor / fitting_net / type_map → Python dataclass，抽取 `ntypes`、`sel`、`neuron`、`type_one_side` 等 tracer 需要的字段 |
| [`neusight/Tracing/trace_deepmd.py`](neusight/Tracing/trace_deepmd.py) | **新增** | ~600 | DeepMD 专用算子图构造器，**替代** NeuSight 原本的 HuggingFace FX tracer。按 7 阶段（neighbor list / env_mat / embedding net / descriptor matmul / fitting net / output / autograd backward）逐步生成 NeuSight 兼容的 (`Linear`/`BMM`/`VECxxx`/`MEM`) op DataFrame，给 MLP_WAVE 算子级 predictor 喂 |
| [`neusight/Prediction/predictor_deepmd.py`](neusight/Prediction/predictor_deepmd.py) | **新增** | ~400 | DeepMD 专用 predictor 主入口，串起 `parse → trace → MLP_WAVE per-op predict → aggregate → overhead_model → 输出 JSON`。v6 重构时新增 `fixed_overhead_ms` / `gpu_overhead_roofline_ms` 字段输出 |
| [`neusight/Prediction/overhead_model.py`](neusight/Prediction/overhead_model.py) | **新增** | ~500 | **本项目核心**。v6 overhead 物理模型：<br>`pred = max(fix, cmp+roof) + bubble`<br>已实装 P0a (roofline 系数) + P0b (transition bubble) + P2 (`type_one_side` 自适应) |
| [`neusight/Prediction/aggregator.py`](neusight/Prediction/aggregator.py) | 改动 | +30 | 上游只支持 Transformer 聚合（按层数乘）；新增 `aggregate_deepmd()`：所有节点 fw/bw/acc 简单求和 |
| [`neusight/__init__.py`](neusight/__init__.py) | 改动 | +2 | export `DeepMDPredictor`，让外部 `import neusight; neusight.DeepMDPredictor(...)` 可用 |

### `overhead_model.py` 关键数据流

```
fix  = α + β·K_modeled + δ·ntypes^p + γ·is_force        # CPU dispatch + kernel launch + autograd setup
cmp  = sum(MLP_WAVE.predict(op))  for op in trace        # NeuSight 原生算子级预测
roof = (C_quad·N·n_all + C_linear·N·n_nei) · 8 / Mem_BW  # 未被 MLP_WAVE 覆盖的 GPU 工作 (memory roofline)

e2e_baseline = max(fix, cmp + roof)                      # 两区间硬切换
ratio        = (cmp + roof) / fix
bubble_factor = sin²(π/2 · t)                            # 平滑过渡 (t 在 [LO,1] 和 [1,HI] 上分段)
pred         = e2e_baseline + BUBBLE_PEAK_FRACTION · fix · bubble_factor
```

p=1 当 `descriptor.type_one_side=True`，否则 p=2（自动检测 + 可由 calibration JSON 显式覆盖）。

---

## 二、可执行脚本 — `scripts/` (18 新增)

### 2.1 校准脚本 (4 个)

| 文件 | 作用 | 当前是否在用 |
|:---|:---|:---:|
| [`scripts/calibrate_fixed_overhead.py`](scripts/calibrate_fixed_overhead.py) | **v6 主校准器**。两段拟合：<br>(a) 从小 N 实测 → 反解 `α + β·K + δ·n² + γ·is_force`<br>(b) `--fit-roofline` 模式：从 `cross_system_report.json` 用 LSQ 反解 `C_quad / C_linear`，若负则 `scipy.optimize.nnls`<br>(c) 输出 `results/calibration/h100_nvl_v6.json` | ✅ 当前主校准 |
| [`scripts/calibrate_analytical.py`](scripts/calibrate_analytical.py) | v3 时期 analytical roofline (手算物理常数) | ⚠️ 已被 v6 替代 |
| [`scripts/calibrate_analytical_a100.py`](scripts/calibrate_analytical_a100.py) | A100 host 端跨 GPU 缩放因子校准 | ⚠️ 跨设备验证用 |
| [`scripts/calibrate_power_law.py`](scripts/calibrate_power_law.py) | v3 power law (`α·N^β`) 拟合 | ❌ v5 后弃用 |

### 2.2 Benchmark / 验证脚本 (6 个)

| 文件 | 作用 | 当前是否在用 |
|:---|:---|:---:|
| [`scripts/benchmark_cross_system.py`](scripts/benchmark_cross_system.py) | **当前主验证**：跑 5 体系 × 多 N，输出 `results/cross_system/cross_system_report.json` 含 `fix/cmp/roof/gap_cmp/gap_gpu`，是 §7-9 实验表的数据来源 | ✅ 主验证 |
| [`scripts/benchmark_deepmd_accuracy.py`](scripts/benchmark_deepmd_accuracy.py) | 早期单体系 (water/copper) 精度 benchmark | ⚠️ 单体系微调用 |
| [`scripts/benchmark_lialocl_accuracy.py`](scripts/benchmark_lialocl_accuracy.py) | LiAlOCl 4 元素体系专用 benchmark | ⚠️ 单体系扩展验证 |
| [`scripts/benchmark_kernel_launch.py`](scripts/benchmark_kernel_launch.py) | 实测 CUDA kernel launch overhead，提供 `α` 校准的物理常数下界 | ✅ 物理常数实证 |
| [`scripts/full_accuracy_test.py`](scripts/full_accuracy_test.py) | v4 全 N 谱精度扫描 | ⚠️ 历史 |
| [`scripts/test_copper_and_large.py`](scripts/test_copper_and_large.py) / [`scripts/test_large_atoms.py`](scripts/test_large_atoms.py) | 大原子数 (N=2048-8192) 单点验证 | ⚠️ 历史 |

### 2.3 工具/数据生成脚本 (4 个)

| 文件 | 作用 |
|:---|:---|
| [`scripts/pred_deepmd.py`](scripts/pred_deepmd.py) | DeepMD 预测 CLI 入口（NeuSight 原生 `pred.py` 的 DeepMD 版本） |
| [`scripts/convert_deepmd_config.py`](scripts/convert_deepmd_config.py) | 从 DeepMD-kit `input.json` 生成 NeuSight 风格 config |
| [`scripts/collect_he6_dpa1_op_samples.py`](scripts/collect_he6_dpa1_op_samples.py) | **P1 scaffolding**：提取 HE6/DPA-1 unique op shape、profile latency、输出 NeuSight trainset 格式 CSV，给 MLP_WAVE_MM retrain 用 |
| [`scripts/verify_confidence_aware.py`](scripts/verify_confidence_aware.py) | overhead_model 的不确定性 bound 自检 |

### 2.4 绘图脚本 (4 个)

| 文件 | 作用 |
|:---|:---|
| [`scripts/plot_slide7_phenomena.py`](scripts/plot_slide7_phenomena.py) / [`scripts/plot_slide7_phenomenon.py`](scripts/plot_slide7_phenomenon.py) | §7 stress-test 误差曲线图 |
| [`scripts/plot_slide8_env.py`](scripts/plot_slide8_env.py) / [`scripts/plot_slide8_results.py`](scripts/plot_slide8_results.py) | §8 fix/cmp/roof 三段拆解 stacked bar |

---

## 三、数据 / 配置 JSON (7 新增)

### 3.1 DeepMD descriptor configs (5 个)

| 文件 | 体系 | ntypes | descriptor | 用途 |
|:---|:---|:---:|:---|:---|
| [`scripts/asplos/data/deepmd_configs/water_se_e2_a.json`](scripts/asplos/data/deepmd_configs/water_se_e2_a.json) | H₂O | 2 | `se_e2_a` | 主基线 |
| [`scripts/asplos/data/deepmd_configs/copper_se_e2_a.json`](scripts/asplos/data/deepmd_configs/copper_se_e2_a.json) | Cu | 1 | `se_e2_a` | 单元素对照 |
| [`scripts/asplos/data/deepmd_configs/LiAlOCl_se_e2_a.json`](scripts/asplos/data/deepmd_configs/LiAlOCl_se_e2_a.json) | LiAlOCl | 4 | `se_e2_a` | 多元素验证 |
| [`scripts/asplos/data/deepmd_configs/he6_se_e2_a.json`](scripts/asplos/data/deepmd_configs/he6_se_e2_a.json) | HE6 | 6 | `se_e2_a` | ntypes² 外推 stress test |
| [`scripts/asplos/data/deepmd_configs/water_dpa1.json`](scripts/asplos/data/deepmd_configs/water_dpa1.json) | H₂O | 2 | `dpa1` (attention) | 描述子换型验证 |

### 3.2 Host configs (2 个)

| 文件 | 作用 |
|:---|:---|
| [`scripts/asplos/data/host_configs/H100_NVL_default.json`](scripts/asplos/data/host_configs/H100_NVL_default.json) | H100 NVL 主机端 CPU 配置 (cpu_freq, mem_bw, ref roofline 系数) |
| [`scripts/asplos/data/host_configs/T4_estimated.json`](scripts/asplos/data/host_configs/T4_estimated.json) | T4 跨设备验证用 |

### 3.3 Calibration 输出 (1 个)

| 文件 | 内容 |
|:---|:---|
| `results/calibration/h100_nvl_v6.json` | 当前生效校准：`alpha=2.0008, beta_per_kernel=0.0294, gamma_force=0.15, delta_ntypes_sq=1.0315, C_quad=27.8995, C_linear=1433.679, transition_lo=0.4, bubble_peak_fraction=0.35, delta_type_one_side_factor="auto"` |

---

## 四、未动的上游部分 (作为工程边界声明)

NeuSight 自带的下列模块**保留原样**，本项目通过新增独立支路复用其能力：

| 模块 | 原作用 | 在本项目里的角色 |
|:---|:---|:---|
| `neusight/Model/mlp_wave_mm.py` | MLP_WAVE_MM 算子 latency predictor | 直接调用预测每个 Linear / BMM 的 fw/bw 时间 |
| `neusight/Model/mlp_wave_vec.py` | MLP_WAVE_VEC / SOFTMAX | 预测 VECtanh / VECmul / VECsoftmax |
| `neusight/Prediction/predictor.py` | NeusightPredictor 主流程 (Transformer) | 不调用，DeepMD 走 `predictor_deepmd.py` 旁路 |
| `neusight/Tracing/trace.py` / `parse.py` | HuggingFace FX 前端 | 不调用，DeepMD 走 `parse_deepmd_input.py` + `trace_deepmd.py` |
| `scripts/train.py` | MLP_WAVE 模型训练入口 | P1 retrain 时复用，不改 |

---

## 五、改动汇总

| 类别 | 数量 |
|:---|:---:|
| 核心库新增 (`neusight/`) | 4 文件 |
| 核心库改动 (`neusight/`) | 2 文件 |
| 校准脚本 | 4 文件 |
| Benchmark 脚本 | 6 文件 |
| 工具/绘图 | 8 文件 |
| 配置/数据 JSON | 7 文件 |
| **合计** | **31 个文件** |

实际"我们的代码"（v6 物理模型 + P0a/P0b/P2/P1 实装）集中在 4 个文件：

1. [`neusight/Prediction/overhead_model.py`](neusight/Prediction/overhead_model.py) — 物理模型
2. [`neusight/Tracing/trace_deepmd.py`](neusight/Tracing/trace_deepmd.py) — 算子图构造
3. [`scripts/calibrate_fixed_overhead.py`](scripts/calibrate_fixed_overhead.py) — 数据驱动校准
4. [`scripts/benchmark_cross_system.py`](scripts/benchmark_cross_system.py) — 主验证 + fix/cmp/roof 分解输出

其余文件均为辅助：predictor 入口、绘图、单体系单点验证、历史阶段产物。

---

## 六、相关文档

- [`DeepMD_Experiment_Report.md`](DeepMD_Experiment_Report.md) — 唯一活跃的实验报告 (§1-9 全量)
- [`docs/archive/`](docs/archive/) — v1-v4 时期的 5 份历史报告，已被 `DeepMD_Experiment_Report.md` 完全覆盖

---

*生成于 2026-05-09*
