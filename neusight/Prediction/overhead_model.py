"""
DeepMD 推理 Overhead 估算模型 (v6 — Kernel-count driven fixed overhead)

NeuSight 的 MLP_WAVE predictor 只预测了建模算子 (Linear/BMM/VEC/MEM) 的
纯 GPU 计算时间。但 DeepMD-kit 实际推理中有两类额外开销:

1. 固定 overhead (overhead-bound region):
   - CUDA kernel launch chain: Python 调度 + CUDA driver enqueue
   - 在小原子数时主导 latency
   - 几乎不随原子数变化, 但 **强烈依赖 kernel 总数**
   - v6 公式:
       fixed = α + β · K_total + γ · is_force
     其中 K_total = K_modeled (来自 tracer) + K_framework (常数)
     (α, β, γ) 通过 1-2 个真实体系一次性拟合, 可推广到任意 ntypes

   v6 之前 (v5 及更早) 用查表 (1_type / 2_type / per_extra_type),
   外推到 4-element (LiAlOCl) 时误差高达 -67%。

2. 未建模 GPU 计算 (compute-bound region):
   基于 DeepMD-kit 源码分析的解析 roofline 模型:

   (a) O(N²) 分量 — neighbor list broadcast distance:
       nlist.py: diff = coord.unsqueeze(1) - coord_local.unsqueeze(2)
       → [nframes, N, nall, 3] tensor, nall = ns × N (ns≈27 ghost cells)
       → 数据量 = N × nall × 8 bytes (float64)
       → latency ∝ C_quad × N × nall × 8 / GPU_MEM_BW

   (b) O(N) 分量 — env_mat + type sort + topk:
       env_mat.py: torch.gather random access [N, nnei, 4]
       nlist.py: topk over distances, type dispatch
       → 数据量 = N × nnei × 8 bytes
       → latency ∝ C_linear × N × nnei × 8 / GPU_MEM_BW

   公式:
     ns = (2 × ceil(rcut / box_face_dist) + 1)^3  (typically 27)
     nall = ns × N
     gpu_oh = (C_quad × N × nall × 8 / bw + C_linear × N × nnei × 8 / bw) × 1000
     e2e = max(fixed_overhead, mlp_compute + gpu_oh)

校准: 使用 scripts/calibrate_fixed_overhead.py 从实测数据拟合
(α, β, γ, C_quad, C_linear), 写入 calibration JSON。
"""

import json
import math
import os


class DeepMDOverheadModel:
    """
    估算 DeepMD-kit PyTorch 后端推理中的总 latency。

    v6: kernel-count driven fixed overhead + 解析 roofline 模型
    - overhead-bound (小原子数): latency ≈ α + β·K_total + γ·force
    - compute-bound (大原子数): latency ≈ mlp_compute + analytical_gpu_overhead
    """

    # ---- 参考硬件 (H100 NVL) — 所有默认常量在此硬件上校准 ----
    REF_GPU_FLOPS = 66908       # H100 NVL SingleFLOPs (GFLOPS)
    REF_GPU_MEM_BW = 3430       # H100 NVL Mem_Bw (GB/s)

    # ---- 解析 roofline 模型参数 (在 H100 NVL 上拟合) ----
    C_QUAD = 28.3284
    C_LINEAR = 2053.2321

    # ---- Ghost cell 复制因子 ----
    DEFAULT_NS = 27

    # ---- v6: kernel-count driven fixed overhead 默认常量 ----
    # fixed_overhead_ms = ALPHA + BETA_PER_KERNEL × K_modeled + (GAMMA_FORCE if force else 0)
    #
    # K_modeled 来自 tracer 的 KERNEL_MULTIPLIER 求和; 它能反映模型结构
    # (尤其是 type_one_side=False 时的 ntypes² 项), 是固定 overhead 的主要驱动
    # 因素。K_framework 仅用于 info, 实测中其差异已隐含在 α 项里。
    #
    # 缺省值由 H100 NVL + deepmd.pt eager-mode 上的 3 个真实测量做最小二乘拟合:
    #   copper  (1-type, K_mod≈40,  real≈4.85 ms)
    #   water   (2-type, K_mod≈56,  real≈5.70 ms)
    #   LiAlOCl (4-type, K_mod≈280, real≈23.0 ms)
    # → α ≈ 3.0 ms, β ≈ 0.065 ms (= 65 μs/kernel)
    # 拟合误差: copper +15%, water +16%, LiAlOCl -8%
    # 用户可通过 scripts/calibrate_fixed_overhead.py 重新拟合本机参数。
    ALPHA_FIXED_MS = 3.0
    BETA_PER_KERNEL_MS = 0.065    # 65 μs / modeled kernel
    GAMMA_FORCE_MS = 0.15         # autograd backward 启动开销
    DELTA_NTYPES_SQ_MS = 0.0      # 额外 ntypes² 项 (子网络个数二次扩展开销)
    DELTA_TYPE_ONE_SIDE_FACTOR = None  # P2: type_one_side=True 时 δ 乘上 n^1 而非 n^2
                                       # None = auto (检测 descriptor.type_one_side)
    BETA_DRIVER = "modeled"       # "modeled" (推荐) | "total"

    # ---- v5/v6 兼容: 旧式查表 (仅当 calibration 显式指定 mode="lookup" 时启用) ----
    FIXED_OVERHEAD_MS = {
        "se_e2_a": {
            "1_type": 4.850,
            "2_type": 5.715,
            "per_extra_type": 0.8,
        },
        "se_atten": {
            "1_type": 5.2,
            "2_type": 6.1,
            "per_extra_type": 0.8,
        },
    }

    # 默认使用 v6 的 kernel-count 模型; 设为 "lookup" 切回 v5 行为
    FIXED_OVERHEAD_MODE = "kernel_count"  # "kernel_count" | "lookup"

    # ---- 每种 NeuSight op 对应的 CUDA kernel 数 ----
    # v6 中此表既用于 info 输出, 也用于 fixed_overhead 计算 (K_modeled)
    KERNEL_MULTIPLIER = {
        "Linear": 2,    # matmul + bias_add
        "BMM": 2,       # matmul + (optional) reshape
        "VECtanh": 1,
        "VECrelu": 1,
        "VECgelu": 1,
        "VECmul": 1,
        "VECadd": 1,
        "VECsoftmax": 3,
        "MEM": 1,
    }

    # ---- 转换区 (transition zone) 置信度标注 ----
    # P0b: 以下三个的默认值可被 calibration JSON 覆盖 (transition_lo/hi/bubble_peak_fraction)
    # 默认 lo=0.4 (不是 0.8) 是为了让 cmp+roof 起源亍到 fix 的一半就进入 transition,
    # 这样 copper/water 在 N=1024 附近 (实测 ratio≈0.5–0.7) 就能被标记为不确定
    # 而不是仅给出 fix 静态点估导致 -25% 误差。
    # bubble_peak_fraction 0.35: 转换区点估上抬 35%·fix·bubble_factor, 实测拟合
    # 跨系统残差 (LiAlOCl 1024 真实差值 7 ms, 0.35·22.92≈8 ms 量级匹配)。
    TRANSITION_LO = 0.4
    TRANSITION_HI = 2.0
    BUBBLE_PEAK_FRACTION = 0.35  # 转换区 pipeline bubble 峰值占 fix 的比例

    # 框架 kernel 常量 (v6: 为 type-dispatch 加入 ntypes² 项)
    # n_framework = base + a × ntypes + b × ntypes^2 + (force_extra if force)
    FRAMEWORK_KERNELS = {
        "se_e2_a": {
            "base": 220,           # neighbor list, env mat, copy, etc.
            "per_type": 4,         # per-type masks, gather index build
            "per_type_sq": 6,      # ntypes² type-dispatch overhead
            "force_extra": 30,
        },
        "se_atten": {
            "base": 260,
            "per_type": 4,
            "per_type_sq": 6,
            "force_extra": 40,
        },
    }

    def __init__(self, calibration_path=None):
        """
        初始化 overhead 模型。

        Parameters
        ----------
        calibration_path : str, optional
            校准数据文件路径 (JSON)。如果提供且文件存在，
            使用校准数据覆盖默认常数。

            支持的字段 (全部可选):
              C_quad, C_linear, default_ns,                  # roofline
              alpha_fixed_ms, beta_per_kernel_ms,             # v6 fixed overhead
              gamma_force_ms,
              fixed_overhead_mode,                            # "kernel_count" | "lookup"
              fixed_overhead,                                 # v5 lookup table
              framework_kernels                               # per-model-type override
        """
        if calibration_path and os.path.isfile(calibration_path):
            with open(calibration_path) as f:
                calib = json.load(f)
            # 覆盖 roofline 参数
            if "C_quad" in calib:
                self.C_QUAD = calib["C_quad"]
            if "C_linear" in calib:
                self.C_LINEAR = calib["C_linear"]
            if "default_ns" in calib:
                self.DEFAULT_NS = calib["default_ns"]
            # v6: 覆盖 kernel-count fixed overhead 参数
            if "alpha_fixed_ms" in calib:
                self.ALPHA_FIXED_MS = calib["alpha_fixed_ms"]
            if "beta_per_kernel_ms" in calib:
                self.BETA_PER_KERNEL_MS = calib["beta_per_kernel_ms"]
            if "gamma_force_ms" in calib:
                self.GAMMA_FORCE_MS = calib["gamma_force_ms"]
            if "delta_ntypes_sq_ms" in calib:
                self.DELTA_NTYPES_SQ_MS = calib["delta_ntypes_sq_ms"]
            if "delta_type_one_side_factor" in calib:
                # P2: 可以取 'auto' / 1 / 2 或 None
                v = calib["delta_type_one_side_factor"]
                if isinstance(v, str) and v.lower() == "auto":
                    self.DELTA_TYPE_ONE_SIDE_FACTOR = None
                else:
                    self.DELTA_TYPE_ONE_SIDE_FACTOR = v
            if "beta_driver" in calib:
                self.BETA_DRIVER = calib["beta_driver"]
            if "fixed_overhead_mode" in calib:
                self.FIXED_OVERHEAD_MODE = calib["fixed_overhead_mode"]
            # P0b: transition / confidence 可调
            if "transition_lo" in calib:
                self.TRANSITION_LO = float(calib["transition_lo"])
            if "transition_hi" in calib:
                self.TRANSITION_HI = float(calib["transition_hi"])
            if "bubble_peak_fraction" in calib:
                self.BUBBLE_PEAK_FRACTION = float(calib["bubble_peak_fraction"])
            # v5 兼容: 覆盖固定 overhead lookup 表
            if "fixed_overhead" in calib:
                # 注意: 复制以避免修改类级常量
                self.FIXED_OVERHEAD_MS = {
                    k: dict(v) for k, v in self.FIXED_OVERHEAD_MS.items()
                }
                for model_type, cfg in calib["fixed_overhead"].items():
                    if model_type in self.FIXED_OVERHEAD_MS:
                        self.FIXED_OVERHEAD_MS[model_type].update(cfg)
            # v6: 覆盖 framework kernel 模板
            if "framework_kernels" in calib:
                self.FRAMEWORK_KERNELS = {
                    k: dict(v) for k, v in self.FRAMEWORK_KERNELS.items()
                }
                for model_type, cfg in calib["framework_kernels"].items():
                    if model_type in self.FRAMEWORK_KERNELS:
                        self.FRAMEWORK_KERNELS[model_type].update(cfg)

    def _extract_num_atoms(self, op_df):
        """
        从 op graph 推断原子数 N。

        通过查看 fitting net 的第一个 Linear op 的 batch 维度来推断。
        """
        for _, row in op_df.iterrows():
            if row["Name"] == "fit_linear_0" and row["OpName"] == "Linear":
                # FwOps format: [("Linear", (N, in_dim, out_dim))]
                fw_ops = row["FwOps"]
                if isinstance(fw_ops, list) and len(fw_ops) > 0:
                    shape = fw_ops[0][1] if isinstance(fw_ops[0], (list, tuple)) else fw_ops[0]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                        return shape[0]

        # Fallback: 从 output_linear 推断
        for _, row in op_df.iterrows():
            if row["Name"] == "output_linear" and row["OpName"] == "Linear":
                fw_ops = row["FwOps"]
                if isinstance(fw_ops, list) and len(fw_ops) > 0:
                    shape = fw_ops[0][1] if isinstance(fw_ops[0], (list, tuple)) else fw_ops[0]
                    if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                        return shape[0]

        # Last resort
        return 128  # default fallback

    def _count_modeled_kernels(self, op_df):
        """统计已建模算子的 kernel 数量 (仅用于 info 输出)"""
        count = 0
        for _, row in op_df.iterrows():
            opname = row["OpName"]
            count += self.KERNEL_MULTIPLIER.get(opname, 1)
        return count

    def _count_framework_kernels(self, deepmd_config, compute_force):
        """
        统计框架 kernel 数量 (用于固定 overhead 估算)。

        v6 公式: base + a × ntypes + b × ntypes² + (force_extra if force)
        ntypes² 项反映 type-dispatch 的二次扩展 (mask/scatter 数量等)。
        """
        model_type = deepmd_config.get(
            "model_type",
            deepmd_config.get("descriptor", {}).get("type", "se_e2_a"),
        )
        if model_type in ("dpa1", "DPA-1"):
            model_type = "se_atten"

        framework_cfg = self.FRAMEWORK_KERNELS.get(
            model_type, self.FRAMEWORK_KERNELS["se_e2_a"]
        )
        num_types = len(deepmd_config.get("type_map", ["X", "Y"]))
        n_framework = (
            framework_cfg.get("base", 220)
            + num_types * framework_cfg.get("per_type", 4)
            + (num_types * num_types) * framework_cfg.get("per_type_sq", 0)
        )
        if compute_force:
            n_framework += framework_cfg.get("force_extra", 30)
        return n_framework

    def _get_fixed_overhead(self, deepmd_config, model_type, compute_force,
                             k_total=None):
        """
        计算固定 overhead (kernel launch chain)。

        v6 默认 (FIXED_OVERHEAD_MODE="kernel_count"):
            fixed = α + β × K_total + δ × ntypes² + (γ if force else 0)
          其中 K_total = K_modeled + K_framework, 由调用方传入。
          δ 项 (默认 0) 可选, 在多元素校准后显著提高跨元素间的准确度。

        v5 兼容 (FIXED_OVERHEAD_MODE="lookup"):
            查表 + per_extra_type 线性外推 (旧行为)。

        Parameters
        ----------
        deepmd_config : dict
        model_type : str
        compute_force : bool
        k_total : int, optional
            总 kernel 数 (modeled + framework)。仅 kernel_count 模式需要。
        """
        if self.FIXED_OVERHEAD_MODE == "kernel_count":
            # 选择 driver: K_modeled (默认) 或 K_total
            k_used = k_total if k_total is not None else 400
            num_types = len(deepmd_config.get("type_map", ["X", "Y"]))

            # P2: 当 descriptor.type_one_side=True 时, embedding 子网络只有 ntypes 个
            # (而非 ntypes²), δ 项的指数应降到 1。
            descriptor = deepmd_config.get("descriptor", {})
            type_one_side = descriptor.get("type_one_side", False)
            if self.DELTA_TYPE_ONE_SIDE_FACTOR is not None:
                # 显式覆盖: factor 直接当作指数 (1=线性, 2=二次)
                exponent = float(self.DELTA_TYPE_ONE_SIDE_FACTOR)
                ntypes_term = num_types ** exponent
            else:
                # 自动: type_one_side=True → n^1, 否则 n^2
                ntypes_term = num_types if type_one_side else num_types ** 2

            fixed = (self.ALPHA_FIXED_MS
                     + self.BETA_PER_KERNEL_MS * k_used
                     + self.DELTA_NTYPES_SQ_MS * ntypes_term)
            if compute_force:
                fixed += self.GAMMA_FORCE_MS
            return fixed

        # ---- 旧的 lookup 路径 ----
        overhead_cfg = self.FIXED_OVERHEAD_MS.get(
            model_type, self.FIXED_OVERHEAD_MS["se_e2_a"]
        )
        num_types = len(deepmd_config.get("type_map", ["X", "Y"]))

        if num_types <= 1:
            fixed = overhead_cfg["1_type"]
        elif num_types == 2:
            fixed = overhead_cfg["2_type"]
        else:
            fixed = overhead_cfg["2_type"] + (num_types - 2) * overhead_cfg.get("per_extra_type", 0.8)

        if compute_force:
            fixed += 0.15
        return fixed

    def _compute_analytical_gpu_overhead(self, num_atoms, nnei, gpu_mem_bw, ns=None):
        """
        基于解析 roofline 模型计算未建模 GPU overhead。

        模型基于 DeepMD-kit 源码分析:
        - O(N²) 分量: nlist broadcast distance [N, nall, 3]
        - O(N) 分量: env_mat gather + type sort [N, nnei, 4]

        Parameters
        ----------
        num_atoms : int
            原子数 N
        nnei : int
            最大邻居数 (sum of sel)
        gpu_mem_bw : float
            GPU 内存带宽 (GB/s)
        ns : int, optional
            Ghost cell 复制因子。默认 27 (box >> rcut)。

        Returns
        -------
        float
            GPU overhead (ms)
        """
        if ns is None:
            ns = self.DEFAULT_NS

        N = num_atoms
        nall = ns * N
        BYTES = 8  # float64

        # O(N²) 分量: broadcast distance 计算
        # 数据量 = N × nall × 8 bytes
        # 实际有多遍访问 (broadcast, norm, topk) → C_QUAD >> 1
        quad_bytes = N * nall * BYTES
        quad_ms = self.C_QUAD * quad_bytes / (gpu_mem_bw * 1e9) * 1000  # bw: GB/s → B/s, *1000 → ms

        # O(N) 分量: env_mat + type sort
        # 数据量 = N × nnei × 8 bytes
        # random gather + per-type mask loop → C_LINEAR >> 1
        linear_bytes = N * nnei * BYTES
        linear_ms = self.C_LINEAR * linear_bytes / (gpu_mem_bw * 1e9) * 1000

        return quad_ms + linear_ms

    def estimate(self, device_config, deepmd_config, op_df, compute_force,
                 compute_latency_ms=0.0, host_config=None, box_size=None):
        """
        估算总 overhead，使预测随原子数和硬件配置正确变化。

        v5 解析 roofline 模型:
          # 固定 overhead (kernel launch chain)
          fixed = f(num_types, model_type)

          # 解析 GPU overhead (基于源码分析)
          ns = 27  (ghost cells, box >> rcut)
          nall = ns × N
          gpu_oh = C_QUAD × N × nall × 8 / bw + C_LINEAR × N × nnei × 8 / bw
          gpu_oh_ms = gpu_oh × 1000

          # GPU 缩放 (跨 GPU 迁移)
          gpu_oh_scaled = gpu_oh_ms × (REF_MEM_BW / target_MEM_BW)

          # 两区间模型
          e2e = max(fixed, mlp_compute + gpu_oh_scaled)
          overhead = e2e - mlp_compute

        Parameters
        ----------
        device_config : dict
            GPU 配置 (必须包含 Mem_Bw)
        deepmd_config : dict
            DeepMD 模型配置
        op_df : pd.DataFrame
            算子图
        compute_force : bool
            是否计算 force
        compute_latency_ms : float
            MLP_WAVE 预测的纯计算时间 (ms)
        host_config : dict, optional
            主机 CPU 配置 (v5 中不再用于 overhead 缩放,
            保留参数以向后兼容)
        box_size : float, optional
            模拟 box 边长 (Å)。v5 中用于计算更精确的 ns
            (ghost cell 因子)。当 box 较小时 ns > 27。

        Returns
        -------
        dict
            overhead 分解
        """
        # ---- 1. 提取模型参数 ----
        model_type = deepmd_config.get(
            "model_type",
            deepmd_config.get("descriptor", {}).get("type", "se_e2_a"),
        )
        if model_type in ("dpa1", "DPA-1"):
            model_type = "se_atten"

        num_types = len(deepmd_config.get("type_map", ["X", "Y"]))
        sel = deepmd_config.get("descriptor", {}).get("sel", [46, 92])
        if isinstance(sel, int):
            sel = [sel]
        nnei = sum(sel)
        rcut = deepmd_config.get("descriptor", {}).get("rcut", 6.0)

        # ---- 2. 统计 kernel count (modeled + framework) ----
        # v6: K_total 用于 fixed_overhead 估算, 是该模型的关键参数
        n_modeled = self._count_modeled_kernels(op_df)
        n_framework = self._count_framework_kernels(deepmd_config, compute_force)
        n_total = n_modeled + n_framework

        # ---- 3. 计算固定 overhead (kernel-count driven) ----
        # 默认用 K_modeled 作为 β 的驱动因子 (跨 ntypes 更稳定),
        # 兼容模式可切换为 K_total。
        if self.FIXED_OVERHEAD_MODE == "kernel_count" and self.BETA_DRIVER == "modeled":
            k_for_fixed = n_modeled
        else:
            k_for_fixed = n_total
        fixed_overhead_ms = self._get_fixed_overhead(
            deepmd_config, model_type, compute_force, k_total=k_for_fixed
        )

        # ---- 4. GPU 内存带宽 (用于 roofline 计算) ----
        gpu_mem_bw = device_config.get("Mem_Bw", self.REF_GPU_MEM_BW)
        if gpu_mem_bw <= 0:
            gpu_mem_bw = self.REF_GPU_MEM_BW
        gpu_bw_scale = self.REF_GPU_MEM_BW / gpu_mem_bw

        # ---- 5. 计算 ghost cell 因子 ns ----
        ns = self.DEFAULT_NS  # 默认 27
        if box_size is not None and box_size > 0:
            n_images_1d = 2 * math.ceil(rcut / box_size) + 1
            ns = n_images_1d ** 3

        # ---- 6. 计算解析 GPU overhead ----
        num_atoms = self._extract_num_atoms(op_df)

        gpu_overhead_ref_ms = self._compute_analytical_gpu_overhead(
            num_atoms, nnei, self.REF_GPU_MEM_BW, ns=ns
        )
        gpu_overhead_ms = gpu_overhead_ref_ms * gpu_bw_scale

        # ---- 7. 两区间模型 + transition bubble 修正 (P0b) ----
        # 朴素 max(fix, cmp+roof) 在转换区会低估实测值，因为 fix 段和 cmp 段
        # 在 GPU pipeline 上是按 kernel 交替执行而非整段串行/并行 —
        # 实测中产生持续的 pipeline bubble，使 real ≥ max(fix, cmp+roof)。
        #
        # 修正: 在转换区把点估计上抬一个 bubble 量, 这个 bubble 在 lo/hi 边界
        # 处衰减到 0, 在 ratio=1 附近最大, 既匹配观察到的物理, 又保持
        # overhead/compute-bound 区段的预测不变。
        adjusted_compute_ms = compute_latency_ms + gpu_overhead_ms
        e2e_baseline_ms = max(fixed_overhead_ms, adjusted_compute_ms)

        if fixed_overhead_ms > 0:
            transition_ratio = adjusted_compute_ms / fixed_overhead_ms
        else:
            transition_ratio = float('inf')

        # bubble 形状: 在 ratio ∈ [TRANSITION_LO, TRANSITION_HI] 之间是一个
        # 平滑钟形, 在边界为 0, 在 ratio=1 处达到 BUBBLE_PEAK_FRACTION × fix
        bubble_correction_ms = 0.0
        if self.TRANSITION_LO <= transition_ratio <= self.TRANSITION_HI:
            # 把 ratio 映射到 [0, π], 在中点 (1.0) 取 sin=1
            # 使用对数映射使非对称 (lo 通常 < 1 < hi)
            if transition_ratio < 1.0:
                # 左侧: 从 lo 升到 1
                if 1.0 - self.TRANSITION_LO > 1e-6:
                    t = (transition_ratio - self.TRANSITION_LO) / (1.0 - self.TRANSITION_LO)
                    t = max(0.0, min(1.0, t))
                else:
                    t = 1.0
                # 用 sin²(π·t/2) 提供光滑边界 (0→1)
                bubble_factor = math.sin(math.pi * t / 2.0) ** 2
            else:
                # 右侧: 从 1 降到 hi
                if self.TRANSITION_HI - 1.0 > 1e-6:
                    t = (self.TRANSITION_HI - transition_ratio) / (self.TRANSITION_HI - 1.0)
                    t = max(0.0, min(1.0, t))
                else:
                    t = 1.0
                bubble_factor = math.sin(math.pi * t / 2.0) ** 2
            bubble_correction_ms = self.BUBBLE_PEAK_FRACTION * fixed_overhead_ms * bubble_factor

        e2e_total_ms = e2e_baseline_ms + bubble_correction_ms
        total_overhead_ms = e2e_total_ms - compute_latency_ms
        total_overhead_ms = max(0.0, total_overhead_ms)

        # ---- 7b. 区间 + 置信区间标注 ----
        if transition_ratio < self.TRANSITION_LO:
            regime = "overhead-bound"
            confidence = "high"
            # overhead-bound: 实际值 ≈ fixed_overhead (非常稳定)
            e2e_lower_ms = e2e_total_ms
            e2e_upper_ms = e2e_total_ms
        elif transition_ratio > self.TRANSITION_HI:
            regime = "compute-bound"
            confidence = "high"
            # compute-bound: 实际值 ≈ mlp + gpu_overhead (power law 区间)
            e2e_lower_ms = e2e_total_ms
            e2e_upper_ms = e2e_total_ms
        else:
            regime = "transition"
            confidence = "low"
            # 转换区: 给一个 ±50% bubble_peak_ms 作为不确定性 bound
            bubble_peak_ms = self.BUBBLE_PEAK_FRACTION * fixed_overhead_ms
            sigma_left = 0.2
            sigma_right = 1.0
            distance = transition_ratio - 1.0
            sigma = sigma_left if distance <= 0 else sigma_right
            decay = math.exp(-0.5 * (distance / sigma) ** 2)
            uncertainty_ms = bubble_peak_ms * decay

            # 对称 bounds: 转换区的误差方向不确定
            e2e_lower_ms = e2e_total_ms - uncertainty_ms
            e2e_upper_ms = e2e_total_ms + uncertainty_ms

        # ---- 8. notes (kernel count 已在步骤 2 计算) ----
        if self.FIXED_OVERHEAD_MODE == "kernel_count":
            num_types = len(deepmd_config.get("type_map", ["X", "Y"]))
            descriptor = deepmd_config.get("descriptor", {})
            type_one_side = descriptor.get("type_one_side", False)
            if self.DELTA_TYPE_ONE_SIDE_FACTOR is not None:
                _exp = float(self.DELTA_TYPE_ONE_SIDE_FACTOR)
                _ntpw = num_types ** _exp
                _label = f"n^{_exp:g}={_ntpw:g}"
            elif type_one_side:
                _ntpw = num_types
                _label = f"n={num_types}"
            else:
                _ntpw = num_types ** 2
                _label = f"n²={num_types**2}"
            fixed_breakdown = (
                f"α={self.ALPHA_FIXED_MS:.2f}"
                f"+β={self.BETA_PER_KERNEL_MS*1e3:.1f}μs×K_{self.BETA_DRIVER}={k_for_fixed}"
                + (f"+δ={self.DELTA_NTYPES_SQ_MS:.2f}×{_label}"
                   if abs(self.DELTA_NTYPES_SQ_MS) > 1e-9 else "")
                + (f"+γ={self.GAMMA_FORCE_MS:.2f}" if compute_force else "")
            )
        else:
            fixed_breakdown = "lookup-table"

        notes = (
            f"[v6-kcount] regime={regime}, confidence={confidence}, "
            f"fixed={fixed_overhead_ms:.2f}ms ({fixed_breakdown}), "
            f"gpu_oh={gpu_overhead_ms:.3f}ms "
            f"(quad={self.C_QUAD:.1f}×N×nall×8/bw, linear={self.C_LINEAR:.1f}×N×nnei×8/bw, "
            f"N={num_atoms}, nall={ns}×N={ns*num_atoms}, nnei={nnei}, "
            f"bw_scale={gpu_bw_scale:.3f}), "
            f"mlp={compute_latency_ms:.2f}ms, "
            f"e2e={e2e_total_ms:.2f}ms"
            + (f", bounds=[{e2e_lower_ms:.2f}, {e2e_upper_ms:.2f}]"
               if confidence == "low" else "")
        )

        return {
            "total_overhead_ms": round(total_overhead_ms, 4),
            "fixed_overhead_ms": round(fixed_overhead_ms, 4),
            "unmodeled_compute_ms": round(gpu_overhead_ms, 4),
            "e2e_estimate_ms": round(e2e_total_ms, 4),
            "regime": regime,
            "confidence": confidence,
            "transition_ratio": round(transition_ratio, 4),
            "e2e_lower_ms": round(e2e_lower_ms, 4),
            "e2e_upper_ms": round(e2e_upper_ms, 4),
            "cpu_scale": 1.0,  # v5+: 固定 overhead 不再依赖 CPU
            "gpu_scale": round(gpu_bw_scale, 4),
            "density_correction": 1.0,
            "kernel_launch_ms": round(self.BETA_PER_KERNEL_MS * n_total, 4)
                                if self.FIXED_OVERHEAD_MODE == "kernel_count"
                                else round(n_total * 5.1 / 1000.0, 4),
            "cpu_dispatch_ms": round(fixed_overhead_ms, 4),  # backward compat alias
            "autograd_ms": round(self.GAMMA_FORCE_MS if compute_force else 0.0, 4),
            "wall_time_ms": round(e2e_total_ms, 4),  # backward compat alias
            "kernel_count": {
                "modeled": n_modeled,
                "framework": n_framework,
                "total": n_total,
            },
            "fixed_model": {
                "mode": self.FIXED_OVERHEAD_MODE,
                "alpha_ms": self.ALPHA_FIXED_MS
                            if self.FIXED_OVERHEAD_MODE == "kernel_count" else None,
                "beta_per_kernel_ms": self.BETA_PER_KERNEL_MS
                            if self.FIXED_OVERHEAD_MODE == "kernel_count" else None,
                "gamma_force_ms": self.GAMMA_FORCE_MS
                            if self.FIXED_OVERHEAD_MODE == "kernel_count" else None,
                "delta_ntypes_sq_ms": self.DELTA_NTYPES_SQ_MS
                            if self.FIXED_OVERHEAD_MODE == "kernel_count" else None,
                "beta_driver": self.BETA_DRIVER
                            if self.FIXED_OVERHEAD_MODE == "kernel_count" else None,
            },
            "analytical_detail": {
                "ns": ns,
                "nall": ns * num_atoms,
                "nnei": nnei,
                "rcut": rcut,
                "C_quad": self.C_QUAD,
                "C_linear": self.C_LINEAR,
                "gpu_mem_bw": gpu_mem_bw,
                "gpu_bw_scale": round(gpu_bw_scale, 4),
                "gpu_oh_quad_ms": round(
                    self.C_QUAD * num_atoms * ns * num_atoms * 8
                    / (self.REF_GPU_MEM_BW * 1e9) * 1000 * gpu_bw_scale, 4
                ),
                "gpu_oh_linear_ms": round(
                    self.C_LINEAR * num_atoms * nnei * 8
                    / (self.REF_GPU_MEM_BW * 1e9) * 1000 * gpu_bw_scale, 4
                ),
            },
            "notes": notes,
        }


def load_calibration(gpu_name, calibration_dir="results/deepmd_benchmark"):
    """
    加载 GPU 校准数据（如果有的话）。

    Parameters
    ----------
    gpu_name : str
        GPU 名称
    calibration_dir : str
        校准数据目录

    Returns
    -------
    str or None
        校准文件路径
    """
    path = os.path.join(calibration_dir, "kernel_launch_cost.json")
    if os.path.isfile(path):
        return path
    return None
