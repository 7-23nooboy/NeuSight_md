"""
DeepMD 推理 Overhead 估算模型 (v5 — 解析 Roofline 模型)

NeuSight 的 MLP_WAVE predictor 只预测了建模算子 (Linear/BMM/VEC/MEM) 的
纯 GPU 计算时间。但 DeepMD-kit 实际推理中有两类额外开销:

1. 固定 overhead (overhead-bound region):
   - CUDA kernel launch chain (~350 kernels × ~16μs/kernel)
   - 在小原子数时主导 latency
   - 几乎不随原子数变化
   - 通过 kernel_count × per_launch_us 建模 (GPU-dependent, not CPU)

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

校准数据 (H100 NVL — 参考硬件):
  GPU:  Mem_Bw=3430 GB/s
  Water (2 types, sel=[46,92], nnei=138, rcut=6.0):
    fixed=5.715ms, N=32-1024 全在 overhead-bound
    N=2048: 11.742ms, N=4096: 35.236ms
  Copper (1 type, sel=[120], nnei=120, rcut=7.0):
    fixed=4.850ms, N=64-1024 全在 overhead-bound
    N=2048: 11.332ms, N=4096: 35.128ms, N=8192: 129.528ms

  拟合: scipy.optimize.minimize (L-BFGS-B), MAE=2.24%, max_error=6.2%
  C_quad=28.3284, C_linear=2053.2321
"""

import json
import math
import os


class DeepMDOverheadModel:
    """
    估算 DeepMD-kit PyTorch 后端推理中的总 latency。

    v5: 解析 roofline 模型 — 基于源码分析的物理建模
    - overhead-bound (小原子数): latency ≈ fixed_overhead
    - compute-bound (大原子数): latency ≈ mlp_compute + analytical_gpu_overhead
    """

    # ---- 参考硬件 (H100 NVL) — 所有默认常量在此硬件上校准 ----
    REF_GPU_FLOPS = 66908       # H100 NVL SingleFLOPs (GFLOPS)
    REF_GPU_MEM_BW = 3430       # H100 NVL Mem_Bw (GB/s)

    # ---- 解析 roofline 模型参数 (在 H100 NVL 上拟合) ----
    # gpu_overhead = (C_QUAD * N * nall * 8 / bw + C_LINEAR * N * nnei * 8 / bw) * 1000
    # 其中 bw 单位 GB/s, 8 = sizeof(float64), *1000 转换为 ms
    #
    # C_QUAD: O(N²) 乘数, 对应 nlist broadcast distance 计算
    #   - diff = coord.unsqueeze(1) - coord_local.unsqueeze(2) → [N, nall, 3]
    #   - dist = diff.norm() → topk → [N, nall]
    #   - 多遍访问 + partial sort → 有效乘数 >> 1
    #
    # C_LINEAR: O(N) 乘数, 对应 env_mat + type sort
    #   - torch.gather random access pattern → low cache hit
    #   - per-type loop with mask → multiple passes
    #   - 有效乘数 >> 1 (random access penalty)
    C_QUAD = 28.3284
    C_LINEAR = 2053.2321

    # ---- Ghost cell 复制因子 ----
    # ns = (2*ceil(rcut/face_dist)+1)^3
    # 当 box >> rcut (典型情况), face_dist > rcut → ns = (2*1+1)^3 = 27
    DEFAULT_NS = 27

    # ---- 固定 overhead (kernel launch chain) ----
    # 从 ground truth 反推 (H100 NVL 参考硬件):
    #   Water (2 types): N=32-1024 平均 ≈ 5.715ms
    #   Copper (1 type): N=64-1024 平均 ≈ 4.850ms
    # 主要是 ~350 CUDA kernels 的 launch overhead
    # 不依赖 CPU 性能 (kernel launch 是 GPU driver/SM 调度)
    FIXED_OVERHEAD_MS = {
        "se_e2_a": {
            "1_type": 4.850,   # 1 atom type (e.g. copper)
            "2_type": 5.715,   # 2 atom types (e.g. water)
            "per_extra_type": 0.8,  # >2 types 时每增加一种类型的额外开销
        },
        "se_atten": {
            "1_type": 5.2,
            "2_type": 6.1,
            "per_extra_type": 0.8,
        },
    }

    # ---- 每种 NeuSight op 的 kernel 数 (仅用于 info 输出) ----
    KERNEL_MULTIPLIER = {
        "Linear": 2,
        "BMM": 2,
        "VECtanh": 1,
        "VECmul": 1,
        "VECadd": 1,
        "VECsoftmax": 3,
        "MEM": 1,
    }

    # ---- 转换区 (transition zone) 置信度标注 ----
    # 当 adjusted_compute / fixed_overhead 在 [TRANSITION_LO, TRANSITION_HI] 区间时,
    # 预测处于 "转换区"，pipeline bubble 导致 max() 模型不准确 (误差可达 ~15%)。
    # 此时标注 confidence="low"，提醒用户此区间预测不可靠。
    #
    # ratio 定义: (mlp_compute + gpu_overhead) / fixed_overhead
    #   ratio=1.0 永远是 crossover point (物理定义, 跨 GPU 通用)
    #   ratio < TRANSITION_LO: overhead-bound, high confidence
    #   ratio > TRANSITION_HI: compute-bound, high confidence
    #   其余: 转换区, low confidence
    #
    # bounds (e2e_lower_ms, e2e_upper_ms) 仅为参考性近似区间,
    # 基于 H100 NVL 实测校准, 在其他 GPU 上宽度可能偏大或偏小。
    # 不应视为严格的 confidence interval。
    TRANSITION_LO = 0.8
    TRANSITION_HI = 2.0
    BUBBLE_PEAK_FRACTION = 0.20  # 近似参考: bubble ≈ 20% of fixed (H100 实测)

    # 框架 kernel 常量 (仅用于 info 输出)
    FRAMEWORK_KERNELS = {
        "se_e2_a": {
            "base": 250,
            "per_type": 8,
            "force_extra": 30,
        },
        "se_atten": {
            "base": 290,
            "per_type": 8,
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
            # 覆盖固定 overhead
            if "fixed_overhead" in calib:
                for model_type, cfg in calib["fixed_overhead"].items():
                    if model_type in self.FIXED_OVERHEAD_MS:
                        self.FIXED_OVERHEAD_MS[model_type].update(cfg)

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
        """统计框架 kernel 数量 (仅用于 info 输出)"""
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
        n_framework = framework_cfg["base"] + num_types * framework_cfg["per_type"]
        if compute_force:
            n_framework += framework_cfg["force_extra"]
        return n_framework

    def _get_fixed_overhead(self, deepmd_config, model_type, compute_force):
        """
        计算固定 overhead (kernel launch chain)。

        基于实测数据: 不同 atom type 数量对应不同的固定开销,
        因为 type dispatch 会产生额外的 kernel launch。
        """
        overhead_cfg = self.FIXED_OVERHEAD_MS.get(
            model_type, self.FIXED_OVERHEAD_MS["se_e2_a"]
        )
        num_types = len(deepmd_config.get("type_map", ["X", "Y"]))

        if num_types <= 1:
            fixed = overhead_cfg["1_type"]
        elif num_types == 2:
            fixed = overhead_cfg["2_type"]
        else:
            # >2 types: 在 2_type 基础上线性外推
            fixed = overhead_cfg["2_type"] + (num_types - 2) * overhead_cfg.get("per_extra_type", 0.8)

        # force 计算额外 overhead (autograd + scatter_reduce)
        if compute_force:
            fixed += 0.15  # ~0.15ms additional for autograd backward pass

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

        # ---- 2. 计算固定 overhead ----
        fixed_overhead_ms = self._get_fixed_overhead(deepmd_config, model_type, compute_force)

        # ---- 3. GPU 内存带宽 (用于 roofline 计算) ----
        gpu_mem_bw = device_config.get("Mem_Bw", self.REF_GPU_MEM_BW)
        if gpu_mem_bw <= 0:
            gpu_mem_bw = self.REF_GPU_MEM_BW

        # GPU 缩放: 解析模型直接使用 bw 参数,
        # 但系数 C_QUAD/C_LINEAR 是在 REF_GPU_MEM_BW 上校准的。
        # 因此需要缩放: gpu_oh(target) = gpu_oh(ref) × (REF_BW / target_BW)
        # 等价于: 在公式中用 REF_BW 计算, 然后乘 REF_BW/target_BW
        # 或者直接用 target_BW 的倒数
        # 我们选择后者: 直接将 REF_BW 传入 _compute_analytical_gpu_overhead,
        # 然后再乘 gpu_scale
        gpu_bw_scale = self.REF_GPU_MEM_BW / gpu_mem_bw

        # ---- 4. 计算 ghost cell 因子 ns ----
        ns = self.DEFAULT_NS  # 默认 27
        if box_size is not None and box_size > 0:
            # 当 box_size 已知时, 精确计算 ns
            # face_dist ≈ box_size (for cubic box)
            # ns = (2*ceil(rcut/face_dist)+1)^3
            n_images_1d = 2 * math.ceil(rcut / box_size) + 1
            ns = n_images_1d ** 3

        # ---- 5. 计算解析 GPU overhead ----
        num_atoms = self._extract_num_atoms(op_df)

        # 使用参考 BW 计算 (系数在此 BW 上校准), 然后缩放
        gpu_overhead_ref_ms = self._compute_analytical_gpu_overhead(
            num_atoms, nnei, self.REF_GPU_MEM_BW, ns=ns
        )
        gpu_overhead_ms = gpu_overhead_ref_ms * gpu_bw_scale

        # ---- 6. 两区间模型 ----
        adjusted_compute_ms = compute_latency_ms + gpu_overhead_ms
        e2e_total_ms = max(fixed_overhead_ms, adjusted_compute_ms)
        total_overhead_ms = e2e_total_ms - compute_latency_ms
        total_overhead_ms = max(0.0, total_overhead_ms)

        # ---- 7. 判断区间 + Confidence-Aware 转换区检测 ----
        #
        # 转换区背景:
        #   在 overhead-bound → compute-bound 的过渡区间,
        #   大小 kernel 交替执行产生 pipeline bubble,
        #   导致 max(Σlaunch, Σcompute) ≠ Σmax(launch_i, compute_i)。
        #   bubble 幅度与模型结构有关 (Water ~25%, Copper ~16%),
        #   无法用统一参数精确校正。
        #
        # 策略: 在转换区给出 [lower, upper] 置信区间 + "low" confidence。
        #
        if fixed_overhead_ms > 0:
            transition_ratio = adjusted_compute_ms / fixed_overhead_ms
        else:
            transition_ratio = float('inf')

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
            # 转换区误差特征 (基于 Water & Copper 实测分析):
            #
            #   ratio < 1.0: point=fixed (偏低), real 可能更高 (bubble)
            #   ratio > 1.0: point=adjusted (可能偏高或偏低, 方向不确定)
            #   ratio ≈ 1.0: 最大不确定性
            #
            # 策略: 对称 bounds — point ± uncertainty
            # uncertainty = BUBBLE_PEAK_FRACTION × fixed × Gaussian_decay(ratio)
            #
            # 非对称 sigma: 左侧 (ratio<1) 陡峭, 右侧 (ratio>1) 缓慢
            # 因为 compute-bound 侧的 GPU 饱和是渐进过程
            bubble_peak_ms = self.BUBBLE_PEAK_FRACTION * fixed_overhead_ms

            sigma_left = 0.2   # overhead-bound 侧: bubble 快速出现/消失
            sigma_right = 1.0  # compute-bound 侧: 误差缓慢衰减到远端
            distance = transition_ratio - 1.0
            sigma = sigma_left if distance <= 0 else sigma_right
            decay = math.exp(-0.5 * (distance / sigma) ** 2)
            uncertainty_ms = bubble_peak_ms * decay

            # 对称 bounds: 转换区的误差方向不确定
            e2e_lower_ms = e2e_total_ms - uncertainty_ms
            e2e_upper_ms = e2e_total_ms + uncertainty_ms

        # ---- 8. Kernel count (仅供参考) ----
        n_modeled = self._count_modeled_kernels(op_df)
        n_framework = self._count_framework_kernels(deepmd_config, compute_force)
        n_total = n_modeled + n_framework

        notes = (
            f"[v5-roofline] regime={regime}, confidence={confidence}, "
            f"fixed={fixed_overhead_ms:.2f}ms, "
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
            "cpu_scale": 1.0,  # v5: 固定 overhead 不再依赖 CPU
            "gpu_scale": round(gpu_bw_scale, 4),
            "density_correction": 1.0,  # v5: 密度修正被 ns 参数替代
            "kernel_launch_ms": round(n_total * 5.1 / 1000.0, 4),  # reference only
            "cpu_dispatch_ms": round(fixed_overhead_ms, 4),  # backward compat alias
            "autograd_ms": round(0.15 if compute_force else 0.0, 4),
            "wall_time_ms": round(e2e_total_ms, 4),  # backward compat alias
            "kernel_count": {
                "modeled": n_modeled,
                "framework": n_framework,
                "total": n_total,
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
