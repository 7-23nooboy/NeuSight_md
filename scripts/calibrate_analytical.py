#!/usr/bin/env python3
"""
解析 Roofline 模型系数校准

基于 DeepMD-kit 源码分析的解析 overhead 模型:
  gpu_oh = C_QUAD × N × nall × 8 / bw + C_LINEAR × N × nnei × 8 / bw

使用实测数据校准 C_QUAD, C_LINEAR, fixed_water, fixed_copper 四个参数。

用法:
  python scripts/calibrate_analytical.py
"""

import sys
import os
import json
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# 实测校准数据 (H100 NVL, energy+force)
# ============================================================
# format: (model, N, nnei, real_ms, mlp_compute_ms)
# mlp_compute_ms 来自 MLP_WAVE predictor
#
# Water: sel=[46,92], nnei=138, rcut=6.0, 2 types
# Copper: sel=[120], nnei=120, rcut=7.0, 1 type
# ns=27 (ghost cell factor, box >> rcut)

CALIBRATION_DATA = [
    # === Water (overhead-bound region) ===
    ("water", 32,   138, 5.601,  0.240),
    ("water", 64,   138, 5.573,  0.358),
    ("water", 128,  138, 5.625,  0.597),
    ("water", 192,  138, 5.660,  0.832),
    ("water", 256,  138, 5.694,  1.065),
    ("water", 512,  138, 5.816,  2.008),
    ("water", 1024, 138, 5.909,  3.892),
    # === Water (compute-bound region) ===
    ("water", 2048, 138, 11.742, 2.172),   # box=40
    ("water", 4096, 138, 35.236, 3.834),   # box=40
    # === Copper (compute-bound region) ===
    ("copper", 2048, 120, 11.332, 2.172),
    ("copper", 4096, 120, 35.128, 3.834),
    ("copper", 8192, 120, 129.528, 7.228),
]

REF_BW = 3430      # H100 NVL Mem_Bw (GB/s)
NS = 27             # ghost cell factor
BYTES = 8           # float64


def analytical_gpu_overhead(N, nnei, C_quad, C_linear, ns=NS, bw=REF_BW):
    """计算解析 GPU overhead (ms)"""
    nall = ns * N
    quad_ms = C_quad * N * nall * BYTES / (bw * 1e9) * 1000
    linear_ms = C_linear * N * nnei * BYTES / (bw * 1e9) * 1000
    return quad_ms + linear_ms


def predict_latency(model, N, nnei, mlp_ms, C_quad, C_linear, fixed_water, fixed_copper):
    """完整预测"""
    gpu_oh = analytical_gpu_overhead(N, nnei, C_quad, C_linear)
    fixed = fixed_water if model == "water" else fixed_copper
    adjusted = mlp_ms + gpu_oh
    return max(fixed, adjusted)


def objective(params):
    """优化目标: minimize weighted MAE

    给 compute-bound 数据点更高权重 (它们是外推预测的基础),
    overhead-bound 数据点权重较低 (固定值本身就无法精确拟合所有小 N)
    """
    C_quad, C_linear, fixed_water, fixed_copper = params
    if C_quad <= 0 or C_linear <= 0 or fixed_water <= 0 or fixed_copper <= 0:
        return 1e6

    errors = []
    for model, N, nnei, real_ms, mlp_ms in CALIBRATION_DATA:
        pred = predict_latency(model, N, nnei, mlp_ms, C_quad, C_linear,
                               fixed_water, fixed_copper)
        rel_err = abs(pred - real_ms) / real_ms
        # compute-bound 点权重更高
        if N >= 2048:
            weight = 3.0
        elif N >= 512:
            weight = 2.0
        else:
            weight = 1.0
        errors.append(rel_err * weight)

    return np.mean(errors)


def main():
    print("=" * 70)
    print("DeepMD 解析 Roofline 模型校准")
    print("=" * 70)
    print(f"校准数据点: {len(CALIBRATION_DATA)}")
    print(f"参考 GPU: H100 NVL (Mem_Bw={REF_BW} GB/s)")
    print(f"Ghost cells: ns={NS}")
    print()

    # 初始猜测
    x0 = [28.3, 2053.0, 5.715, 4.850]
    bounds = [(1, 200), (100, 50000), (3, 10), (3, 10)]

    result = minimize(
        objective, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50000, 'ftol': 1e-15}
    )

    C_quad, C_linear, fixed_water, fixed_copper = result.x

    print(f"优化结果 (converged={result.success}):")
    print(f"  C_QUAD    = {C_quad:.4f}")
    print(f"  C_LINEAR  = {C_linear:.4f}")
    print(f"  fixed_water  = {fixed_water:.3f} ms")
    print(f"  fixed_copper = {fixed_copper:.3f} ms")
    print(f"  MAE = {result.fun * 100:.2f}%")
    print()

    # 详细对比
    print(f"{'Model':>7} | {'N':>6} | {'nnei':>5} | {'Real (ms)':>10} | {'Pred (ms)':>10} | {'Error':>7} | {'Regime':>14}")
    print("-" * 75)

    all_errors = []
    for model, N, nnei, real_ms, mlp_ms in CALIBRATION_DATA:
        pred = predict_latency(model, N, nnei, mlp_ms, C_quad, C_linear,
                               fixed_water, fixed_copper)
        err = (pred - real_ms) / real_ms * 100
        all_errors.append(err)

        gpu_oh = analytical_gpu_overhead(N, nnei, C_quad, C_linear)
        fixed = fixed_water if model == "water" else fixed_copper
        regime = "overhead" if fixed >= mlp_ms + gpu_oh else "compute"

        print(f"{model:>7} | {N:>6} | {nnei:>5} | {real_ms:>10.3f} | {pred:>10.3f} | {err:>+6.1f}% | {regime:>14}")

    print("-" * 75)
    print(f"MAE: {np.mean(np.abs(all_errors)):.2f}%")
    print(f"Max error: {max(all_errors, key=abs):+.1f}%")
    print(f"Median error: {np.median(all_errors):+.1f}%")

    # 保存校准结果
    os.makedirs("results/deepmd_benchmark", exist_ok=True)
    calib_result = {
        "model_version": "v5_analytical_roofline",
        "C_quad": round(C_quad, 4),
        "C_linear": round(C_linear, 4),
        "fixed_overhead": {
            "se_e2_a": {
                "1_type": round(fixed_copper, 3),
                "2_type": round(fixed_water, 3),
            }
        },
        "default_ns": NS,
        "ref_gpu_mem_bw": REF_BW,
        "calibration_stats": {
            "mae_pct": round(np.mean(np.abs(all_errors)), 2),
            "max_error_pct": round(max(all_errors, key=abs), 1),
            "n_points": len(CALIBRATION_DATA),
        },
        "calibration_data": [
            {
                "model": m, "N": n, "nnei": nn,
                "real_ms": r, "mlp_ms": mlp,
                "pred_ms": round(predict_latency(m, n, nn, mlp, C_quad, C_linear,
                                                  fixed_water, fixed_copper), 3),
            }
            for m, n, nn, r, mlp in CALIBRATION_DATA
        ],
    }

    calib_path = "results/deepmd_benchmark/analytical_calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calib_result, f, indent=2)
    print(f"\n校准结果已保存到 {calib_path}")


if __name__ == "__main__":
    main()
