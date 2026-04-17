#!/usr/bin/env python3
"""
Power Law 校准脚本 — 用所有可用 ground truth 重新拟合 alpha/beta

用法:
  python3 scripts/calibrate_power_law.py

数据来源:
  - results/deepmd_fulltest/full_accuracy_report.json  (water 32-2048)
  - results/deepmd_fulltest/large_atoms_report.json    (copper 2048-8192, water 4096)

只使用 compute-bound 区间的数据点 (N≥2048)，因为小原子数处于
overhead-bound 区间，unmodeled compute 被 fixed_overhead 遮盖。

拟合模型: log(unmodeled) = log(alpha) + beta * log(N)
输出:     results/deepmd_benchmark/power_law_calibration.json
"""

import json
import math
import os
import sys
import numpy as np

# ============================================================
# 校准数据: (model, N, real_ms, mlp_compute_ms)
# unmodeled = real_ms - mlp_compute_ms
# ============================================================
# 注意: 只使用一致条件下的数据 (合理密度, 非 OOM)

CALIBRATION_DATA = [
    # Water se_e2_a, H100 NVL, box=40.0 (高密度, 与原校准一致)
    # N=2048: full_accuracy_report -> real=11.742, compute=2.1724
    ("water", 2048, 11.742, 2.1724),
    # N=4096: test_copper_and_large -> real=35.236, compute=3.8343 (box=40.0)
    # 注: large_atoms_report 中 water N=4096 用了 box=55.5, 不用那个
    ("water", 4096, 35.236, 3.8343),

    # Copper se_e2_a, H100 NVL
    # N=2048: large_atoms_report -> real=11.332, compute=2.1724
    ("copper", 2048, 11.332, 2.1724),
    # N=4096: large_atoms_report -> real=35.128, compute=3.8343
    ("copper", 4096, 35.128, 3.8343),
    # N=8192: large_atoms_report -> real=129.528, compute=7.2279
    ("copper", 8192, 129.528, 7.2279),
]


def fit_power_law(data):
    """
    拟合 unmodeled = alpha * N^beta

    用 log-log 线性回归:
      log(unmodeled) = log(alpha) + beta * log(N)

    Parameters
    ----------
    data : list of (model, N, real_ms, compute_ms)

    Returns
    -------
    alpha, beta, r_squared, residuals
    """
    points = []
    for model, N, real_ms, compute_ms in data:
        unmodeled = real_ms - compute_ms
        if unmodeled <= 0:
            print(f"  WARNING: skipping {model} N={N}: unmodeled={unmodeled:.3f} <= 0")
            continue
        points.append((N, unmodeled))

    if len(points) < 2:
        raise ValueError("Need at least 2 data points for fitting")

    log_N = np.array([math.log(n) for n, _ in points])
    log_U = np.array([math.log(u) for _, u in points])

    # Least squares: log_U = log_alpha + beta * log_N
    A = np.vstack([log_N, np.ones(len(log_N))]).T
    result = np.linalg.lstsq(A, log_U, rcond=None)
    beta, log_alpha = result[0]
    alpha = math.exp(log_alpha)

    # R-squared
    U_pred = np.array([alpha * (n ** beta) for n, _ in points])
    U_real = np.array([u for _, u in points])
    ss_res = np.sum((U_real - U_pred) ** 2)
    ss_tot = np.sum((U_real - np.mean(U_real)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Per-point residuals
    residuals = []
    for i, (n, u) in enumerate(points):
        pred = alpha * (n ** beta)
        err_pct = (pred - u) / u * 100
        residuals.append({
            "model": data[i][0],
            "N": int(n),
            "real_unmodeled_ms": round(u, 4),
            "pred_unmodeled_ms": round(pred, 4),
            "error_pct": round(err_pct, 1),
        })

    return alpha, beta, r_squared, residuals


def main():
    print("=" * 60)
    print("Power Law 校准: unmodeled = alpha * N^beta")
    print("=" * 60)

    # Old values
    OLD_ALPHA = 2.14e-5
    OLD_BETA = 1.708

    print(f"\n旧参数: alpha={OLD_ALPHA:.6e}, beta={OLD_BETA:.4f}")
    print(f"\n校准数据 ({len(CALIBRATION_DATA)} 个点):")
    print(f"  {'Model':<8} {'N':>6} {'real_ms':>10} {'compute_ms':>12} {'unmodeled_ms':>14}")
    for model, N, real, comp in CALIBRATION_DATA:
        unmod = real - comp
        print(f"  {model:<8} {N:>6} {real:>10.3f} {comp:>12.4f} {unmod:>14.4f}")

    # Fit
    alpha, beta, r2, residuals = fit_power_law(CALIBRATION_DATA)

    print(f"\n新参数: alpha={alpha:.6e}, beta={beta:.4f}")
    print(f"R² = {r2:.6f}")
    print(f"\n旧 vs 新对比:")
    print(f"  alpha: {OLD_ALPHA:.6e} → {alpha:.6e} (变化 {(alpha/OLD_ALPHA-1)*100:+.1f}%)")
    print(f"  beta:  {OLD_BETA:.4f} → {beta:.4f} (变化 {(beta/OLD_BETA-1)*100:+.1f}%)")

    print(f"\n逐点拟合残差:")
    print(f"  {'Model':<8} {'N':>6} {'real_unmod':>12} {'pred_unmod':>12} {'error':>8}")
    for r in residuals:
        print(f"  {r['model']:<8} {r['N']:>6} {r['real_unmodeled_ms']:>12.4f} "
              f"{r['pred_unmodeled_ms']:>12.4f} {r['error_pct']:>+7.1f}%")

    # Compare old vs new on all points
    print(f"\n新旧模型在各数据点的预测对比:")
    print(f"  {'Model':<8} {'N':>6} {'real_unmod':>12} {'old_pred':>10} {'old_err':>9} "
          f"{'new_pred':>10} {'new_err':>9}")
    for model, N, real_ms, compute_ms in CALIBRATION_DATA:
        unmod = real_ms - compute_ms
        old_pred = OLD_ALPHA * (N ** OLD_BETA)
        new_pred = alpha * (N ** beta)
        old_err = (old_pred - unmod) / unmod * 100
        new_err = (new_pred - unmod) / unmod * 100
        print(f"  {model:<8} {N:>6} {unmod:>12.3f} {old_pred:>10.3f} {old_err:>+8.1f}% "
              f"{new_pred:>10.3f} {new_err:>+8.1f}%")

    # Extrapolation preview
    print(f"\n外推预测 (新 vs 旧):")
    for N in [1024, 2048, 4096, 8192, 16384, 32768]:
        old = OLD_ALPHA * (N ** OLD_BETA)
        new = alpha * (N ** beta)
        diff = (new - old) / old * 100
        print(f"  N={N:>6}: old={old:>10.3f}ms  new={new:>10.3f}ms  diff={diff:>+6.1f}%")

    # Save calibration
    output = {
        "unmodeled_alpha": float(alpha),
        "unmodeled_beta": float(beta),
        "r_squared": float(r2),
        "old_alpha": OLD_ALPHA,
        "old_beta": OLD_BETA,
        "calibration_points": [
            {
                "model": m, "N": int(n), "real_ms": float(r),
                "compute_ms": float(c), "unmodeled_ms": float(r - c)
            }
            for m, n, r, c in CALIBRATION_DATA
        ],
        "residuals": residuals,
    }

    out_dir = "results/deepmd_benchmark"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "power_law_calibration.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ 校准结果已保存到: {out_path}")
    print(f"\n请更新 overhead_model.py:")
    print(f"  UNMODELED_COMPUTE_ALPHA = {alpha:.6e}")
    print(f"  UNMODELED_COMPUTE_BETA = {beta:.4f}")

    return alpha, beta


if __name__ == "__main__":
    main()
