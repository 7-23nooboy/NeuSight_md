#!/usr/bin/env python3
"""
A100 解析 Roofline 模型系数重校准

从 results/a100_experiment/ 下的实测 JSON 提取 (real_ms, mlp_compute_ms),
重新拟合 C_QUAD, C_LINEAR, fixed_water, fixed_copper 四个参数,
并与 H100 校准版本 + A100 未重校准版本对比。

用法:
  python scripts/calibrate_analytical_a100.py
"""

import os
import sys
import json
import glob
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
A100_DIR = os.path.join(ROOT, "results", "a100_experiment")

# A100 PCIe device params
REF_BW_A100 = 1935     # Mem_Bw GB/s
REF_BW_H100 = 3430     # for comparison
NS = 27
BYTES = 8

# H100 校准常数 (当前 v5 值)
H100_C_QUAD = 28.3284
H100_C_LINEAR = 2053.2321
H100_FIXED_WATER = 5.715
H100_FIXED_COPPER = 4.850


# ============================================================
# 数据加载: 从 A100 实验 JSON 抽出 (model, N, nnei, real, mlp)
# ============================================================

WATER_NNEI = 46 + 92      # sel=[46,92]
COPPER_NNEI = 120         # sel=[120]


def load_a100_data():
    """返回 list of (model, N, nnei, real_ms, mlp_ms, source_file)"""
    files = [
        # 平台段
        "water_small.json",
        "transition_water_pre_transition.json",
        "transition_copper_floor.json",
        # 转换段
        "transition_water_1024_2048_step128.json",
        "transition_copper_1024_2048_step128.json",
        # 后转换段
        "transition_water_post_transition.json",
        "transition_copper_post_transition.json",
        # 大原子数
        "water_large.json",
        "copper.json",
    ]
    data = []
    seen = set()     # (model, N) 去重，按首次出现保留
    for fn in files:
        path = os.path.join(A100_DIR, fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        mc = d.get("model_config", "")
        if "water" in mc:
            model, nnei = "water", WATER_NNEI
        elif "copper" in mc:
            model, nnei = "copper", COPPER_NNEI
        else:
            continue
        for r in d["results"]:
            N = r["num_atoms"]
            key = (model, N)
            if key in seen:
                continue
            seen.add(key)
            data.append((
                model, N, nnei,
                r["real_mean_ms"],
                r["pred_force_compute_ms"],
                fn,
            ))
    # 按 (model, N) 排序
    data.sort(key=lambda x: (x[0], x[1]))
    return data


# ============================================================
# 解析模型
# ============================================================

def analytical_gpu_overhead(N, nnei, C_quad, C_linear, ns=NS, bw=REF_BW_A100):
    nall = ns * N
    quad_ms = C_quad * N * nall * BYTES / (bw * 1e9) * 1000
    linear_ms = C_linear * N * nnei * BYTES / (bw * 1e9) * 1000
    return quad_ms + linear_ms


def predict(model, N, nnei, mlp_ms, C_quad, C_linear, fixed_w, fixed_c, bw=REF_BW_A100):
    gpu_oh = analytical_gpu_overhead(N, nnei, C_quad, C_linear, bw=bw)
    fixed = fixed_w if model == "water" else fixed_c
    return max(fixed, mlp_ms + gpu_oh)


# ============================================================
# 分区间权重
# ============================================================

def regime_weight(N):
    """给不同区间不同权重: compute-bound 最重, plateau 次之, transition 稍低"""
    if N >= 4096:
        return 3.0
    if N >= 2048:
        return 2.5
    if N <= 512:
        return 2.0      # plateau: A100 上平台本来就被低估最严重, 值得校准
    return 1.0          # transition


def objective(params, data):
    C_quad, C_linear, fixed_w, fixed_c = params
    if C_quad <= 0 or C_linear <= 0 or fixed_w <= 0 or fixed_c <= 0:
        return 1e9
    weighted = []
    for model, N, nnei, real, mlp, _ in data:
        pred = predict(model, N, nnei, mlp, C_quad, C_linear, fixed_w, fixed_c)
        rel = abs(pred - real) / real
        weighted.append(rel * regime_weight(N))
    return float(np.mean(weighted))


# ============================================================
# 报表
# ============================================================

def evaluate(data, C_quad, C_linear, fixed_w, fixed_c, label):
    rows = []
    for model, N, nnei, real, mlp, src in data:
        pred = predict(model, N, nnei, mlp, C_quad, C_linear, fixed_w, fixed_c)
        err = (pred - real) / real * 100
        rows.append((model, N, real, pred, err))
    abs_errs = [abs(r[4]) for r in rows]
    return {
        "label": label,
        "rows": rows,
        "mae": float(np.mean(abs_errs)),
        "max": float(max(abs_errs)),
    }


def print_table(report, data):
    print(f"\n--- {report['label']} ---")
    print(f"{'Model':>7} | {'N':>5} | {'Real':>8} | {'Pred':>8} | {'Err':>7}")
    print("-" * 50)
    for model, N, real, pred, err in report["rows"]:
        print(f"{model:>7} | {N:>5d} | {real:>8.3f} | {pred:>8.3f} | {err:>+6.2f}%")
    print("-" * 50)
    print(f"MAE: {report['mae']:.2f}%   Max: {report['max']:.2f}%")


def partition_stats(report, data):
    """按区间 + 按模型统计 MAE"""
    bins = {
        ("water",   "plateau (N<=512)"):   [],
        ("water",   "transition (1024-2048)"): [],
        ("water",   "post-trans (2304-3072)"): [],
        ("water",   "compute (N>=4096)"):   [],
        ("copper",  "plateau (N<=768)"):    [],
        ("copper",  "transition (1024-2048)"): [],
        ("copper",  "post-trans (2304-3072)"): [],
        ("copper",  "compute (N>=4096)"):    [],
    }
    for (model, N, real, pred, err), (m2, N2, *_) in zip(report["rows"], data):
        assert model == m2 and N == N2
        if model == "water":
            if N <= 512:
                k = ("water", "plateau (N<=512)")
            elif N <= 2048:
                k = ("water", "transition (1024-2048)")
            elif N < 4096:
                k = ("water", "post-trans (2304-3072)")
            else:
                k = ("water", "compute (N>=4096)")
        else:
            if N <= 768:
                k = ("copper", "plateau (N<=768)")
            elif N <= 2048:
                k = ("copper", "transition (1024-2048)")
            elif N < 4096:
                k = ("copper", "post-trans (2304-3072)")
            else:
                k = ("copper", "compute (N>=4096)")
        bins[k].append(err)
    result = {}
    for k, errs in bins.items():
        if not errs:
            continue
        result[k] = {
            "n": len(errs),
            "mae": float(np.mean([abs(e) for e in errs])),
            "max": float(max(errs, key=abs)),
        }
    return result


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 74)
    print("A100 PCIe 重校准 — NeuSight DeepMD v5 analytical roofline")
    print("=" * 74)

    data = load_a100_data()
    print(f"加载 A100 数据点: {len(data)}")
    for model, N, nnei, real, mlp, src in data:
        pass  # 数据量大, 不全打
    print()

    # -------- (1) H100 校准值直接套 A100 (未重校准) --------
    # 为公平对比, 这里 BW 用 H100->A100 简单缩放, 等价于 v5 现行行为:
    # gpu_oh_scaled = gpu_oh_H100 * (REF_BW_H100 / REF_BW_A100)
    # 用 C_QUAD / C_LINEAR + bw=A100 就自动实现了。
    # fixed 沿用 H100 查表 (这是 v5 跨 GPU 的问题所在)。
    baseline = evaluate(
        data,
        H100_C_QUAD, H100_C_LINEAR, H100_FIXED_WATER, H100_FIXED_COPPER,
        label="Baseline: H100 常数 + A100 BW 缩放 (v5 原貌)",
    )
    baseline_bins = partition_stats(baseline, data)

    # -------- (2) A100 重校准 --------
    x0 = [H100_C_QUAD, H100_C_LINEAR, 8.30, 6.85]     # 初值: H100 + 实测 plateau 估值
    bounds = [(1, 200), (100, 50000), (4, 15), (4, 15)]
    result = minimize(
        objective, x0, args=(data,),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 50000, "ftol": 1e-15},
    )
    C_q, C_l, fw, fc = result.x
    calibrated = evaluate(
        data, C_q, C_l, fw, fc,
        label=f"Recalibrated A100 (converged={result.success})",
    )
    calib_bins = partition_stats(calibrated, data)

    # -------- 打印详细表 --------
    print_table(baseline, data)
    print_table(calibrated, data)

    # -------- 打印常数对比 --------
    print("\n" + "=" * 74)
    print("常数对比")
    print("=" * 74)
    print(f"{'Constant':<20} | {'H100 (ref)':>13} | {'A100 re-calibrated':>20}")
    print("-" * 74)
    print(f"{'C_QUAD':<20} | {H100_C_QUAD:>13.4f} | {C_q:>20.4f}")
    print(f"{'C_LINEAR':<20} | {H100_C_LINEAR:>13.4f} | {C_l:>20.4f}")
    print(f"{'fixed_2type (water)':<20} | {H100_FIXED_WATER:>10.3f} ms | {fw:>17.3f} ms")
    print(f"{'fixed_1type (copper)':<20} | {H100_FIXED_COPPER:>10.3f} ms | {fc:>17.3f} ms")

    # -------- 打印区间对比表 --------
    print("\n" + "=" * 74)
    print("区间对比 (MAE / max|err|)")
    print("=" * 74)
    print(f"{'Model':>7} | {'Region':<25} | {'n':>3} | {'Baseline MAE':>13} | {'Calibrated MAE':>15} | {'Max|err|':>9}")
    print("-" * 88)
    for k in sorted(set(baseline_bins) | set(calib_bins)):
        bb = baseline_bins.get(k, {})
        cb = calib_bins.get(k, {})
        print(
            f"{k[0]:>7} | {k[1]:<25} | {cb.get('n', bb.get('n', 0)):>3} | "
            f"{bb.get('mae', 0):>12.2f}% | {cb.get('mae', 0):>14.2f}% | "
            f"{cb.get('max', 0):>+8.2f}%"
        )

    # -------- 整体总结 --------
    print("\n" + "=" * 74)
    print("整体 MAE 对比")
    print("=" * 74)
    for model in ("water", "copper"):
        sub_b = [r[4] for r in baseline["rows"] if r[0] == model]
        sub_c = [r[4] for r in calibrated["rows"] if r[0] == model]
        print(
            f"  {model:>7}: baseline MAE = {np.mean(np.abs(sub_b)):6.2f}%  "
            f"max {max(sub_b, key=abs):+6.2f}%    "
            f"calibrated MAE = {np.mean(np.abs(sub_c)):6.2f}%  "
            f"max {max(sub_c, key=abs):+6.2f}%"
        )
    print(
        f"  overall: baseline {baseline['mae']:.2f}%   "
        f"calibrated {calibrated['mae']:.2f}%   "
        f"improvement {baseline['mae'] / max(calibrated['mae'], 1e-9):.2f}x"
    )

    # -------- 保存 JSON --------
    out = {
        "model_version": "v5_analytical_roofline_A100_recalibrated",
        "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_mem_bw": REF_BW_A100,
        "recalibrated": {
            "C_quad": round(C_q, 4),
            "C_linear": round(C_l, 4),
            "fixed_overhead": {
                "se_e2_a": {
                    "1_type": round(fc, 3),
                    "2_type": round(fw, 3),
                },
            },
            "mae_pct": round(calibrated["mae"], 2),
            "max_err_pct": round(calibrated["max"], 2),
        },
        "baseline_h100_constants": {
            "C_quad": H100_C_QUAD,
            "C_linear": H100_C_LINEAR,
            "fixed_2type": H100_FIXED_WATER,
            "fixed_1type": H100_FIXED_COPPER,
            "mae_pct": round(baseline["mae"], 2),
            "max_err_pct": round(baseline["max"], 2),
        },
        "partition_stats": {
            "baseline": {f"{m}/{r}": v for (m, r), v in baseline_bins.items()},
            "calibrated": {f"{m}/{r}": v for (m, r), v in calib_bins.items()},
        },
        "rows": [
            {
                "model": m, "N": N, "nnei": nnei,
                "real_ms": real, "mlp_ms": mlp,
                "baseline_pred": round(
                    predict(m, N, nnei, mlp,
                            H100_C_QUAD, H100_C_LINEAR,
                            H100_FIXED_WATER, H100_FIXED_COPPER), 3),
                "calibrated_pred": round(
                    predict(m, N, nnei, mlp, C_q, C_l, fw, fc), 3),
                "baseline_err_pct": round(
                    (predict(m, N, nnei, mlp,
                             H100_C_QUAD, H100_C_LINEAR,
                             H100_FIXED_WATER, H100_FIXED_COPPER) - real) / real * 100, 2),
                "calibrated_err_pct": round(
                    (predict(m, N, nnei, mlp, C_q, C_l, fw, fc) - real) / real * 100, 2),
                "source": src,
            }
            for m, N, nnei, real, mlp, src in data
        ],
    }
    out_dir = os.path.join(ROOT, "results", "a100_experiment")
    out_path = os.path.join(out_dir, "recalibration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n写入 {out_path}")


if __name__ == "__main__":
    main()
