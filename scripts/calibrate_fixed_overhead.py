#!/usr/bin/env python3
"""
校准 NeuSight DeepMD overhead 模型的 fixed_overhead 参数 (α, β, γ)。

Overhead 模型 v6:
    fixed_overhead_ms = α + β × K_modeled + γ × is_force

K_modeled 由 tracer 的 KERNEL_MULTIPLIER 表对算子图求和得到，
天然反映 ntypes (1/2/3/4 元体系) 的差异。

通过 1-3 个真实测量做最小二乘拟合, 即可在新机器/新框架版本上重新校准。
拟合结果可被 DeepMDOverheadModel.__init__(calibration_path=...) 加载。

用法 1 — 从现成 benchmark 报告校准:
    python scripts/calibrate_fixed_overhead.py \\
        --measurement results/deepmd_benchmark/accuracy_report.json:water_se_e2_a \\
        --measurement results/lialocl/lialocl_accuracy_report.json \\
        --output results/calibration/fixed_overhead.json

用法 2 — 直接传 (config_path, num_atoms, real_ms[, force]) 三/四元组:
    python scripts/calibrate_fixed_overhead.py \\
        --point scripts/asplos/data/deepmd_configs/copper_se_e2_a.json,128,4.85 \\
        --point scripts/asplos/data/deepmd_configs/water_se_e2_a.json,128,5.7 \\
        --point scripts/asplos/data/deepmd_configs/LiAlOCl_se_e2_a.json,128,23.0 \\
        --output results/calibration/fixed_overhead.json

要求: 用于校准的体系必须处于 overhead-bound 区间 (即真实 latency
几乎不随 N 变化)。建议 N=64-256 之间的 small-system 平均值。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 直接加载 tracer + overhead 模型
from neusight.Tracing.parse_deepmd_input import parse_deepmd_input
from neusight.Tracing.trace_deepmd import build_deepmd_opgraph
from neusight.Prediction.overhead_model import DeepMDOverheadModel


def kmod_for(config_path, num_atoms, compute_force):
    """计算给定 config + N + force 的 (K_modeled, K_framework, K_total, df)."""
    cfg = parse_deepmd_input(config_path)
    df = build_deepmd_opgraph(cfg, num_atoms, compute_force=compute_force)
    m = DeepMDOverheadModel()
    k_mod = m._count_modeled_kernels(df)
    k_fw = m._count_framework_kernels(cfg, compute_force)
    return k_mod, k_fw, k_mod + k_fw, df


# ----------------------------------------------------------------------
# P0a: roofline (C_quad, C_linear) LSQ fitting on compute-bound points
# ----------------------------------------------------------------------
def fit_roofline_from_report(report_path, ref_mem_bw=3430.0,
                              regime_filter=("compute-bound", "transition"),
                              min_atoms=2048):
    """
    从 cross_system_report.json 拟合 C_quad 与 C_linear。

    模型 (per row):
        roof_observed = real - max(fix, cmp)        (compute-bound 时 ≈ real - cmp)
        roof_observed ≈ (C_quad·N·n_all + C_linear·N·n_nei) · 8B / Mem_Bw  [ms]
                      = (C_quad·N·n_all + C_linear·N·n_nei) · 8/Mem_Bw_GBps × 1000  [ms? no]

    实际 overhead_model._compute_analytical_gpu_overhead 公式:
        roof_ms = (C_quad·N·n_all·8 + C_linear·N·n_nei·8) / (Mem_Bw_GBps · 1e9) × 1e3
                = (C_quad·N·n_all + C_linear·N·n_nei) · 8e-6 / Mem_Bw_GBps × 1e3
                = (C_quad·N·n_all + C_linear·N·n_nei) · 8e-3 / Mem_Bw_GBps  (ms)

    令 y = roof_observed × Mem_Bw_GBps / 8e-3  =  C_quad·N·n_all + C_linear·N·n_nei
    然后对 [N·n_all, N·n_nei] LSQ → (C_quad, C_linear)

    Parameters
    ----------
    report_path : str
        cross_system_report.json (含 fix/cmp/real/system/num_atoms 等列)
    ref_mem_bw : float
        参考 GPU 带宽 (GB/s); 默认 H100 NVL = 3430
    regime_filter : tuple
        哪些 regime 的行用于拟合 (默认 transition + compute-bound)
    min_atoms : int
        最小 N 阈值 (排除小 N 干扰)

    Returns
    -------
    dict with C_quad, C_linear, residuals, fit_meta
    """
    with open(report_path) as f:
        rep = json.load(f)

    rows = rep.get("rows", rep) if isinstance(rep, dict) else rep
    fit_pts = []
    for r in rows:
        regime = r.get("regime")
        N = r.get("num_atoms")
        real = r.get("real_mean_ms")
        cmp = r.get("pred_force_compute_ms")
        fix = r.get("pred_force_fixed_ms")
        sysname = r.get("system")
        if not (regime in regime_filter and N and real and cmp is not None and fix is not None):
            continue
        if N < min_atoms:
            continue
        # 找该 system 的 config
        cfg_path = ROOT / "scripts/asplos/data/deepmd_configs"
        candidates = [cfg_path / f"{sysname}_se_e2_a.json",
                      cfg_path / f"{sysname}.json",
                      cfg_path / f"{sysname}_dpa1.json"]
        cfg_file = next((c for c in candidates if c.is_file()), None)
        if cfg_file is None:
            print(f"[warn] no config for system={sysname}, skip")
            continue
        with open(cfg_file) as fh:
            cfg = json.load(fh)
        sel = cfg.get("descriptor", {}).get("sel", [60])
        if isinstance(sel, list):
            n_nei = sum(sel)
        else:
            n_nei = int(sel)
        # ns: ghost cell factor — 估为 27 (与 overhead_model 一致)
        ns = 27
        n_all = ns * N
        # roof_observed = real - max(fix, cmp)
        roof_obs = real - max(fix, cmp)
        roof_obs = max(0.0, roof_obs)
        fit_pts.append({
            "system": sysname,
            "N": N,
            "n_nei": n_nei,
            "n_all": n_all,
            "real": real,
            "fix": fix,
            "cmp": cmp,
            "roof_obs": roof_obs,
            "regime": regime,
        })

    if len(fit_pts) < 2:
        raise ValueError(
            f"Need ≥ 2 compute-bound points (N≥{min_atoms}, regime∈{regime_filter}); "
            f"got {len(fit_pts)} from {report_path}")

    # y = roof_obs[ms] · Mem_Bw[GB/s] / 8e-3   →  在 (C_quad·N·n_all + C_linear·N·n_nei) 量纲下
    # 这里需要与 overhead_model._compute_analytical_gpu_overhead 数值口径完全一致:
    #     roof_ms = (C_quad·N·nall + C_linear·N·nnei) × 8 / (Mem_Bw_GBps × 1e6)  (ms)
    # ⇒ y = roof_ms × Mem_Bw_GBps × 1e6 / 8
    scale = ref_mem_bw * 1e6 / 8.0
    y = np.array([p["roof_obs"] * scale for p in fit_pts], dtype=float)
    A = np.array([[p["N"] * p["n_all"], p["N"] * p["n_nei"]] for p in fit_pts],
                 dtype=float)

    # LSQ with non-negativity safeguard: try plain LSQ; if any negative, redo NNLS
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    if any(v < 0 for v in x):
        try:
            from scipy.optimize import nnls
            x, _ = nnls(A, y)
            print("[info] used scipy.optimize.nnls (negative coef in plain LSQ)")
        except ImportError:
            print(f"[warn] LSQ produced negative coef {x.tolist()}; "
                  f"clamping to 0 (install scipy for NNLS)")
            x = np.maximum(x, 0.0)

    C_quad, C_linear = float(x[0]), float(x[1])

    # residuals on training set (in ms space)
    residuals = []
    for p in fit_pts:
        roof_pred_ms = (C_quad * p["N"] * p["n_all"]
                        + C_linear * p["N"] * p["n_nei"]) * 8.0 / (ref_mem_bw * 1e6)
        e2e_pred_ms = max(p["fix"], p["cmp"] + roof_pred_ms)
        residuals.append({
            "system": p["system"],
            "N": p["N"],
            "real_ms": round(p["real"], 4),
            "fix_ms": round(p["fix"], 4),
            "cmp_ms": round(p["cmp"], 4),
            "roof_obs_ms": round(p["roof_obs"], 4),
            "roof_pred_ms": round(roof_pred_ms, 4),
            "e2e_pred_ms": round(e2e_pred_ms, 4),
            "e2e_err_pct": round((e2e_pred_ms - p["real"]) / p["real"] * 100, 2),
        })

    return {
        "C_quad": C_quad,
        "C_linear": C_linear,
        "ref_mem_bw_gbps": ref_mem_bw,
        "n_points": len(fit_pts),
        "residuals": residuals,
    }


def parse_point(spec):
    """解析 'config,atoms,real_ms[,force]' 字符串."""
    parts = spec.split(",")
    if len(parts) < 3:
        raise ValueError(f"--point expects 'config,atoms,real_ms[,force]', got {spec}")
    config = parts[0].strip()
    atoms = int(parts[1])
    real_ms = float(parts[2])
    force = False
    if len(parts) >= 4:
        force = parts[3].strip().lower() in ("1", "true", "yes", "force")
    return {"config": config, "atoms": atoms, "real_ms": real_ms, "force": force}


def measurements_from_report(spec):
    """
    从 benchmark JSON 报告抽取测量点。

    spec 格式: 'path/to/report.json[:model_name]'
    支持两种报告:
      - benchmark_deepmd_accuracy.py 输出 (有 model 字段)
      - benchmark_lialocl_accuracy.py 输出 (rows 列表, 单一系统)
    """
    if ":" in spec:
        path, model_filter = spec.split(":", 1)
    else:
        path, model_filter = spec, None

    with open(path) as f:
        rep = json.load(f)

    # 顶层 dict, rows 列表 (benchmark_lialocl 或 benchmark_cross_system 风格)
    if isinstance(rep, dict) and "rows" in rep:
        top_config = rep.get("config")
        out = []
        for r in rep["rows"]:
            real = r.get("real_mean_ms") or r.get("real_median_ms")
            if real is None:
                continue

            # 解析 config 路径: 三种来源
            #   1) row 自带 'config' 字段
            #   2) 顶层 'config' 字段 (benchmark_lialocl)
            #   3) row 'system' → scripts/asplos/data/deepmd_configs/<system>_se_e2_a.json
            row_config = r.get("config")
            if row_config:
                cfg = row_config
            elif top_config:
                cfg = top_config
            elif r.get("system"):
                cfg = str(ROOT / "scripts/asplos/data/deepmd_configs" /
                          f"{r['system']}_se_e2_a.json")
            else:
                print(f"[warn] no config resolvable for row: {r}")
                continue

            if not os.path.isabs(cfg):
                cfg = str(ROOT / cfg)

            out.append({
                "config": cfg,
                "atoms": r["num_atoms"],
                "real_ms": real,
                "force": r.get("force", True),  # 这些 benchmark 默认测 E+F
                "tag": f"{r.get('system') or Path(cfg).stem}_n{r['num_atoms']}",
            })
        return out

    # benchmark_deepmd 风格: list of dicts with 'model' field
    if isinstance(rep, list):
        out = []
        for entry in rep:
            if model_filter and entry.get("model") != model_filter:
                continue
            real = entry.get("real_mean_ms") or entry.get("real_median_ms")
            if real is None:
                continue
            # 找 config 路径
            model_name = entry.get("model", "water_se_e2_a")
            config = ROOT / "scripts/asplos/data/deepmd_configs" / f"{model_name}.json"
            if not config.is_file():
                print(f"[warn] config not found for model={model_name}: {config}")
                continue
            out.append({
                "config": str(config),
                "atoms": entry["num_atoms"],
                "real_ms": real,
                "force": True,
                "tag": f"{model_name}_n{entry['num_atoms']}",
            })
        return out

    raise ValueError(f"Unknown report format: {path}")


def fit_alpha_beta_gamma(points, fix_gamma=None, driver="modeled", fit_ntypes_sq=False):
    """
    最小二乘拟合 overhead 参数。

    默认模型: real_ms ≈ α + β × K_driver + γ × force_flag

    拓展模型 (fit_ntypes_sq=True):
        real_ms ≈ α + β × K_driver + δ × ntypes² + γ × force_flag
      —— ntypes² 项显式捕捉多元素体系中子网络个数的二次扩展。

    Parameters
    ----------
    points : list[dict]
        每个元素需含: config, atoms, real_ms, force (bool)
    fix_gamma : float, optional
        若提供, γ 视为已知常量。默认启动 force-degenerate 检测：
        若所有点的 force 状态一致, 自动固定 γ=0.15。
    driver : {'modeled', 'total'}
        控制 β 乘以 K_modeled 还是 K_total = K_modeled + K_framework。
    fit_ntypes_sq : bool
        启用 ntypes² 额外项 (需要成份包含必要变化)。

    Returns
    -------
    dict
    """
    # 计算所有 K 与 ntypes
    rows = []
    for p in points:
        K_mod, K_fw, K_tot, df = kmod_for(p["config"], p["atoms"], p["force"])
        with open(p["config"]) as fh:
            cfg = json.load(fh)
        ntypes = len(cfg.get("type_map", []))
        p["K_modeled"] = K_mod
        p["K_framework"] = K_fw
        p["K_total"] = K_tot
        p["ntypes"] = ntypes
        rows.append(p)

    K = np.array([p["K_total" if driver == "total" else "K_modeled"] for p in rows],
                  dtype=float)
    y = np.array([p["real_ms"] for p in rows], dtype=float)
    f = np.array([1.0 if p["force"] else 0.0 for p in rows], dtype=float)
    n2 = np.array([p["ntypes"] ** 2 for p in rows], dtype=float)

    # 自动检测 force degenerate
    force_uniform = (len(set(int(x) for x in f)) == 1)
    if fix_gamma is None and force_uniform:
        print(f"[info] all points have force={bool(f[0])}; γ not identifiable, fixing γ=0.15")
        fix_gamma = 0.15

    # 拼接设计矩阵
    cols = [np.ones_like(K), K]
    col_names = ["alpha", "beta"]
    if fit_ntypes_sq:
        # 检查 ntypes 是否多样
        if len(set(p["ntypes"] for p in rows)) < 2:
            raise ValueError("fit_ntypes_sq requires at least 2 distinct ntypes in measurements")
        cols.append(n2)
        col_names.append("delta_ntypes_sq")
    if fix_gamma is None:
        cols.append(f)
        col_names.append("gamma")

    A = np.column_stack(cols)
    if A.shape[0] < A.shape[1]:
        raise ValueError(f"Need at least {A.shape[1]} measurements to fit "
                         f"{col_names}; got {A.shape[0]}")

    y_corr = y - (fix_gamma * f if fix_gamma is not None else 0.0)
    x, *_ = np.linalg.lstsq(A, y_corr, rcond=None)

    alpha = float(x[0])
    beta = float(x[1])
    delta = float(x[2]) if fit_ntypes_sq else 0.0
    if fix_gamma is None:
        gamma = float(x[3 if fit_ntypes_sq else 2])
    else:
        gamma = float(fix_gamma)

    residuals = []
    for p in rows:
        k_used = p["K_total"] if driver == "total" else p["K_modeled"]
        pred = alpha + beta * k_used + delta * (p["ntypes"] ** 2) + (gamma if p["force"] else 0.0)
        residuals.append({
            "tag": p.get("tag", Path(p["config"]).stem),
            "atoms": p["atoms"],
            "force": p["force"],
            "ntypes": p["ntypes"],
            "K_modeled": p["K_modeled"],
            "K_framework": p["K_framework"],
            "K_total": p["K_total"],
            "K_driver": p["K_total" if driver == "total" else "K_modeled"],
            "real_ms": round(p["real_ms"], 4),
            "predicted_fixed_ms": round(pred, 4),
            "residual_ms": round(p["real_ms"] - pred, 4),
            "residual_pct": round((p["real_ms"] - pred) / p["real_ms"] * 100, 2),
        })

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta_ntypes_sq": delta,
        "driver": driver,
        "fit_ntypes_sq": fit_ntypes_sq,
        "residuals": residuals,
    }


def main():
    p = argparse.ArgumentParser(description="Fit fixed-overhead α, β, γ from real measurements")
    p.add_argument("--measurement", action="append", default=[],
                   help="Path to existing benchmark report JSON (can repeat). "
                        "Append ':model_name' to filter (for accuracy_report.json).")
    p.add_argument("--point", action="append", default=[],
                   help="Single measurement 'config,atoms,real_ms[,force]'. Repeatable.")
    p.add_argument("--fix-gamma", type=float, default=None,
                   help="Fix γ_force to this value instead of fitting "
                        "(default: fit if force varies, else 0.15).")
    p.add_argument("--driver", choices=["modeled", "total"], default="modeled",
                   help="Which K to drive β: modeled-only (default) or total "
                        "(K_modeled + K_framework, captures per_type_sq term).")
    p.add_argument("--fit-ntypes-sq", action="store_true",
                   help="Add an explicit δ·ntypes² term to the fit (recommended "
                        "when measurements span 1+ different ntypes values).")
    p.add_argument("--output", "-o", required=True,
                   help="Output calibration JSON path")
    p.add_argument("--no-write", action="store_true",
                   help="Print fit but do not save")
    # ---- P0a: roofline fitting ----
    p.add_argument("--fit-roofline", default=None,
                   help="Path to a cross_system_report.json containing fix/cmp/real "
                        "rows. Will fit C_quad and C_linear by LSQ on compute-bound "
                        "rows and write them into the output calibration JSON.")
    p.add_argument("--roofline-min-atoms", type=int, default=2048,
                   help="Minimum N for compute-bound roofline fit (default 2048).")
    p.add_argument("--ref-mem-bw", type=float, default=3430.0,
                   help="Reference GPU memory BW in GB/s (default 3430 = H100 NVL).")
    # ---- P0b: transition zone overrides ----
    p.add_argument("--transition-lo", type=float, default=None,
                   help="Override transition_lo (default 0.4 in v6 model).")
    p.add_argument("--transition-hi", type=float, default=None,
                   help="Override transition_hi (default 2.0 in v6 model).")
    p.add_argument("--bubble-peak-fraction", type=float, default=None,
                   help="Override bubble_peak_fraction (default 0.20).")
    # ---- P2: type_one_side factor ----
    p.add_argument("--delta-type-one-side-factor", default=None,
                   help="Override δ exponent for ntypes term: 'auto' (default), 1, or 2.")
    args = p.parse_args()

    # P0a: 如果只 fit roofline 不需要 fix-overhead 测量
    only_roofline = args.fit_roofline and not args.measurement and not args.point
    if only_roofline:
        print(f"\n=== Fitting roofline only from {args.fit_roofline} ===\n")
        roof_fit = fit_roofline_from_report(
            args.fit_roofline,
            ref_mem_bw=args.ref_mem_bw,
            min_atoms=args.roofline_min_atoms,
        )
        print(f"  n_points          = {roof_fit['n_points']}")
        print(f"  C_quad            = {roof_fit['C_quad']:.4f}")
        print(f"  C_linear          = {roof_fit['C_linear']:.4f}")
        print(f"  ref_mem_bw_gbps   = {roof_fit['ref_mem_bw_gbps']:.1f}")
        print("\n=== Roofline fit residuals (e2e) ===")
        print(f"{'system':12s} {'N':>5s} {'real':>7s} {'pred':>7s} {'err%':>7s}")
        for r in roof_fit["residuals"]:
            print(f"{r['system']:12s} {r['N']:>5d} {r['real_ms']:>7.2f} "
                  f"{r['e2e_pred_ms']:>7.2f} {r['e2e_err_pct']:>+7.1f}%")
        # 仍写出 calibration JSON, 但不带 alpha/beta (用户应在已有 calibration 上叠加)
        partial = {
            "C_quad": round(roof_fit["C_quad"], 6),
            "C_linear": round(roof_fit["C_linear"], 6),
            "fit_meta": {"roofline": {
                "n_points": roof_fit["n_points"],
                "ref_mem_bw_gbps": roof_fit["ref_mem_bw_gbps"],
                "residuals": roof_fit["residuals"],
            }},
        }
        if args.transition_lo is not None:
            partial["transition_lo"] = args.transition_lo
        if args.transition_hi is not None:
            partial["transition_hi"] = args.transition_hi
        if args.bubble_peak_fraction is not None:
            partial["bubble_peak_fraction"] = args.bubble_peak_fraction
        if args.delta_type_one_side_factor is not None:
            partial["delta_type_one_side_factor"] = args.delta_type_one_side_factor
        if args.no_write:
            print("\n[--no-write] skipping save")
            return
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果输出文件已存在 (例如 v6 fixed-overhead calibration), 合并而不是覆盖
        existing = {}
        if out_path.is_file():
            with open(out_path) as f:
                existing = json.load(f)
            print(f"[merge] existing {out_path.name} keys: {sorted(existing.keys())}")
        existing.update(partial)
        # 合并 fit_meta 而不是覆盖
        if "fit_meta" in existing and "roofline" in partial.get("fit_meta", {}):
            if isinstance(existing.get("fit_meta"), dict):
                existing["fit_meta"]["roofline"] = partial["fit_meta"]["roofline"]
        with open(out_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\n[saved] {out_path}")
        return

    points = []
    for spec in args.measurement:
        sub = measurements_from_report(spec)
        print(f"[load] {spec} → {len(sub)} points")
        points.extend(sub)
    for spec in args.point:
        points.append(parse_point(spec))

    if not points:
        p.error("Provide at least one --measurement or --point")

    print("\n=== Calibration input points ===")
    for q in points:
        K_mod, K_fw, K_tot, _ = kmod_for(q["config"], q["atoms"], q["force"])
        q["K_modeled"] = K_mod
        q["K_framework"] = K_fw
        q["K_total"] = K_tot
        with open(q["config"]) as fh:
            ntypes = len(json.load(fh).get("type_map", []))
        q["ntypes"] = ntypes
        print(f"  {Path(q['config']).stem:25s} ntypes={ntypes} N={q['atoms']:>5d} "
              f"force={q['force']!s:5s} K_mod={K_mod:>4d} K_fw={K_fw:>4d} "
              f"K_tot={K_tot:>4d} real={q['real_ms']:.3f} ms")

    fit = fit_alpha_beta_gamma(points, fix_gamma=args.fix_gamma,
                                driver=args.driver,
                                fit_ntypes_sq=args.fit_ntypes_sq)
    alpha, beta, gamma, delta = fit["alpha"], fit["beta"], fit["gamma"], fit["delta_ntypes_sq"]
    residuals = fit["residuals"]

    print("\n=== Fitted parameters ===")
    print(f"  driver                    = K_{args.driver}")
    print(f"  α (alpha_fixed_ms)        = {alpha:.4f} ms")
    print(f"  β (beta_per_kernel_ms)    = {beta:.6f} ms  ({beta*1e3:.2f} μs/kernel)")
    if args.fit_ntypes_sq:
        print(f"  δ (delta_ntypes_sq_ms)    = {delta:.4f} ms / ntypes²")
    print(f"  γ (gamma_force_ms)        = {gamma:.4f} ms"
          + ("  [fixed]" if args.fix_gamma is not None or
             len(set(int(p['force']) for p in points)) == 1 else "  [fit]"))

    print("\n=== Residuals on training set ===")
    print(f"{'tag':32s} {'nt':>3s} {'Kdrv':>5s} {'real':>8s} {'pred':>8s} {'err':>8s} {'err%':>7s}")
    print("-" * 80)
    abs_pcts = []
    for r in residuals:
        print(f"{r['tag']:32s} {r['ntypes']:>3d} {r['K_driver']:>5d} {r['real_ms']:>8.3f} "
              f"{r['predicted_fixed_ms']:>8.3f} {r['residual_ms']:>+8.3f} "
              f"{r['residual_pct']:>+7.1f}%")
        abs_pcts.append(abs(r["residual_pct"]))
    print(f"\nMAE %: {np.mean(abs_pcts):.2f}%  Max abs %: {max(abs_pcts):.2f}%")

    calib = {
        "fixed_overhead_mode": "kernel_count",
        "beta_driver": args.driver,
        "alpha_fixed_ms": round(alpha, 6),
        "beta_per_kernel_ms": round(beta, 8),
        "gamma_force_ms": round(gamma, 6),
        "fit_meta": {
            "n_points": len(residuals),
            "MAE_pct": round(float(np.mean(abs_pcts)), 4),
            "max_abs_pct": round(float(max(abs_pcts)), 4),
            "fix_gamma": args.fix_gamma,
            "driver": args.driver,
            "fit_ntypes_sq": args.fit_ntypes_sq,
            "delta_ntypes_sq_ms": round(delta, 6) if args.fit_ntypes_sq else None,
            "training_points": residuals,
        },
    }
    if args.fit_ntypes_sq:
        # 在 calibration JSON 中设置 ntypes² 加项 (需要 overhead_model 支持)
        calib["delta_ntypes_sq_ms"] = round(delta, 6)

    # P0a: 同时跑 roofline 拟合
    if args.fit_roofline:
        print(f"\n=== Also fitting roofline from {args.fit_roofline} ===")
        roof_fit = fit_roofline_from_report(
            args.fit_roofline,
            ref_mem_bw=args.ref_mem_bw,
            min_atoms=args.roofline_min_atoms,
        )
        calib["C_quad"] = round(roof_fit["C_quad"], 6)
        calib["C_linear"] = round(roof_fit["C_linear"], 6)
        calib["fit_meta"]["roofline"] = {
            "n_points": roof_fit["n_points"],
            "ref_mem_bw_gbps": roof_fit["ref_mem_bw_gbps"],
            "residuals": roof_fit["residuals"],
        }
        print(f"  C_quad   = {roof_fit['C_quad']:.4f}")
        print(f"  C_linear = {roof_fit['C_linear']:.4f}")

    # P0b/P2: 透传 transition / type_one_side 配置
    if args.transition_lo is not None:
        calib["transition_lo"] = args.transition_lo
    if args.transition_hi is not None:
        calib["transition_hi"] = args.transition_hi
    if args.bubble_peak_fraction is not None:
        calib["bubble_peak_fraction"] = args.bubble_peak_fraction
    if args.delta_type_one_side_factor is not None:
        calib["delta_type_one_side_factor"] = args.delta_type_one_side_factor

    if args.no_write:
        print("\n[--no-write] skipping save")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n[saved] {out_path}")
    print(f"\nUse it via:\n"
          f"  predictor = neusight.DeepMDPredictor(...)\n"
          f"  predictor.overhead_model = DeepMDOverheadModel(calibration_path='{out_path}')")


if __name__ == "__main__":
    main()
