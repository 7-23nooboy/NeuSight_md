#!/usr/bin/env python3
"""
跨体系 (1/2/4 元) NeuSight DeepMD 精度回归测试。

分别在 H100 NVL 上 profile copper / water / LiAlOCl, 与 NeuSight 预测对比。
配合 scripts/calibrate_fixed_overhead.py 使用:

  step 1) 跑本脚本 → 得 results/cross_system/cross_system_report.json
  step 2) 用 1-2 个体系的实测值校准:
            python scripts/calibrate_fixed_overhead.py \\
                --measurement results/cross_system/cross_system_report.json \\
                --output results/calibration/h100_nvl.json
  step 3) NEUSIGHT_DEEPMD_CALIBRATION=results/calibration/h100_nvl.json \\
            python scripts/benchmark_cross_system.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE_CONFIG_PATH = ROOT / "scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json"
HOST_CONFIG_PATH = ROOT / "scripts/asplos/data/host_configs/H100_NVL_default.json"
PREDICTOR_PATH = ROOT / "scripts/asplos/data/predictor/MLP_WAVE"
TILE_DATASET_DIR = ROOT / "scripts/asplos/data/dataset/train"
RESULT_DIR = ROOT / "results/cross_system"

SYSTEMS = [
    {
        "name": "copper",
        "config": ROOT / "scripts/asplos/data/deepmd_configs/copper_se_e2_a.json",
        "atoms": [64, 128, 256],
        "box": 15.0,
    },
    {
        "name": "water",
        "config": ROOT / "scripts/asplos/data/deepmd_configs/water_se_e2_a.json",
        "atoms": [64, 128, 256],
        "box": 15.0,
    },
    {
        "name": "LiAlOCl",
        "config": ROOT / "scripts/asplos/data/deepmd_configs/LiAlOCl_se_e2_a.json",
        "atoms": [64, 128, 256],
        "box": 15.0,
    },
    # ---- 压力测试点 (可选 / 需 --include-stress 开启) ----
    {
        "name": "he6",
        "config": ROOT / "scripts/asplos/data/deepmd_configs/he6_se_e2_a.json",
        "atoms": [64, 128, 256],
        "box": 18.0,
        "stress": True,        # 6 元素 × type_one_side=False → 36 子网络
    },
    {
        "name": "water_dpa1",
        "config": ROOT / "scripts/asplos/data/deepmd_configs/water_dpa1.json",
        "atoms": [64, 128, 256],
        "box": 15.0,
        "stress": True,        # 切换描述子: se_e2_a → dpa1 (se_atten)
    },
]


def build_model(config):
    import torch
    from deepmd.pt.model.model import get_model
    from deepmd.pt.utils.env import DEVICE

    desc = config["descriptor"]
    fit = config["fitting_net"]
    desc_type = desc.get("type", "se_e2_a")
    desc_type_norm = desc_type.lower().replace("-", "").replace("_", "")

    desc_spec = {
        "type": desc_type,
        "sel": desc["sel"],
        "rcut": desc.get("rcut", 6.0),
        "rcut_smth": desc.get("rcut_smth", 0.5),
        "neuron": desc["neuron"],
        "axis_neuron": desc.get("axis_neuron", 16),
        "resnet_dt": desc.get("resnet_dt", False),
        "activation_function": desc.get("activation_function", "tanh"),
        "precision": desc.get("precision", "float32"),
    }
    if desc_type_norm in ("dpa1", "seatten"):
        # DPA-1 (se_atten) 不接受 type_one_side, 需要 attn / attn_layer 等
        desc_spec["attn"] = desc.get("attn", 128)
        desc_spec["attn_layer"] = desc.get("attn_layer", 2)
        desc_spec["attn_dotr"] = desc.get("attn_dotr", True)
        desc_spec["attn_mask"] = desc.get("attn_mask", False)
    else:
        desc_spec["type_one_side"] = desc.get("type_one_side", False)

    model_params = {
        "type_map": config["type_map"],
        "descriptor": desc_spec,
        "fitting_net": {
            "type": "ener",
            "neuron": fit["neuron"],
            "resnet_dt": fit.get("resnet_dt", True),
            "activation_function": fit.get("activation_function", "tanh"),
            "precision": fit.get("precision", "float32"),
        },
    }
    return get_model(model_params).to(DEVICE).eval()


def profile(model, num_atoms, num_types, box_size, warmup, runs):
    import torch
    from deepmd.pt.utils.env import DEVICE

    coord = torch.rand(1, num_atoms, 3, dtype=torch.float64, device=DEVICE) * box_size
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, num_atoms), device=DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box_size
    box.requires_grad_(True)

    for _ in range(warmup):
        _ = model(coord, atype, box)
        if coord.grad is not None: coord.grad.zero_()
        if box.grad is not None: box.grad.zero_()
    torch.cuda.synchronize()

    lat_ms = []
    for _ in range(runs):
        if coord.grad is not None: coord.grad.zero_()
        if box.grad is not None: box.grad.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(coord, atype, box)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)
    return lat_ms


def neusight_pred(config_path, num_atoms, compute_force, box_size, result_dir):
    """Run pred_deepmd.py and return a breakdown dict.

    Returns
    -------
    dict | None
        {
          'e2e':            预测总 latency (ms),
          'compute':        MLP_WAVE 给出的 compute 部分 (ms),
          'overhead_total': v6 overhead 总量 (ms),
          'fixed_overhead': fixed_overhead_ms (α+βK+δn²+γ, ms),
          'gpu_overhead':   roofline (ms),
          'regime':         'overhead-bound' | 'transition' | 'compute-bound',
          'k_modeled': int, 'k_framework': int, 'k_total': int,
        }
    """
    cmd = [
        sys.executable, str(ROOT / "scripts/pred_deepmd.py"),
        "--predictor_path", str(PREDICTOR_PATH),
        "--device_config_path", str(DEVICE_CONFIG_PATH),
        "--deepmd_config_path", str(config_path),
        "--num_atoms", str(num_atoms),
        "--tile_dataset_dir", str(TILE_DATASET_DIR),
        "--result_dir", str(result_dir),
        "--host_config_path", str(HOST_CONFIG_PATH),
        "--box_size", f"{box_size}",
    ]
    if compute_force:
        cmd.append("--compute_force")
    out = subprocess.run(cmd, capture_output=True, text=True)

    # 从 JSON 读详细分解 (不仅仅是总 latency)
    model_type_map = {"se_e2_a": "se_e2_a", "dpa1": "dpa1", "se_atten": "dpa1"}
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = model_type_map.get(
        cfg.get("descriptor", {}).get("type", cfg.get("model_type", "se_e2_a")),
        "se_e2_a",
    )
    suffix = "_force" if compute_force else ""
    json_path = (Path(result_dir) / "prediction" / DEVICE_CONFIG_PATH.stem
                 / f"deepmd_{model_type}_n{num_atoms}{suffix}.json")
    if not json_path.exists():
        # fallback: parse stdout total only
        line = next((ln for ln in out.stdout.splitlines() if "DeepMD E2E latency" in ln), None)
        if not line:
            return None
        return {"e2e": float(line.partition(":")[2].strip().split(" ms")[0].split()[-1])}

    with open(json_path) as f:
        d = json.load(f)
    oh = d.get("overhead", {})
    # gpu_oh_roofline = roofline 解析模型 (我们 v6 的另一半)
    # total_ms - fixed = back-derived; 当 overhead-bound 时 total = fix-compute (会被 max() 截断)
    # 真正想要的是模型内部那个 unmodeled_compute_ms (= gpu_overhead_roofline_ms)
    gpu_oh_roofline = oh.get("gpu_overhead_roofline_ms", None)
    return {
        "e2e": d["e2e_latency"],                        # = max(fix, compute + gpu_oh_roofline)
        "compute": d["compute_latency"],                # MLP_WAVE per-op sum
        "fixed_overhead": oh.get("fixed_overhead_ms",   # v6 fixed (我们的)
                                 oh.get("cpu_dispatch_ms")),
        "gpu_oh_roofline": gpu_oh_roofline,             # roofline GPU oh (我们的)
        "overhead_total": oh.get("total_ms"),           # back-derived (cosmetic)
        "regime": d.get("confidence", {}).get("regime"),
        "transition_ratio": d.get("confidence", {}).get("transition_ratio"),
        "k_modeled": d.get("kernel_count", {}).get("modeled"),
        "k_framework": d.get("kernel_count", {}).get("framework"),
        "k_total": d.get("kernel_count", {}).get("total"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=80)
    p.add_argument("--skip_real", action="store_true")
    p.add_argument("--include-stress", action="store_true",
                   help="启用 stress test 体系 (he6, water_dpa1)。"
                        "默认只跑 copper / water / LiAlOCl 3 个主体系。")
    p.add_argument("--only", default=None,
                   help="逗号分隔的体系名过滤: 例如 'he6,water_dpa1'")
    args = p.parse_args()

    # 过滤 systems
    systems = []
    for s in SYSTEMS:
        if args.only:
            if s["name"] in args.only.split(","):
                systems.append(s)
        elif s.get("stress") and not args.include_stress:
            continue
        else:
            systems.append(s)

    import torch
    print("=" * 78)
    print("Cross-system NeuSight DeepMD accuracy (copper / water / LiAlOCl)")
    print("=" * 78)
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"DeepMD-kit  : {__import__('deepmd').__version__}")
    calib_env = os.environ.get("NEUSIGHT_DEEPMD_CALIBRATION", "(none)")
    print(f"Calibration : {calib_env}")
    print()

    all_rows = []
    for sys_info in systems:
        with open(sys_info["config"]) as f:
            config = json.load(f)
        num_types = len(config["type_map"])

        if not args.skip_real:
            print(f"[{sys_info['name']}] building model…")
            model = build_model(config)
        else:
            model = None

        for n in sys_info["atoms"]:
            real_mean = real_std = None
            if not args.skip_real:
                lat = profile(model, n, num_types, sys_info["box"], args.warmup, args.runs)
                real_mean = float(np.mean(lat))
                real_std = float(np.std(lat))

            pred_force = neusight_pred(sys_info["config"], n, True, sys_info["box"],
                                        RESULT_DIR / sys_info["name"])
            pred_e = neusight_pred(sys_info["config"], n, False, sys_info["box"],
                                    RESULT_DIR / sys_info["name"])

            err = None
            if real_mean and pred_force and pred_force.get("e2e"):
                err = (pred_force["e2e"] - real_mean) / real_mean * 100

            row = {
                "system": sys_info["name"],
                "num_atoms": n,
                "real_mean_ms": round(real_mean, 4) if real_mean else None,
                "real_std_ms": round(real_std, 4) if real_std else None,
                # ---- v6 三部分拆解 (force=True 路径) ----
                #   pred_e2e = max(fix, compute + gpu_oh_roofline)
                "pred_force_e2e_ms":   round(pred_force["e2e"], 4) if pred_force else None,
                "pred_force_fixed_ms": round(pred_force.get("fixed_overhead") or 0, 4) if pred_force else None,
                "pred_force_compute_ms": round(pred_force.get("compute") or 0, 4) if pred_force else None,
                "pred_force_gpu_oh_roofline_ms": round(pred_force.get("gpu_oh_roofline") or 0, 4) if pred_force else None,
                "regime": pred_force.get("regime") if pred_force else None,
                "transition_ratio": pred_force.get("transition_ratio") if pred_force else None,
                "k_modeled": pred_force.get("k_modeled") if pred_force else None,
                "k_total": pred_force.get("k_total") if pred_force else None,
                # ---- E only ----
                "pred_energy_e2e_ms": round(pred_e["e2e"], 4) if pred_e else None,
                "error_pct": round(err, 1) if err is not None else None,
            }
            # 归因分析: 在假设 fixed 货真价实下, MLP_WAVE compute 差多少?
            #   real_compute_implied = real - fix
            #   compute_gap = real_compute_implied - compute   (>0 表示 MLP 低估)
            if real_mean and pred_force and pred_force.get("fixed_overhead") is not None:
                real_compute_implied = real_mean - pred_force["fixed_overhead"]
                row["real_minus_fixed_ms"] = round(real_compute_implied, 4)
                row["compute_gap_ms"] = round(real_compute_implied - pred_force["compute"], 4)
                # 加上 roofline 后的 gap
                row["gpu_compute_gap_ms"] = round(
                    real_compute_implied
                    - pred_force["compute"]
                    - (pred_force.get("gpu_oh_roofline") or 0),
                    4)
            all_rows.append(row)

            # 控制台实时输出: 3 部分 + 归因
            if pred_force and real_mean:
                fix = pred_force.get("fixed_overhead") or 0
                cmp_ = pred_force.get("compute") or 0
                roof = pred_force.get("gpu_oh_roofline") or 0
                rgme = (pred_force.get("regime") or "?")[:7]
                print(f"  {sys_info['name']:10s} N={n:>4d}  real={real_mean:>7.2f}  "
                      f"pred={pred_force['e2e']:>7.2f}  "
                      f"[fix={fix:>5.2f} | cmp={cmp_:>5.2f} +roof={roof:>5.2f}]  "
                      f"reg={rgme:7s}  err={err:+6.1f}%")
            else:
                print(f"  {sys_info['name']:10s} N={n:>4d}  pred={pred_force}")

    print("\n" + "=" * 122)
    print("Summary  (E+F path)  —  pred = max(fix, cmp + roof)")
    print("=" * 122)
    print(f"{'system':11s} {'N':>5s} {'real':>7s} {'std':>5s} | "
          f"{'pred':>7s} | {'fix':>6s} {'cmp':>6s} {'roof':>6s} | "
          f"{'regime':<13s} {'err%':>7s} | {'gap_cmp':>8s} {'gap_gpu':>8s}")
    print("-" * 122)
    for r in all_rows:
        err = "N/A" if r["error_pct"] is None else f"{r['error_pct']:+.1f}%"
        real = "N/A" if r["real_mean_ms"] is None else f"{r['real_mean_ms']:.2f}"
        std  = "N/A" if r["real_std_ms"] is None else f"{r['real_std_ms']:.2f}"
        pe2e = r['pred_force_e2e_ms']
        pfix = r['pred_force_fixed_ms']
        pcmp = r['pred_force_compute_ms']
        proof = r['pred_force_gpu_oh_roofline_ms']
        rgme = (r['regime'] or "?")
        gap  = r.get('compute_gap_ms')
        gpu_gap = r.get('gpu_compute_gap_ms')
        gap_s = f"{gap:+.2f}" if gap is not None else "N/A"
        gpu_gap_s = f"{gpu_gap:+.2f}" if gpu_gap is not None else "N/A"
        print(f"{r['system']:11s} {r['num_atoms']:>5d} {real:>7s} {std:>5s} | "
              f"{pe2e:>7.2f} | {pfix:>6.2f} {pcmp:>6.2f} {proof:>6.2f} | "
              f"{rgme:<13s} {err:>7s} | {gap_s:>8s} {gpu_gap_s:>8s}")
    print("\nLegend (责任划分):")
    print("  fix     = v6 fixed_overhead_ms   (α+β·K_mod+δ·n²+γ)            — 我们的模型")
    print("  cmp     = compute_latency_ms     (MLP_WAVE 逐算子加总)         — NeuSight 原生")
    print("  roof    = gpu_oh_roofline_ms     (C_quad/C_linear roofline)    — 我们的模型")
    print("  pred    = max(fix, cmp + roof)")
    print("  gap_cmp = (real - fix) - cmp                                   — 仅 MLP 责任 (不含 roofline)")
    print("  gap_gpu = (real - fix) - cmp - roof                            — MLP+roofline 之后还不能解释的部分")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "cross_system_report.json"
    with open(out, "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "calibration": calib_env,
            "warmup": args.warmup,
            "runs": args.runs,
            "rows": all_rows,
        }, f, indent=2)
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()
