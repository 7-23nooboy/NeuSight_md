#!/usr/bin/env python3
"""
LiAlOCl (4-element 体系) NeuSight 精度验证

针对 LiAlOCl-compressed.pb 提取出的模型架构 (se_e2_a, sel=[512]*4,
emb=[16,32,64], axis_neuron=16, fitting=[240,240,240]/relu, fp32) 做:

  1) 用 deepmd.pt 在 H100 上构造同结构 PyTorch 模型 + 随机数据
  2) 真实 wall-time profiling (warmup + N runs)
  3) 调用 scripts/pred_deepmd.py 取得 NeuSight 预测
  4) 多个 num_atoms 下汇总 真实 vs 预测 对比

输出: results/lialocl/lialocl_accuracy_report.json + 终端表格
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

CONFIG_PATH = ROOT / "scripts/asplos/data/deepmd_configs/LiAlOCl_se_e2_a.json"
DEVICE_CONFIG_PATH = ROOT / "scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json"
HOST_CONFIG_PATH = ROOT / "scripts/asplos/data/host_configs/H100_NVL_default.json"
PREDICTOR_PATH = ROOT / "scripts/asplos/data/predictor/MLP_WAVE"
TILE_DATASET_DIR = ROOT / "scripts/asplos/data/dataset/train"
RESULT_DIR = ROOT / "results/lialocl"


def build_lialocl_model(config):
    import torch
    from deepmd.pt.model.model import get_model
    from deepmd.pt.utils.env import DEVICE

    desc = config["descriptor"]
    fit = config["fitting_net"]

    model_params = {
        "type_map": config["type_map"],
        "descriptor": {
            "type": desc.get("type", "se_e2_a"),
            "sel": desc["sel"],
            "rcut": desc.get("rcut", 6.0),
            "rcut_smth": desc.get("rcut_smth", 0.5),
            "neuron": desc["neuron"],
            "axis_neuron": desc.get("axis_neuron", 16),
            "resnet_dt": desc.get("resnet_dt", False),
            "type_one_side": desc.get("type_one_side", False),
            "activation_function": desc.get("activation_function", "tanh"),
            "precision": desc.get("precision", "float32"),
        },
        "fitting_net": {
            "type": "ener",
            "neuron": fit["neuron"],
            "resnet_dt": fit.get("resnet_dt", True),
            "activation_function": fit.get("activation_function", "tanh"),
            "precision": fit.get("precision", "float32"),
        },
    }

    model = get_model(model_params).to(DEVICE).eval()
    return model


def gen_random_system(num_atoms, num_types, box_size):
    import torch
    from deepmd.pt.utils.env import DEVICE

    coord = torch.rand(1, num_atoms, 3, dtype=torch.float64, device=DEVICE) * box_size
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, num_atoms), device=DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box_size
    box.requires_grad_(True)
    return coord, atype, box


def profile_latency(model, coord, atype, box, warmup, runs):
    import torch

    # warmup (DeepMD computes force via autograd)
    for _ in range(warmup):
        _ = model(coord, atype, box)
        if coord.grad is not None:
            coord.grad.zero_()
        if box.grad is not None:
            box.grad.zero_()
    torch.cuda.synchronize()

    lat_ms = []
    for _ in range(runs):
        if coord.grad is not None:
            coord.grad.zero_()
        if box.grad is not None:
            box.grad.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(coord, atype, box)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)
    return lat_ms


def neusight_pred(num_atoms, compute_force, box_size):
    cmd = [
        sys.executable,
        str(ROOT / "scripts/pred_deepmd.py"),
        "--predictor_path", str(PREDICTOR_PATH),
        "--device_config_path", str(DEVICE_CONFIG_PATH),
        "--deepmd_config_path", str(CONFIG_PATH),
        "--num_atoms", str(num_atoms),
        "--tile_dataset_dir", str(TILE_DATASET_DIR),
        "--result_dir", str(RESULT_DIR),
        "--host_config_path", str(HOST_CONFIG_PATH),
        "--box_size", f"{box_size}",
    ]
    if compute_force:
        cmd.append("--compute_force")

    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next(
        (ln for ln in out.stdout.splitlines() if "DeepMD E2E latency" in ln),
        None,
    )
    if not line:
        print("[neusight] FAILED to parse output:")
        print(out.stdout)
        print(out.stderr)
        return None, None

    # "... : 7.315 ms (compute=1.3344, overhead=5.9806)"
    head, _, rest = line.partition(":")
    e2e_ms = float(rest.strip().split(" ms")[0].split()[-1])
    compute_ms = None
    if "compute=" in rest:
        compute_ms = float(rest.split("compute=")[1].split(",")[0])
    return e2e_ms, compute_ms


def fmt(x, n=4):
    return f"{x:.{n}f}" if x is not None else "N/A"


def main():
    parser = argparse.ArgumentParser(description="LiAlOCl 4-element NeuSight accuracy test")
    parser.add_argument(
        "--atoms",
        type=int,
        nargs="+",
        default=[64, 96, 192, 384],
        help="Atom counts to test (default: 64 96 192 384)",
    )
    parser.add_argument("--box_size", type=float, default=15.0,
                        help="Cubic box side length in Å for random system (default: 15)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--skip_real", action="store_true",
                        help="Skip real GPU profiling (only NeuSight prediction)")
    args = parser.parse_args()

    import torch

    print("=" * 78)
    print("LiAlOCl (Li/Al/Cl/O · se_e2_a · sel=[512]*4 · 4-element) accuracy test")
    print("=" * 78)
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"DeepMD-kit  : {__import__('deepmd').__version__}")
    print(f"Config      : {CONFIG_PATH.relative_to(ROOT)}")
    print(f"Device cfg  : {DEVICE_CONFIG_PATH.name}")
    print(f"Atom counts : {args.atoms}")
    print(f"Box (rand)  : {args.box_size} Å")
    print()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    num_types = len(config["type_map"])

    # Build model (architecture independent of num_atoms; reuse for all sizes)
    if not args.skip_real:
        print("Constructing PyTorch DeepMD model (deepmd.pt)…")
        model = build_lialocl_model(config)
        print("  done.\n")
    else:
        model = None

    rows = []
    for n in args.atoms:
        print(f"--- num_atoms = {n} ---")

        real_mean = real_med = real_std = real_p99 = None
        if not args.skip_real:
            try:
                coord, atype, box = gen_random_system(n, num_types, args.box_size)
                lat = profile_latency(model, coord, atype, box, args.warmup, args.runs)
                real_mean = float(np.mean(lat))
                real_med = float(np.median(lat))
                real_std = float(np.std(lat))
                real_p99 = float(np.percentile(lat, 99))
                print(f"  real  : mean={real_mean:.4f}  median={real_med:.4f}"
                      f"  std={real_std:.4f}  p99={real_p99:.4f} (ms)")
            except Exception as e:
                print(f"  real profiling FAILED: {e}")
                real_mean = None

        # NeuSight prediction (energy+force matches deepmd.pt output behaviour)
        pred_force_e2e, pred_force_compute = neusight_pred(n, compute_force=True, box_size=args.box_size)
        pred_e_e2e, pred_e_compute = neusight_pred(n, compute_force=False, box_size=args.box_size)
        print(f"  pred  : E+F e2e={fmt(pred_force_e2e)}  (compute={fmt(pred_force_compute)}) | "
              f"E   e2e={fmt(pred_e_e2e)}  (compute={fmt(pred_e_compute)})")

        err_pct = None
        if real_mean is not None and pred_force_e2e is not None and real_mean > 0:
            err_pct = (pred_force_e2e - real_mean) / real_mean * 100
            print(f"  error : {err_pct:+.1f}% (NeuSight E+F vs real mean)")

        rows.append({
            "num_atoms": n,
            "real_mean_ms": round(real_mean, 4) if real_mean is not None else None,
            "real_median_ms": round(real_med, 4) if real_med is not None else None,
            "real_std_ms": round(real_std, 4) if real_std is not None else None,
            "real_p99_ms": round(real_p99, 4) if real_p99 is not None else None,
            "pred_force_e2e_ms": round(pred_force_e2e, 4) if pred_force_e2e else None,
            "pred_force_compute_ms": round(pred_force_compute, 4) if pred_force_compute else None,
            "pred_energy_e2e_ms": round(pred_e_e2e, 4) if pred_e_e2e else None,
            "pred_energy_compute_ms": round(pred_e_compute, 4) if pred_e_compute else None,
            "error_pct": round(err_pct, 1) if err_pct is not None else None,
        })

    # Summary table
    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'atoms':>6} {'real(ms)':>10} {'std':>8} {'pred E+F':>10} {'pred E':>10} {'error':>8}")
    print("-" * 78)
    for r in rows:
        err_str = f"{r['error_pct']:+.1f}%" if r["error_pct"] is not None else "N/A"
        print(f"{r['num_atoms']:>6} "
              f"{fmt(r['real_mean_ms']):>10} "
              f"{fmt(r['real_std_ms'], 3):>8} "
              f"{fmt(r['pred_force_e2e_ms']):>10} "
              f"{fmt(r['pred_energy_e2e_ms']):>10} "
              f"{err_str:>8}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "lialocl_accuracy_report.json"
    with open(out_path, "w") as f:
        json.dump({
            "system": "LiAlOCl",
            "config": str(CONFIG_PATH.relative_to(ROOT)),
            "device_config": str(DEVICE_CONFIG_PATH.relative_to(ROOT)),
            "host_config": str(HOST_CONFIG_PATH.relative_to(ROOT)),
            "gpu": torch.cuda.get_device_name(0),
            "box_size": args.box_size,
            "warmup": args.warmup,
            "runs": args.runs,
            "rows": rows,
        }, f, indent=2)
    print(f"\nReport written to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
