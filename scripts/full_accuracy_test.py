#!/usr/bin/env python3
"""
全面精度测试: 多原子数 × energy-only / energy+force

在实际 GPU 上跑 DeepMD 推理，同时收集 NeuSight 预测值，
输出 Markdown 表格 + JSON。
"""

import sys
import os
import time
import json
import subprocess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_deepmd_model(config):
    import torch
    from deepmd.pt.model.model import get_model
    from deepmd.pt.utils.env import DEVICE

    desc_cfg = config["descriptor"]
    fit_cfg = config["fitting_net"]
    type_map = config.get("type_map", ["O", "H"])

    model_params = {
        "type_map": type_map,
        "descriptor": {
            "type": desc_cfg.get("type", "se_e2_a"),
            "sel": desc_cfg["sel"],
            "rcut": desc_cfg.get("rcut", 6.0),
            "rcut_smth": desc_cfg.get("rcut_smth", 0.5),
            "neuron": desc_cfg["neuron"],
            "axis_neuron": desc_cfg.get("axis_neuron", 16),
            "resnet_dt": False,
            "type_one_side": True,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": fit_cfg["neuron"],
            "resnet_dt": True,
        },
    }

    model = get_model(model_params)
    model = model.to(DEVICE)
    model.eval()
    return model


def generate_test_data(num_atoms, num_types, box_size=20.0):
    import torch
    from deepmd.pt.utils.env import DEVICE

    coord = torch.rand(1, num_atoms, 3, dtype=torch.float64, device=DEVICE) * box_size
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, num_atoms), device=DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box_size
    box.requires_grad_(True)
    return coord, atype, box


def profile_inference(model, coord, atype, box, num_warmup=30, num_runs=100):
    """Profile with autograd enabled (DeepMD computes force by default)"""
    import torch

    for _ in range(num_warmup):
        _ = model(coord, atype, box)
        if coord.grad is not None:
            coord.grad.zero_()
        if box.grad is not None:
            box.grad.zero_()

    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_runs):
        if coord.grad is not None:
            coord.grad.zero_()
        if box.grad is not None:
            box.grad.zero_()

        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(coord, atype, box)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return latencies


def get_prediction(config_path, num_atoms, device_config_path, compute_force):
    cmd = [
        sys.executable, "scripts/pred_deepmd.py",
        "--predictor_path", "scripts/asplos/data/predictor/MLP_WAVE",
        "--device_config_path", device_config_path,
        "--deepmd_config_path", config_path,
        "--num_atoms", str(num_atoms),
        "--tile_dataset_dir", "scripts/asplos/data/dataset/train",
        "--result_dir", "results/deepmd_fulltest/",
    ]
    if compute_force:
        cmd.append("--compute_force")
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "E2E latency" in line:
            parts = line.strip().split(":")
            lat_str = parts[-1].strip()
            # 提取总 latency 和 compute/overhead
            total_str = lat_str.split(" ms")[0].strip().split()[-1]
            total = float(total_str)
            # 提取 compute
            compute = None
            if "compute=" in lat_str:
                compute = float(lat_str.split("compute=")[1].split(",")[0])
            overhead = None
            if "overhead=" in lat_str:
                overhead = float(lat_str.split("overhead=")[1].split(")")[0])
            return {"total": total, "compute": compute, "overhead": overhead}
    return None


def main():
    import torch

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"DeepMD-kit: {__import__('deepmd').__version__}")
    print(f"PyTorch: {torch.__version__}")
    print()

    config_path = "scripts/asplos/data/deepmd_configs/water_se_e2_a.json"
    device_config_path = "scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json"

    with open(config_path) as f:
        config = json.load(f)

    type_map = config.get("type_map", ["O", "H"])
    num_types = len(type_map)

    # 测试的原子数列表
    atoms_list = [32, 64, 128, 192, 256, 512, 1024, 2048]

    print("构建 DeepMD 模型...")
    model = build_deepmd_model(config)

    results = []

    for num_atoms in atoms_list:
        print(f"\n{'='*60}")
        print(f"Testing {num_atoms} atoms...")
        print(f"{'='*60}")

        # --- Ground truth profiling ---
        coord, atype, box = generate_test_data(num_atoms, num_types)

        # 根据原子数调整 runs 数量（大原子数跑少一点）
        if num_atoms >= 2048:
            warmup, runs = 10, 30
        elif num_atoms >= 1024:
            warmup, runs = 20, 50
        else:
            warmup, runs = 30, 100

        print(f"  Profiling (warmup={warmup}, runs={runs})...")
        try:
            latencies = profile_inference(model, coord, atype, box, warmup, runs)
            real_mean = np.mean(latencies)
            real_median = np.median(latencies)
            real_std = np.std(latencies)
            real_p5 = np.percentile(latencies, 5)
            real_p95 = np.percentile(latencies, 95)
            print(f"  实测: mean={real_mean:.3f}ms  median={real_median:.3f}ms  "
                  f"std={real_std:.3f}ms  [p5={real_p5:.3f}, p95={real_p95:.3f}]")
        except Exception as e:
            print(f"  Profiling 失败: {e}")
            real_mean = real_median = real_std = real_p5 = real_p95 = None

        # --- NeuSight 预测 ---
        pred_force = get_prediction(config_path, num_atoms, device_config_path, True)
        pred_energy = get_prediction(config_path, num_atoms, device_config_path, False)

        if pred_force:
            print(f"  预测(E+F): total={pred_force['total']:.3f}ms  "
                  f"compute={pred_force['compute']:.3f}ms  overhead={pred_force['overhead']:.3f}ms")
        if pred_energy:
            print(f"  预测(E):   total={pred_energy['total']:.3f}ms  "
                  f"compute={pred_energy['compute']:.3f}ms  overhead={pred_energy['overhead']:.3f}ms")

        # --- 误差 ---
        err_force = None
        if real_mean and pred_force:
            err_force = (pred_force["total"] - real_mean) / real_mean * 100
            print(f"  误差(E+F vs 实测): {err_force:+.1f}%")

        results.append({
            "num_atoms": num_atoms,
            "real_mean_ms": round(real_mean, 3) if real_mean else None,
            "real_median_ms": round(real_median, 3) if real_median else None,
            "real_std_ms": round(real_std, 3) if real_std else None,
            "real_p5_ms": round(real_p5, 3) if real_p5 else None,
            "real_p95_ms": round(real_p95, 3) if real_p95 else None,
            "pred_force_total_ms": pred_force["total"] if pred_force else None,
            "pred_force_compute_ms": pred_force["compute"] if pred_force else None,
            "pred_force_overhead_ms": pred_force["overhead"] if pred_force else None,
            "pred_energy_total_ms": pred_energy["total"] if pred_energy else None,
            "pred_energy_compute_ms": pred_energy["compute"] if pred_energy else None,
            "error_force_pct": round(err_force, 1) if err_force is not None else None,
        })

    # ==================== 输出 Markdown 表格 ====================
    print(f"\n\n{'='*90}")
    print("精度验证汇总表")
    print(f"{'='*90}")
    print(f"GPU: {gpu_name}")
    print(f"模型: water_se_e2_a (sel=[46,92], emb=[25,50,100], fit=[240,240,240])")
    print(f"DeepMD 默认计算 energy + force (autograd)")
    print()

    # Table 1: 主表
    print("### 预测 vs 实测 (energy+force)")
    print()
    print("| Atoms | 实测 mean (ms) | 实测 std | 预测 E+F (ms) | compute | overhead | 误差 |")
    print("|------:|---------------:|---------:|--------------:|--------:|---------:|-----:|")
    for r in results:
        real = f"{r['real_mean_ms']:.3f}" if r['real_mean_ms'] else "N/A"
        std = f"{r['real_std_ms']:.3f}" if r['real_std_ms'] else "N/A"
        pred = f"{r['pred_force_total_ms']:.3f}" if r['pred_force_total_ms'] else "N/A"
        comp = f"{r['pred_force_compute_ms']:.3f}" if r['pred_force_compute_ms'] else "N/A"
        ovh = f"{r['pred_force_overhead_ms']:.3f}" if r['pred_force_overhead_ms'] else "N/A"
        err = f"{r['error_force_pct']:+.1f}%" if r['error_force_pct'] is not None else "N/A"
        print(f"| {r['num_atoms']:>5} | {real:>14} | {std:>8} | {pred:>13} | {comp:>7} | {ovh:>8} | {err:>5} |")

    # Table 2: energy-only 预测
    print()
    print("### Energy-only 预测参考")
    print()
    print("| Atoms | 预测 E-only (ms) | compute | overhead |")
    print("|------:|-----------------:|--------:|---------:|")
    for r in results:
        pred = f"{r['pred_energy_total_ms']:.3f}" if r['pred_energy_total_ms'] else "N/A"
        comp = f"{r['pred_energy_compute_ms']:.3f}" if r['pred_energy_compute_ms'] else "N/A"
        ovh_val = None
        if r['pred_energy_total_ms'] and r['pred_energy_compute_ms']:
            ovh_val = r['pred_energy_total_ms'] - r['pred_energy_compute_ms']
        ovh = f"{ovh_val:.3f}" if ovh_val else "N/A"
        print(f"| {r['num_atoms']:>5} | {pred:>16} | {comp:>7} | {ovh:>8} |")

    # 统计
    errors = [r['error_force_pct'] for r in results if r['error_force_pct'] is not None]
    if errors:
        print()
        print(f"### 误差统计")
        print(f"- 平均绝对误差 (MAE): {np.mean(np.abs(errors)):.1f}%")
        print(f"- 最大误差: {max(errors, key=abs):+.1f}%")
        print(f"- 最小误差: {min(errors, key=abs):+.1f}%")
        print(f"- 中位误差: {np.median(errors):+.1f}%")

    # 保存 JSON
    os.makedirs("results/deepmd_fulltest", exist_ok=True)
    report = {
        "gpu": gpu_name,
        "model": "water_se_e2_a",
        "config": {
            "sel": config["descriptor"]["sel"],
            "emb_neuron": config["descriptor"]["neuron"],
            "fit_neuron": config["fitting_net"]["neuron"],
        },
        "results": results,
        "error_stats": {
            "mae_pct": round(np.mean(np.abs(errors)), 1) if errors else None,
            "max_err_pct": round(max(errors, key=abs), 1) if errors else None,
            "min_err_pct": round(min(errors, key=abs), 1) if errors else None,
            "median_err_pct": round(float(np.median(errors)), 1) if errors else None,
        },
    }
    with open("results/deepmd_fulltest/full_accuracy_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n结果已保存到 results/deepmd_fulltest/full_accuracy_report.json")


if __name__ == "__main__":
    main()
