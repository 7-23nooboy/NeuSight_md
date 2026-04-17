#!/usr/bin/env python3
"""
DeepMD 预测精度验证: 真实 GPU profiling vs NeuSight 预测

在实际 GPU 上运行 DeepMD 推理, 测量 wall-time latency,
和 NeuSight 的预测值做逐阶段对比。

用法:
    python scripts/benchmark_deepmd_accuracy.py

输出: 表格 + 总误差分析
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_deepmd_model(config, num_atoms):
    """用 DeepMD-kit API 构建一个真实模型并生成测试数据"""
    import torch
    from deepmd.pt.model.model import get_model
    from deepmd.pt.utils.env import DEVICE

    # 构造 DeepMD 完整的 model_params
    desc_cfg = config["descriptor"]
    fit_cfg = config["fitting_net"]
    type_map = config.get("type_map", ["O", "H"])
    num_types = len(type_map)

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
    """生成随机测试数据"""
    import torch
    from deepmd.pt.utils.env import DEVICE

    # 随机坐标 — requires_grad=True 因为 DeepMD 需要计算 force = dE/dcoord
    coord = torch.rand(1, num_atoms, 3, dtype=torch.float64, device=DEVICE) * box_size
    coord.requires_grad_(True)

    # 随机原子类型 (均匀分布)
    atype = torch.randint(0, num_types, (1, num_atoms), device=DEVICE)

    # 正交 box
    box = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box_size
    box.requires_grad_(True)

    return coord, atype, box


def profile_deepmd_inference(model, coord, atype, box, num_warmup=50, num_runs=200):
    """
    Profile DeepMD 推理 latency。

    注意: DeepMD 默认计算 force (通过 autograd)，不能用 torch.no_grad()。
    每次推理后需要清零梯度。
    """
    import torch

    # Warmup (不用 no_grad，因为 DeepMD 内部需要 autograd 计算 force)
    for _ in range(num_warmup):
        _ = model(coord, atype, box)
        if coord.grad is not None:
            coord.grad.zero_()
        if box.grad is not None:
            box.grad.zero_()

    torch.cuda.synchronize()

    # Timed runs
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
        latencies.append((end - start) * 1000)  # ms

    return latencies


def profile_with_torch_profiler(model, coord, atype, box):
    """用 torch.profiler 获取逐 kernel 的时间"""
    import torch
    from torch.profiler import profile, ProfilerActivity

    with torch.no_grad():
        # warmup
        for _ in range(20):
            model(coord, atype, box)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            model(coord, atype, box)
            torch.cuda.synchronize()

    return prof


def get_neusight_prediction(config, num_atoms, device_config_path):
    """获取 NeuSight 预测值"""
    import neusight

    predictor = neusight.DeepMDPredictor(
        predictor_path="scripts/asplos/data/predictor/MLP_WAVE",
        tile_dataset_dir="scripts/asplos/data/dataset/train",
    )
    result = predictor.predict(
        device_config_path=device_config_path,
        deepmd_config_path=None,  # 直接传 config dict
        deepmd_config=config,
        num_atoms=num_atoms,
        result_dir="results/deepmd_benchmark/",
        compute_force=False,
    )
    return result


def get_neusight_prediction_via_cli(config_path, num_atoms, device_config_path,
                                    compute_force=False):
    """通过已有 CLI 获取 NeuSight 预测值"""
    import subprocess
    cmd = [
        sys.executable, "scripts/pred_deepmd.py",
        "--predictor_path", "scripts/asplos/data/predictor/MLP_WAVE",
        "--device_config_path", device_config_path,
        "--deepmd_config_path", config_path,
        "--num_atoms", str(num_atoms),
        "--tile_dataset_dir", "scripts/asplos/data/dataset/train",
        "--result_dir", "results/deepmd_benchmark/",
    ]
    if compute_force:
        cmd.append("--compute_force")
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 从输出中解析 latency
    for line in result.stdout.split("\n"):
        if "E2E latency" in line:
            # "DeepMD E2E latency for xxx: 6.7117 ms (compute=0.6621, overhead=6.0496)"
            # 或旧格式 "DeepMD E2E latency for xxx: 0.4157 ms"
            parts = line.strip().split(":")
            lat_str = parts[-1].strip()
            # 提取第一个数字 (总 latency)
            lat_str = lat_str.split(" ms")[0].strip().split()[-1]
            return float(lat_str)
    print("Warning: could not parse prediction output:")
    print(result.stdout)
    print(result.stderr)
    return None


def main():
    import torch

    print("=" * 70)
    print("DeepMD 预测精度验证: 真实 Profiling vs NeuSight 预测")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"DeepMD-kit: {__import__('deepmd').__version__}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # 测试配置
    configs = [
        {
            "name": "water_se_e2_a",
            "config_path": "scripts/asplos/data/deepmd_configs/water_se_e2_a.json",
            "atoms_list": [64, 192, 512, 1024],
        },
    ]

    # H100 NVL 最接近的 device config
    # 注意: 我们的 config 是 H100 80GB HBM3, 机器上是 H100 NVL 95GB
    device_config_path = "scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json"

    results = []

    for cfg_info in configs:
        with open(cfg_info["config_path"]) as f:
            config = json.load(f)

        print(f"\n{'='*70}")
        print(f"模型: {cfg_info['name']}")
        print(f"Descriptor: sel={config['descriptor']['sel']}, "
              f"neuron={config['descriptor']['neuron']}")
        print(f"Fitting: neuron={config['fitting_net']['neuron']}")
        print(f"{'='*70}")

        type_map = config.get("type_map", ["O", "H"])
        num_types = len(type_map)

        # 构建模型 (只需一次)
        print("\n构建 DeepMD 模型...")
        model = build_deepmd_model(config, cfg_info["atoms_list"][0])

        for num_atoms in cfg_info["atoms_list"]:
            print(f"\n--- {num_atoms} atoms ---")

            # 1. 真实 profiling
            print(f"  生成测试数据...")
            coord, atype, box = generate_test_data(num_atoms, num_types)

            print(f"  Profiling (warmup=50, runs=200)...")
            try:
                latencies = profile_deepmd_inference(model, coord, atype, box)
                real_mean = np.mean(latencies)
                real_median = np.median(latencies)
                real_std = np.std(latencies)
                real_p99 = np.percentile(latencies, 99)
                print(f"  真实 latency: mean={real_mean:.4f} ms, "
                      f"median={real_median:.4f} ms, std={real_std:.4f} ms")
            except Exception as e:
                print(f"  Profiling 失败: {e}")
                real_mean = real_median = real_std = real_p99 = None

            # 2. NeuSight 预测 (energy+force，因为 DeepMD 默认计算 force)
            print(f"  NeuSight 预测 (energy+force)...")
            predicted_force = get_neusight_prediction_via_cli(
                cfg_info["config_path"], num_atoms, device_config_path,
                compute_force=True,
            )
            if predicted_force is not None:
                print(f"  预测 latency (energy+force): {predicted_force:.4f} ms")

            print(f"  NeuSight 预测 (energy-only)...")
            predicted_energy = get_neusight_prediction_via_cli(
                cfg_info["config_path"], num_atoms, device_config_path,
                compute_force=False,
            )
            if predicted_energy is not None:
                print(f"  预测 latency (energy-only):  {predicted_energy:.4f} ms")

            # 3. 计算误差 (用 energy+force 对比实测，因为实测包含 force)
            predicted = predicted_force
            if real_mean is not None and predicted is not None:
                error_pct = (predicted - real_mean) / real_mean * 100
                print(f"  误差: {error_pct:+.1f}% (energy+force预测 vs 实测均值)")
            else:
                error_pct = None

            results.append({
                "model": cfg_info["name"],
                "num_atoms": num_atoms,
                "real_mean_ms": round(real_mean, 4) if real_mean else None,
                "real_median_ms": round(real_median, 4) if real_median else None,
                "real_std_ms": round(real_std, 4) if real_std else None,
                "predicted_force_ms": round(predicted_force, 4) if predicted_force else None,
                "predicted_energy_ms": round(predicted_energy, 4) if predicted_energy else None,
                "error_pct": round(error_pct, 1) if error_pct is not None else None,
            })

    # 汇总表格
    print(f"\n\n{'='*70}")
    print("汇总结果")
    print(f"{'='*70}")
    print(f"{'模型':<20} {'atoms':>6} {'实测(ms)':>10} {'预测E+F':>10} {'预测E':>10} {'误差':>8}")
    print(f"{'-'*70}")
    for r in results:
        real = f"{r['real_mean_ms']:.4f}" if r['real_mean_ms'] else "N/A"
        pred_f = f"{r['predicted_force_ms']:.4f}" if r.get('predicted_force_ms') else "N/A"
        pred_e = f"{r['predicted_energy_ms']:.4f}" if r.get('predicted_energy_ms') else "N/A"
        err = f"{r['error_pct']:+.1f}%" if r['error_pct'] is not None else "N/A"
        print(f"{r['model']:<20} {r['num_atoms']:>6} {real:>10} {pred_f:>10} {pred_e:>10} {err:>8}")

    # 保存结果
    os.makedirs("results/deepmd_benchmark", exist_ok=True)
    with open("results/deepmd_benchmark/accuracy_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 results/deepmd_benchmark/accuracy_report.json")


if __name__ == "__main__":
    main()
