#!/usr/bin/env python3
"""
补充测试: copper 模型 + 大原子数
"""

import sys, os, time, json, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_model(config):
    import torch
    from deepmd.pt.model.model import get_model
    from deepmd.pt.utils.env import DEVICE
    desc_cfg = config["descriptor"]
    fit_cfg = config["fitting_net"]
    model_params = {
        "type_map": config.get("type_map", ["Cu"]),
        "descriptor": {
            "type": desc_cfg.get("type", "se_e2_a"),
            "sel": desc_cfg["sel"],
            "rcut": desc_cfg.get("rcut", 7.0),
            "rcut_smth": desc_cfg.get("rcut_smth", 0.5),
            "neuron": desc_cfg["neuron"],
            "axis_neuron": desc_cfg.get("axis_neuron", 16),
            "resnet_dt": False, "type_one_side": True,
        },
        "fitting_net": {"type": "ener", "neuron": fit_cfg["neuron"], "resnet_dt": True},
    }
    model = get_model(model_params)
    model = model.to(DEVICE).eval()
    return model


def gen_data(num_atoms, num_types, box_size=25.0):
    import torch
    from deepmd.pt.utils.env import DEVICE
    coord = torch.rand(1, num_atoms, 3, dtype=torch.float64, device=DEVICE) * box_size
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, num_atoms), device=DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box_size
    box.requires_grad_(True)
    return coord, atype, box


def profile(model, coord, atype, box, warmup=20, runs=50):
    import torch
    for _ in range(warmup):
        _ = model(coord, atype, box)
        if coord.grad is not None: coord.grad.zero_()
        if box.grad is not None: box.grad.zero_()
    torch.cuda.synchronize()

    lats = []
    for _ in range(runs):
        if coord.grad is not None: coord.grad.zero_()
        if box.grad is not None: box.grad.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(coord, atype, box)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
    return lats


def get_pred(config_path, num_atoms, device_cfg, force):
    cmd = [sys.executable, "scripts/pred_deepmd.py",
           "--predictor_path", "scripts/asplos/data/predictor/MLP_WAVE",
           "--device_config_path", device_cfg,
           "--deepmd_config_path", config_path,
           "--num_atoms", str(num_atoms),
           "--tile_dataset_dir", "scripts/asplos/data/dataset/train",
           "--result_dir", "results/deepmd_fulltest/"]
    if force: cmd.append("--compute_force")
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.split("\n"):
        if "E2E latency" in line:
            parts = line.strip().split(":")
            s = parts[-1].strip()
            total = float(s.split(" ms")[0].strip().split()[-1])
            compute = float(s.split("compute=")[1].split(",")[0]) if "compute=" in s else None
            return {"total": total, "compute": compute}
    return None


def main():
    import torch
    gpu = torch.cuda.get_device_name(0)
    dcfg = "scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json"

    tests = [
        ("copper_se_e2_a", "scripts/asplos/data/deepmd_configs/copper_se_e2_a.json",
         [64, 128, 256, 512, 1024]),
    ]

    for model_name, config_path, atoms_list in tests:
        with open(config_path) as f:
            config = json.load(f)
        nt = len(config.get("type_map", ["Cu"]))

        print(f"\n{'='*70}")
        print(f"模型: {model_name}  (types={config.get('type_map')}, sel={config['descriptor']['sel']})")
        print(f"{'='*70}")

        model = build_model(config)

        print(f"\n| Atoms | 实测 (ms) | std | 预测 E+F | compute | 误差 |")
        print(f"|------:|----------:|----:|---------:|--------:|-----:|")

        for n in atoms_list:
            coord, atype, box = gen_data(n, nt)
            w, r = (10, 20) if n >= 1024 else (20, 50)
            try:
                lats = profile(model, coord, atype, box, w, r)
                real = np.mean(lats)
                std = np.std(lats)
            except Exception as e:
                print(f"| {n:>5} | FAIL | - | - | - | {e} |")
                continue

            pred = get_pred(config_path, n, dcfg, True)
            if pred:
                err = (pred["total"] - real) / real * 100
                print(f"| {n:>5} | {real:>9.3f} | {std:.3f} | {pred['total']:>8.3f} | {pred['compute']:>7.3f} | {err:+.1f}% |")
            else:
                print(f"| {n:>5} | {real:>9.3f} | {std:.3f} | N/A | N/A | N/A |")

    # 补充 water 大原子数 (4096)
    print(f"\n{'='*70}")
    print(f"补充: water_se_e2_a 大原子数")
    print(f"{'='*70}")
    wconfig_path = "scripts/asplos/data/deepmd_configs/water_se_e2_a.json"
    with open(wconfig_path) as f:
        wconfig = json.load(f)
    wmodel = build_model(wconfig)
    wnt = len(wconfig.get("type_map", ["O", "H"]))

    print(f"\n| Atoms | 实测 (ms) | std | 预测 E+F | compute | 误差 |")
    print(f"|------:|----------:|----:|---------:|--------:|-----:|")
    for n in [4096]:
        coord, atype, box = gen_data(n, wnt, box_size=40.0)
        try:
            lats = profile(wmodel, coord, atype, box, 5, 15)
            real = np.mean(lats)
            std = np.std(lats)
        except Exception as e:
            print(f"| {n:>5} | FAIL | {e}")
            continue
        pred = get_pred(wconfig_path, n, dcfg, True)
        if pred:
            err = (pred["total"] - real) / real * 100
            print(f"| {n:>5} | {real:>9.3f} | {std:.3f} | {pred['total']:>8.3f} | {pred['compute']:>7.3f} | {err:+.1f}% |")


if __name__ == "__main__":
    main()
