#!/usr/bin/env python3
"""
超大原子数测试: 验证 power law 外推精度

测试 DeepMD 预测器在 8192 / 16384 / 32768 等超大原子数下的预测准确度。
power law 公式 unmodeled = alpha * N^beta 仅在 N=2048/4096 上拟合,
需要验证外推到更大 N 时是否仍然准确。

用法:
  # 纯预测模式 (不需要 DeepMD/GPU, 快速查看 power law 外推趋势)
  python scripts/test_large_atoms.py --predict-only

  # 完整对比模式 (需要 GPU + DeepMD)
  python scripts/test_large_atoms.py --full

  # 限制最大原子数 (避免 OOM 或太慢)
  python scripts/test_large_atoms.py --full --max_atoms 16384

  # 只测 water 模型
  python scripts/test_large_atoms.py --full --model water
"""

import sys
import os
import time
import json
import argparse
import subprocess
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 解析 Roofline 模型参数 (与 overhead_model.py v5 保持一致)
# ============================================================
C_QUAD = 28.3284
C_LINEAR = 2053.2321
REF_BW = 3430  # H100 NVL Mem_Bw (GB/s)
DEFAULT_NS = 27  # ghost cell factor


def roofline_predict(n, nnei=138, ns=DEFAULT_NS):
    """用解析 roofline 公式计算 GPU overhead (ms), H100 参考"""
    nall = ns * n
    BYTES = 8
    quad_ms = C_QUAD * n * nall * BYTES / (REF_BW * 1e9) * 1000
    linear_ms = C_LINEAR * n * nnei * BYTES / (REF_BW * 1e9) * 1000
    return quad_ms + linear_ms


# Backward compat alias
def power_law_predict(n):
    """Legacy power law — for comparison only"""
    return roofline_predict(n)


# ============================================================
# 预测: 调用 pred_deepmd.py
# ============================================================
def get_prediction(config_path, num_atoms, device_config_path, compute_force):
    """调用 pred_deepmd.py 获取预测值"""
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    for line in result.stdout.split("\n"):
        if "E2E latency" in line:
            parts = line.strip().split(":")
            s = parts[-1].strip()
            total = float(s.split(" ms")[0].strip().split()[-1])
            compute = None
            if "compute=" in s:
                compute = float(s.split("compute=")[1].split(",")[0])
            overhead = None
            if "overhead=" in s:
                overhead = float(s.split("overhead=")[1].split(")")[0])
            return {"total": total, "compute": compute, "overhead": overhead}

    # 如果解析失败, 打印 stderr 帮助调试
    if result.returncode != 0:
        print(f"  [ERROR] pred_deepmd.py failed (rc={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
    return None


# ============================================================
# 实测: 构建 DeepMD 模型并 profile
# ============================================================
def build_deepmd_model(config):
    """构建 DeepMD-kit 模型"""
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
    """生成随机测试数据, box_size 自适应原子数以保持合理密度"""
    import torch
    from deepmd.pt.utils.env import DEVICE

    coord = torch.rand(1, num_atoms, 3, dtype=torch.float64, device=DEVICE) * box_size
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, num_atoms), device=DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box_size
    box.requires_grad_(True)
    return coord, atype, box


def adaptive_box_size(num_atoms, base_box=20.0, base_atoms=192):
    """
    根据原子数自适应调整 box_size, 保持接近水的常温密度。
    密度 ∝ N / V = N / box^3 → box ∝ N^(1/3)
    """
    return base_box * (num_atoms / base_atoms) ** (1.0 / 3.0)


def adaptive_runs(num_atoms):
    """根据原子数自适应调整 warmup 和 runs"""
    if num_atoms >= 32768:
        return 2, 3
    elif num_atoms >= 16384:
        return 3, 5
    elif num_atoms >= 8192:
        return 5, 10
    elif num_atoms >= 4096:
        return 5, 15
    elif num_atoms >= 2048:
        return 10, 30
    else:
        return 20, 50


def profile_inference(model, coord, atype, box, num_warmup, num_runs):
    """Profile DeepMD inference with autograd (energy+force)"""
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


# ============================================================
# 测试配置
# ============================================================
TEST_CONFIGS = {
    "water": {
        "config_path": "scripts/asplos/data/deepmd_configs/water_se_e2_a.json",
        "atoms_list": [4096, 8192, 16384, 32768],
        "base_box": 20.0,
        "base_atoms": 192,
        "nnei": 138,  # sum(sel) = 46+92
        "label": "water_se_e2_a (sel=[46,92], 2 types)",
    },
    "copper": {
        "config_path": "scripts/asplos/data/deepmd_configs/copper_se_e2_a.json",
        "atoms_list": [2048, 4096, 8192, 16384],
        "base_box": 25.0,
        "base_atoms": 256,
        "nnei": 120,  # sum(sel) = 120
        "label": "copper_se_e2_a (sel=[120], 1 type)",
    },
}


def run_predict_only(models, max_atoms, device_config_path):
    """纯预测模式: 只调用 pred_deepmd.py, 无需 GPU/DeepMD"""
    print("=" * 80)
    print("超大原子数预测 (predict-only 模式)")
    print("=" * 80)
    print(f"Device config: {device_config_path}")
    print()

    all_results = {}

    for model_name in models:
        cfg = TEST_CONFIGS[model_name]
        atoms_list = [n for n in cfg["atoms_list"] if n <= max_atoms]

        print(f"\n{'─' * 70}")
        print(f"模型: {cfg['label']}")
        print(f"{'─' * 70}")

        # 表头
        print(f"\n| {'Atoms':>7} | {'预测 E+F (ms)':>14} | {'compute':>9} | {'overhead':>10} | "
              f"{'roofline_oh':>13} | {'mlp+roofline':>13} | {'regime':>14} |")
        print(f"|{'-' * 8}:|{'-' * 15}:|{'-' * 10}:|{'-' * 11}:|"
              f"{'-' * 14}:|{'-' * 14}:|{'-' * 15}:|")

        model_results = []
        for n in atoms_list:
            pred = get_prediction(cfg["config_path"], n, device_config_path, True)
            pw = roofline_predict(n)

            if pred:
                mlp_plus_unmod = (pred["compute"] or 0) + pw
                regime = "overhead" if pred["overhead"] and pred["overhead"] > pred.get("compute", 0) + pw else "compute"
                print(f"| {n:>7} | {pred['total']:>14.3f} | {pred['compute']:>9.3f} | "
                      f"{pred['overhead']:>10.3f} | {pw:>13.2f} | {mlp_plus_unmod:>13.2f} | {regime:>14} |")
                model_results.append({
                    "num_atoms": n,
                    "pred_total_ms": pred["total"],
                    "pred_compute_ms": pred["compute"],
                    "pred_overhead_ms": pred["overhead"],
                    "roofline_unmodeled_ms": round(pw, 3),
                })
            else:
                print(f"| {n:>7} | {'FAIL':>14} | {'-':>9} | {'-':>10} | {pw:>13.2f} | {'-':>10} | {'-':>14} |")
                model_results.append({
                    "num_atoms": n,
                    "pred_total_ms": None,
                    "error": "prediction failed",
                })

        all_results[model_name] = model_results

    # Analytical roofline 趋势分析
    print(f"\n\n{'=' * 70}")
    print(f"Roofline 模型外推趋势分析 (C_quad={C_QUAD}, C_linear={C_LINEAR})")
    print(f"{'=' * 70}")
    print(f"\n| {'N':>7} | {'gpu_oh (ms)':>12} | {'ratio vs 4096':>14} |")
    print(f"|{'-' * 8}:|{'-' * 13}:|{'-' * 15}:|")
    base_pw = roofline_predict(4096)
    for n in [4096, 8192, 16384, 32768, 65536]:
        if n > max_atoms * 2:
            break
        pw = roofline_predict(n)
        ratio = pw / base_pw
        print(f"| {n:>7} | {pw:>12.2f} | {ratio:>14.2f}x |")

    print(f"\n注: O(N²) 分量主导 → 每翻倍约增长 ~4x")

    return all_results


def run_full_test(models, max_atoms, device_config_path):
    """完整测试模式: 预测 + 实测对比"""
    import torch

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 80)
    print("超大原子数测试 (完整模式: 预测 vs 实测)")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    try:
        print(f"DeepMD-kit: {__import__('deepmd').__version__}")
    except ImportError:
        print("DeepMD-kit: NOT INSTALLED")
        print("请先安装 deepmd-kit, 或使用 --predict-only 模式")
        return None
    print(f"Device config: {device_config_path}")
    print()

    all_results = {}

    for model_name in models:
        cfg = TEST_CONFIGS[model_name]
        atoms_list = [n for n in cfg["atoms_list"] if n <= max_atoms]

        print(f"\n{'═' * 70}")
        print(f"模型: {cfg['label']}")
        print(f"{'═' * 70}")

        with open(cfg["config_path"]) as f:
            config = json.load(f)
        num_types = len(config.get("type_map", ["X"]))

        print(f"\n构建 DeepMD 模型...")
        model = build_deepmd_model(config)

        # 表头
        print(f"\n| {'Atoms':>7} | {'实测 (ms)':>10} | {'std':>7} | {'预测 (ms)':>10} | "
              f"{'compute':>9} | {'overhead':>10} | {'误差':>7} | {'roofline':>10} |")
        print(f"|{'-' * 8}:|{'-' * 11}:|{'-' * 8}:|{'-' * 11}:|"
              f"{'-' * 10}:|{'-' * 11}:|{'-' * 8}:|{'-' * 11}:|")

        model_results = []
        for n in atoms_list:
            # 自适应参数
            box_size = adaptive_box_size(n, cfg["base_box"], cfg["base_atoms"])
            warmup, runs = adaptive_runs(n)
            pw = roofline_predict(n)

            print(f"  Testing N={n} (box={box_size:.1f}, warmup={warmup}, runs={runs})...",
                  end="", flush=True)

            # --- 预测 ---
            pred = get_prediction(cfg["config_path"], n, device_config_path, True)

            # --- 实测 ---
            real_mean = real_std = None
            try:
                coord, atype, box = generate_test_data(n, num_types, box_size)
                latencies = profile_inference(model, coord, atype, box, warmup, runs)
                real_mean = np.mean(latencies)
                real_std = np.std(latencies)
                # 释放内存
                del coord, atype, box
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f" OOM!", flush=True)
                model_results.append({
                    "num_atoms": n,
                    "error": "CUDA OOM",
                    "pred_total_ms": pred["total"] if pred else None,
                })
                print(f"| {n:>7} | {'OOM':>10} | {'-':>7} | "
                      f"{pred['total'] if pred else 'N/A':>10} | {'-':>9} | {'-':>10} | {'OOM':>7} | {pw:>10.1f} |")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f" FAIL: {e}", flush=True)
                model_results.append({
                    "num_atoms": n,
                    "error": str(e),
                    "pred_total_ms": pred["total"] if pred else None,
                })
                print(f"| {n:>7} | {'FAIL':>10} | {'-':>7} | "
                      f"{pred['total'] if pred else 'N/A':>10} | {'-':>9} | {'-':>10} | {'FAIL':>7} | {pw:>10.1f} |")
                continue

            print(" done", flush=True)

            # --- 误差 ---
            err = None
            if pred and real_mean:
                err = (pred["total"] - real_mean) / real_mean * 100

            # 格式化输出
            err_str = f"{err:+.1f}%" if err is not None else "N/A"
            pred_str = f"{pred['total']:.3f}" if pred else "N/A"
            comp_str = f"{pred['compute']:.3f}" if pred and pred['compute'] else "N/A"
            ovh_str = f"{pred['overhead']:.3f}" if pred and pred['overhead'] else "N/A"

            print(f"| {n:>7} | {real_mean:>10.3f} | {real_std:>7.3f} | {pred_str:>10} | "
                  f"{comp_str:>9} | {ovh_str:>10} | {err_str:>7} | {pw:>10.1f} |")

            model_results.append({
                "num_atoms": n,
                "real_mean_ms": round(real_mean, 3) if real_mean else None,
                "real_std_ms": round(real_std, 3) if real_std else None,
                "pred_total_ms": pred["total"] if pred else None,
                "pred_compute_ms": pred["compute"] if pred else None,
                "pred_overhead_ms": pred["overhead"] if pred else None,
                "error_pct": round(err, 1) if err is not None else None,
                "roofline_unmodeled_ms": round(pw, 3),
                "box_size": round(box_size, 1),
            })

        all_results[model_name] = model_results

    # ==================== 误差统计 ====================
    print(f"\n\n{'=' * 70}")
    print("误差统计汇总")
    print(f"{'=' * 70}")

    all_errors = []
    for model_name, results in all_results.items():
        errors = [r["error_pct"] for r in results if r.get("error_pct") is not None]
        if errors:
            print(f"\n{model_name}:")
            print(f"  MAE: {np.mean(np.abs(errors)):.1f}%")
            print(f"  Max: {max(errors, key=abs):+.1f}%")
            print(f"  Min: {min(errors, key=abs):+.1f}%")
            all_errors.extend(errors)

    if all_errors:
        print(f"\n综合:")
        print(f"  MAE: {np.mean(np.abs(all_errors)):.1f}%")
        print(f"  Max: {max(all_errors, key=abs):+.1f}%")
        print(f"  Median: {np.median(all_errors):+.1f}%")

    # ==================== Roofline 分析 ====================
    print(f"\n\n{'=' * 70}")
    print("Roofline 模型外推精度分析")
    print(f"{'=' * 70}")
    print(f"公式: gpu_oh = C_quad×N×nall×8/bw + C_linear×N×nnei×8/bw")
    print(f"C_quad={C_QUAD}, C_linear={C_LINEAR}, REF_BW={REF_BW} GB/s")

    for model_name, results in all_results.items():
        valid = [r for r in results if r.get("real_mean_ms") and r.get("pred_total_ms")]
        if len(valid) >= 2:
            ns = [r["num_atoms"] for r in valid]
            reals = [r["real_mean_ms"] for r in valid]
            preds = [r["pred_total_ms"] for r in valid]

            # 计算实际 scaling exponent (从最小到最大)
            if len(valid) >= 2:
                n1, n2 = ns[0], ns[-1]
                r1, r2 = reals[0], reals[-1]
                actual_beta = math.log(r2 / r1) / math.log(n2 / n1) if r1 > 0 and n1 > 0 else 0
                print(f"\n{model_name}:")
                print(f"  实测 scaling exponent (N={n1}→{n2}): {actual_beta:.3f}")
                print(f"  Roofline 模型: O(N²) 主导 → 理论 exponent ≈ 2.0")
                if actual_beta > 2.1:
                    print(f"  ⚠ 实测超二次方增长 — 可能有内存瓶颈")
                elif actual_beta < 1.7:
                    print(f"  ⚠ 实测低于预期 — GPU 并行化可能降低了有效指数")
                else:
                    print(f"  ✅ 实测与 roofline 模型 O(N²) 预期一致")

    # ==================== 保存 JSON ====================
    os.makedirs("results/deepmd_fulltest", exist_ok=True)
    report = {
        "test_type": "large_atoms",
        "gpu": gpu_name if 'gpu_name' in dir() else "unknown",
        "max_atoms": max_atoms,
        "roofline_model": {"C_quad": C_QUAD, "C_linear": C_LINEAR, "REF_BW": REF_BW},
        "results": all_results,
        "error_stats": {
            "overall_mae_pct": round(np.mean(np.abs(all_errors)), 1) if all_errors else None,
            "overall_max_err_pct": round(max(all_errors, key=abs), 1) if all_errors else None,
        },
    }
    report_path = "results/deepmd_fulltest/large_atoms_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n结果已保存到 {report_path}")

    return all_results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="超大原子数测试 — 验证 power law 外推精度"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predict-only", action="store_true",
                      help="纯预测模式 (不需要 DeepMD/GPU)")
    mode.add_argument("--full", action="store_true",
                      help="完整模式 (预测 + 实测对比, 需要 GPU + DeepMD)")

    parser.add_argument("--max_atoms", type=int, default=32768,
                        help="最大测试原子数 (default: 32768)")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "water", "copper"],
                        help="测试模型 (default: all)")
    parser.add_argument("--device_config", type=str,
                        default="scripts/asplos/data/device_configs/NVIDIA_H100_80GB_HBM3.json",
                        help="GPU device config path")

    args = parser.parse_args()

    models = list(TEST_CONFIGS.keys()) if args.model == "all" else [args.model]

    if args.predict_only:
        run_predict_only(models, args.max_atoms, args.device_config)
    else:
        run_full_test(models, args.max_atoms, args.device_config)


if __name__ == "__main__":
    main()
