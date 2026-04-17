#!/usr/bin/env python3
"""
GPU Kernel Launch Overhead Micro-benchmark

测量目标 GPU 上的 kernel launch 开销，用于校准 DeepMD overhead 模型。

测量方法：
  1. 空 kernel launch cost: 在小 tensor 上执行最简单的操作
  2. 不同 kernel 类型的 launch cost: addmm, bmm, elementwise, copy
  3. 连续 launch 的平均 overhead (模拟 DeepMD 的 343 次连续 kernel)

用法:
    python scripts/benchmark_kernel_launch.py
"""

import torch
import time
import json
import os


def measure_elementwise_launch(num_trials=5000):
    """最简单的 elementwise kernel (add)"""
    x = torch.ones(1, device='cuda', dtype=torch.float64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_trials):
        x + x
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_trials * 1e6  # microseconds


def measure_addmm_launch(num_trials=2000):
    """Linear kernel (addmm) — small matrices"""
    a = torch.ones(1, 4, device='cuda', dtype=torch.float64)
    b = torch.ones(4, 4, device='cuda', dtype=torch.float64)
    bias = torch.ones(4, device='cuda', dtype=torch.float64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_trials):
        torch.addmm(bias, a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_trials * 1e6


def measure_bmm_launch(num_trials=2000):
    """BMM kernel — small matrices"""
    a = torch.ones(1, 4, 4, device='cuda', dtype=torch.float64)
    b = torch.ones(1, 4, 4, device='cuda', dtype=torch.float64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_trials):
        torch.bmm(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_trials * 1e6


def measure_copy_launch(num_trials=5000):
    """Memory copy kernel"""
    src = torch.ones(1, device='cuda', dtype=torch.float64)
    dst = torch.zeros(1, device='cuda', dtype=torch.float64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_trials):
        dst.copy_(src)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_trials * 1e6


def measure_mixed_chain(num_trials=500, chain_length=343):
    """
    模拟 DeepMD 推理中 343 次混合 kernel launch 的实际场景。
    交替执行不同类型的 kernel，测量平均 per-kernel overhead。
    """
    # 准备各种 tensor
    x_small = torch.ones(1, device='cuda', dtype=torch.float64)
    a_mm = torch.ones(4, 16, device='cuda', dtype=torch.float64)
    b_mm = torch.ones(16, 4, device='cuda', dtype=torch.float64)
    bias = torch.ones(4, device='cuda', dtype=torch.float64)
    a_bmm = torch.ones(2, 4, 4, device='cuda', dtype=torch.float64)
    b_bmm = torch.ones(2, 4, 4, device='cuda', dtype=torch.float64)
    x_vec = torch.ones(32, device='cuda', dtype=torch.float64)

    torch.cuda.synchronize()

    # 按 DeepMD 实际比例混合: ~35% elementwise, ~30% mm, ~15% bmm, ~20% misc
    n_elem = int(chain_length * 0.35)
    n_mm = int(chain_length * 0.30)
    n_bmm = int(chain_length * 0.15)
    n_misc = chain_length - n_elem - n_mm - n_bmm

    latencies = []
    for _ in range(num_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(n_elem):
            x_vec + x_vec
        for _ in range(n_mm):
            torch.addmm(bias, a_mm, b_mm)
        for _ in range(n_bmm):
            torch.bmm(a_bmm, b_bmm)
        for _ in range(n_misc):
            x_small.copy_(x_small)

        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1e6)  # microseconds

    import numpy as np
    avg_total = np.mean(latencies)
    per_kernel = avg_total / chain_length

    return per_kernel, avg_total


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Warmup
    print("Warming up...")
    x = torch.ones(100, device='cuda')
    for _ in range(1000):
        x + x
    torch.cuda.synchronize()

    print("=" * 60)
    print("Kernel Launch Cost Measurements")
    print("=" * 60)

    elem_us = measure_elementwise_launch()
    print(f"Elementwise (add 1-elem):        {elem_us:.1f} μs")

    addmm_us = measure_addmm_launch()
    print(f"Linear (addmm 1×4 @ 4×4):       {addmm_us:.1f} μs")

    bmm_us = measure_bmm_launch()
    print(f"BMM (1×4×4 @ 1×4×4):            {bmm_us:.1f} μs")

    copy_us = measure_copy_launch()
    print(f"Memory copy (1-elem):            {copy_us:.1f} μs")

    print()
    print("=" * 60)
    print(f"Mixed chain (simulating DeepMD's 343 kernel launches)")
    print("=" * 60)

    per_kernel_us, total_us = measure_mixed_chain()
    print(f"Per-kernel average:              {per_kernel_us:.1f} μs")
    print(f"Total chain (343 kernels):       {total_us:.0f} μs = {total_us/1000:.2f} ms")

    # 用这个值来估算 DeepMD 的 overhead
    print()
    print("=" * 60)
    print("校准结果")
    print("=" * 60)
    print(f"推荐 per-launch cost for {gpu_name}: {per_kernel_us:.1f} μs")
    print(f"343 kernels 的预期 overhead: {343 * per_kernel_us / 1000:.2f} ms")

    # 保存结果
    result = {
        "gpu": gpu_name,
        "pytorch_version": torch.__version__,
        "elementwise_launch_us": round(elem_us, 2),
        "addmm_launch_us": round(addmm_us, 2),
        "bmm_launch_us": round(bmm_us, 2),
        "copy_launch_us": round(copy_us, 2),
        "mixed_chain_per_kernel_us": round(per_kernel_us, 2),
        "mixed_chain_343_total_us": round(total_us, 2),
        "recommended_per_launch_us": round(per_kernel_us, 1),
    }

    os.makedirs("results/deepmd_benchmark", exist_ok=True)
    with open("results/deepmd_benchmark/kernel_launch_cost.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n保存到 results/deepmd_benchmark/kernel_launch_cost.json")


if __name__ == "__main__":
    main()
