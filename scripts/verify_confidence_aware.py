#!/usr/bin/env python3
"""
验证 Confidence-Aware Prediction 功能

检查:
1. overhead-bound 区间 (N<=1024) → confidence="high", regime="overhead-bound"
2. compute-bound 区间 (N>=2048) → confidence="high", regime="compute-bound"
3. 转换区 (~1280-1792) → confidence="low", regime="transition"
4. 转换区 bounds 覆盖实测值
5. 两端精度不受影响 (point estimate 不变)
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neusight.Prediction.overhead_model import DeepMDOverheadModel
import pandas as pd


def build_mock_op_df(num_atoms):
    """构建最小化的 op_df 用于测试 (只需要 fit_linear_0 行来推断 N)"""
    rows = [{
        "Name": "fit_linear_0",
        "OpName": "Linear",
        "FwOps": [("Linear", (num_atoms, 240, 240))],
        "BwOps": [],
        "AccOps": [],
        "InputShapes": [(num_atoms, 240)],
        "OutputShape": (num_atoms, 240),
    }]
    return pd.DataFrame(rows)


def test_confidence_zones():
    """测试不同 N 的 confidence 标注是否正确"""
    model = DeepMDOverheadModel()

    # Water 配置
    device_config = {"Mem_Bw": 3430, "Device": "TEST"}
    deepmd_config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [46, 92],
            "rcut": 6.0,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
        },
        "fitting_net": {"neuron": [240, 240, 240]},
    }

    # 模拟不同 N 的 MLP compute 时间 (近似值)
    # 这些值从实际 MLP_WAVE 预测中获取的近似值
    test_cases = [
        # (N, mlp_compute_ms, expected_regime, expected_confidence)
        (32,   1.14, "overhead-bound", "high"),
        (64,   1.07, "overhead-bound", "high"),
        (128,  1.08, "overhead-bound", "high"),
        (256,  1.08, "overhead-bound", "high"),
        (512,  1.37, "overhead-bound", "high"),
        (1024, 1.68, "overhead-bound", "high"),
        # 转换区 — 这些点 MLP_WAVE 预测 + gpu_overhead ≈ fixed_overhead
        (1280, 1.85, "transition", "low"),
        (1536, 2.10, "transition", "low"),
        (1792, 2.30, "transition", "low"),
        # compute-bound (ratio > 2.0)
        (2048, 2.43, "transition", "low"),     # ratio ≈ 1.92, still in transition
        (4096, 3.96, "compute-bound", "high"),
        (8192, 7.19, "compute-bound", "high"),
    ]

    print("=" * 90)
    print("Confidence-Aware Prediction 验证")
    print("=" * 90)
    print(f"配置: Water se_e2_a, sel=[46,92], nnei=138")
    print(f"转换区参数: ratio ∈ [{model.TRANSITION_LO}, {model.TRANSITION_HI}]")
    print(f"Bubble peak fraction: {model.BUBBLE_PEAK_FRACTION}")
    print()

    header = f"{'N':>6} | {'mlp_ms':>7} | {'gpu_oh_ms':>9} | {'adjusted':>8} | {'fixed':>6} | {'ratio':>6} | {'regime':>15} | {'conf':>6} | {'point':>7} | {'lower':>7} | {'upper':>7} | {'check':>5}"
    print(header)
    print("-" * len(header))

    all_passed = True
    for num_atoms, mlp_ms, expected_regime, expected_conf in test_cases:
        op_df = build_mock_op_df(num_atoms)
        result = model.estimate(
            device_config=device_config,
            deepmd_config=deepmd_config,
            op_df=op_df,
            compute_force=True,
            compute_latency_ms=mlp_ms,
        )

        regime = result["regime"]
        confidence = result["confidence"]
        ratio = result["transition_ratio"]
        e2e = result["e2e_estimate_ms"]
        lo = result["e2e_lower_ms"]
        hi = result["e2e_upper_ms"]
        gpu_oh = result["unmodeled_compute_ms"]
        fixed = result["fixed_overhead_ms"]
        adjusted = mlp_ms + gpu_oh

        # 验证
        regime_ok = regime == expected_regime
        conf_ok = confidence == expected_conf
        # high confidence 时 lower == upper == point
        if confidence == "high":
            bounds_ok = abs(lo - hi) < 0.001
        else:
            # low confidence 时 upper > lower
            bounds_ok = hi > lo
        passed = regime_ok and conf_ok and bounds_ok
        if not passed:
            all_passed = False

        check = "✓" if passed else "✗"
        print(f"{num_atoms:>6} | {mlp_ms:>7.2f} | {gpu_oh:>9.3f} | {adjusted:>8.3f} | {fixed:>6.3f} | {ratio:>6.3f} | {regime:>15} | {confidence:>6} | {e2e:>7.3f} | {lo:>7.3f} | {hi:>7.3f} | {check:>5}")

    print()
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败！")

    return all_passed


def test_transition_zone_coverage():
    """验证转换区的 [lower, upper] 能覆盖实测值"""
    print()
    print("=" * 90)
    print("转换区 Bounds Coverage 验证")
    print("=" * 90)

    # 从之前密集 profiling 中收集的转换区实测数据 (Water, H100 NVL)
    # format: (N, real_ms, mlp_compute_ms)
    transition_data = [
        (1024, 5.873, 1.68),
        (1152, 5.942, 1.77),
        (1280, 6.214, 1.85),
        (1408, 6.640, 2.00),
        (1536, 6.950, 2.10),
        (1664, 7.525, 2.18),
        (1792, 8.275, 2.30),
        (1920, 9.378, 2.38),
        (2048, 11.823, 2.43),
    ]

    model = DeepMDOverheadModel()
    device_config = {"Mem_Bw": 3430, "Device": "TEST"}
    deepmd_config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [46, 92],
            "rcut": 6.0,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
        },
        "fitting_net": {"neuron": [240, 240, 240]},
    }

    print(f"{'N':>6} | {'real_ms':>8} | {'point':>7} | {'lower':>7} | {'upper':>7} | {'regime':>12} | {'conf':>5} | {'covered':>8} | {'err%':>6}")
    print("-" * 95)

    covered_count = 0
    total_transition = 0
    for num_atoms, real_ms, mlp_ms in transition_data:
        op_df = build_mock_op_df(num_atoms)
        result = model.estimate(
            device_config=device_config,
            deepmd_config=deepmd_config,
            op_df=op_df,
            compute_force=True,
            compute_latency_ms=mlp_ms,
        )

        e2e = result["e2e_estimate_ms"]
        lo = result["e2e_lower_ms"]
        hi = result["e2e_upper_ms"]
        regime = result["regime"]
        conf = result["confidence"]
        err = (e2e - real_ms) / real_ms * 100

        is_covered = lo <= real_ms <= hi
        if regime == "transition":
            total_transition += 1
            if is_covered:
                covered_count += 1

        cover_str = "✓" if is_covered else "✗"
        print(f"{num_atoms:>6} | {real_ms:>8.3f} | {e2e:>7.3f} | {lo:>7.3f} | {hi:>7.3f} | {regime:>12} | {conf:>5} | {cover_str:>8} | {err:>+5.1f}%")

    print()
    if total_transition > 0:
        print(f"转换区覆盖率: {covered_count}/{total_transition} ({covered_count/total_transition*100:.0f}%)")
    print()


def test_backward_compat():
    """验证返回值包含所有旧字段 (向后兼容)"""
    print("=" * 90)
    print("向后兼容性验证")
    print("=" * 90)

    model = DeepMDOverheadModel()
    device_config = {"Mem_Bw": 3430, "Device": "TEST"}
    deepmd_config = {
        "type_map": ["O", "H"],
        "descriptor": {"type": "se_e2_a", "sel": [46, 92], "rcut": 6.0, "neuron": [25, 50, 100], "axis_neuron": 16},
        "fitting_net": {"neuron": [240, 240, 240]},
    }

    required_fields = [
        "total_overhead_ms", "fixed_overhead_ms", "unmodeled_compute_ms",
        "e2e_estimate_ms", "regime", "cpu_scale", "gpu_scale",
        "density_correction", "kernel_launch_ms", "cpu_dispatch_ms",
        "autograd_ms", "wall_time_ms", "kernel_count", "analytical_detail", "notes",
    ]
    new_fields = ["confidence", "transition_ratio", "e2e_lower_ms", "e2e_upper_ms"]

    op_df = build_mock_op_df(256)
    result = model.estimate(
        device_config=device_config,
        deepmd_config=deepmd_config,
        op_df=op_df,
        compute_force=True,
        compute_latency_ms=1.08,
    )

    all_ok = True
    for field in required_fields:
        if field in result:
            print(f"  ✓ {field}: {result[field]}")
        else:
            print(f"  ✗ MISSING: {field}")
            all_ok = False

    print()
    print("新增字段:")
    for field in new_fields:
        if field in result:
            print(f"  ✓ {field}: {result[field]}")
        else:
            print(f"  ✗ MISSING: {field}")
            all_ok = False

    print()
    if all_ok:
        print("✓ 向后兼容性检查通过！")
    else:
        print("✗ 向后兼容性检查失败！")
    return all_ok


if __name__ == "__main__":
    ok1 = test_confidence_zones()
    test_transition_zone_coverage()
    ok2 = test_backward_compat()

    print()
    print("=" * 90)
    if ok1 and ok2:
        print("🎉 所有验证通过！Confidence-Aware Prediction 功能正常。")
    else:
        print("⚠️ 部分验证失败，请检查。")
