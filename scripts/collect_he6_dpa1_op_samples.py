#!/usr/bin/env python3
"""
P1: 为 HE6 / DPA-1 风格的 op shape 生成 MLP_WAVE 训练样本。

Motivation
----------
DeepMD §8 报告显示 N=4096 时:
  - water/copper/LiAlOCl: ±5%
  - HE6:        -16% (gap_cmp = +18.6 ms, MLP_WAVE 漏算)
  - water_dpa1: -33% (gap_cmp = +39.2 ms, attention 完全漏算)

原因: NeuSight MLP_WAVE_LINEAR / MLP_WAVE_BMM 训练集是
asplos/data/dataset/train/collect/{linear,bmm}/...的 (B, M, N, K) 形状,
覆盖了 Transformer 大 batch 的 GEMM, 但缺以下两类:

1. HE6 embedding subnet:
   - ntypes=6, sel=80, neuron=[16,32,64]
   - 每个子网 batch = N×80 (per-pair), MNK = (N×80, in, out), in/out ∈ {1,16,32,64}
   - 36 个 subnet → 极小 batch, 极小 K (1)

2. DPA-1 attention:
   - Q@K^T:   B = N, M = sel(60), N = 60, K = attn_dim(128)
   - softmax: B = N×60, H = 60
   - V@attn:  B = N, M = 60, N = 128, K = 60

这个脚本:
  (a) 对每个 config 跑 build_deepmd_opgraph 收集所有 (op_type, B, M, N, K)
  (b) 用 torch.profiler 实测每个 unique shape 的 latency
  (c) 输出符合 NeuSight trainset 格式的 CSV (OPName/Latency/Device/B/M/N/K + 元数据)

Usage
-----
    # 收集 HE6 + DPA-1 的所有 unique linear/bmm 形状
    python scripts/collect_he6_dpa1_op_samples.py \\
        --configs scripts/asplos/data/deepmd_configs/he6_se_e2_a.json \\
                  scripts/asplos/data/deepmd_configs/water_dpa1.json \\
        --atoms 256 1024 4096 \\
        --runs 30 --warmup 5 \\
        --out-linear scripts/asplos/data/dataset/train/collect/linear/p2.4.1/${HOSTNAME}_he6_dpa1.csv \\
        --out-bmm    scripts/asplos/data/dataset/train/collect/bmm/p2.4.1/${HOSTNAME}_he6_dpa1.csv

After: 把生成的 CSV 加到 trainset 里, 然后重新跑 scripts/train.py 训
MLP_WAVE_LINEAR + MLP_WAVE_BMM, 即可让大 N HE6/DPA-1 的 cmp 段不再漏算。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neusight.Tracing.parse_deepmd_input import parse_deepmd_input
from neusight.Tracing.trace_deepmd import build_deepmd_opgraph


def collect_unique_shapes(configs, atoms_list, compute_force=True):
    """
    对每个 (config, N) 跑 tracer, 收集 (op_type, B, M, N, K) unique 元组。

    Linear: FwOps 形如 [("Linear", (batch, in_dim, out_dim))]
    BMM:    FwOps 形如 [("BMM", (B, M, N, K))]

    Returns
    -------
    dict
        {"Linear": set([(B, M, N, K), ...]),
         "BMM":    set([(B, M, N, K), ...])}
    """
    bucket = defaultdict(set)

    for cfg_path in configs:
        for N in atoms_list:
            cfg = parse_deepmd_input(cfg_path)
            df = build_deepmd_opgraph(cfg, N, compute_force=compute_force)
            for _, row in df.iterrows():
                op_name = row["OpName"]
                fw_ops = row["FwOps"]
                if not isinstance(fw_ops, list):
                    continue
                for op_entry in fw_ops:
                    if not (isinstance(op_entry, (list, tuple)) and len(op_entry) >= 2):
                        continue
                    sub_op, shape = op_entry[0], op_entry[1]
                    if sub_op == "Linear":
                        # shape = (batch, in_dim, out_dim)
                        # NeuSight CSV 列: B=1, M=batch, N=out_dim, K=in_dim
                        if len(shape) == 3:
                            B_, M_, N_, K_ = 1, int(shape[0]), int(shape[2]), int(shape[1])
                            bucket["Linear"].add((B_, M_, N_, K_))
                    elif sub_op == "BMM":
                        # shape = (B, M, N, K)
                        if len(shape) == 4:
                            B_, M_, N_, K_ = (int(s) for s in shape)
                            bucket["BMM"].add((B_, M_, N_, K_))

    return bucket


def profile_linear(B, M, N, K, runs=30, warmup=5, device="cuda:0", dtype=None):
    """实测一个 Linear shape 的 latency (ms)."""
    import torch
    if dtype is None:
        dtype = torch.float32
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)
    for _ in range(warmup):
        torch.nn.functional.linear(x, w, b)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    for i in range(runs):
        starts[i].record()
        torch.nn.functional.linear(x, w, b)
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.median([s.elapsed_time(e) for s, e in zip(starts, ends)]))


def profile_bmm(B, M, N, K, runs=30, warmup=5, device="cuda:0", dtype=None):
    """实测一个 BMM shape 的 latency (ms)."""
    import torch
    if dtype is None:
        dtype = torch.float32
    a = torch.randn(B, M, K, device=device, dtype=dtype)
    b_ = torch.randn(B, K, N, device=device, dtype=dtype)
    for _ in range(warmup):
        torch.bmm(a, b_)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    for i in range(runs):
        starts[i].record()
        torch.bmm(a, b_)
        ends[i].record()
    torch.cuda.synchronize()
    return float(np.median([s.elapsed_time(e) for s, e in zip(starts, ends)]))


def emit_csv(rows, out_path, op_name):
    """按 NeuSight trainset 列顺序写 CSV."""
    import torch
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(0)
    torch_ver = "p" + torch.__version__.split("+")[0]
    cuda_ver = f"cu{torch.version.cuda.replace('.', '')}" if torch.version.cuda else "cu128"
    cudnn_ver = torch.backends.cudnn.version() or 0

    columns = [
        "OPName", "Latency", "Device", "Torch Version", "CUDNN Version",
        "Kernel Name", "Warps per SM", "Blocks per SM",
        "Grid x", "Grid y", "Grid z", "Block x", "Block y", "Block z",
        "CUDA Version", "Kernels", "B", "M", "N", "K",
    ]
    header_present = out_path.is_file() and out_path.stat().st_size > 0
    with open(out_path, "a") as f:
        if not header_present:
            f.write(",".join(columns) + "\n")
        for r in rows:
            # 这些列对训练 MLP_WAVE_MM 不是必须的 (MLP 只用 B/M/N/K + 设备特征),
            # 留 placeholder 以满足 read_csv 的 schema。
            line = ",".join([
                op_name,
                f"{r['latency']:.10f}",
                device_name,
                torch_ver,
                str(cudnn_ver),
                "ampere_sgemm_unknown",  # placeholder kernel name
                "0", "0", "0", "0", "0", "0", "0", "0",
                cuda_ver,
                ";ampere_sgemm_unknown",
                str(r["B"]), str(r["M"]), str(r["N"]), str(r["K"]),
            ])
            f.write(line + "\n")
    print(f"[saved] {out_path}: {len(rows)} new rows ({op_name})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True,
                   help="DeepMD config JSON files to extract shapes from")
    p.add_argument("--atoms", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-linear", required=True,
                   help="Output CSV path for Linear samples")
    p.add_argument("--out-bmm", required=True,
                   help="Output CSV path for BMM samples")
    p.add_argument("--no-profile", action="store_true",
                   help="Only print unique shapes, don't profile")
    args = p.parse_args()

    print(f"=== Collecting unique op shapes from {len(args.configs)} configs × "
          f"{len(args.atoms)} atom sizes ===\n")
    bucket = collect_unique_shapes(args.configs, args.atoms, compute_force=True)
    n_lin = len(bucket["Linear"])
    n_bmm = len(bucket["BMM"])
    print(f"  unique Linear shapes (B,M,N,K) : {n_lin}")
    print(f"  unique BMM shapes    (B,M,N,K) : {n_bmm}")

    if args.no_profile:
        print("\n[--no-profile] sample unique shapes:")
        for op, shapes in bucket.items():
            print(f"\n  {op}:")
            for s in sorted(shapes)[:10]:
                print(f"    B={s[0]} M={s[1]} N={s[2]} K={s[3]}")
            if len(shapes) > 10:
                print(f"    ... and {len(shapes)-10} more")
        return

    import torch
    print(f"\n=== Profiling on {torch.cuda.get_device_name(0)} (warmup={args.warmup}, runs={args.runs}) ===")

    # Linear
    print(f"\n[Linear] profiling {n_lin} shapes...")
    lin_rows = []
    for i, (B, M, N, K) in enumerate(sorted(bucket["Linear"])):
        try:
            t = profile_linear(B, M, N, K, runs=args.runs, warmup=args.warmup, device=args.device)
            lin_rows.append({"B": B, "M": M, "N": N, "K": K, "latency": t})
            if i < 5 or i % 20 == 0:
                print(f"  Linear[{i+1}/{n_lin}]  B={B} M={M} N={N} K={K}  "
                      f"latency={t:.4f} ms")
        except Exception as exc:
            print(f"  Linear B={B} M={M} N={N} K={K} FAILED: {exc}")
    emit_csv(lin_rows, args.out_linear, "linear")

    # BMM
    print(f"\n[BMM] profiling {n_bmm} shapes...")
    bmm_rows = []
    for i, (B, M, N, K) in enumerate(sorted(bucket["BMM"])):
        try:
            t = profile_bmm(B, M, N, K, runs=args.runs, warmup=args.warmup, device=args.device)
            bmm_rows.append({"B": B, "M": M, "N": N, "K": K, "latency": t})
            if i < 5 or i % 20 == 0:
                print(f"  BMM[{i+1}/{n_bmm}]  B={B} M={M} N={N} K={K}  "
                      f"latency={t:.4f} ms")
        except Exception as exc:
            print(f"  BMM B={B} M={M} N={N} K={K} FAILED: {exc}")
    emit_csv(bmm_rows, args.out_bmm, "bmm")

    print("\nNext steps:")
    print(f"  1. Inspect {args.out_linear} and {args.out_bmm}")
    print(f"  2. Append to existing trainset:")
    print(f"     cat {args.out_linear} >> scripts/asplos/data/dataset/train/linear.csv")
    print(f"     cat {args.out_bmm}    >> scripts/asplos/data/dataset/train/bmm.csv")
    print(f"  3. Re-train MLP_WAVE_LINEAR / MLP_WAVE_BMM via scripts/train.py")


if __name__ == "__main__":
    main()
