#!/usr/bin/env python3
"""
Independently measure the REAL fix / cmp / roof decomposition of DeepMD-pt
inference, without relying on any of our model predictions.

Method
------
Run torch.profiler around model(coord, atype, box) and bucket every kernel by
its operator name:

  * MLP_real    = sum(GPU time of) aten::{mm, addmm, bmm, linear, addbmm,
                   matmul, _matmul_impl, gemm*}                      → "cmp"
  * DESC_real   = sum(GPU time of) descriptor / neighbor / scatter / gather
                   kernels: aten::{scatter*, gather, index_select,
                   sort, unique, cat, stack, where, masked_select,
                   pdist, cdist, segment_reduce, env_mat_a, prod_env_mat*,
                   border_op, format_nlist*}                          → "roof"
  * OTHER_gpu   = everything else still on GPU (copy_, fill_, mul_, add_,
                   tanh, exp, pow, mean, sum, etc.)                   → noise
  * FIX_real    = wall_total - (sum of all GPU device time)
                   = host_dispatch + sync + launch overhead          → "fix"

This is fully independent of the calibration JSON / NeuSight predictor.
"""
import argparse, json, os, sys, time, gc, re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cross_system import SYSTEMS, build_model

# DESC = neighbor / descriptor / env_mat / scatter-gather / sort.
# Everything else on GPU is treated as compute (MLP + activation + elementwise),
# because NeuSight's MLP_WAVE per-op predictor is meant to cover all of it.
DESC_PAT = re.compile(
    r"(scatter|gather|index_select|index_add|index_put|sort|unique|"
    r"masked_select|nonzero|pdist|cdist|segment_reduce|env_mat|prod_env|"
    r"border_op|format_nlist|nlist|nnei|neighbor|build_descrpt|cumsum)",
    re.IGNORECASE,
)
def bucket(name: str) -> str:
    return "DESC" if DESC_PAT.search(name) else "CMP"


def run_one(model, n, num_types, box, warmup, runs):
    """Return per-run breakdown of fix/cmp/roof in ms (median over runs)."""
    import torch
    from deepmd.pt.utils.env import DEVICE
    from torch.profiler import profile, ProfilerActivity

    coord = torch.rand(1, n, 3, dtype=torch.float64, device=DEVICE) * box
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, n), device=DEVICE)
    box_t = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box
    box_t.requires_grad_(True)

    # warmup
    for _ in range(warmup):
        _ = model(coord, atype, box_t)
        if coord.grad is not None: coord.grad.zero_()
    torch.cuda.synchronize()

    per_run = []  # list of dicts
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(runs):
            if coord.grad is not None: coord.grad.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(coord, atype, box_t)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_run.append((t1 - t0) * 1000.0)

    # Aggregate kernel-time buckets across all `runs` (then divide by runs)
    bucket_us = defaultdict(float)
    total_cuda_us = 0.0
    total_cpu_us = 0.0
    for evt in prof.key_averages():
        # cuda_time_total is in microseconds, summed across all invocations
        cu = evt.self_device_time_total if hasattr(evt, "self_device_time_total") else evt.self_cuda_time_total
        cp = evt.self_cpu_time_total
        bucket_us[bucket(evt.key)] += cu
        total_cuda_us += cu
        total_cpu_us += cp

    runs_real_ms = float(np.mean(per_run))
    runs_std_ms  = float(np.std(per_run))
    wall_total_ms = sum(per_run)

    # convert to per-run ms
    cmp_ms_raw  = bucket_us["CMP"]  / 1000.0 / runs
    desc_ms_raw = bucket_us["DESC"] / 1000.0 / runs
    gpu_sum_ms  = total_cuda_us    / 1000.0 / runs

    # Handle stream overlap: cap GPU usable share at wall, keep MLP:DESC ratio.
    gpu_capped = min(gpu_sum_ms, runs_real_ms)
    if gpu_sum_ms > 0:
        scale = gpu_capped / gpu_sum_ms
    else:
        scale = 0.0
    cmp_ms  = cmp_ms_raw  * scale
    desc_ms = desc_ms_raw * scale
    fix_ms  = max(0.0, runs_real_ms - gpu_capped)

    return {
        "N": n, "real_mean_ms": runs_real_ms, "real_std_ms": runs_std_ms,
        "gpu_sum_ms": gpu_sum_ms, "gpu_capped_ms": gpu_capped,
        "cmp_real_ms": cmp_ms, "roof_real_ms": desc_ms, "fix_real_ms": fix_ms,
        "frac_fix":  fix_ms  / runs_real_ms,
        "frac_cmp":  cmp_ms  / runs_real_ms,
        "frac_roof": desc_ms / runs_real_ms,
        "overlap_factor": gpu_sum_ms / max(runs_real_ms, 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["copper","water","LiAlOCl","he6"])
    ap.add_argument("--atoms",   type=int, nargs="+",
                    default=[64, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs",   type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results/cross_system/real_breakdown.json"))
    args = ap.parse_args()

    import torch
    sys_map = {s["name"]: s for s in SYSTEMS}
    rows = []
    for sname in args.systems:
        s = sys_map[sname]
        cfg = json.load(open(s["config"]))
        nt = len(cfg["type_map"])
        for n in args.atoms:
            print(f"[{sname:10s} N={n:>5d}] building & profiling…", end=" ", flush=True)
            try:
                model = build_model(cfg)
                r = run_one(model, n, nt, s["box"], args.warmup, args.runs)
                r["system"] = sname
                rows.append(r)
                print(f"real={r['real_mean_ms']:.2f}  "
                      f"gpu_sum={r['gpu_sum_ms']:.2f} (ovlp={r['overlap_factor']:.2f}x)  "
                      f"[fix={r['fix_real_ms']:.2f} cmp={r['cmp_real_ms']:.2f} roof={r['roof_real_ms']:.2f}]")
                del model; torch.cuda.empty_cache(); gc.collect()
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM: {e}")
                torch.cuda.empty_cache(); gc.collect()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0),
               "warmup": args.warmup, "runs": args.runs, "rows": rows},
              open(out, "w"), indent=2)
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
