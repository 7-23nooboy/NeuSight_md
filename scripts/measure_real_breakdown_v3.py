#!/usr/bin/env python3
"""
v3: real_breakdown with CORRECTED DESC bucket.

Differences from v2:
  - DESC now includes:
      * topk / radixFindKthValues / computeBlockwiseWithinKCounts  (neighbor selection)
      * linalg_vector_norm / NormTwoOps                             (pairwise distance)
      * cat / CatArrayBatchedCopy / copy_ / direct_copy             (descriptor tensor concat)
  - Reason: DeepMD-pt builds neighbor list via topk over per-pair distances,
            which v2 wrongly attributed to CMP.

Output schema matches measure_real_breakdown.py so analysis scripts can switch by
swapping the input file name.
"""
import argparse, json, os, sys, time, gc, re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cross_system import SYSTEMS, build_model

DESC_PAT = re.compile(
    r"(scatter|gather|index_select|index_add|index_put|sort|unique|"
    r"masked_select|nonzero|pdist|cdist|segment_reduce|env_mat|prod_env|"
    r"border_op|format_nlist|nlist|neighbor|build_descrpt|cumsum|"
    r"topk|radixFindKthValues|computeBlockwiseWithinKCounts|"
    r"linalg_vector_norm|NormTwoOps|distance|"
    r"cat\b|CatArrayBatchedCopy|copy_|direct_copy"
    r")", re.IGNORECASE)


def classify(name: str) -> str:
    return "DESC" if DESC_PAT.search(name) else "CMP"


def run_one(model, n, num_types, box, warmup, runs):
    import torch
    from deepmd.pt.utils.env import DEVICE
    from torch.profiler import profile, ProfilerActivity

    coord = torch.rand(1, n, 3, dtype=torch.float64, device=DEVICE) * box
    coord.requires_grad_(True)
    atype = torch.randint(0, num_types, (1, n), device=DEVICE)
    box_t = torch.eye(3, dtype=torch.float64, device=DEVICE).unsqueeze(0) * box
    box_t.requires_grad_(True)

    for _ in range(warmup):
        _ = model(coord, atype, box_t)
        if coord.grad is not None:
            coord.grad.zero_()
    torch.cuda.synchronize()

    per_run = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(runs):
            if coord.grad is not None:
                coord.grad.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(coord, atype, box_t)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_run.append((t1 - t0) * 1000.0)

    bucket_us = defaultdict(float)
    total_cuda_us = 0.0
    for evt in prof.key_averages():
        cu = (evt.self_device_time_total if hasattr(evt, "self_device_time_total")
              else evt.self_cuda_time_total)
        if cu <= 0:
            continue
        bucket_us[classify(evt.key)] += cu
        total_cuda_us += cu

    runs_real_ms = float(np.mean(per_run))
    runs_std_ms = float(np.std(per_run))
    cmp_ms_raw = bucket_us["CMP"] / 1000.0 / runs
    desc_ms_raw = bucket_us["DESC"] / 1000.0 / runs
    gpu_sum_ms = total_cuda_us / 1000.0 / runs

    gpu_capped = min(gpu_sum_ms, runs_real_ms)
    scale = gpu_capped / gpu_sum_ms if gpu_sum_ms > 0 else 0.0
    cmp_ms = cmp_ms_raw * scale
    desc_ms = desc_ms_raw * scale
    fix_ms = max(0.0, runs_real_ms - gpu_capped)

    return {
        "N": n,
        "real_mean_ms": runs_real_ms, "real_std_ms": runs_std_ms,
        "gpu_sum_ms": gpu_sum_ms, "gpu_capped_ms": gpu_capped,
        "cmp_real_ms": cmp_ms, "roof_real_ms": desc_ms, "fix_real_ms": fix_ms,
        "cmp_raw_ms": cmp_ms_raw, "roof_raw_ms": desc_ms_raw,
        "frac_fix":  fix_ms / runs_real_ms,
        "frac_cmp":  cmp_ms / runs_real_ms,
        "frac_roof": desc_ms / runs_real_ms,
        "overlap_factor": gpu_sum_ms / max(runs_real_ms, 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["copper", "water", "LiAlOCl", "he6"])
    ap.add_argument("--atoms", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results/cross_system/real_breakdown_v3.json"))
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
                      f"[fix={r['fix_real_ms']:.2f} cmp={r['cmp_real_ms']:.2f} "
                      f"roof={r['roof_real_ms']:.2f}]")
                del model
                torch.cuda.empty_cache()
                gc.collect()
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM: {e}")
                torch.cuda.empty_cache()
                gc.collect()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0),
               "warmup": args.warmup, "runs": args.runs,
               "desc_pattern": DESC_PAT.pattern,
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
