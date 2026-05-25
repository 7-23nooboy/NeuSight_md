#!/usr/bin/env python3
"""
Large-N extrapolation test for the NeuSight DeepMD overhead model (v6).

Re-uses build_model / profile / neusight_pred from benchmark_cross_system.py
but overrides the atom-count list to push beyond 4096.

Usage:
  conda activate gpu_sim
  NEUSIGHT_DEEPMD_CALIBRATION=results/calibration/h100_nvl_v6.json \
    python scripts/benchmark_large_atoms_v6.py \
      --atoms 8192 16384 32768 \
      --systems copper water LiAlOCl he6 \
      --warmup 5 --runs 20
"""
import argparse, json, os, sys, time, gc
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cross_system import (
    SYSTEMS, build_model, profile, neusight_pred, RESULT_DIR
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, nargs="+",
                    default=[8192, 16384, 32768])
    ap.add_argument("--systems", nargs="+",
                    default=["copper", "water", "LiAlOCl", "he6"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", default=str(RESULT_DIR / "cross_system_report_large.json"))
    args = ap.parse_args()

    import torch
    sys_map = {s["name"]: s for s in SYSTEMS}
    print(f"GPU         : {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"DeepMD      : {__import__('deepmd').__version__}")
    print(f"Calibration : {os.environ.get('NEUSIGHT_DEEPMD_CALIBRATION', '(none)')}")
    print(f"Atoms       : {args.atoms}")
    print(f"Systems     : {args.systems}")
    print()

    rows = []
    for sname in args.systems:
        if sname not in sys_map:
            print(f"!! unknown system '{sname}', skipping"); continue
        s = sys_map[sname]
        with open(s["config"]) as f: cfg = json.load(f)
        num_types = len(cfg["type_map"])

        for n in args.atoms:
            # try-catch OOM, fall back to skipping this point
            model = None; real_mean = real_std = None
            t0 = time.time()
            try:
                if not args.skip_real:
                    model = build_model(cfg)
                    lat = profile(model, n, num_types, s["box"], args.warmup, args.runs)
                    real_mean = float(np.mean(lat)); real_std = float(np.std(lat))
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [{sname} N={n}] OOM during profile: {e}")
            except Exception as e:
                print(f"  [{sname} N={n}] error during profile: {e!r}")
            finally:
                if model is not None: del model
                torch.cuda.empty_cache(); gc.collect()

            pred_force = neusight_pred(s["config"], n, True, s["box"],
                                       RESULT_DIR / sname)
            pred_e     = neusight_pred(s["config"], n, False, s["box"],
                                       RESULT_DIR / sname)

            err = None
            if real_mean and pred_force and pred_force.get("e2e"):
                err = (pred_force["e2e"] - real_mean) / real_mean * 100

            row = {
                "system": sname, "num_atoms": n,
                "real_mean_ms": round(real_mean, 4) if real_mean else None,
                "real_std_ms":  round(real_std, 4) if real_std else None,
                "pred_force_e2e_ms":     round(pred_force["e2e"], 4) if pred_force else None,
                "pred_force_fixed_ms":   round(pred_force.get("fixed_overhead") or 0, 4) if pred_force else None,
                "pred_force_compute_ms": round(pred_force.get("compute") or 0, 4) if pred_force else None,
                "pred_force_gpu_oh_roofline_ms": round(pred_force.get("gpu_oh_roofline") or 0, 4) if pred_force else None,
                "regime":           pred_force.get("regime") if pred_force else None,
                "transition_ratio": pred_force.get("transition_ratio") if pred_force else None,
                "k_modeled":        pred_force.get("k_modeled") if pred_force else None,
                "k_total":          pred_force.get("k_total") if pred_force else None,
                "pred_energy_e2e_ms": round(pred_e["e2e"], 4) if pred_e else None,
                "error_pct":         round(err, 1) if err is not None else None,
                "wall_s":            round(time.time() - t0, 1),
            }
            rows.append(row)
            print(f"  {sname:10s} N={n:>6d}  "
                  f"real={('%.2f' % real_mean) if real_mean else 'N/A':>9s}  "
                  f"pred={('%.2f' % pred_force['e2e']) if pred_force else 'N/A':>9s}  "
                  f"err={('%+.1f%%' % err) if err is not None else 'N/A':>8s}  "
                  f"({row['wall_s']}s)")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0),
                   "calibration": os.environ.get("NEUSIGHT_DEEPMD_CALIBRATION"),
                   "warmup": args.warmup, "runs": args.runs,
                   "rows": rows}, f, indent=2)
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
