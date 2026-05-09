#!/usr/bin/env python3
"""Slide 8 — H100 / A100 校准后预测 vs 实测对比图

输出:
  ppt_assets/fig_slide8_h100.png
  ppt_assets/fig_slide8_a100.png

每张图: parity plot (predicted vs measured, log-log) + 残差柱状图
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "ppt_assets")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = "#1F3A68"
TEAL = "#3BA2B5"
CORAL = "#D9846C"
GRAY = "#3D4A5C"


def parity_plot(rows, title, mae, max_err, n_points, gpu_label, out_path):
    """rows: list of (model, N, real, pred)"""
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)

    # ---- Left: parity plot ----
    ax = axes[0]
    water = [(N, r, p) for m, N, r, p in rows if m == "water"]
    copper = [(N, r, p) for m, N, r, p in rows if m == "copper"]

    if water:
        wN, wr, wp = zip(*water)
        ax.scatter(wr, wp, s=55, color=TEAL, edgecolors=NAVY, linewidth=0.8,
                   label=f"Water ({len(water)} pts)", zorder=3)
    if copper:
        cN, cr, cp = zip(*copper)
        ax.scatter(cr, cp, s=55, color=CORAL, edgecolors=NAVY, linewidth=0.8,
                   marker="s", label=f"Copper ({len(copper)} pts)", zorder=3)

    all_real = [r for _, _, r, _ in rows]
    all_pred = [p for _, _, _, p in rows]
    lo = min(min(all_real), min(all_pred)) * 0.7
    hi = max(max(all_real), max(all_pred)) * 1.4
    xs = np.array([lo, hi])
    ax.plot(xs, xs, "--", color=GRAY, linewidth=1.2, zorder=2, label="y = x")
    ax.fill_between(xs, xs * 0.95, xs * 1.05, color=GRAY, alpha=0.10,
                    label="±5% band", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured latency (ms)", fontsize=11)
    ax.set_ylabel("Predicted latency (ms)", fontsize=11)
    ax.set_title(f"{gpu_label} — Predicted vs Measured", fontsize=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

    # Annotate MAE/max box
    ax.text(0.97, 0.05,
            f"MAE = {mae:.2f}%\nMax|err| = {max_err:.2f}%\nN points = {n_points}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=NAVY, linewidth=0.8))

    # ---- Right: per-N error bars ----
    ax2 = axes[1]
    # group by model, sort by N
    rows_sorted = sorted(rows, key=lambda x: (x[0], x[1]))
    labels, errs, colors = [], [], []
    for m, N, r, p in rows_sorted:
        labels.append(f"{m[0].upper()}-{N}")
        errs.append((p - r) / r * 100)
        colors.append(TEAL if m == "water" else CORAL)

    xpos = np.arange(len(labels))
    ax2.bar(xpos, errs, color=colors, edgecolor=NAVY, linewidth=0.6)
    ax2.axhline(0, color=GRAY, linewidth=0.8)
    ax2.axhspan(-5, 5, color=GRAY, alpha=0.10, label="±5%")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, rotation=75, fontsize=7.5)
    ax2.set_ylabel("Relative error (%)", fontsize=11)
    ax2.set_title(f"{gpu_label} — Per-point relative error", fontsize=12)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax2.legend(fontsize=9, loc="upper right")
    # symmetric ylim
    ymax = max(15, max(abs(e) for e in errs) * 1.15)
    ax2.set_ylim(-ymax, ymax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ============================================================
# H100 — combine full_accuracy_report (water 32..2048) +
#         large_atoms_report (water 4096/8192, copper 2048/4096/8192)
# ============================================================
h100_rows = []
seen = set()

# Source 1: full_accuracy_report.json  (water small/medium)
fa = json.load(open(os.path.join(
    ROOT, "results", "deepmd_fulltest", "full_accuracy_report.json")))
model_name = "water" if "water" in fa["model"] else "copper"
for r in fa["results"]:
    key = (model_name, r["num_atoms"])
    if key in seen:
        continue
    seen.add(key)
    h100_rows.append((model_name, r["num_atoms"],
                      r["real_mean_ms"], r["pred_force_total_ms"]))

# Source 2: large_atoms_report.json  (water 4k/8k + copper 2k/4k/8k)
la = json.load(open(os.path.join(
    ROOT, "results", "deepmd_fulltest", "large_atoms_report.json")))
for m, lst in la["results"].items():
    for r in lst:
        key = (m, r["num_atoms"])
        if key in seen:
            continue
        seen.add(key)
        h100_rows.append((m, r["num_atoms"],
                          r["real_mean_ms"], r["pred_total_ms"]))

# Sort
h100_rows.sort(key=lambda x: (x[0], x[1]))

# Recompute stats
errs = [(p - r) / r * 100 for _, _, r, p in h100_rows]
h100_mae = float(np.mean(np.abs(errs)))
h100_max = float(max(abs(e) for e in errs))

parity_plot(
    h100_rows,
    title="Overhead Model — H100 NVL (calibrated on H100)",
    mae=h100_mae,
    max_err=h100_max,
    n_points=len(h100_rows),
    gpu_label="H100 NVL",
    out_path=os.path.join(OUT_DIR, "fig_slide8_h100.png"),
)

# ============================================================
# A100
# ============================================================
a100 = json.load(open(os.path.join(
    ROOT, "results", "a100_experiment", "recalibration.json")))
a100_rows = [
    (d["model"], d["N"], d["real_ms"], d["calibrated_pred"])
    for d in a100["rows"]
]
parity_plot(
    a100_rows,
    title="Overhead Model — A100 80GB PCIe (re-calibrated on A100)",
    mae=a100["recalibrated"]["mae_pct"],
    max_err=a100["recalibrated"]["max_err_pct"],
    n_points=len(a100_rows),
    gpu_label="A100 80GB PCIe",
    out_path=os.path.join(OUT_DIR, "fig_slide8_a100.png"),
)
