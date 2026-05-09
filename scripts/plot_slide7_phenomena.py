#!/usr/bin/env python3
"""Slide 7 (phenomena) — 三张图说明 overhead model 的发现逻辑

Fig 1: small-N plateau (折线在 ~5.7ms 横走)
Fig 2: MLP vs Real gap (柱状对比, 揭示未建模算子)
Fig 3: gap ~ N^beta scaling (log-log 拟合 + roofline 公式叠加)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "ppt_assets")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = "#1F3A68"
TEAL = "#3BA2B5"
CORAL = "#D9846C"
AMBER = "#E0B458"
GRAY = "#3D4A5C"
LIGHT = "#F0F2F5"

# ============================================================
# Load H100 data
# ============================================================
fa = json.load(open(os.path.join(
    ROOT, "results", "deepmd_fulltest", "full_accuracy_report.json")))
la = json.load(open(os.path.join(
    ROOT, "results", "deepmd_fulltest", "large_atoms_report.json")))

# Water small/medium: from full_accuracy_report
water_rows = []  # (N, real_ms, mlp_ms)
for r in fa["results"]:
    water_rows.append((
        r["num_atoms"],
        r["real_mean_ms"],
        r["pred_force_compute_ms"],   # MLP-only compute
    ))
# Water large: from large_atoms_report
for r in la["results"]["water"]:
    water_rows.append((
        r["num_atoms"],
        r["real_mean_ms"],
        r["pred_compute_ms"],
    ))
water_rows.sort()

# Copper from large_atoms_report
copper_rows = []
for r in la["results"]["copper"]:
    copper_rows.append((
        r["num_atoms"],
        r["real_mean_ms"],
        r["pred_compute_ms"],
    ))
copper_rows.sort()

# ============================================================
# Fig 1 — Small-N plateau
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.8))
small = [r for r in water_rows if r[0] <= 2048]
Ns = [r[0] for r in small]
reals = [r[1] for r in small]

ax.plot(Ns, reals, "-o", color=NAVY, linewidth=2.0, markersize=8,
        markerfacecolor=TEAL, markeredgecolor=NAVY, markeredgewidth=1.2,
        label="Measured end-to-end latency (Water)")

# Plateau region shading
ax.axvspan(28, 1100, color=GRAY, alpha=0.10)
ax.text(180, 7.6, "plateau region: latency ≈ const",
        fontsize=11, style="italic", color=GRAY)

# Plateau reference line
plateau_val = float(np.mean([r[1] for r in small if r[0] <= 1024]))
ax.axhline(plateau_val, linestyle="--", color=CORAL, linewidth=1.5,
           label=f"plateau ≈ {plateau_val:.2f} ms (independent of N)")

ax.set_xscale("log")
ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
ax.set_xticklabels([32, 64, 128, 256, 512, 1024, 2048])
ax.set_xlabel("Number of atoms N", fontsize=12)
ax.set_ylabel("End-to-end latency (ms)", fontsize=12)
ax.set_title("Phenomenon 1 — Small-N plateau\n"
             "Latency stays flat for N ≤ 1024, suggesting an N-independent floor",
             fontsize=12, fontweight="bold", color=NAVY)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
ax.set_ylim(0, max(reals) * 1.4)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "fig_phenom_plateau.png")
plt.savefig(out1, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {out1}")

# ============================================================
# Fig 2 — MLP vs Real gap (compute-bound region)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 4.8))

# Combine water large + copper for "compute-bound" view
big = [(f"W-{r[0]}", r[0], r[1], r[2]) for r in water_rows if r[0] >= 2048]
big += [(f"Cu-{r[0]}", r[0], r[1], r[2]) for r in copper_rows]
big.sort(key=lambda x: (x[0][0], x[1]))

labels = [b[0] for b in big]
reals = np.array([b[2] for b in big])
mlps = np.array([b[3] for b in big])
gaps = reals - mlps

xpos = np.arange(len(labels))
w = 0.38

ax.bar(xpos - w / 2, mlps, w, color=TEAL, edgecolor=NAVY, linewidth=0.8,
       label="MLP-only compute (predicted by MLP_WAVE)")
ax.bar(xpos + w / 2, gaps, w, color=CORAL, edgecolor=NAVY, linewidth=0.8,
       label="Gap = Real − MLP  (un-modeled GPU work)")

# Annotate gap values on top of coral bars
for i, g in enumerate(gaps):
    ax.text(xpos[i] + w / 2, g + max(gaps) * 0.02,
            f"{g:.1f}", ha="center", va="bottom", fontsize=9,
            color=CORAL, fontweight="bold")

ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_xlabel("Test point (model − N)", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_title("Phenomenon 2 — A large gap between MLP-only and measured latency\n"
             "Reveals NeuSight is missing significant GPU work (nlist / env_mat / force backward)",
             fontsize=12, fontweight="bold", color=NAVY)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "fig_phenom_gap.png")
plt.savefig(out2, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {out2}")

# ============================================================
# Fig 3 — Gap scales as power law of N → motivates roofline
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.0))

# Compute-bound points: use water 2048/4096/8192 + copper 2048/4096/8192
points = []
for N, real, mlp in water_rows:
    if N >= 2048:
        points.append(("water", N, real - mlp))
for N, real, mlp in copper_rows:
    points.append(("copper", N, real - mlp))

# Fit log-log
all_N = np.array([p[1] for p in points])
all_gap = np.array([p[2] for p in points])
log_N = np.log(all_N)
log_g = np.log(all_gap)
beta, log_alpha = np.polyfit(log_N, log_g, 1)
alpha = np.exp(log_alpha)

# Plot
w_pts = [p for p in points if p[0] == "water"]
c_pts = [p for p in points if p[0] == "copper"]
ax.scatter([p[1] for p in w_pts], [p[2] for p in w_pts],
           s=80, color=TEAL, edgecolors=NAVY, linewidth=1.0,
           zorder=3, label="Water (gap)")
ax.scatter([p[1] for p in c_pts], [p[2] for p in c_pts],
           s=80, color=CORAL, edgecolors=NAVY, linewidth=1.0,
           marker="s", zorder=3, label="Copper (gap)")

# Fit line
xs = np.logspace(np.log10(2048 * 0.9), np.log10(8192 * 1.1), 50)
ys = alpha * xs ** beta
ax.plot(xs, ys, "--", color=NAVY, linewidth=1.6,
        label=f"Power-law fit: gap ∝ N^{beta:.2f}")

# Annotation: roofline interpretation
ax.text(0.04, 0.96,
        "Two-term Roofline:\n"
        "   gap ≈  C_quad · N · nall · 8 / BW\n"
        "          + C_linear · N · nnei · 8 / BW\n"
        "(nall = ns · N  →  first term is N², second is N)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, family="monospace", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                  edgecolor=NAVY, linewidth=0.8))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of atoms N", fontsize=12)
ax.set_ylabel("Gap = Real − MLP  (ms)", fontsize=12)
ax.set_title("Phenomenon 3 — The gap follows a clean power law\n"
             "→ closed-form analytical (Roofline) model is sufficient",
             fontsize=12, fontweight="bold", color=NAVY)
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "fig_phenom_scaling.png")
plt.savefig(out3, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {out3}")

print(f"\nFitted power-law exponent beta = {beta:.3f}, alpha = {alpha:.3e}")
