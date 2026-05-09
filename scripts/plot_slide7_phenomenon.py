#!/usr/bin/env python3
"""Slide 7 (phenomenon) — 一张图说明 overhead 现象

横轴 N (log)，纵轴 latency (log)，画三条曲线：
  - measured real_ms (实测)
  - mlp-only compute (NeuSight 原本能预测的部分)
  - 二者差值 (= unmodeled overhead)
并在 N≈1024 处标注 "regime change"。
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "ppt_assets", "fig_phenom_overview.png")

NAVY = "#1F3A68"
TEAL = "#3BA2B5"
CORAL = "#D9846C"
AMBER = "#E0B458"
GRAY = "#3D4A5C"
LIGHT = "#F0F2F5"

# Load Water (覆盖 N=32..8192 全段最完整的一组)
fa = json.load(open(os.path.join(
    ROOT, "results", "deepmd_fulltest", "full_accuracy_report.json")))
la = json.load(open(os.path.join(
    ROOT, "results", "deepmd_fulltest", "large_atoms_report.json")))

rows = []
for r in fa["results"]:
    rows.append((r["num_atoms"], r["real_mean_ms"], r["pred_force_compute_ms"]))
for r in la["results"]["water"]:
    rows.append((r["num_atoms"], r["real_mean_ms"], r["pred_compute_ms"]))
rows.sort()

Ns = np.array([r[0] for r in rows])
real = np.array([r[1] for r in rows])
mlp = np.array([r[2] for r in rows])
gap = real - mlp

# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.6))

# Three curves
ax.plot(Ns, real, "-o", color=NAVY, linewidth=2.2, markersize=8,
        markerfacecolor=CORAL, markeredgecolor=NAVY, markeredgewidth=1.0,
        label="Measured end-to-end latency", zorder=4)

ax.plot(Ns, mlp, "-s", color=TEAL, linewidth=1.8, markersize=7,
        markerfacecolor="white", markeredgecolor=TEAL, markeredgewidth=1.5,
        label="MLP-only compute (what NeuSight predicts)", zorder=3)

ax.plot(Ns, gap, "--^", color=GRAY, linewidth=1.6, markersize=7,
        markerfacecolor=LIGHT, markeredgecolor=GRAY,
        label="Gap = Measured − MLP  (un-modeled by NeuSight)", zorder=2)

# Region shading
ax.axvspan(28, 1100, color=TEAL, alpha=0.08)
ax.axvspan(1100, 9500, color=CORAL, alpha=0.08)

# Region labels at top
ymax_pos = 380
ax.text(180, ymax_pos, "Plateau region\n(latency ~ const)",
        ha="center", va="top", fontsize=11, style="italic", color=GRAY)
ax.text(3500, ymax_pos, "Anomalous super-linear growth\n(latency ≫ MLP cost)",
        ha="center", va="top", fontsize=11, style="italic", color=GRAY)

# Vertical regime-change marker
ax.axvline(1024, color=AMBER, linewidth=1.5, linestyle=":")
ax.text(1024, 0.4, "  regime change\n  ≈ N=1024",
        rotation=0, fontsize=10, color=AMBER, fontweight="bold",
        ha="left", va="bottom")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xticklabels([32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
ax.set_xlabel("Number of atoms N", fontsize=12)
ax.set_ylabel("Latency (ms, log scale)", fontsize=12)
ax.set_title("Where does the un-modeled overhead come from?\n"
             "Measured latency stays flat for small N, then grows much faster than MLP cost",
             fontsize=13, fontweight="bold", color=NAVY)
ax.grid(True, which="both", linestyle=":", alpha=0.45)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
ax.set_ylim(0.3, 500)

plt.tight_layout()
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {OUT}")
