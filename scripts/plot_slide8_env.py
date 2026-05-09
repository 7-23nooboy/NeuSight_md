#!/usr/bin/env python3
"""Slide 8 — Experimental environment table (rendered as image)."""

import os
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "ppt_assets", "fig_slide8_env.png")

NAVY = "#1F3A68"
TEAL = "#3BA2B5"
CORAL = "#D9846C"
LIGHT = "#F0F2F5"
GRAY = "#3D4A5C"

rows = [
    ("GPU",                "NVIDIA H100 NVL (94 GB HBM3)",  "NVIDIA A100 80GB PCIe"),
    ("Memory bandwidth",   "3430 GB/s",                     "1935 GB/s"),
    ("Streaming MPs",      "132",                           "108"),
    ("Device config file", "NVIDIA_H100_NVL.json",          "NVIDIA_A100_80GB_PCIe.json"),
    ("DeepMD-kit",         "3.1.3",                         "3.1.3"),
    ("PyTorch / CUDA",     "2.10.0 / CUDA 12.8",            "2.10.0 / CUDA 12.8"),
    ("Float precision",    "float64",                       "float64"),
    ("Test models",        "Water se_e2_a  (2-type, sel=[46,92])\nCopper se_e2_a (1-type, sel=[120])",
                            "Water se_e2_a  (2-type, sel=[46,92])\nCopper se_e2_a (1-type, sel=[120])"),
    ("Task",               "Energy + Force inference",      "Energy + Force inference"),
    ("N (atoms) range",    "32 – 8192",                     "32 – 8192"),
    ("Repeats per point",  "100 (warm-up 20)",              "100 (warm-up 20)"),
    ("Test points",        "13",                            "43"),
    ("MAE",                "2.27 %",                        "3.11 %"),
    ("Max |err|",          "7.35 %",                        "7.13 %"),
]

# Build table
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_axis_off()

col_labels = ["", "H100 experiment", "A100 experiment"]
cell_text = [list(r) for r in rows]

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="left",
    colLoc="left",
    loc="center",
    colWidths=[0.24, 0.38, 0.38],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.55)

# Style header row
for j in range(3):
    c = table[(0, j)]
    c.set_facecolor(NAVY)
    c.set_text_props(color="white", fontweight="bold", fontsize=12)
    c.set_edgecolor("white")
    c.set_height(0.08)

# Style body rows
for i in range(1, len(rows) + 1):
    label = rows[i - 1][0]
    is_metric = label in ("MAE", "Max |err|", "Test points")
    for j in range(3):
        c = table[(i, j)]
        c.set_edgecolor("white")
        if j == 0:
            c.set_facecolor(LIGHT)
            c.set_text_props(color=NAVY, fontweight="bold")
        else:
            c.set_facecolor("white")
            if is_metric:
                color = TEAL if j == 1 else CORAL
                c.set_text_props(color=color, fontweight="bold", fontsize=12)
            else:
                c.set_text_props(color=GRAY)

# Title
fig.suptitle("Slide 8 — Experimental Setup",
             fontsize=15, fontweight="bold", color=NAVY, y=0.985)

plt.tight_layout()
plt.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {OUT}")
