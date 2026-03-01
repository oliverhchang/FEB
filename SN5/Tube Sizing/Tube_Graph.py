"""
FSAE 2026 Tube Candidate Selection Chart
VR3 Engineering 4130N Catalog — Round, Square, and Rectangular tubes
Plots Cross-Sectional Area vs. MOI, with point size scaled by mass/meter
Each FSAE chassis size (A/B/C/D) is color-coded throughout:
  shaded legal zone, dashed threshold lines, corner dot, label box, legend patch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────
# MATERIAL PROPERTIES
# ─────────────────────────────────────────────────────────────────
DENSITY_STEEL_KG_M3 = 7850  # 4130N treated as 1000-series per FSAE rules

# ─────────────────────────────────────────────────────────────────
# FSAE 2026 RULES MINIMUMS  (confirm against your current rulebook)
# ─────────────────────────────────────────────────────────────────
RULES = {
    "A": {"min_area_mm2": 173.0, "min_moi_mm4": 11320.0},
    "B": {"min_area_mm2": 114.0, "min_moi_mm4":  8080.0},
    "C": {"min_area_mm2":  91.0, "min_moi_mm4":  6400.0},
    "D": {"min_area_mm2": 126.0, "min_moi_mm4": 17900.0},
}

# One distinct color per chassis size — applied to ALL size-related visuals
SIZE_COLORS = {
    "A": "#d62728",   # red
    "B": "#e6a817",   # amber
    "C": "#2ca02c",   # green
    "D": "#1f77b4",   # blue
}

# ─────────────────────────────────────────────────────────────────
# VR3 CATALOG DATA  (all dimensions in mm)
# ─────────────────────────────────────────────────────────────────

ROUND_TUBES_RAW = [
    (4.78, 0.71),
    (6.35, 0.71), (6.35, 0.89), (6.35, 1.65),
    (7.92, 0.89), (7.92, 1.47), (7.92, 1.65), (7.92, 2.11),
    (9.53, 0.71), (9.53, 0.89), (9.53, 1.24), (9.53, 1.47), (9.53, 1.65), (9.53, 2.11),
    (11.10, 0.89), (11.10, 1.47), (11.10, 1.65), (11.10, 2.11),
    (12.70, 0.71), (12.70, 0.89), (12.70, 1.24), (12.70, 1.47), (12.70, 1.65),
    (12.70, 2.11), (12.70, 2.41), (12.70, 3.05),
    (14.27, 0.89), (14.27, 1.65),
    (15.88, 0.71), (15.88, 0.89), (15.88, 1.24), (15.88, 1.47), (15.88, 1.65),
    (15.88, 2.11), (15.88, 2.41), (15.88, 3.05),
    (19.05, 0.71), (19.05, 0.89), (19.05, 1.24), (19.05, 1.47), (19.05, 1.65),
    (19.05, 2.11), (19.05, 2.41), (19.05, 3.05), (19.05, 4.78),
    (22.23, 0.89), (22.23, 1.24), (22.23, 1.47), (22.23, 1.65),
    (22.23, 2.11), (22.23, 2.41), (22.23, 3.05),
    (25.40, 0.71), (25.40, 0.89), (25.40, 1.24), (25.40, 1.47), (25.40, 1.65),
    (25.40, 2.11), (25.40, 2.41), (25.40, 3.05),
    (28.58, 0.89), (28.58, 1.24), (28.58, 1.47), (28.58, 1.65),
    (28.58, 2.11), (28.58, 2.41), (28.58, 3.05),
    (31.75, 0.89), (31.75, 1.24), (31.75, 1.47), (31.75, 1.65),
    (31.75, 2.11), (31.75, 2.41), (31.75, 3.05),
    (34.93, 0.89), (34.93, 1.47), (34.93, 1.65),
    (34.93, 2.11), (34.93, 2.41), (34.93, 3.05),
    (38.10, 0.89), (38.10, 1.24), (38.10, 1.47), (38.10, 1.65),
    (38.10, 2.11), (38.10, 2.41), (38.10, 3.05),
    (41.28, 1.47), (41.28, 1.65), (41.28, 2.11), (41.28, 2.41), (41.28, 3.05),
    (44.45, 1.24), (44.45, 1.65), (44.45, 2.41), (44.45, 3.05),
    (50.80, 1.24), (50.80, 1.65), (50.80, 2.41), (50.80, 3.05),
]

SQUARE_TUBES_RAW = [
    (9.53,  0.71),
    (12.70, 0.71),
    (15.88, 0.71),
    (19.05, 0.71), (19.05, 0.89), (19.05, 1.65),
    (22.23, 0.89),
    (25.40, 0.71), (25.40, 0.89), (25.40, 1.65),
    (31.75, 0.89),
    (38.10, 0.89),
    (50.80, 0.89),
]

RECT_TUBES_RAW = [
    (12.70, 25.40, 0.71), (12.70, 25.40, 0.89),
    (19.05, 38.10, 0.71),
    (25.40, 38.10, 0.71), (25.40, 38.10, 0.89),
    (25.40, 44.45, 0.89),
    (25.40, 50.80, 0.89),
]

# ─────────────────────────────────────────────────────────────────
# GEOMETRY CALCULATIONS
# ─────────────────────────────────────────────────────────────────

def round_props(od_mm, wall_mm):
    id_mm = od_mm - 2 * wall_mm
    if id_mm <= 0: return None
    area = np.pi / 4 * (od_mm**2 - id_mm**2)
    moi  = np.pi / 64 * (od_mm**4 - id_mm**4)
    return area, moi, area * 1e-6 * DENSITY_STEEL_KG_M3

def square_props(side_mm, wall_mm):
    inner = side_mm - 2 * wall_mm
    if inner <= 0: return None
    area = side_mm**2 - inner**2
    moi  = (side_mm**4 - inner**4) / 12
    return area, moi, area * 1e-6 * DENSITY_STEEL_KG_M3

def rect_props(w_mm, h_mm, wall_mm):
    wi, hi = w_mm - 2*wall_mm, h_mm - 2*wall_mm
    if wi <= 0 or hi <= 0: return None
    area = w_mm*h_mm - wi*hi
    moi  = (w_mm*h_mm**3 - wi*hi**3) / 12
    return area, moi, area * 1e-6 * DENSITY_STEEL_KG_M3

def build_dataset(raw_list, prop_fn, label_fn):
    data = []
    for entry in raw_list:
        result = prop_fn(*entry)
        if result is None: continue
        area, moi, mass = result
        data.append({"label": label_fn(*entry), "area": area, "moi": moi, "mass": mass})
    return data

round_data  = build_dataset(ROUND_TUBES_RAW,
    lambda od, t: round_props(od, t),   lambda od, t: f"Round {od:.2f}x{t:.2f}")
square_data = build_dataset(SQUARE_TUBES_RAW,
    lambda s, t: square_props(s, t),    lambda s, t: f"Sq {s:.2f}x{t:.2f}")
rect_data   = build_dataset(RECT_TUBES_RAW,
    lambda w, h, t: rect_props(w, h, t), lambda w, h, t: f"Rect {w:.2f}x{h:.2f}x{t:.2f}")

# ─────────────────────────────────────────────────────────────────
# FIGURE SETUP
# ─────────────────────────────────────────────────────────────────

WALL_COLORS = {
    0.71: "#e41a1c", 0.89: "#ff7f00", 1.24: "#4daf4a",
    1.47: "#377eb8", 1.65: "#984ea3", 2.11: "#a65628",
    2.41: "#f781bf", 3.05: "#999999", 4.78: "#111111",
}

fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#f8f8f8")

all_masses = [d["mass"] for d in round_data + square_data + rect_data]
mass_min, mass_max = min(all_masses), max(all_masses)
def mass_to_size(m, s_min=20, s_max=300):
    return s_min + (s_max - s_min) * (m - mass_min) / (mass_max - mass_min)

X_MAX = 320
Y_MAX = 45000

# ─────────────────────────────────────────────────────────────────
# LAYER 1 — Background reference curves
# ─────────────────────────────────────────────────────────────────
od_sweep = np.linspace(5, 55, 300)

for wall_mm, lc in [(0.89, "#bbbbbb"), (1.24, "#999999"),
                     (1.65, "#777777"), (2.11, "#555555")]:
    areas, mois = [], []
    for od in od_sweep:
        r = round_props(od, wall_mm)
        if r: areas.append(r[0]); mois.append(r[1])
    ax.plot(areas, mois, color=lc, lw=0.8, ls="-", alpha=0.28, zorder=1)

for od_mm, lc in [(19.05, "#cc9999"), (25.40, "#9999cc"),
                   (31.75, "#99cc99"), (38.10, "#cccc55")]:
    walls = np.linspace(0.5, od_mm / 2 - 0.1, 200)
    areas, mois = [], []
    for t in walls:
        r = round_props(od_mm, t)
        if r: areas.append(r[0]); mois.append(r[1])
    ax.plot(areas, mois, color=lc, lw=0.9, ls="--", alpha=0.32, zorder=1)

# ─────────────────────────────────────────────────────────────────
# LAYER 2 — Color-coded rules zones
# Each size: shaded legal quadrant → threshold lines → corner dot → label box
# ─────────────────────────────────────────────────────────────────

# Shading first so lines and labels render on top
for size, r in RULES.items():
    c = SIZE_COLORS[size]
    ax.fill_betweenx(
        [r["min_moi_mm4"], Y_MAX],
        r["min_area_mm2"], X_MAX,
        color=c, alpha=0.06, zorder=0
    )

# Offset each label slightly so they don't stack on top of each other
LABEL_OFFSETS = {"A": (5, 500), "B": (5, 500), "C": (5, 500), "D": (5, 500)}

for size, r in RULES.items():
    a  = r["min_area_mm2"]
    m  = r["min_moi_mm4"]
    c  = SIZE_COLORS[size]
    dx, dy = LABEL_OFFSETS[size]

    ax.axvline(a, color=c, linestyle="--", linewidth=1.8, alpha=0.92, zorder=2)
    ax.axhline(m, color=c, linestyle="--", linewidth=1.8, alpha=0.92, zorder=2)

    # Corner dot
    ax.scatter([a], [m], color=c, s=90, zorder=6, clip_on=False)

    # Label box — white fill, colored border, colored bold text
    ax.annotate(
        f"Size {size}\nA ≥ {a:.0f} mm²\nI ≥ {m:.0f} mm⁴",
        xy=(a, m), xytext=(a + dx, m + dy),
        fontsize=7.8, fontweight="bold", color=c,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                  edgecolor=c, linewidth=2.2, alpha=0.96),
        zorder=7
    )

# ─────────────────────────────────────────────────────────────────
# LAYER 3 — Tube scatter points
# ─────────────────────────────────────────────────────────────────
for d in round_data:
    wall  = float(d["label"].split("x")[1])
    color = WALL_COLORS.get(round(wall, 2), "#333333")
    ax.scatter(d["area"], d["moi"], s=mass_to_size(d["mass"]),
               color=color, marker="o", alpha=0.78,
               linewidths=0.5, edgecolors="white", zorder=4)

for d in square_data:
    ax.scatter(d["area"], d["moi"], s=mass_to_size(d["mass"]),
               color="#1f78b4", marker="s", alpha=0.82,
               linewidths=0.5, edgecolors="white", zorder=4)

for d in rect_data:
    ax.scatter(d["area"], d["moi"], s=mass_to_size(d["mass"]),
               color="#33a02c", marker="D", alpha=0.82,
               linewidths=0.5, edgecolors="white", zorder=4)

# ─────────────────────────────────────────────────────────────────
# LEGENDS
# ─────────────────────────────────────────────────────────────────

# Tube type & wall thickness (upper left)
wall_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
           markersize=8, label=f"Round  t = {t:.2f} mm")
    for t, c in sorted(WALL_COLORS.items())
]
shape_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f78b4",
           markersize=8, label="Square  4130N"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#33a02c",
           markersize=8, label="Rectangular  4130N"),
]
legend1 = ax.legend(handles=wall_handles + shape_handles,
                    loc="upper left", fontsize=7.5,
                    title="Tube Type & Wall  (4130N)", title_fontsize=8,
                    framealpha=0.93, ncol=2)
ax.add_artist(legend1)

# FSAE size color key (upper center)
size_handles = [
    mpatches.Patch(facecolor=SIZE_COLORS[s], edgecolor=SIZE_COLORS[s],
                   alpha=0.88, label=f"Size {s}")
    for s in ["A", "B", "C", "D"]
]
legend2 = ax.legend(handles=size_handles,
                    loc="upper center", fontsize=9.5,
                    title="FSAE 2026 Chassis Size", title_fontsize=9,
                    framealpha=0.93, ncol=4, borderpad=0.8)
ax.add_artist(legend2)

# Mass dot size note (lower left)
ax.text(0.01, 0.01,
        f"Dot size  ∝  mass / meter\nMin: {mass_min*1000:.1f} g/m     Max: {mass_max*1000:.1f} g/m",
        transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="#cccccc",
                  boxstyle="round,pad=0.3"))

# ─────────────────────────────────────────────────────────────────
# AXES & TITLE
# ─────────────────────────────────────────────────────────────────
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.set_xlabel("Cross-Sectional Area  (mm²)", fontsize=11)
ax.set_ylabel("Area Moment of Inertia  (mm⁴)", fontsize=11)
ax.set_title(
    "VR3 Engineering 4130N Tube Catalog — Area vs. Moment of Inertia\n"
    "Round (●)  ·  Square (■)  ·  Rectangular (◆)     "
    "Dot size = mass/meter     FSAE 2026 chassis sizes color-coded",
    fontsize=11, fontweight="bold", pad=12
)
ax.grid(True, color="#cccccc", linewidth=0.5, alpha=0.55)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("fsae_tube_chart.png", dpi=180, bbox_inches="tight")
plt.savefig("fsae_tube_chart.pdf", bbox_inches="tight")
print("Saved: fsae_tube_chart.png  and  fsae_tube_chart.pdf")
plt.show()