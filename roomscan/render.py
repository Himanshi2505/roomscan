"""Render a plan dict to SVG/PNG: magicplan-style dimensioned whole-property plan."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import polylabel

ROOM_FILL = "#f4efe6"
WALL = "#3b3329"
DIM = "#8a6d3b"
DOOR = "#e2703a"
WINDOW = "#3fa7d6"
DAMAGE = "#c0392b"


def _fmt(m):
    if m is None:
        return "n/a"
    lo, hi = m["ci90"]
    return f"{m['value']:.2f} m (±{(hi - lo) / 2 * 100:.0f} cm)"


def render(plan: dict, out_path: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")
    ax.axis("off")
    for r in plan["rooms"]:
        V = np.array(r["polygon"])
        ax.add_patch(Polygon(V, closed=True, fc=ROOM_FILL, ec=WALL, lw=3, zorder=1))
        ctr = np.array(polylabel(ShapelyPolygon(V), 0.01).coords[0])
        H = r["ceiling_height"]
        label = f"{r['name']}\n{r['floor_area']['value']:.1f} m²"
        label += f"\nH {H['value']:.2f} m" if H else "\nH not observed"
        ax.text(*ctr, label, ha="center", va="center", fontsize=9, color=WALL,
                weight="bold", zorder=5)
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            L = w["length"]["value"]
            if L < 0.35:
                continue
            mid = 0.5 * (a + b)
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            nrm = np.array([-d[1], d[0]])
            # put the dimension inside the room
            if not Polygon(V).get_path().contains_point(mid + 0.1 * nrm):
                nrm = -nrm
            p = mid + 0.22 * nrm
            ang = np.degrees(np.arctan2(d[1], d[0]))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
            ax.text(*p, f"{L:.2f}", ha="center", va="center", fontsize=6.5,
                    color=DIM if w["measured"] else "#b0a090", rotation=ang, zorder=4)
        for o in r["openings"]:
            _draw_opening(ax, r, o)
    for dmg in plan.get("damage", []):
        if "plan_xy" in dmg:
            ax.plot(*dmg["plan_xy"], marker="X", color=DAMAGE, ms=10, zorder=6)
    fa = plan["footprint_area"]
    ax.set_title(f"{title}\n{len(plan['rooms'])} rooms · footprint {fa['value']:.1f} m² "
                 f"(90% CI {fa['ci90'][0]:.1f}–{fa['ci90'][1]:.1f}) · tier: {plan['tier']}"
                 + ("" if plan.get("calibrated") else " · intervals uncalibrated"),
                 fontsize=10, color=WALL)
    ax.autoscale_view()
    ax.invert_yaxis()       # world +Z drawn downward, like the top-down debug views
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if out_path.endswith(".png"):
        fig.savefig(out_path[:-4] + ".svg", bbox_inches="tight")
    plt.close(fig)


def _draw_opening(ax, r, o):
    w = next(x for x in r["walls"] if x["id"] == o["wall_id"])
    a, b = np.array(w["start"]), np.array(w["end"])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    c = np.array(o["center"])
    half = o["width"]["value"] / 2
    p, q = c - half * d, c + half * d
    color = WINDOW if o["kind"] == "window" else DOOR
    ax.plot(*np.stack([p, q]).T, color="white", lw=4, zorder=2, solid_capstyle="butt")
    ax.plot(*np.stack([p, q]).T, color=color, lw=1.5, zorder=3)
    if o["kind"] == "door":
        inward = -np.array(o["outward"])
        ang0 = np.degrees(np.arctan2(d[1], d[0]))
        sweep = np.degrees(np.arctan2(inward[1], inward[0]))
        start = min(ang0, sweep) if abs(ang0 - sweep) <= 180 else max(ang0, sweep)
        ax.add_patch(Arc(p, 2 * o["width"]["value"], 2 * o["width"]["value"],
                         theta1=start, theta2=start + 90, color=color, lw=0.8, zorder=3))
