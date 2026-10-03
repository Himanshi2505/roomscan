"""Compare two results of the same capture (e.g. video vs LiDAR) frame-independently.

Rooms are paired by closest floor area; walls >= 0.5 m are compared sorted by
length (frames differ, so wall identity is not available).
"""
import json
import sys

import numpy as np
from PIL import Image

a_dir, b_dir = sys.argv[1], sys.argv[2]
A = json.load(open(f"{a_dir}/result.json"))
B = json.load(open(f"{b_dir}/result.json"))
print(f"{'':10s} {A['tier']:>10s} {B['tier']:>10s}")
print(f"{'rooms':10s} {len(A['rooms']):10d} {len(B['rooms']):10d}")
print(f"{'footprint':10s} {A['footprint_area']['value']:10.2f} {B['footprint_area']['value']:10.2f}")
used = set()
for ra in sorted(A["rooms"], key=lambda r: -r["floor_area"]["value"]):
    cand = [rb for rb in B["rooms"] if rb["id"] not in used]
    if not cand:
        break
    rb = min(cand, key=lambda r: abs(r["floor_area"]["value"] - ra["floor_area"]["value"]))
    used.add(rb["id"])
    la = sorted([w["length"]["value"] for w in ra["walls"] if w["length"]["value"] >= 0.5], reverse=True)
    lb = sorted([w["length"]["value"] for w in rb["walls"] if w["length"]["value"] >= 0.5], reverse=True)
    n = min(len(la), len(lb))
    d = np.array(la[:n]) - np.array(lb[:n])
    print(f"\n{ra['id']} ({ra['floor_area']['value']:.2f} m2) vs {rb['id']} ({rb['floor_area']['value']:.2f} m2)")
    print("  walls A:", la)
    print("  walls B:", lb)
    if n:
        print(f"  sorted-wall diff: median {100 * np.median(np.abs(d)):.1f} cm, "
              f"max {100 * np.abs(d).max():.1f} cm, median rel {100 * np.median(np.abs(d) / np.array(lb[:n])):.1f}%")
ia, ib = Image.open(f"{a_dir}/plan.png"), Image.open(f"{b_dir}/plan.png")
im = Image.new("RGB", (ia.width + ib.width, max(ia.height, ib.height)), "white")
im.paste(ia, (0, 0))
im.paste(ib, (ia.width, 0))
im.thumbnail((1800, 1000))
im.save("out/compare_tiers.png")
