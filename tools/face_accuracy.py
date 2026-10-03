"""Face-level accuracy of a LiDAR recording against laser ground truth.

    python tools/face_accuracy.py data/public/stray/42445021 out/public/42445021 \
        benchmark/public_gt/arkitscenes_421380.json

Needs <result_dir>/laser_registration.json (tools/register_to_laser.py). For
each GT wall, the recording's best-supported face with the same axis and
normal within 15 cm (after registration) is taken; room dimensions are the
distances between opposite faces *in the recording's own cloud*. A rigid
registration cannot change distances, so this is not circular. Independent of
room segmentation: it measures the sensor + face extraction layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from roomscan.geometry.levels import floor_ceiling, manhattan_yaw, yaw_matrix
from roomscan.geometry.walls import extract_faces
from roomscan.pipeline import _lidar_cloud


def main():
    cap, res_dir, gt_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    reg = json.loads((res_dir / "laser_registration.json").read_text())
    T = np.array(reg["T_plan_to_laser"])
    gt = json.loads(gt_path.read_text())
    c, _ = _lidar_cloud(cap, reg["drift"], {}, {})
    c = c.transformed(yaw_matrix(manhattan_yaw(c)))
    fl, ce = floor_ceiling(c)
    faces = extract_faces(c, fl.y, ce.y if ce else None)
    spec = gt["gt_faces"]                       # name -> [axis, sign, position]
    found = {}
    for name, (ax, sg, pos) in spec.items():
        best = None
        for f in faces:
            mid = np.array([f.pos, 0.5 * (f.lo + f.hi)]) if f.axis == "x" else \
                np.array([0.5 * (f.lo + f.hi), f.pos])
            m = mid @ T[:2, :2].T + T[:2, 2]
            p = m[0] if ax == "x" else m[1]
            if f.axis == ax and f.sign == sg and abs(p - pos) < 0.15 and \
                    (best is None or f.support > best[1].support):
                best = (p, f)
        found[name] = best
    out = {"registration": {k: reg[k] for k in ("inliers_5cm", "rmse_m", "accepted")}}
    for a, b, gname in (("N", "S", "N-S"), ("W", "E", "W-E")):
        if found.get(a) and found.get(b):
            ours = found[b][0] - found[a][0]
            truth = spec[b][2] - spec[a][2]
            out[gname] = {"ours": round(ours, 4), "gt": round(truth, 4), "err_cm": round(100 * (ours - truth), 2)}
        else:
            out[gname] = None
    if ce is not None:
        h = ce.y - fl.y
        out["ceiling"] = {"ours": round(h, 4), "gt": gt["laser_ceiling_height"],
                          "err_cm": round(100 * (h - gt["laser_ceiling_height"]), 2)}
    (res_dir / "face_accuracy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
