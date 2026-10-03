"""Diagnose ceiling-height error: where do floor/ceiling plane heights come from?

    python tools/ceiling_diag.py data/public/stray/42445021 [laser_gt_height]

Uses raw per-frame LiDAR points (confidence 2), not the fused cloud, so each
point keeps its measurement range and incidence angle. Reports:
  * floor and ceiling height histograms around their peaks (5 mm bins): extra
    peaks = rugs, fixtures, mouldings;
  * plane height as a function of range and incidence angle: a range- or
    angle-dependent bias shows up as a trend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from roomscan.geometry.cloud import depth_to_cloud
from roomscan.io import strayscanner as ss


def main():
    cap = ss.load(Path(sys.argv[1]))
    gt = float(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = []
    for f in cap.depth_frames[::4]:
        d, K = cap.load_depth(f, min_conf=2)
        c = depth_to_cloud(d, K, f.T_wc, f.index, stride=2, max_grazing_deg=89)
        cam = f.T_wc[:3, 3]
        v = c.xyz - cam
        rng = np.linalg.norm(v, axis=1)
        inc = np.degrees(np.arccos(np.clip(np.abs(np.sum(v / rng[:, None] * c.normals, 1)), 0, 1)))
        rows.append(np.c_[c.xyz[:, 1], c.normals[:, 1], rng, inc])
    A = np.concatenate(rows)
    y, ny, rng, inc = A.T

    def peak(m):
        h, e = np.histogram(y[m], bins=np.arange(y[m].min(), y[m].max() + 0.005, 0.005))
        i = np.argmax(h)
        y0 = 0.5 * (e[i] + e[i + 1])
        near = y[m][np.abs(y[m] - y0) < 0.03]
        return float(np.median(near)), h, e

    up, down = ny > 0.9, ny < -0.9
    fl, hf, ef = peak(up)
    ce, hc, ec = peak(down & (y > fl + 1.5))
    print(f"floor {fl:.4f}  ceiling {ce:.4f}  height {ce - fl:.4f}" + (f"  (laser {gt:.4f}, err {100 * (ce - fl - gt):+.2f} cm)" if gt else ""))
    for name, h, e, y0 in (("floor", hf, ef, fl), ("ceiling", hc, ec, ce)):
        sel = np.abs(0.5 * (e[:-1] + e[1:]) - y0) < 0.06
        print(f"\n{name} histogram (+-6 cm, 5 mm bins; offset from peak in cm: count):")
        print("  " + "  ".join(f"{100 * (0.5 * (a + b) - y0):+.1f}:{n}" for a, b, n in zip(e[:-1][sel], e[1:][sel], h[sel])))
    for name, m, y0 in (("floor", up & (np.abs(y - fl) < 0.05), fl), ("ceiling", down & (np.abs(y - ce) < 0.05), ce)):
        print(f"\n{name}: median plane height (cm from peak) by range / incidence")
        for lo, hi in ((0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 6)):
            s = m & (rng >= lo) & (rng < hi)
            if s.sum() > 200:
                print(f"  range {lo}-{hi} m: {100 * (np.median(y[s]) - y0):+.2f} cm  (n={s.sum()})")
        for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 75), (75, 90)):
            s = m & (inc >= lo) & (inc < hi)
            if s.sum() > 200:
                print(f"  incidence {lo}-{hi} deg: {100 * (np.median(y[s]) - y0):+.2f} cm  (n={s.sum()})")


if __name__ == "__main__":
    main()
