"""Overlay a result.json plan on the laser ground-truth map to see what it covers.

    python tools/align_plan_to_laser.py out/public/42445021 data/public/arkitscenes/laser/170107

Both frames are Manhattan-aligned, so they differ by a quarter turn and a 2D
shift. For each of the 4 quarter turns, the plan's wall outline raster is
cross-correlated (FFT) with the laser wall-height raster; the best peak gives
the alignment. Writes <result_dir>/laser_overlay.png (laser walls red, laser
floor green, plan outline white) and prints the transform and its score.
Diagnostic only: decides which predicted room corresponds to the GT room.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tools.laser_gt import load_aligned

RES = 0.02


def laser_maps(stem: Path):
    c = load_aligned(stem)
    y = c.xyz[:, 1]
    floor = np.percentile(y, 2)
    lo = c.xyz[:, [0, 2]].min(0) - 1.0
    hi = c.xyz[:, [0, 2]].max(0) + 1.0
    shape = tuple(((hi - lo) / RES).astype(int)[::-1] + 1)

    def raster(m):
        img = np.zeros(shape, np.float32)
        ij = ((c.xyz[m][:, [0, 2]] - lo) / RES).astype(int)
        np.add.at(img, (ij[:, 1], ij[:, 0]), 1)
        return img
    wall = raster((y > floor + 0.3) & (y < floor + 2.0) & (np.abs(c.normals[:, 1]) < 0.3))
    flo = raster(y < floor + 0.05)
    return (wall > 2).astype(np.float32), (flo > 0).astype(np.float32), lo


def rot(k, P):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    return P @ np.array([[c, -s], [s, c]]).T


def main():
    res_dir, stem = Path(sys.argv[1]), Path(sys.argv[2])
    plan = json.loads((res_dir / "result.json").read_text())
    wall, flo, lo = laser_maps(stem)
    H, W = wall.shape
    best = None
    for k in range(4):
        img = np.zeros_like(wall)
        polys = [rot(k, np.array(r["polygon"])) for r in plan["rooms"]]
        allp = np.concatenate(polys)
        off = allp.min(0)
        for P in polys:
            q = ((P - off) / RES).astype(np.int32)
            cv2.polylines(img, [q.reshape(-1, 1, 2)], True, 1.0, 2)
        F = np.fft.rfft2(wall) * np.conj(np.fft.rfft2(img))
        corr = np.fft.irfft2(F, s=wall.shape)
        r, c = np.unravel_index(np.argmax(corr), corr.shape)
        score = corr[r, c] / max(img.sum(), 1)
        if best is None or score > best[0]:
            best = (score, k, off, (c, r), polys)
    score, k, off, (dc, dr), polys = best
    shift = lo + np.array([dc, dr]) * RES - off          # plan(rotated) -> laser frame
    vis = np.zeros((H, W, 3), np.uint8)
    vis[..., 1] = (flo * 90).astype(np.uint8)
    vis[..., 2] = (wall * 255).astype(np.uint8)
    for r_, P in zip(plan["rooms"], polys):
        q = ((P + shift - lo) / RES).astype(np.int32)
        cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
        ctr = q.mean(0).astype(int)
        cv2.putText(vis, f"{r_['id']} {r_['floor_area']['value']:.1f}m2", tuple(ctr - [40, 0]), 0, 0.5,
                    (255, 255, 0), 1)
    cv2.imwrite(str(res_dir / "laser_overlay.png"), vis)
    print(f"quarter turns {k}, shift {np.round(shift, 3)}, overlap score {score:.3f}")


if __name__ == "__main__":
    main()
