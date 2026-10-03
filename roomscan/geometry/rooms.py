"""Split a fused multi-room floor into rooms.

1. Raster the observed floor + object footprints (3 cm cells) into an interior mask;
   wall faces are burned in as barriers.
2. Watershed on the distance transform: seeds are the "deep" parts of the floor
   (> SEED_R from any wall), so doorways (~0.8-0.9 m wide, distance <= 0.45 m)
   never hold a seed and become the cut lines between rooms.
3. Merge neighbouring regions whose shared boundary is longer than MAX_DOOR_W:
   a cut that long is an open-plan transition or an over-segmented corridor,
   not a door. What remains are rooms connected through door-width cuts.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from roomscan.geometry.cloud import OrientedCloud
from roomscan.geometry.walls import WallFace

RES = 0.03
SEED_R = 0.55
MAX_DOOR_W = 1.6
MIN_ROOM_AREA = 1.2


@dataclass
class Grid:
    origin: np.ndarray      # world (x, z) of cell (0, 0)
    shape: tuple[int, int]  # rows (z), cols (x)
    res: float = RES

    def to_px(self, x, z):
        return ((np.asarray(x) - self.origin[0]) / self.res).astype(int), \
               ((np.asarray(z) - self.origin[1]) / self.res).astype(int)

    def to_world(self, col, row):
        return self.origin[0] + (np.asarray(col) + 0.5) * self.res, \
               self.origin[1] + (np.asarray(row) + 0.5) * self.res


def make_grid(c: OrientedCloud, pad: float = 0.5) -> Grid:
    lo = c.xyz[:, [0, 2]].min(0) - pad
    hi = c.xyz[:, [0, 2]].max(0) + pad
    cols, rows = ((hi - lo) / RES).astype(int) + 1
    return Grid(lo, (rows, cols))


def _raster(g: Grid, xyz: np.ndarray, min_hits: int = 1) -> np.ndarray:
    m = np.zeros(g.shape, np.int32)
    cx, rz = g.to_px(xyz[:, 0], xyz[:, 2])
    np.add.at(m, (rz, cx), 1)
    return m >= min_hits


def wall_raster(g: Grid, faces: list[WallFace], thick_px: int = 2) -> np.ndarray:
    img = np.zeros(g.shape, np.uint8)
    for f in faces:
        if f.axis == "x":
            p, q = g.to_px([f.pos, f.pos], [f.lo, f.hi])
        else:
            p, q = g.to_px([f.lo, f.hi], [f.pos, f.pos])
        cv2.line(img, (int(p[0]), int(q[0])), (int(p[1]), int(q[1])), 1, thick_px)
    return img.astype(bool)


def carve_free(c: OrientedCloud, g: Grid, floor_y: float, max_rays: int = 400_000,
               seed: int = 0) -> np.ndarray:
    """2D free space swept by sensor rays (camera -> hit point, minus the last 10 cm).

    Only hits between 0.3 m and 1.8 m above the floor are used, so rays travel
    roughly horizontally through the room. Needs per-point source frames and
    camera poses (c.frame, c.cams); returns an all-False grid otherwise.
    """
    free = np.zeros(g.shape, np.uint8)
    if not c.cams:
        return free.astype(bool)
    y = c.xyz[:, 1]
    idx = np.flatnonzero((y > floor_y + 0.3) & (y < floor_y + 1.8)
                         & np.isin(c.frame, list(c.cams)))
    if len(idx) > max_rays:     # fixed seed: same capture in, same plan out
        idx = np.random.default_rng(seed).choice(idx, max_rays, replace=False)
    cam = np.array([c.cams[f][[0, 2], 3] for f in c.frame[idx]])
    hit = c.xyz[idx][:, [0, 2]]
    v = hit - cam
    L = np.linalg.norm(v, axis=1, keepdims=True)
    end = cam + v * np.clip((L - 0.10) / np.maximum(L, 1e-6), 0, 1)
    c0, r0 = g.to_px(cam[:, 0], cam[:, 1])
    c1, r1 = g.to_px(end[:, 0], end[:, 1])
    for a, b, p, q in zip(c0, r0, c1, r1):
        cv2.line(free, (int(a), int(b)), (int(p), int(q)), 1, 1)
    return free.astype(bool)


def interior_mask(c: OrientedCloud, g: Grid, floor_y: float, walls: np.ndarray,
                  cam_xz: np.ndarray | None = None) -> np.ndarray:
    y = c.xyz[:, 1]
    floor = _raster(g, c.xyz[(c.normals[:, 1] > 0.85) & (np.abs(y - floor_y) < 0.04)], 2)
    # Furniture hides floor; its footprint is still room interior.
    objs = _raster(g, c.xyz[(y > floor_y + 0.05) & (y < floor_y + 1.5)
                            & (np.abs(c.normals[:, 1]) > 0.85)], 2)
    # Barrier = every wall-height point with a horizontal normal, not just the
    # fitted faces: a window sill or a cupboard stops rays that escaped through
    # glass from leaking "free space" outside the building.
    barrier = walls | _raster(g, c.xyz[(y > floor_y + 0.25) & (y < floor_y + 2.0)
                                       & (np.abs(c.normals[:, 1]) < 0.3)], 2)
    m = floor | objs | (carve_free(c, g, floor_y) & ~barrier)
    if cam_xz is None and c.cams:
        cam_xz = np.array([T[[0, 2], 3] for T in c.cams.values()])
    if cam_xz is not None:            # where the operator walked is interior
        cx, rz = g.to_px(cam_xz[:, 0], cam_xz[:, 1])
        m[rz, cx] = True
    m = ndi.binary_closing(m, np.ones((5, 5)), iterations=2)
    m &= ~walls
    m = ndi.binary_opening(m, np.ones((3, 3)))
    # keep components touched by the walked trajectory (drops reflections
    # behind mirrors / through windows) or, without poses, the big ones.
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1)) * RES * RES
    keep = sizes >= MIN_ROOM_AREA
    if cam_xz is not None:
        cx, rz = g.to_px(cam_xz[:, 0], cam_xz[:, 1])
        walked = np.unique(lab[rz, cx])
        keep &= np.isin(np.arange(1, n + 1), walked)
    return np.isin(lab, np.flatnonzero(keep) + 1)


def segment_rooms(interior: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Label image: 0 = outside, 1..K = rooms."""
    free = interior & ~walls
    dist = ndi.distance_transform_edt(free) * RES
    seeds, ns = ndi.label(dist > SEED_R)
    if ns == 0:
        return free.astype(np.int32)
    # Masked watershed: floods outward from seeds over -distance, never leaves
    # free space, so labels can only meet where two seeds' floods collide,
    # i.e. at the narrowest point between rooms (the doorway).
    lab = watershed(-dist, seeds, mask=free)
    lab = _fill_boundaries(lab, free)
    return _merge_wide_cuts(lab)


def _fill_boundaries(lab: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Assign watershed boundary pixels (-1/0 inside free space) to a neighbour."""
    hole = free & (lab == 0)
    if hole.any():
        idx = ndi.distance_transform_edt(~(lab > 0), return_distances=False,
                                         return_indices=True)
        lab = lab.copy()
        lab[hole] = lab[idx[0][hole], idx[1][hole]]
    return lab


def _shared_boundary(lab: np.ndarray) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        for p, q in zip(a[m], b[m]):
            k = (min(p, q), max(p, q))
            out[k] = out.get(k, 0) + 1
    return out


def _merge_wide_cuts(lab: np.ndarray) -> np.ndarray:
    lab = lab.copy()
    while True:
        cuts = _shared_boundary(lab)
        wide = [(n, k) for k, n in cuts.items() if n * RES > MAX_DOOR_W]
        if not wide:
            break
        _, (a, b) = max(wide)
        lab[lab == b] = a
    # drop slivers, relabel 1..K
    out = np.zeros_like(lab)
    k = 0
    for r in np.unique(lab[lab > 0]):
        if (lab == r).sum() * RES * RES >= MIN_ROOM_AREA:
            k += 1
            out[lab == r] = k
    return out
