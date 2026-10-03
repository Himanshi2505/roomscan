"""Room mask -> rectilinear polygon whose edges are snapped to measured wall faces."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from roomscan.geometry.rooms import Grid
from roomscan.geometry.walls import WallFace

SNAP_TOL = 0.30          # how far outside the mask bbox a wall face may lie


@dataclass
class Edge:
    axis: str            # 'x' -> edge lies at constant x (runs along z); 'z' likewise
    coord: float
    sigma: float         # 1-sigma of coord
    inward: int          # +1 if the room lies on the +axis side of the edge
    face: WallFace | None = None   # measured face it was snapped to (None = virtual)
    s_lo: float = 0.0    # extent along the edge
    s_hi: float = 0.0


@dataclass
class RoomPolygon:
    edges: list[Edge]
    vertices: np.ndarray = field(default=None)   # (K,2) world (x,z)
    shape: object = None                          # shapely Polygon
    lines: list = field(default_factory=list)     # candidate lines it was built from

    def area(self) -> float:
        x, z = self.vertices[:, 0], self.vertices[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))

    def edge_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.roll(self.vertices, -1, 0) - self.vertices, axis=1)


@dataclass
class _Line:
    axis: str
    pos: float
    face: WallFace | None


def _candidate_lines(mask: np.ndarray, g: Grid, faces: list[WallFace]) -> dict:
    rows, cols = np.nonzero(mask)
    x0, z0 = g.to_world(cols.min(), rows.min())
    x1, z1 = g.to_world(cols.max(), rows.max())
    lines = {"x": [], "z": []}
    for f in faces:
        lo, hi = (z0, z1) if f.axis == "x" else (x0, x1)
        plo, phi = (x0, x1) if f.axis == "x" else (z0, z1)
        if f.pos < plo - SNAP_TOL or f.pos > phi + SNAP_TOL:
            continue
        if min(f.hi, hi) - max(f.lo, lo) < 0.3:
            continue
        lines[f.axis].append(_Line(f.axis, f.pos, f))
    # mask-extent lines only where no measured face backs that side (open cuts)
    for axis, side in (("x", x0 - g.res / 2), ("x", x1 + g.res / 2),
                       ("z", z0 - g.res / 2), ("z", z1 + g.res / 2)):
        if not any(abs(l.pos - side) < 0.15 for l in lines[axis]):
            lines[axis].append(_Line(axis, float(side), None))
    for k in lines:   # dedupe: within 4 cm keep the measured / better supported line
        ls = sorted(lines[k], key=lambda l: l.pos)
        out = []
        for l in ls:
            if out and l.pos - out[-1].pos < 0.04:
                sup = lambda q: q.face.support if q.face else -1
                if sup(l) > sup(out[-1]):
                    out[-1] = l
            else:
                out.append(l)
        lines[k] = out
    return lines


def _cells(mask, g, xs, zs):
    """Coverage of the room mask in each arrangement cell, plus cell areas (px)."""
    cx = np.clip(((np.array([l.pos for l in xs]) - g.origin[0]) / g.res).round().astype(int), 0, mask.shape[1])
    rz = np.clip(((np.array([l.pos for l in zs]) - g.origin[1]) / g.res).round().astype(int), 0, mask.shape[0])
    S = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    tot = S[rz[1:, None], cx[None, 1:]] - S[rz[:-1, None], cx[None, 1:]]         - S[rz[1:, None], cx[None, :-1]] + S[rz[:-1, None], cx[None, :-1]]
    area = np.outer(np.diff(rz), np.diff(cx))
    return tot, area


def _score(mask, g, xs, zs, lam):
    tot, area = _cells(mask, g, xs, zs)
    inside = tot > 0.5 * np.maximum(area, 1)
    inter = tot[inside].sum()
    union = mask.sum() + area[inside].sum() - inter
    return inter / max(union, 1) - lam * (len(xs) + len(zs)), inside


LAMBDA = 0.010   # IoU traded per extra line (MDL-style simplicity prior)


def fit_polygon(mask: np.ndarray, g: Grid, faces: list[WallFace]) -> RoomPolygon | None:
    from shapely.geometry import box
    from shapely.ops import unary_union
    if mask.sum() < 50:
        return None
    # the wall raster eats ~1 cell next to each face; give it back
    mask = ndi.binary_dilation(mask, np.ones((3, 3)), iterations=1)
    lines = _candidate_lines(mask, g, faces)
    xs, zs = lines["x"], lines["z"]
    best, _ = _score(mask, g, xs, zs, LAMBDA)
    improved = True
    while improved:                      # greedy backward elimination of lines
        improved = False
        for k, ls in (("x", xs), ("z", zs)):
            for i in range(len(ls)):
                if len(ls) <= 2:
                    break
                trial = ls[:i] + ls[i + 1:]
                sc, _ = _score(mask, g, trial if k == "x" else xs, zs if k == "x" else trial, LAMBDA)
                if sc >= best:
                    best, improved = sc, True
                    if k == "x":
                        xs = trial
                    else:
                        zs = trial
                    break
            if improved:
                break
    _, inside = _score(mask, g, xs, zs, LAMBDA)
    boxes = [box(xs[j].pos, zs[i].pos, xs[j + 1].pos, zs[i + 1].pos)
             for i, j in zip(*np.nonzero(inside))]
    if not boxes:
        return None
    poly = unary_union(boxes)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    poly = poly.simplify(1e-6).buffer(0)
    return polygon_from_shape(poly, xs + zs)


def polygon_from_shape(poly, lines: list[_Line]) -> RoomPolygon:
    """Edges of a rectilinear shapely polygon, each tagged with its source face."""
    V = np.array(poly.exterior.coords)[:-1]
    by_pos = {"x": {}, "z": {}}
    for l in lines:
        by_pos[l.axis][round(l.pos, 6)] = l
    edges = []
    for a, b in zip(V, np.roll(V, -1, 0)):
        axis = "x" if abs(b[0] - a[0]) < 1e-6 else "z"
        k = 0 if axis == "x" else 1
        l = by_pos[axis].get(round(a[k], 6))
        mid = 0.5 * (a + b)
        step = np.zeros(2); step[k] = 0.02
        inward = 1 if poly.contains(_pt(mid + step)) else -1
        face = l.face if l else None
        # a measured face only counts if its normal points into this room
        if face is not None and face.sign != inward:
            face = None
        sig = face.pos_sigma if face else 0.03
        edges.append(Edge(axis, float(a[k]), sig, inward, face,
                          float(min(a[1 - k], b[1 - k])), float(max(a[1 - k], b[1 - k]))))
    rp = RoomPolygon(edges, V)
    rp.shape, rp.lines = poly, lines
    return rp


def _pt(p):
    from shapely.geometry import Point
    return Point(p[0], p[1])


def resolve_overlaps(polys: list[RoomPolygon], masks: list[np.ndarray], g: Grid) -> list[RoomPolygon]:
    """Give each overlap to the room whose observed floor covers more of it."""
    from shapely.geometry import box
    def cover(mask, shp):
        x0, z0, x1, z1 = shp.bounds
        c0, r0 = g.to_px(x0, z0)
        c1, r1 = g.to_px(x1, z1)
        return mask[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1].sum()
    shapes = [p.shape for p in polys]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            ov = shapes[i].intersection(shapes[j])
            if ov.area < 1e-4:
                continue
            ci, cj = cover(masks[i], ov.envelope), cover(masks[j], ov.envelope)
            if ci >= cj:
                shapes[j] = shapes[j].difference(shapes[i])
            else:
                shapes[i] = shapes[i].difference(shapes[j])
    out = []
    for p, shp in zip(polys, shapes):
        if shp.geom_type != "Polygon":
            shp = max(shp.geoms, key=lambda q: q.area)
        lines = p.lines + [l for q in polys for l in q.lines]
        out.append(polygon_from_shape(shp.simplify(1e-6).buffer(0), lines))
    return out
