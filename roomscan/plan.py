"""OrientedCloud (any tier) -> floor plan dict following schema/plan.schema.json."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from roomscan import uncertainty as U
from roomscan.geometry import rooms as R
from roomscan.geometry.cloud import OrientedCloud
from roomscan.geometry.levels import _plane_height, floor_ceiling
from roomscan.geometry.openings import edge_openings
from roomscan.geometry.polygon import RoomPolygon, fit_polygon, resolve_overlaps
from roomscan.geometry.walls import extract_faces


def _room_levels(c: OrientedCloud, P: RoomPolygon, floor_y: float):
    """Local floor and ceiling inside the room footprint (floors are not flat
    across a whole flat; thresholds and sunken bathrooms differ by 1-3 cm)."""
    from matplotlib.path import Path as MPath
    inside = MPath(P.vertices).contains_points(c.xyz[:, [0, 2]])
    sub = c.xyz[inside]
    n = c.normals[inside]
    fl = _plane_height(sub[(n[:, 1] > 0.9) & (np.abs(sub[:, 1] - floor_y) < 0.15), 1],
                       "floor", min_support=300)
    ce = _plane_height(sub[(n[:, 1] < -0.9) & (sub[:, 1] > floor_y + 1.8), 1],
                       "ceiling", min_support=300)
    return fl, ce


def build_plan(c: OrientedCloud, tier: str, path_length: float = 0.0,
               single_room: bool = False, extra_rel: float = 0.0) -> dict:
    """single_room=True: the capture is one room by construction (a photo-tier
    folder), so skip watershed splitting and keep the largest interior region."""
    model = U.load_model(tier)
    if extra_rel:   # e.g. trajectory distortion: widens every length/area interval
        model["scale_rel"] = float(np.hypot(model["scale_rel"], extra_rel))
    floor, ceil = floor_ceiling(c)
    if floor is None:
        raise RuntimeError("no floor plane found: capture must show the floor")
    faces = extract_faces(c, floor.y, ceil.y if ceil else None)
    g = R.make_grid(c)
    W = R.wall_raster(g, faces, floor_y=floor.y, ceil_y=ceil.y if ceil else None)
    I = R.interior_mask(c, g, floor.y, W)
    if single_room:
        from scipy import ndimage as ndi
        lab, n = ndi.label(I)
        if n > 1:
            sizes = ndi.sum(I, lab, range(1, n + 1))
            lab = (lab == 1 + int(np.argmax(sizes))).astype(np.int32)
    else:
        lab = R.segment_rooms(I, W)
    ks = list(range(1, lab.max() + 1))
    masks = [lab == k for k in ks]
    polys = [fit_polygon(m, g, faces) for m in masks]
    keep = [i for i, p in enumerate(polys) if p is not None]
    polys = resolve_overlaps([polys[i] for i in keep], [masks[i] for i in keep], g)
    polys = [p for p in polys if p.area() >= R.MIN_ROOM_AREA]

    rooms, all_openings = [], []
    for ri, P in enumerate(polys):
        rid = f"R{ri + 1}"
        fl, ce = _room_levels(c, P, floor.y)
        fy = fl.y if fl else floor.y
        # ceiling height: local ceiling, else global ceiling, else unobserved
        if ce is not None:
            H = ce.y - fy
            Hs = np.hypot(np.hypot(ce.sigma, fl.sigma if fl else floor.sigma),
                          np.sqrt(2) * model["sigma_face"])
        elif ceil is not None:
            H = ceil.y - fy
            Hs = np.hypot(0.01, np.sqrt(2) * model["sigma_face"])
        else:
            H, Hs = None, None
        walls = []
        n = len(P.vertices)
        perim = 0.0
        sig_len = []
        for i, e in enumerate(P.edges):
            a, b = P.vertices[i], P.vertices[(i + 1) % n]
            L = float(np.linalg.norm(b - a))
            perim += L
            # length is set by the two neighbouring edges' positions
            ep, en = P.edges[i - 1], P.edges[(i + 1) % n]
            stat = np.hypot(ep.sigma, en.sigma)
            nf = int(ep.face is not None) + int(en.face is not None)
            s = U.length_sigma(L, stat, model, n_faces=max(nf, 1), path=path_length)
            sig_len.append(s)
            walls.append({"id": f"{rid}.W{i + 1}", "start": [round(float(a[0]), 3), round(float(a[1]), 3)],
                          "end": [round(float(b[0]), 3), round(float(b[1]), 3)],
                          "length": U.measure(L, s, model),
                          "height": U.measure(H, Hs, model) if H is not None else None,
                          "measured": e.face is not None, "openings": []})
        A = P.area()
        As = U.area_sigma(A, float(np.mean(sig_len)), perim, model)
        ops = []
        for i, e in enumerate(P.edges):
            for o in edge_openings(c, e, i, fy):
                oid = f"{rid}.O{len(ops) + 1}"
                ws = float(np.hypot(o.width_sigma, np.sqrt(2) * model["sigma_face"]))
                mid = 0.5 * (o.s_lo + o.s_hi)
                ctr = (e.coord, mid) if e.axis == "x" else (mid, e.coord)
                ops.append({"id": oid, "kind": o.kind, "wall_id": walls[i]["id"],
                            "width": U.measure(o.width, ws, model),
                            "height": U.measure(o.head, 0.03, model) if o.head else None,
                            "sill": U.measure(o.sill, 0.03, model) if o.sill is not None else None,
                            "center": [round(float(ctr[0]), 3), round(float(ctr[1]), 3)],
                            "outward": _outward(e), "connects_to": None})
                walls[i]["openings"].append(oid)
        rooms.append({"id": rid, "name": rid, "polygon": np.round(P.vertices, 3).tolist(),
                      "floor_area": U.measure(A, As, model, 2),
                      "ceiling_height": U.measure(H, Hs, model) if H is not None else None,
                      "ceiling_observed": ce is not None or ceil is not None,
                      "walls": walls, "openings": ops, "_shape": P.shape})
        all_openings += [(rid, o) for o in ops]

    adjacency = _adjacency(rooms)
    for r in rooms:
        r.pop("_shape")
    fp = sum(r["floor_area"]["value"] for r in rooms)
    fps = float(np.sqrt(sum(r["floor_area"]["sigma"] ** 2 for r in rooms)))
    return {"tier": tier, "calibrated": model["calibrated"],
            "rooms": rooms, "adjacency": adjacency,
            "footprint_area": U.measure(fp, fps, model, 2),
            "levels": {"floor_y": round(floor.y, 4),
                       "ceiling_y": round(ceil.y, 4) if ceil else None}}


def _outward(e) -> list[float]:
    v = [0.0, 0.0]
    v[0 if e.axis == "x" else 1] = -float(e.inward)
    return v


def _adjacency(rooms: list[dict]) -> list[dict]:
    """Two rooms are adjacent through an opening if the point 0.35 m beyond the
    opening's centre falls inside the other room's footprint. When neither
    side's opening was detected but footprints touch along >= 0.6 m, they're
    still reported adjacent (via=None) so the stitched plan stays connected."""
    adj, seen = [], set()
    for r in rooms:
        for o in r["openings"]:
            p = np.array(o["center"]) + 0.35 * np.array(o["outward"])
            for q in rooms:
                if q is r or not q["_shape"].buffer(0.05).contains(Point(*p)):
                    continue
                o["connects_to"] = q["id"]
                key = tuple(sorted((r["id"], q["id"])))
                if key not in seen:
                    seen.add(key)
                    adj.append({"rooms": list(key), "via": o["id"], "kind": o["kind"]})
    for i, r in enumerate(rooms):
        for q in rooms[i + 1:]:
            key = tuple(sorted((r["id"], q["id"])))
            if key in seen:
                continue
            shared = r["_shape"].buffer(0.25).intersection(q["_shape"].buffer(0.25))
            if not shared.is_empty and max(shared.bounds[2] - shared.bounds[0],
                                           shared.bounds[3] - shared.bounds[1]) >= 0.6:
                if r["_shape"].distance(q["_shape"]) < 0.05:   # open-plan cut
                    seen.add(key)
                    adj.append({"rooms": list(key), "via": None, "kind": "open"})
    return adj
