"""Per-surface damage regions, concealed-damage flags and scope line items.

Detection (open_clip ViT-B-32, laion2b weights; zero-shot, no training):
  * Each view is tiled into square patches. A patch is only scored if most of
    its pixels lie on a plan surface (a wall within 8 cm of its plane, or the
    ceiling): furniture, floors, doors and windows are never classified.
  * Patch is "damage class c" if softmax(prompts) puts > P_MIN on c's prompts
    against a bank of benign wall prompts (clean paint, switch, frame, ...).
Metric extent:
  * Inside positive patches, pixels whose colour deviates from the surface's
    own median colour (Lab distance > DE_MIN) are the damage mask; their 3D
    points are projected onto the wall plane -> (along-wall, height) box.
  * Detections on the same surface whose boxes overlap are merged across
    views; for multi-view tiers a region must be seen in >= 2 views.
Rules (concealed damage) and scope are deterministic tables below, every flag
naming the rule that fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from roomscan import uncertainty as U

CLASSES = {
    "water_stain": ["a brown water stain on a wall", "a yellow water damage stain on a ceiling",
                    "water damage on drywall"],
    "mold": ["black mold growing on a wall", "mould spots on a damp wall corner"],
    "crack": ["a crack in a plaster wall", "a long crack in drywall"],
    "hole": ["a hole punched in drywall", "a hole in a wall"],
    "peeling_paint": ["peeling paint on a wall", "blistered flaking paint"],
}
BENIGN = ["a clean painted wall", "a plain white wall", "a wall with a light switch",
          "a wall with a picture frame", "a wall with a power socket", "a shadow on a wall",
          "a smooth ceiling", "a ceiling with a lamp", "wall tiles", "a wooden wardrobe door",
          "a curtain", "a window", "wood grain texture", "a wooden cabinet", "a paper calendar",
          "a poster on a wall", "a dark corner of a room", "the edge of a door frame",
          "a shelf", "a mirror", "a blurry photo", "a skirting board", "dark furniture",
          "a bathtub", "white tiles with grout lines"]
MARGIN = 0.02             # cosine: best damage prompt must beat best benign prompt by this
DEV_FRAC = (0.03, 0.50)   # stain = blob on a uniform surface, not a whole patch of material
BLUR_MIN = 60.0
PATCH = 160               # px at the 640-px analysis resolution
P_MIN = 0.45
DE_MIN = 12.0             # CIE Lab distance from the surface's median colour
MIN_AREA = 0.004          # m2 (~6x6 cm)

_clip = None


def _load_clip():
    global _clip
    if _clip is None:
        import open_clip
        import torch
        cache = Path(__file__).resolve().parent.parent / "weights" / "clip"
        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(cache))
        tok = open_clip.get_tokenizer("ViT-B-32")
        prompts = [p for c in CLASSES for p in CLASSES[c]] + BENIGN
        with torch.no_grad():
            t = model.encode_text(tok(prompts))
            t /= t.norm(dim=-1, keepdim=True)
        owner = [c for c in CLASSES for _ in CLASSES[c]] + [None] * len(BENIGN)
        _clip = (model.eval(), pre, t, owner)
    return _clip


@dataclass
class DView:
    """One image with per-pixel world geometry in the plan frame."""
    rgb: np.ndarray            # (H,W,3) uint8
    xyz: np.ndarray            # (H,W,3) plan frame (x, y-up, z)
    normal: np.ndarray         # (H,W,3)
    valid: np.ndarray          # (H,W) bool
    vid: int = 0


@dataclass
class Region:
    surface: str
    cls: str
    s_lo: float
    s_hi: float
    h_lo: float
    h_hi: float
    area: float
    prob: float
    views: set = field(default_factory=set)
    crop: np.ndarray | None = None      # best patch, for audit/debug output


def _surfaces(plan: dict, floor_y: float):
    out = []
    for r in plan["rooms"]:
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            out.append(("wall", w["id"], r, a, b))
        out.append(("ceiling", f"{r['id']}.CEIL", r, None, None))
    return out


def _assign(v: DView, plan: dict, floor_y: float, ceil_y: float | None):
    """Per-pixel surface index (-1 = none) and (s, h) coordinates on it."""
    from matplotlib.path import Path as MPath
    H, W = v.valid.shape
    sid = np.full((H, W), -1, int)
    s_c = np.zeros((H, W))
    h_c = v.xyz[..., 1] - floor_y
    surf = _surfaces(plan, floor_y)
    P = v.xyz[..., [0, 2]]
    best = np.full((H, W), np.inf)
    for k, (kind, _, room, a, b) in enumerate(surf):
        if kind == "wall":
            d = b - a
            L = np.linalg.norm(d)
            if L < 0.3:
                continue
            u = d / L
            rel = P - a
            s = rel @ u
            dist = np.abs(rel @ np.array([-u[1], u[0]]))
            ok = v.valid & (dist < 0.08) & (s > 0.02) & (s < L - 0.02) & \
                (np.abs(v.normal[..., [0, 2]] @ u) < 0.4) & (h_c > 0.05) & (dist < best)
            sid[ok], s_c[ok], best[ok] = k, s[ok], dist[ok]
        elif ceil_y is not None:
            inside = MPath(np.array(room["polygon"])).contains_points(P.reshape(-1, 2)).reshape(H, W)
            ok = v.valid & inside & (v.normal[..., 1] < -0.85) & \
                (np.abs(v.xyz[..., 1] - ceil_y) < 0.08)
            sid[ok] = k
            s_c[ok] = P[..., 0][ok]
            h_c = np.where(ok, P[..., 1], h_c)      # ceiling: (x, z) coordinates
    return surf, sid, s_c, h_c


def detect(views: list[DView], plan: dict, floor_y: float, ceil_y: float | None,
           min_views: int = 2) -> list[Region]:
    import torch
    from PIL import Image
    model, pre, T, owner = _load_clip()
    regions: list[Region] = []
    for v in views:
        if cv2.Laplacian(cv2.cvtColor(v.rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() < BLUR_MIN:
            continue                      # motion blur: textures smear into "stains"
        surf, sid, s_c, h_c = _assign(v, plan, floor_y, ceil_y)
        H, W = sid.shape
        lab = cv2.cvtColor(v.rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        tiles, boxes = [], []
        for y0 in range(0, H - PATCH + 1, PATCH // 2):
            for x0 in range(0, W - PATCH + 1, PATCH // 2):
                blk = sid[y0:y0 + PATCH, x0:x0 + PATCH]
                vals, cnt = np.unique(blk[blk >= 0], return_counts=True)
                if len(vals) == 0 or cnt.max() < 0.6 * PATCH * PATCH:
                    continue
                tiles.append(pre(Image.fromarray(v.rgb[y0:y0 + PATCH, x0:x0 + PATCH])))
                boxes.append((y0, x0, int(vals[np.argmax(cnt)])))
        if not tiles:
            continue
        with torch.no_grad():
            f = model.encode_image(torch.stack(tiles))
            f /= f.norm(dim=-1, keepdim=True)
            sim = (f @ T.T).numpy()
            p = (100.0 * f @ T.T).softmax(-1).numpy()
        benign = np.array([o is None for o in owner])
        for (y0, x0, k), pr, sm in zip(boxes, p, sim):
            cls_p = {c: pr[[i for i, o in enumerate(owner) if o == c]].sum() for c in CLASSES}
            c, pc = max(cls_p.items(), key=lambda x: x[1])
            if pc < P_MIN or sm[~benign].max() - sm[benign].max() < MARGIN:
                continue
            on = sid == k
            # deviation from the patch's own surface colour: a stain is a blob
            # on a uniform surface; wood grain / posters fill the patch instead
            win = (slice(y0, y0 + PATCH), slice(x0, x0 + PATCH))
            on_p = on[win]
            med = np.median(lab[win][on_p], axis=0)
            dev = np.linalg.norm(lab[win] - med, axis=-1) > DE_MIN
            frac = (dev & on_p).sum() / max(on_p.sum(), 1)
            if not DEV_FRAC[0] <= frac <= DEV_FRAC[1]:
                continue
            m = np.zeros_like(on)
            m[win] = dev & on_p
            if m.sum() < 30:
                continue
            ss, hh = s_c[m], h_c[m]
            s_lo, s_hi = np.percentile(ss, [2, 98])
            h_lo, h_hi = np.percentile(hh, [2, 98])
            # area: fraction of the box covered by deviating pixels
            box_px = (on[y0:y0 + PATCH, x0:x0 + PATCH]).sum()
            area = (s_hi - s_lo) * (h_hi - h_lo) * min(1.0, 2.0 * m.sum() / max(box_px, 1))
            if area < MIN_AREA:
                continue
            regions.append(Region(surf[k][1], c, s_lo, s_hi, h_lo, h_hi, float(area),
                                  float(pc), {v.vid},
                                  v.rgb[y0:y0 + PATCH, x0:x0 + PATCH].copy()))
    return _merge(regions, min_views)


def _merge(regs: list[Region], min_views: int) -> list[Region]:
    out: list[Region] = []
    for r in sorted(regs, key=lambda r: -r.prob):
        for o in out:
            if o.surface == r.surface and o.cls == r.cls and \
                    r.s_lo < o.s_hi + 0.05 and o.s_lo < r.s_hi + 0.05 and \
                    r.h_lo < o.h_hi + 0.05 and o.h_lo < r.h_hi + 0.05:
                o.s_lo, o.s_hi = min(o.s_lo, r.s_lo), max(o.s_hi, r.s_hi)
                o.h_lo, o.h_hi = min(o.h_lo, r.h_lo), max(o.h_hi, r.h_hi)
                o.area = max(o.area, r.area)
                o.views |= r.views
                break
        else:
            out.append(r)
    return [r for r in out if len(r.views) >= min_views]


# ---------------------------------------------------------------- rules + scope

RULES = [
    ("CD1", lambda r, ctx: r.cls == "water_stain" and r.surface.endswith("CEIL"),
     "Water stain on ceiling: active or past leak above; ceiling cavity, joists and "
     "insulation likely wet beyond the visible stain"),
    ("CD2", lambda r, ctx: r.cls in ("water_stain", "mold") and not r.surface.endswith("CEIL")
     and r.h_lo < 0.30,
     "Moisture damage starting within 0.30 m of the floor: wicking from floor level; "
     "bottom plate, insulation and drywall back face likely affected up to 0.6 m above "
     "the visible line"),
    ("CD3", lambda r, ctx: r.cls == "mold",
     "Visible mould: growth on the drywall face implies colonisation of the cavity side"),
    ("CD4", lambda r, ctx: r.cls == "crack" and ctx["near_opening_corner"](r),
     "Crack starting at a door/window corner: lintel or framing movement"),
    ("CD5", lambda r, ctx: r.cls == "peeling_paint" and ctx["wet_surface"](r.surface),
     "Peeling paint on a surface that also shows water damage: moisture-driven "
     "delamination, substrate likely saturated"),
]


def _wall(plan, sid):
    for r in plan["rooms"]:
        for w in r["walls"]:
            if w["id"] == sid:
                return r, w
    return None, None


def build_outputs(regs: list[Region], plan: dict, tier: str):
    model = U.load_model(tier)
    se = model["sigma_face"] * 2           # extent edges are image-segmentation edges
    damage, flags, scope = [], [], []
    wet = {r.surface for r in regs if r.cls in ("water_stain", "mold")}

    def near_opening_corner(r):
        _, w = _wall(plan, r.surface)
        if not w:
            return False
        room = next(x for x in plan["rooms"] if any(q["id"] == w["id"] for q in x["walls"]))
        a = np.array(w["start"])
        for oid in w["openings"]:
            o = next(q for q in room["openings"] if q["id"] == oid)
            c = np.linalg.norm(np.array(o["center"]) - a)
            half = o["width"]["value"] / 2
            for edge in (c - half, c + half):
                if min(abs(r.s_lo - edge), abs(r.s_hi - edge)) < 0.3:
                    return True
        return False

    ctx = {"near_opening_corner": near_opening_corner, "wet_surface": lambda s: s in wet}
    for i, r in enumerate(regs):
        did = f"D{i + 1}"
        w_, h_ = r.s_hi - r.s_lo, r.h_hi - r.h_lo
        damage.append({
            "id": did, "surface_id": r.surface, "class": r.cls,
            "area": U.measure(r.area, np.hypot(se * (w_ + h_), 0.2 * r.area), model, 3),
            "extent": {"width": U.measure(w_, np.sqrt(2) * se, model),
                       "height": U.measure(h_, np.sqrt(2) * se, model),
                       "bottom_above_floor": U.measure(r.h_lo, se, model)},
            "confidence": round(r.prob, 3), "views": len(r.views)})
        for rid, cond, text in RULES:
            if cond(r, ctx):
                flags.append({"id": f"F{len(flags) + 1}", "surface_id": r.surface,
                              "rule_id": rid, "rule": text,
                              "evidence": [did, f"class={r.cls}",
                                           f"bottom={r.h_lo:.2f} m above floor"]})
        scope += _scope_for(r, did, plan, model, len(scope))
    for f in flags:
        scope.append({"id": f"S{len(scope) + 1}", "surface_id": f["surface_id"],
                      "code": "INSP-CAV", "description": f"Open and inspect cavity ({f['rule_id']})",
                      "quantity": U.measure(1, 0, model, 0), "unit": "ea",
                      "triggered_by": f["id"]})
    return damage, flags, scope


def _scope_for(r: Region, did: str, plan: dict, model: dict, n0: int) -> list[dict]:
    items = []
    add = lambda code, desc, q, qs, unit: items.append(
        {"id": f"S{n0 + len(items) + 1}", "surface_id": r.surface, "code": code,
         "description": desc, "quantity": U.measure(q, qs, model, 2), "unit": unit,
         "triggered_by": did})
    room, w = _wall(plan, r.surface)
    if r.surface.endswith("CEIL"):
        room = next(x for x in plan["rooms"] if r.surface.startswith(x["id"] + "."))
        A = room["floor_area"]
        add("CEIL-PATCH", "Cut out and patch ceiling board (damage + 0.3 m margin)",
            (r.s_hi - r.s_lo + 0.6) * (r.h_hi - r.h_lo + 0.6), 0.1, "m2")
        add("PNT-CEIL", "Seal stain and repaint whole ceiling", A["value"], A["sigma"], "m2")
        return items
    L = w["length"]["value"]
    H = w["height"]["value"] if w.get("height") else 2.5
    if r.cls in ("water_stain", "mold", "hole"):
        cut_w = min(L, (r.s_hi - r.s_lo) + 0.6)
        cut_h = min(H, r.h_hi + 0.6) if r.cls != "hole" else (r.h_hi - r.h_lo) + 0.4
        add("DRY-RR", "Remove and replace drywall (damage + 0.3 m margin; flood cut "
            "0.6 m above highest moisture line)", cut_w * cut_h, 0.05 * cut_w * cut_h + 0.02, "m2")
    if r.cls == "mold":
        add("MLD-TRT", "Antimicrobial treatment of exposed framing",
            (r.s_hi - r.s_lo + 0.6) * (r.h_hi + 0.6), 0.1, "m2")
    if r.cls == "crack":
        add("CRK-RPR", "Rake out, tape and fill crack", max(r.s_hi - r.s_lo, r.h_hi - r.h_lo),
            0.03, "m")
    if r.h_lo < 0.3 and r.cls in ("water_stain", "mold"):
        add("BSB-RR", "Remove and replace baseboard", min(L, r.s_hi - r.s_lo + 0.6), 0.03, "m")
    add("PNT-WALL", "Prime and paint whole wall (colour match)", L * H, 0.05 * L * H, "m2")
    return items


# ---------------------------------------------------------------- view builders

def _pix_normals(X: np.ndarray) -> np.ndarray:
    dx = np.gradient(X, axis=1)
    dy = np.gradient(X, axis=0)
    n = np.cross(dx, dy)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    flip = np.sum(n * X, -1) > 0
    n[flip] *= -1
    return n


def lidar_views(cap, poses: dict | None, M: np.ndarray, every_s: float = 1.0,
                width: int = 640) -> list[DView]:
    """RGB keyframes (one per `every_s`) with LiDAR depth upsampled to RGB size."""
    frames = cap.depth_frames
    if not frames:
        return []
    t0, picks, last = frames[0].timestamp, {}, -1e9
    for f in frames:
        if f.timestamp - last >= every_s:
            picks[f.index], last = f, f.timestamp
    vid = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
    out, i = [], 0
    while picks:
        if not vid.grab():
            break
        if i in picks:
            f = picks.pop(i)
            _, img = vid.retrieve()
            h = int(width * img.shape[0] / img.shape[1])
            rgb = cv2.cvtColor(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            d, _ = cap.load_depth(f)
            d = cv2.resize(d, (width, h), interpolation=cv2.INTER_NEAREST)
            K = f.K_rgb * np.array([[width / 1920], [h / 1440], [1.0]])
            v_, u_ = np.mgrid[0:h, 0:width]
            X = np.stack([(u_ - K[0, 2]) / K[0, 0] * d, (v_ - K[1, 2]) / K[1, 1] * d, d], -1)
            n = _pix_normals(X)
            T = M @ (poses.get(f.index, f.T_wc) if poses else f.T_wc)
            R, t = T[:3, :3], T[:3, 3]
            out.append(DView(rgb, X @ R.T + t, n @ R.T, d > 0.2, f.index))
        i += 1
    return out


def mono_views(views, M: np.ndarray) -> list[DView]:
    """Photo/video keyframes: MoGe point maps placed by their registered pose."""
    out = []
    for k, v in enumerate(views):
        if v.T is None:
            continue
        A = v.T[:3, :3] @ v.R_g
        R = M[:3, :3]
        X = (v.mv.points @ A.T + v.T[:3, 3]) @ R.T + M[:3, 3]
        n = (v.mv.normals @ (A / max(v.scale, 1e-9)).T) @ R.T
        out.append(DView(v.mv.rgb, X, n, v.mv.mask, k))
    return out
