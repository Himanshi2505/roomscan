"""Tier dispatch: capture -> OrientedCloud -> plan -> damage/scope -> result dict."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from roomscan.geometry.levels import manhattan_yaw, yaw_matrix
from roomscan.plan import build_plan


def _lidar_cloud(capture: Path, drift: bool, timing: dict, info: dict):
    from roomscan.io import strayscanner as ss
    from roomscan.tiers.lidar import build_cloud
    cap = ss.load(capture)
    t = time.perf_counter()
    poses = None
    info["drift"] = {"method": "none (poses as-is)", "enabled": False,
                     "correction_m": None, "loop_closures": 0}
    if drift:
        from roomscan.drift import correct_drift
        poses, info["drift"] = correct_drift(cap)
        timing["drift"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    c = build_cloud(cap, every=3, poses=poses)
    timing["fusion"] = round(time.perf_counter() - t, 2)
    P = np.array([f.T_wc[:3, 3] for f in cap.frames])
    path = float(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1).sum())
    info["n_inputs"] = len(cap.depth_frames)
    info["cap"], info["poses"] = cap, poses
    return c, path


def _photo(capture: Path, timing: dict, warnings: list) -> dict:
    """Per-room photo folders -> per-room plans -> doorway links -> stitched plan."""
    from roomscan import stitch
    from roomscan.tiers import photo
    folders = sorted(d for d in capture.iterdir() if d.is_dir())
    t = time.perf_counter()
    recons, plans, Ms, names = [], [], [], []
    from roomscan.tiers import photo_layout
    for k, d in enumerate(folders):
        rec = photo.reconstruct_room(d, fid0=1000 * k)
        n_all = len(rec.views) + len(rec.unregistered)
        if len(rec.views) >= max(3, n_all // 2):
            # enough texture: fused multi-view reconstruction, any rectilinear shape
            M = yaw_matrix(manhattan_yaw(rec.cloud))
            try:
                pl = build_plan(rec.cloud.transformed(M), "photo", single_room=True)
            except RuntimeError as e:
                warnings.append(f"{d.name}: {e}")
                continue
            dv = (rec.views, M)
        else:
            # textureless room: one wall per photo, rectangle (notes D11)
            allv = sorted(rec.views + rec.unregistered, key=lambda v: v.name)
            pl, notes = photo_layout.room_plan(allv)
            warnings += [f"{d.name}: {x}" for x in notes]
            if pl is None:
                warnings.append(f"{d.name}: no wall could be measured, room skipped")
                continue
            warnings.append(f"{d.name}: layout path (rectangle from {pl['wall_photos']} wall photos)")
            rec.views = [v for v in allv if v.T is not None and v.T.shape == (4, 4)
                         and not np.allclose(v.T, np.eye(4))]
            rec.unregistered = [v for v in allv if v not in rec.views]
            M = np.eye(4)
            dv = (rec.views, M)
        if not pl["rooms"]:
            warnings.append(f"{d.name}: no room footprint found")
            continue
        pl["rooms"] = [max(pl["rooms"], key=lambda r: r["floor_area"]["value"])]
        from roomscan import damage as DM
        regs = DM.detect(DM.mono_views(*dv), pl, pl["levels"]["floor_y"],
                         pl["levels"]["ceiling_y"], min_views=1)
        pl["damage"], pl["concealed_flags"], pl["scope"] = DM.build_outputs(regs, pl, "photo")
        recons.append(rec); plans.append(pl); Ms.append(M); names.append(d.name)
        if rec.unregistered:
            warnings.append(f"{d.name}: {len(rec.unregistered)} photo(s) did not register "
                            f"in-room (expected for doorway photos)")
    timing["rooms"] = round(time.perf_counter() - t, 2)
    if not plans:
        raise RuntimeError("no room could be reconstructed from the photos")
    # doorway photos of room a that register into room b -> which door of b
    t = time.perf_counter()
    links = []
    for a, ra in enumerate(recons):
        for u in ra.unregistered:
            best = None
            for b, rb in enumerate(recons):
                if b == a:
                    continue
                for v in rb.views:
                    r = photo.register_pair(v, u)
                    if r and (best is None or r[3] > best[0]):
                        best = (r[3], b, v, r)
            if best is None:
                continue
            _, b, v, (s_, R_, t_, _) = best
            T = v.T @ photo._sim_to_T(s_, R_, t_)
            cam = (Ms[b] @ np.r_[T[:3, 3], 1.0])[[0, 2]]
            doors = [o for o in plans[b]["rooms"][0]["openings"] if o["kind"] != "window"]
            if doors:
                d = min(doors, key=lambda o: np.linalg.norm(np.array(o["center"]) - cam))
                if np.linalg.norm(np.array(d["center"]) - cam) < 1.5:
                    links.append((a, b, d["id"]))
                    continue
            links.append((a, b, None))
    rooms, adjacency, w2 = stitch.stitch(plans, names, links)
    timing["stitch"] = round(time.perf_counter() - t, 2)
    warnings += w2
    for r, n in zip(rooms, names):
        r["name"] = n
    # per-room damage/flags/scope: prefix ids with the room folder name
    dmg, flg, scp = [], [], []
    for pl, n in zip(plans, names):
        ren = {}
        for d in pl["damage"]:
            ren[d["id"]] = f"{n}.{d['id']}"
            dmg.append({**d, "id": ren[d["id"]], "surface_id": f"{n}.{d['surface_id']}"})
        for f in pl["concealed_flags"]:
            ren[f["id"]] = f"{n}.{f['id']}"
            flg.append({**f, "id": ren[f["id"]], "surface_id": f"{n}.{f['surface_id']}",
                        "evidence": [ren.get(e, e) for e in f["evidence"]]})
        for x in pl["scope"]:
            scp.append({**x, "id": f"{n}.{x['id']}", "surface_id": f"{n}.{x['surface_id']}",
                        "triggered_by": ren.get(x["triggered_by"], x["triggered_by"])})
    from roomscan import uncertainty as U
    model = U.load_model("photo")
    fp = sum(r["floor_area"]["value"] for r in rooms)
    fps = float(np.sqrt(sum(r["floor_area"]["sigma"] ** 2 for r in rooms)))
    return {"tier": "photo", "calibrated": model["calibrated"], "rooms": rooms,
            "adjacency": [{k: a[k] for k in ("rooms", "via", "kind")} for a in adjacency],
            "footprint_area": U.measure(fp, fps, model, 2),
            "n_inputs": sum(len(r.views) + len(r.unregistered) for r in recons),
            "photo_links": len(links), "damage": dmg, "concealed_flags": flg, "scope": scp}


def run(capture: Path, tier: str, drift: bool = True) -> dict:
    timing, info, warnings = {}, {}, []
    if tier == "photo":
        plan = _photo(capture, timing, warnings)
        n_inputs = plan.pop("n_inputs")
        info["drift"] = {"method": "n/a: rooms are reconstructed independently and joined "
                                   "through doors (no accumulated trajectory)",
                         "enabled": False, "correction_m": None,
                         "loop_closures": plan.pop("photo_links")}
        return _result(capture, plan, info, timing, warnings, n_inputs)
    if tier == "lidar":
        c, path = _lidar_cloud(capture, drift, timing, info)
    else:
        from roomscan.tiers import video
        vid = capture if capture.is_file() else next(
            f for f in sorted(capture.iterdir()) if f.suffix.lower() in {".mov", ".mp4", ".m4v"})
        t = time.perf_counter()
        c, path, info["drift"], info["n_inputs"], info["views"] = video.build(vid, drift=drift)
        timing["reconstruction"] = round(time.perf_counter() - t, 2)
        d = info["drift"]
        if d["segments"] > 1:
            warnings.append(f"video tracking broke into {d['segments']} pieces; the plan covers "
                            f"only the largest ({d['kept_keyframes']}/{d['total_keyframes']} "
                            "keyframes). Re-record walking slower, ~1 m from walls")
    M = yaw_matrix(manhattan_yaw(c))
    c = c.transformed(M)
    t = time.perf_counter()
    # drift scales with walked path only when poses are used uncorrected
    plan = build_plan(c, tier, path_length=0.0 if info["drift"]["enabled"] else path,
                      extra_rel=info["drift"].get("distortion_rel", 0.0))
    timing["plan"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    from roomscan import damage as DM
    if tier == "lidar":
        dviews = DM.lidar_views(info.pop("cap"), info.pop("poses"), M)
    else:
        dviews = DM.mono_views(info.pop("views"), M)
    regs = DM.detect(dviews, plan, plan["levels"]["floor_y"], plan["levels"]["ceiling_y"],
                     min_views=2)
    plan["damage"], plan["concealed_flags"], plan["scope"] = DM.build_outputs(regs, plan, tier)
    timing["damage"] = round(time.perf_counter() - t, 2)
    for r in plan["rooms"]:
        if r["ceiling_height"] is None:
            warnings.append(f"{r['id']}: ceiling not observed; sweep the phone over the ceiling")
    return _result(capture, plan, info, timing, warnings, info.get("n_inputs", 0))


def _result(capture, plan, info, timing, warnings, n_inputs) -> dict:
    for r in plan["rooms"]:
        if r["ceiling_height"] is None and not any(r["id"] in w for w in warnings):
            warnings.append(f"{r['id']}: ceiling not observed")
    if not plan["calibrated"]:
        warnings.append("intervals use prior error model (no benchmark calibration found)")
    return {"schema_version": "1.0",
            "capture": {"id": capture.name, "path": str(capture), "device": None,
                        "n_inputs": n_inputs},
            "damage": [], "concealed_flags": [], "scope": [], **plan,
            "drift": info["drift"], "timing_s": timing, "warnings": warnings}
