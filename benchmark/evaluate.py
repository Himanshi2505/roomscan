"""Benchmark: run every capture, score against tape/laser ground truth, write the report.

    python -m benchmark.evaluate                # run pipeline (cached results reused) + score
    python -m benchmark.evaluate --rerun        # force pipeline re-runs
    python -m benchmark.evaluate --calibrate    # also fit per-tier interval factor k

Inputs
  benchmark/captures.json   list of {id, tier, path, gt, room_map{pred_id: gt_room},
                            [repeat_of], [no_drift_ablation: true]}
  benchmark/gt/<gt>.json    ground truth (see benchmark/gt/EXAMPLE.json)
  benchmark/competitor/<capture_id>.json   consumer-app export in GT format (head-to-head)
Outputs
  benchmark/out/results.json, benchmark/out/REPORT.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
Z90 = 1.645

GATES = {
    "opening_width_m": 0.02, "opening_frac": 0.85,
    "ceiling_m": 0.015, "ceiling_spread_m": 0.01,
    "repeat_m": 0.01, "repeat_rel": 0.005,
    "wall_rel": {"photo": 0.08, "video": 0.03, "lidar": None},
    "footprint_rel_photo": 0.08,
}


# ------------------------------------------------------------------ matching

def match_walls(pred: list[float], gt: list[float]) -> list[tuple[int, int]]:
    """Best cyclic alignment (either direction) of predicted vs GT clockwise wall
    lengths. Predicted short jogs (< 0.25 m) that GT doesn't list are skipped."""
    p_idx = [i for i, L in enumerate(pred) if L >= 0.25]
    if not p_idx or not gt:
        return []
    best = None
    for direction in (1, -1):
        seq = p_idx[::direction]
        for shift in range(len(seq)):
            rot = seq[shift:] + seq[:shift]
            pairs = list(zip(rot, range(len(gt))))
            err = sum(abs(pred[i] - gt[j]) for i, j in pairs) + 0.5 * abs(len(rot) - len(gt))
            if best is None or err < best[0]:
                best = (err, pairs)
    return best[1]


def match_openings(pred: list[dict], gt: list[dict], max_dw: float = 0.30):
    """One-to-one by kind and width; returns pairs, missed gt, phantom preds."""
    if not pred or not gt:
        return [], list(range(len(gt))), list(range(len(pred)))
    C = np.full((len(pred), len(gt)), 1e3)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            kind_ok = (p["kind"] == "window") == (g["kind"] == "window")
            dw = abs(p["width"]["value"] - g["width"])
            if kind_ok and dw <= max_dw:
                C[i, j] = dw
    r, c = linear_sum_assignment(C)
    pairs = [(i, j) for i, j in zip(r, c) if C[i, j] < 1e3]
    mi = [j for j in range(len(gt)) if j not in {b for _, b in pairs}]
    ph = [i for i in range(len(pred)) if i not in {a for a, _ in pairs}]
    return pairs, mi, ph


# ------------------------------------------------------------------ scoring

def _cov(m: dict, truth: float) -> bool:
    return m["ci90"][0] <= truth <= m["ci90"][1]


def score_capture(cap: dict, res: dict, gt: dict) -> dict:
    rows, opens, ceil, cal = [], [], [], []
    rooms = {r["id"]: r for r in res["rooms"]}
    for pid, gname in cap["room_map"].items():
        r, g = rooms.get(pid), gt["rooms"].get(gname)
        if r is None or g is None:
            continue
        plens = [w["length"]["value"] for w in r["walls"]]
        glens = [w["length"] for w in g["walls"]]
        for i, j in match_walls(plens, glens):
            m, truth = r["walls"][i]["length"], glens[j]
            e = m["value"] - truth
            rows.append({"room": gname, "wall": g["walls"][j].get("id", j), "gt": truth,
                         "pred": m["value"], "err": e, "rel": e / truth, "covered": _cov(m, truth),
                         "pred_wall_id": r["walls"][i]["id"]})
            cal.append((e, m["sigma"]))
        if g.get("ceiling_height") is not None:
            ch = r.get("ceiling_height")
            if ch is None:
                ceil.append({"room": gname, "gt": g["ceiling_height"], "pred": None, "err": None})
            else:
                e = ch["value"] - g["ceiling_height"]
                ceil.append({"room": gname, "gt": g["ceiling_height"], "pred": ch["value"],
                             "err": e, "covered": _cov(ch, g["ceiling_height"])})
                cal.append((e, ch["sigma"]))
        pairs, missed, phantom = match_openings(r["openings"], g.get("openings", []))
        for i, j in pairs:
            m, truth = r["openings"][i]["width"], g["openings"][j]["width"]
            e = m["value"] - truth
            opens.append({"room": gname, "kind": g["openings"][j]["kind"], "gt": truth,
                          "pred": m["value"], "err": e, "status": "matched",
                          "covered": _cov(m, truth)})
            cal.append((e, m["sigma"]))
        opens += [{"room": gname, "kind": g["openings"][j]["kind"], "gt": g["openings"][j]["width"],
                   "pred": None, "err": None, "status": "missed"} for j in missed]
        opens += [{"room": gname, "kind": r["openings"][i]["kind"], "gt": None,
                   "pred": r["openings"][i]["width"]["value"], "err": None, "status": "phantom"}
                  for i in phantom]
    # stitched plan checks
    shapes = [Polygon(r["polygon"]) for r in res["rooms"]]
    overlap = sum(a.intersection(b).area for a, b in itertools.combinations(shapes, 2))
    gt_adj = {tuple(sorted(p)) for p in gt.get("adjacency", [])}
    inv = cap["room_map"]
    pred_adj = {tuple(sorted((inv.get(a["rooms"][0]), inv.get(a["rooms"][1]))))
                for a in res["adjacency"] if inv.get(a["rooms"][0]) and inv.get(a["rooms"][1])}
    fp = res["footprint_area"]
    fp_gt = gt.get("footprint_area")
    return {"walls": rows, "openings": opens, "ceiling": ceil, "calibration": cal,
            "overlap_m2": overlap,
            "adjacency": {"gt": sorted(gt_adj), "pred": sorted(pred_adj),
                          "correct": pred_adj == gt_adj, "missing": sorted(gt_adj - pred_adj),
                          "extra": sorted(pred_adj - gt_adj)},
            "footprint": {"gt": fp_gt, "pred": fp["value"],
                          "rel": (fp["value"] - fp_gt) / fp_gt if fp_gt else None,
                          "covered": _cov(fp, fp_gt) if fp_gt else None},
            "timing_s": res["timing_s"].get("total")}


def gates(tier: str, s: dict) -> dict:
    g = {}
    ow = [o for o in s["openings"]]
    if ow:
        ok = sum(o["status"] == "matched" and abs(o["err"]) <= GATES["opening_width_m"] for o in ow)
        g["opening_widths"] = {"value": ok / len(ow), "pass": ok / len(ow) >= GATES["opening_frac"],
                               "detail": f"{ok}/{len(ow)} within 2 cm (missed+phantom count as fails)"}
    ce = [c for c in s["ceiling"]]
    if ce:
        worst = max((abs(c["err"]) if c["err"] is not None else np.inf) for c in ce)
        g["ceiling_height"] = {"value": worst, "pass": worst <= GATES["ceiling_m"],
                               "detail": "worst |error| per room (m)"}
    lim = GATES["wall_rel"][tier]
    if lim and s["walls"]:
        worst = max(abs(w["rel"]) for w in s["walls"])
        cov = np.mean([w["covered"] for w in s["walls"]])
        g["wall_lengths"] = {"value": worst, "pass": worst <= lim,
                             "detail": f"worst relative error; CI coverage {cov:.0%}"}
    if tier == "photo":
        fp = s["footprint"]
        ok = s["adjacency"]["correct"] and s["overlap_m2"] < 0.05 and fp["rel"] is not None \
            and abs(fp["rel"]) <= GATES["footprint_rel_photo"]
        g["photo_stitch"] = {"value": fp["rel"], "pass": bool(ok),
                             "detail": f"adjacency correct={s['adjacency']['correct']}, "
                                       f"overlap={s['overlap_m2']:.3f} m2, footprint err={fp['rel']}"}
    return g


def repeatability(a: dict, b: dict) -> dict:
    """Same room, same tier, two captures: per-wall agreement (matched via GT ids)."""
    wa = {(w["room"], w["wall"]): w["pred"] for w in a["walls"]}
    wb = {(w["room"], w["wall"]): w["pred"] for w in b["walls"]}
    rows = []
    for k in sorted(set(wa) & set(wb)):
        d = abs(wa[k] - wb[k])
        rows.append({"room": k[0], "wall": k[1], "a": wa[k], "b": wb[k], "diff": d,
                     "pass": d <= max(GATES["repeat_m"], GATES["repeat_rel"] * wa[k])})
    ca = {c["room"]: c["pred"] for c in a["ceiling"] if c["pred"] is not None}
    cb = {c["room"]: c["pred"] for c in b["ceiling"] if c["pred"] is not None}
    ceil = []
    for room in sorted(set(ca) & set(cb)):
        gt = next(c["gt"] for c in a["ceiling"] if c["room"] == room)
        vals = np.array([ca[room], cb[room]])
        spread, bias = float(vals.max() - vals.min()), float(vals.mean() - gt)
        kind = ("ok" if spread <= GATES["ceiling_spread_m"] and abs(bias) <= GATES["ceiling_m"]
                else "repeatable-but-biased" if spread <= GATES["ceiling_spread_m"]
                else "unrepeatable")
        ceil.append({"room": room, "spread": spread, "bias": bias, "verdict": kind})
    return {"walls": rows, "pass": bool(rows) and all(r["pass"] for r in rows), "ceiling": ceil}


def calibrate(cal_by_tier: dict) -> dict:
    """Conformal: smallest k with >= 90% of |err| <= 1.645 * k * sigma."""
    out = {}
    for tier, cal in cal_by_tier.items():
        if len(cal) < 5:
            continue
        z = np.array([abs(e) / max(s, 1e-6) for e, s in cal]) / Z90
        n = len(z)
        q = np.sort(z)[min(n - 1, int(np.ceil(0.9 * (n + 1))) - 1)]
        out[tier] = {"k": float(max(q, 0.5)), "n": n}
    return out


# ------------------------------------------------------------------ driver

def run_capture(cap: dict, rerun: bool) -> dict:
    from roomscan.cli import run
    out = OUT / "runs" / cap["id"]
    f = out / "result.json"
    if f.exists() and not rerun:
        return json.loads(f.read_text())
    t = time.perf_counter()
    res = run(Path(cap["path"]), out, cap["tier"], drift=not cap.get("no_drift_ablation", False))
    res["timing_s"]["wall_clock"] = round(time.perf_counter() - t, 2)
    f.write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--captures", default=str(ROOT / "captures.json"))
    a = ap.parse_args(argv)
    caps = json.loads(Path(a.captures).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    scored, cal = {}, {}
    for cap in caps:
        gt = json.loads((ROOT / "gt" / f"{cap['gt']}.json").read_text())
        res = run_capture(cap, a.rerun)
        s = score_capture(cap, res, gt)
        s["gates"] = gates(cap["tier"], s)
        s["tier"], s["drift"] = cap["tier"], res["drift"]
        scored[cap["id"]] = s
        if not cap.get("no_drift_ablation"):
            cal.setdefault(cap["tier"], []).extend(s["calibration"])
        comp = ROOT / "competitor" / f"{cap['id']}.json"
        if comp.exists():
            s["head_to_head"] = head_to_head(s, json.loads(comp.read_text()), gt, cap)
    reps = {c["id"]: repeatability(scored[c["repeat_of"]], scored[c["id"]])
            for c in caps if c.get("repeat_of") in scored}
    k = calibrate(cal)
    if a.calibrate and k:
        from roomscan.uncertainty import CAL_PATH
        CAL_PATH.parent.mkdir(exist_ok=True)
        old = json.loads(CAL_PATH.read_text()) if CAL_PATH.exists() else {}
        for tier, v in k.items():
            old[tier] = {"k": round(v["k"] * old.get(tier, {}).get("k", 1.0), 3)}
        CAL_PATH.write_text(json.dumps(old, indent=1))
    results = {"captures": scored, "repeatability": reps, "calibration_fit": k}
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    from benchmark.report import write_report
    write_report(results, caps, OUT / "REPORT.md")
    print((OUT / "REPORT.md").read_text())


def head_to_head(s: dict, comp: dict, gt: dict, cap: dict) -> dict:
    """Competitor export in GT format with the same room names -> per-dimension errors."""
    rows = []
    ours = {(w["room"], w["wall"]): w for w in s["walls"]}
    for gname, g in gt["rooms"].items():
        c = comp["rooms"].get(gname)
        if not c:
            continue
        clens = [w["length"] for w in c["walls"]]
        glens = [w["length"] for w in g["walls"]]
        for i, j in match_walls(clens, glens):
            wid = g["walls"][j].get("id", j)
            o = ours.get((gname, wid))
            if o is None:
                continue
            rows.append({"dim": f"{gname} wall {wid}", "gt": glens[j], "ours_err": o["err"],
                         "theirs_err": clens[i] - glens[j]})
        if g.get("ceiling_height") and c.get("ceiling_height"):
            oc = next((x for x in s["ceiling"] if x["room"] == gname and x["err"] is not None), None)
            if oc:
                rows.append({"dim": f"{gname} ceiling", "gt": g["ceiling_height"],
                             "ours_err": oc["err"], "theirs_err": c["ceiling_height"] - g["ceiling_height"]})
    for r in rows:
        r["beat_or_tie"] = abs(r["ours_err"]) <= abs(r["theirs_err"]) + 0.005
    frac = np.mean([r["beat_or_tie"] for r in rows]) if rows else None
    return {"app": comp.get("app"), "version": comp.get("version"), "rows": rows,
            "beat_or_tie_frac": frac, "pass": frac is not None and frac >= 0.7}


if __name__ == "__main__":
    main()
