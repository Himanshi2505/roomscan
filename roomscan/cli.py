"""One command per capture:

    python -m roomscan run <capture> [--out DIR] [--no-drift]

<capture> may be
  * a Stray Scanner folder (odometry.csv + depth/)      -> LiDAR tier
  * a video file (.mov/.mp4) or a folder holding one    -> video tier
  * a folder of per-room photo folders                  -> photo tier
The tier is detected from what is there; --tier forces it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

VIDEO_EXT = {".mov", ".mp4", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".heic", ".png"}


def detect_tier(p: Path) -> str:
    from roomscan.io.strayscanner import is_stray_capture
    if p.is_file() and p.suffix.lower() in VIDEO_EXT:
        return "video"
    if p.is_dir():
        if is_stray_capture(p):
            return "lidar"
        if any(f.suffix.lower() in VIDEO_EXT for f in p.iterdir() if f.is_file()):
            return "video"
        subs = [d for d in p.iterdir() if d.is_dir()]
        if subs and all(any(f.suffix.lower() in IMAGE_EXT for f in d.iterdir()) for d in subs):
            return "photo"
    raise SystemExit(f"cannot tell which tier {p} is: expected a Stray Scanner folder, "
                     "a video file, or a folder of per-room photo folders")


def run(capture: Path, out: Path, tier: str | None = None, drift: bool = True) -> dict:
    from roomscan import pipeline
    tier = tier or detect_tier(capture)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = pipeline.run(capture, tier, drift=drift)
    result["timing_s"]["total"] = round(time.perf_counter() - t0, 2)
    _validate(result)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    from roomscan.render import render
    render(result, str(out / "plan.png"), title=f"{capture.name} ({tier})")
    return result


def _validate(result: dict) -> None:
    import jsonschema
    schema = json.loads((Path(__file__).parent / "schema" / "plan.schema.json").read_text())
    jsonschema.validate(result, schema)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roomscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture")
    r.add_argument("capture", type=Path)
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--tier", choices=["lidar", "video", "photo"])
    r.add_argument("--no-drift", action="store_true", help="ablation: use poses as-is")
    a = ap.parse_args(argv)
    out = a.out or Path("out") / a.capture.name
    res = run(a.capture, out, a.tier, drift=not a.no_drift)
    print(f"{len(res['rooms'])} rooms, footprint {res['footprint_area']['value']} m2, "
          f"{res['timing_s']['total']} s -> {out / 'result.json'}, {out / 'plan.png'}")
    for w in res["warnings"]:
        print("warning:", w, file=sys.stderr)


if __name__ == "__main__":
    main()
