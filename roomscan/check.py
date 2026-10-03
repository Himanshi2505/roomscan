"""Pre-flight check of a capture before running the pipeline (seconds, no models).

    python -m roomscan check <capture>

Prints OK / WARN / FAIL lines against the capture protocol; returns the number
of FAILs (the CLI exits 1 if any). Meant for the walk-in: catch a wrong folder
layout or a missing file before a multi-minute run, not after.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from roomscan import mono

PHOTOS_PER_ROOM = (2, 8)
VIDEO_S = (10.0, 360.0)
LIDAR_MAX_S = 240.0


class Report:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def ok(self, msg):
        self.lines.append(("OK", msg))

    def warn(self, msg):
        self.lines.append(("WARN", msg))

    def fail(self, msg):
        self.lines.append(("FAIL", msg))

    @property
    def fails(self) -> int:
        return sum(level == "FAIL" for level, _ in self.lines)


def _weights(r: Report):
    w = mono.WEIGHTS[mono.MODEL_SIZE]
    if w.exists():
        r.ok(f"MoGe-2 weights present ({mono.MODEL_SIZE})")
    else:
        r.fail(f"missing {w}: run  python scripts/fetch_weights.py --large")


def _photo(p: Path, r: Report):
    from roomscan.tiers.photo import IMAGE_EXT, exif_fov_x
    rooms = sorted(d for d in p.iterdir() if d.is_dir())
    r.ok(f"photo tier: {len(rooms)} room folder(s)")
    for d in rooms:
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT)
        lo, hi = PHOTOS_PER_ROOM
        if not lo <= len(imgs) <= hi:
            r.warn(f"{d.name}: {len(imgs)} photos (protocol: {lo}-{hi})")
        else:
            r.ok(f"{d.name}: {len(imgs)} photos")
        heic = [f for f in imgs if f.suffix.lower() in {".heic", ".heif"}]
        if heic:
            try:
                import pillow_heif  # noqa: F401
            except ImportError:
                r.fail(f"{d.name}: {len(heic)} HEIC photo(s) but pillow-heif is not installed")
        no_focal = [f.name for f in imgs if f not in heic and exif_fov_x(f) is None]
        if no_focal:
            r.warn(f"{d.name}: no EXIF focal length in {len(no_focal)} photo(s) "
                   f"(e.g. {no_focal[0]}): screenshots/edited copies? scale will be less accurate")
    _weights(r)


def _video(p: Path, r: Report):
    from roomscan.cli import VIDEO_EXT
    f = p if p.is_file() else next(x for x in sorted(p.iterdir()) if x.suffix.lower() in VIDEO_EXT)
    cap = cv2.VideoCapture(str(f))
    ok, img = cap.read()
    if not ok:
        r.fail(f"{f.name}: cannot decode video")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps else 0.0
    lo, hi = VIDEO_S
    msg = f"{f.name}: {dur:.0f} s at {fps:.0f} fps, {img.shape[1]}x{img.shape[0]}"
    if not lo <= dur <= hi:
        r.warn(msg + f" (protocol: {lo:.0f}-{hi:.0f} s)")
    else:
        r.ok(msg)
    if fps and fps < 24:
        r.warn(f"{f.name}: {fps:.0f} fps; tracking is more robust at 30 fps")
    _weights(r)


def _lidar(p: Path, r: Report):
    from roomscan.io import strayscanner as ss
    for name in ("camera_matrix.csv", "odometry.csv", "rgb.mp4"):
        if (p / name).exists():
            r.ok(f"{name} present")
        else:
            r.fail(f"{name} missing: copy the whole Stray Scanner recording folder")
    if r.fails:
        return
    cap = ss.load(p)
    nd = len(cap.depth_frames)
    if nd == 0:
        r.fail("no depth frames: was this recorded on a LiDAR (Pro) iPhone?")
        return
    dur = cap.frames[-1].timestamp - cap.frames[0].timestamp
    r.ok(f"{len(cap.frames)} poses, {nd} depth frames, {dur:.0f} s")
    if dur > LIDAR_MAX_S:
        r.warn(f"{dur:.0f} s recording (protocol: <= {LIDAR_MAX_S:.0f} s; one recording per floor)")


def check(capture: Path) -> Report:
    from roomscan.cli import detect_tier
    r = Report()
    if not capture.exists():
        r.fail(f"{capture} does not exist")
        return r
    try:
        tier = detect_tier(capture)
    except SystemExit as e:
        r.fail(str(e))
        return r
    r.ok(f"detected tier: {tier}")
    {"photo": _photo, "video": _video, "lidar": _lidar}[tier](capture, r)
    return r
