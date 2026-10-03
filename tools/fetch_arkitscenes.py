"""Download a small, disclosed third-party validation set from Apple's ARKitScenes.

    python tools/fetch_arkitscenes.py

Not our capture: recorded by Apple on an iPad Pro (2020), with a static laser
scanner for ground truth. Used only as clearly labelled supplementary
validation (see docs/PUBLIC_VALIDATION.md). Licence: ARKitScenes LICENSE
(commercial use allowed below 700M monthly active users). Never committed:
lands in data/public/ (git-ignored via data/).

One visit (room) recorded twice -> LiDAR tier + repeatability; one .mov ->
video tier; one laser scan -> ground truth.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

BASE = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
VISIT = "421380"
VIDEOS = ["42445021", "42445022"]
MOV_VIDEO = "42445021"
LASER = "170107"
ASSETS = ["lowres_depth.zip", "confidence.zip", "lowres_wide.zip",
          "lowres_wide_intrinsics.zip", "lowres_wide.traj"]
OUT = Path(__file__).resolve().parent.parent / "data" / "public" / "arkitscenes"


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print("have", dst.name)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("get ", url, flush=True)
    subprocess.run(["curl", "-sSL", "--retry", "3", "-C", "-", "-o", str(dst), url], check=True)


def main():
    for vid in VIDEOS:
        d = OUT / vid
        for a in ASSETS:
            fetch(f"{BASE}/Validation/{vid}/{a}", d / a)
            if a.endswith(".zip") and not (d / a[:-4]).exists():
                with zipfile.ZipFile(d / a) as z:
                    z.extractall(d)
                (d / a).unlink()                 # keep disk use down (disk is ~94% full)
    fetch(f"{BASE}/Validation/{MOV_VIDEO}/{MOV_VIDEO}.mov", OUT / MOV_VIDEO / f"{MOV_VIDEO}.mov")
    for ext in (".ply", "_pose.txt"):
        fetch(f"{BASE}/laser_scanner_point_clouds/{VISIT}/{LASER}{ext}",
              OUT / "laser" / f"{LASER}{ext}")
    print("done:", OUT)


if __name__ == "__main__":
    sys.exit(main())
