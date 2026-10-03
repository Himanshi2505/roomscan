"""Regenerate the ceiling fix loop: run tags fix-before and fix-after, compare.

    python tools/fixloop_ceiling.py run fix-before target     # one pipeline pass each,
    python tools/fixloop_ceiling.py run fix-after target      # run separately to keep
    python tools/fixloop_ceiling.py run fix-before apartment  # peak memory low
    python tools/fixloop_ceiling.py run fix-after apartment
    python tools/fixloop_ceiling.py run fix-after-2 target     # iteration 2
    python tools/fixloop_ceiling.py run fix-after-2 apartment
    python tools/fixloop_ceiling.py report

Each tag is checked out into its own git worktree (fixloop/wt_<tag>), the
pipeline runs there on:
  * ARKitScenes 42445021 (third-party, laser GT ceiling 2.3149 m), the target;
  * the 9-room sample apartment, side-effect check (room count, areas).
Writes fixloop/CEILING_DELTA.md and fixloop/code.diff. Needs the data from
tools/fetch_arkitscenes.py + tools/arkitscenes_to_stray.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixloop"
GT_CEILING = 2.3149
TARGET = ROOT / "data/public/stray/42445021"
SIDE = ROOT / "single_scan_floor_only/1a8384c3f6"


def sh(*cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


SCANS = {"target": TARGET, "apartment": SIDE}


def worktree(tag: str) -> Path:
    wt = OUT / f"wt_{tag}"
    if not wt.exists():
        OUT.mkdir(exist_ok=True)
        sh("git", "worktree", "add", "--detach", str(wt), tag)
    return wt


def run_one(tag: str, scan: str) -> None:
    wt = worktree(tag)
    out = OUT / tag / scan
    env = dict(os.environ, PYTHONPATH=str(wt))
    subprocess.run([sys.executable, "-m", "roomscan", "run", str(SCANS[scan]), "--out", str(out)],
                   cwd=wt, env=env, check=True, capture_output=True)
    print("done", tag, scan, "->", out / "result.json")


def load(tag: str, scan: str) -> dict | None:
    f = OUT / tag / scan / "result.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        return run_one(sys.argv[2], sys.argv[3])
    r = {tag: {scan: load(tag, scan) for scan in SCANS} for tag in ("fix-before", "fix-after", "fix-after-2")}
    L = ["# Fix loop: ceiling height (LiDAR tier)", "",
         "Target: ARKitScenes 42445021 (third-party), laser GT ceiling 2.3149 m; gate <= 1.5 cm.", "",
         "| tag | room | ceiling (m) | error (cm) | gate |", "|---|---|---|---|---|"]
    for tag in r:
        if r[tag]["target"] is None:
            L.append(f"| {tag} | not run yet | | | |")
            continue
        for room in sorted(r[tag]["target"]["rooms"], key=lambda x: -x["floor_area"]["value"]):
            h = room["ceiling_height"]
            if h is None:
                continue
            e = 100 * (h["value"] - GT_CEILING)
            L.append(f"| {tag} | {room['id']} ({room['floor_area']['value']:.1f} m2) | {h['value']:.4f} | "
                     f"{e:+.2f} | {'PASS' if abs(e) <= 1.5 else 'FAIL'} |")
    L += ["", "Side effects, 9-room sample apartment:", "",
          "| tag | rooms | footprint (m2) | room areas |", "|---|---|---|---|"]
    for tag in r:
        a = r[tag]["apartment"]
        if a is None:
            L.append(f"| {tag} | not run yet | | |")
            continue
        areas = sorted((x["floor_area"]["value"] for x in a["rooms"]), reverse=True)
        L.append(f"| {tag} | {len(a['rooms'])} | {a['footprint_area']['value']} | {areas} |")
    (OUT / "CEILING_DELTA.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "code.diff").write_text(sh("git", "diff", "fix-before", "fix-after-2", "--", "roomscan"),
                                   encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
