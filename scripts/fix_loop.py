"""Regenerate the fix loop: benchmark at tag `fix-before` and `fix-after`, then diff.

    python scripts/fix_loop.py [--before fix-before] [--after fix-after]

Each tag is checked out into its own git worktree under fixloop/, the benchmark
runs there from the same raw data (data/raw is shared, never copied) with
--rerun so no cached result from the other side leaks in, and we write:

  fixloop/BEFORE_REPORT.md, fixloop/AFTER_REPORT.md
  fixloop/GATES_DELTA.md      gate-by-gate before -> after
  fixloop/code.diff           git diff before..after over roomscan/ benchmark/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixloop"


def sh(*cmd, cwd=ROOT, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


def bench(tag: str) -> dict:
    wt = OUT / f"wt_{tag}"
    if wt.exists():
        sh("git", "worktree", "remove", "--force", str(wt))
    sh("git", "worktree", "add", "--detach", str(wt), tag)
    # share raw data, ground truth from the *current* checkout (identical inputs on both sides)
    for d in ("data", "weights"):
        src = ROOT / d
        if src.exists() and not (wt / d).exists():
            try:
                os.symlink(src, wt / d, target_is_directory=True)
            except OSError:                       # Windows without symlink rights
                sh("cmd", "/c", "mklink", "/J", str(wt / d), str(src))
    shutil.copytree(ROOT / "benchmark" / "gt", wt / "benchmark" / "gt", dirs_exist_ok=True)
    shutil.copy(ROOT / "benchmark" / "captures.json", wt / "benchmark" / "captures.json")
    env = dict(os.environ, PYTHONPATH=str(wt))
    subprocess.run([sys.executable, "-m", "benchmark.evaluate", "--rerun"], cwd=wt, env=env, check=True)
    shutil.copy(wt / "benchmark" / "out" / "REPORT.md", OUT / f"{tag.upper()}_REPORT.md")
    return json.loads((wt / "benchmark" / "out" / "results.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="fix-before")
    ap.add_argument("--after", default="fix-after")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rb, ra = bench(a.before), bench(a.after)
    L = [f"# Fix loop: {a.before} -> {a.after}", "",
         "| capture | gate | before | after | before result | after result |",
         "|---|---|---|---|---|---|"]
    for cid, sb in rb["captures"].items():
        sa = ra["captures"].get(cid, {"gates": {}})
        for g, vb in sb["gates"].items():
            va = sa["gates"].get(g, {"value": None, "pass": None})
            fmt = lambda v: "–" if v is None else f"{v:.4f}"
            pf = lambda p: "–" if p is None else ("PASS" if p else "FAIL")
            L.append(f"| {cid} | {g} | {fmt(vb['value'])} | {fmt(va['value'])} | "
                     f"{pf(vb['pass'])} | {pf(va['pass'])} |")
    (OUT / "GATES_DELTA.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "code.diff").write_text(sh("git", "diff", a.before, a.after, "--", "roomscan", "benchmark"),
                                   encoding="utf-8")
    print((OUT / "GATES_DELTA.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
