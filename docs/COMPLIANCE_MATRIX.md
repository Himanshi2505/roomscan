# Compliance matrix

Status key:
- **done**: implemented and exercised on real data in this repo
- **built**: implemented and runs end to end, but accuracy is not yet verified against ground truth
- **needs captures**: blocked on benchmark captures and tape ground truth (physical work)

| # | Requirement (spec section) | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | Capture route: stock protocol (Part 1, Route 2) | `docs/CAPTURE_PROTOCOL.md` | one-page protocol, all three tiers | done; walk-through by a non-engineer pending |
| 2 | Photo tier: 2–8 stills per room, no depth/poses (Part 1.1) | `roomscan/tiers/photo.py`, `roomscan/tiers/photo_layout.py`, `roomscan/mono.py` | per-room plan from photos | built |
| 3 | Photo folders → one stitched whole-property plan (Part 1.1, gate) | `roomscan/stitch.py`, `roomscan/pipeline.py::_photo` | stitched plan, adjacency, no overlaps | built |
| 4 | Video tier: handheld walkthrough (Part 1.2) | `roomscan/tiers/video.py` | multi-room plan from `.MOV` | built |
| 5 | LiDAR tier: depth, poses, intrinsics (Part 1.3) | `roomscan/io/strayscanner.py`, `roomscan/tiers/lidar.py` | multi-room plan | done (3 sample captures) |
| 6 | Device matrix with honest accuracy per tier (Part 1) | `docs/DEVICE_MATRIX.md` | table | built; accuracy column needs captures |
| 7 | Per-room plan: walls, ceiling height, floor area, openings (Part 2) | `roomscan/plan.py`, `roomscan/geometry/*` | `result.json` → `rooms[]` | done (LiDAR), built (photo/video) |
| 8 | Stitched multi-room plan with correct adjacency (Part 2) | `roomscan/geometry/rooms.py`, `roomscan/plan.py::_adjacency`, `roomscan/stitch.py` | `adjacency[]`, `plan.png` | done (LiDAR: 9-room apartment), built (photo/video) |
| 9 | Per-surface damage regions with class and metric extent (Part 2) | `roomscan/damage.py::detect` | `damage[]` keyed to wall/ceiling ids | built; false-positive audit done (11 → 2), recall needs the staged-damage room |
| 10 | Concealed-damage flags with the rule that fired (Part 2) | `roomscan/damage.py::RULES` | `concealed_flags[]` with `rule_id`, `rule`, `evidence` | built |
| 11 | Scope line items keyed to surfaces (Part 2) | `roomscan/damage.py::_scope_for` | `scope[]` with `surface_id`, quantity ± interval | built |
| 12 | Confidence interval on every measurement (Part 2) | `roomscan/uncertainty.py` | `{value, sigma, ci90}` everywhere | done; calibration factor needs GT |
| 13 | One command per capture (Part 2) | `roomscan/cli.py` | `python -m roomscan run <capture>` | done |
| 14 | JSON to the published schema (Part 2) | `roomscan/schema/plan.schema.json` | validated on every run | done |
| 15 | Rendered plan (Part 2) | `roomscan/render.py` | `plan.png`, `plan.svg` | done |
| 16 | Benchmark set composition (Part 2) | `benchmark/captures.json`, `data/raw/` | captures + GT | **needs captures** |
| 17 | Gate: opening widths ≤2 cm on ≥85%, misses and phantoms scored | `benchmark/evaluate.py::gates` | REPORT.md | harness done, tested; numbers need captures |
| 18 | Gate: ceiling ≤1.5 cm; spread ≤1 cm; biased vs unrepeatable verdict | `benchmark/evaluate.py::repeatability` | REPORT.md | harness done, tested; numbers need captures |
| 19 | Gate: repeatability ≤1 cm or 0.5% per wall | `benchmark/evaluate.py::repeatability` | REPORT.md | harness done, tested; numbers need captures |
| 20 | Gate: drift accountability + on/off ablation | `roomscan/drift.py`, `--no-drift` | `drift{}` in result, ablation rows | done on sample (crispness evidence); footprint-vs-GT needs captures |
| 21 | Gate: photo-tier whole-property stitch ±8% | `benchmark/evaluate.py::gates` | REPORT.md | harness done; numbers need captures |
| 22 | Photo ±8% / video ±3% wall gates, calibration scored per tier | `benchmark/evaluate.py`, `--calibrate` | REPORT.md, `config/calibration.json` | harness done; numbers need captures |
| 23 | Head-to-head vs consumer app on 2 rooms, ≥70% beat/tie (Part 3) | `benchmark/evaluate.py::head_to_head`, `benchmark/competitor/` | table in REPORT.md | harness done; **needs app export** |
| 24 | Fix loop: declaration, before/after, readable diff (Part 4) | `docs/FIX_LOOP.md`, tagged commits | before/after REPORT.md | **needs benchmark** to pick the worst gate |
| 25 | Process evidence: commit as you work (Part 5) | git history | incremental commits | ongoing |
| 26 | README: running on a fresh capture in <15 min (Deliverable 3) | `README.md` | setup + one command | done; clean-machine timing check pending |
| 27 | Reproduction bundle (Deliverable 4) | `benchmark/`, `scripts/fetch_weights.py`, `out/cache` | regenerate every number | built |
| 28 | Technical report ≤6 pages (Deliverable 7) | `docs/TECHNICAL_REPORT.md` | report | pending |
| 29 | Raw benchmark data: sensor logs, GT, app exports (Deliverable 8) | `data/raw/`, `benchmark/gt/`, `benchmark/competitor/` | files | **needs captures** |
| 30 | Weights fetched by script (Constraints) | `scripts/fetch_weights.py` | MoGe-2 weights | done |
| 31 | Mirrors, glass, wet-look surfaces, low light (Constraints) | `roomscan/geometry/rooms.py` (barrier + trajectory filter), `docs/CAPTURE_PROTOCOL.md` | handling + protocol guidance | partly: glass/mirror leak handling done on samples; wet/low-light need captures |
| 32 | Pretrained models disclosed, no own infrastructure (Constraints) | `README.md` § Pretrained models | disclosure table | done |
