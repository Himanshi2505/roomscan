# roomscan

Phone capture → dimensioned whole-property floor plan, damage regions, concealed-damage flags and scope line items, with a 90% interval on every measurement. Three input tiers (photos, video, LiDAR) share one output contract.

## Fresh machine to first result (≈10 min)

Requires Python 3.11–3.13 and git. CPU only; a CUDA GPU makes the photo/video tiers much faster.

```bash
git clone <this repo> roomscan && cd roomscan
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt          # ~3 min (CPU PyTorch)
python scripts/fetch_weights.py --large  # ~1.5 GB: MoGe-2 (photo/video tiers)
python -m roomscan check <capture>       # optional: seconds-long pre-flight check
python -m roomscan run <capture>         # one command per capture
```

`<capture>` is whatever the [capture protocol](docs/CAPTURE_PROTOCOL.md) produces:

| tier | input | detected by |
|---|---|---|
| LiDAR | Stray Scanner folder (`rgb.mp4`, `depth/`, `odometry.csv`) | `odometry.csv` + `depth/` |
| video | `IMG_xxxx.MOV` (or a folder holding it) | video extension |
| photo | folder of per-room folders of photos | sub-folders of images |

Output goes to `out/<capture>/`:
- `result.json` follows [`roomscan/schema/plan.schema.json`](roomscan/schema/plan.schema.json) and is validated before writing
- `plan.png` / `plan.svg` is the rendered stitched plan

Options: `--tier` to force a tier, `--no-drift` for the drift ablation (poses as-is), `--out DIR`.

## Benchmark

```bash
python -m benchmark.evaluate               # score every capture in benchmark/captures.json
python -m benchmark.evaluate --calibrate   # also fit the per-tier interval factor
```

Ground truth format: [`benchmark/gt/EXAMPLE.json`](benchmark/gt/EXAMPLE.json). Capture list: [`benchmark/captures.EXAMPLE.json`](benchmark/captures.EXAMPLE.json). The report is written to `benchmark/out/REPORT.md`.

## Layout

```
roomscan/
  cli.py, pipeline.py      one command, tier dispatch
  io/strayscanner.py       LiDAR capture loader
  tiers/lidar.py           depth fusion
  tiers/video.py           keyframes, visual-metric odometry, pose graph
  tiers/photo.py           per-photo geometry, feature registration
  tiers/photo_layout.py    textureless rooms: one wall measured per photo
  mono.py                  MoGe-2 monocular metric geometry (cached)
  drift.py                 2D pose graph: odometry + loop closure + Manhattan prior
  geometry/                levels, wall faces, room segmentation, polygons, openings
  plan.py                  cloud -> per-room plan with intervals
  stitch.py                photo tier: rooms joined through doors
  damage.py                damage regions, concealed-damage rules, scope
  uncertainty.py           per-tier error model + calibration
  render.py                plan rendering
benchmark/                 ground truth, gates, report
docs/                      capture protocol, device matrix, compliance matrix, report
notes/decisions.md         every design decision with its evidence
```

## Pretrained models (disclosure)

| model | use | licence | fetched by |
|---|---|---|---|
| MoGe-2 ViT-L / ViT-S (Microsoft) | metric point maps, normals, intrinsics for photo/video | MIT | `scripts/fetch_weights.py` |
| OpenCLIP ViT-B-32, LAION-2B (LAION) | zero-shot damage classification | MIT | auto-downloaded on first run to `weights/clip` |

**Third-party data:** Apple's ARKitScenes (one scene, iPad Pro LiDAR with laser ground truth) is used *only* as clearly labelled supplementary validation, fetched by `tools/fetch_arkitscenes.py`, never committed. See `docs/PUBLIC_VALIDATION.md`.

Nothing calls our own infrastructure. Model outputs are cached under `out/cache` keyed by input hash, so re-runs replay deterministically. Delete that folder to force the live path.
