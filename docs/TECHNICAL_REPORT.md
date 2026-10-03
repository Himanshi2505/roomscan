# roomscan: technical report

*Draft. Sections marked ⏳ are filled from `benchmark/out/REPORT.md` once the benchmark captures exist. Design evidence lives in `notes/decisions.md` (D1–D15).*

## 1. Architecture

One capture in, one schema-validated `result.json` and one rendered plan out: `python -m roomscan run <capture>`. The tier is detected from the input.

```
 photos/ ─► MoGe-2 per photo ─► gravity + Manhattan ─┬─► wall-per-photo layout ──┐
                                                    └─► SIFT registration ──────┤ per room
 video  ─► KLT keyframing ─► MoGe-2 per keyframe ─► visual-metric odometry ─┐   │
 LiDAR  ─► Stray Scanner depth + ARKit poses ─────────────────────────────┤   │
                                       2D pose graph (drift) ◄────────────┘   │
                                               │                              │
                    oriented point cloud (gravity-aligned, metric) ◄──────────┘
                                               │
   floor/ceiling levels ─► wall faces ─► free-space carving ─► room segmentation
     ─► line-arrangement polygons ─► openings (see-through test) ─► adjacency
                                               │
          photo tier: door-based stitching ◄───┤
                                               ▼
      damage (CLIP zero-shot, geometry-gated) ─► concealed-damage rules ─► scope
                                               ▼
              intervals (per-tier error model × calibrated k) ─► JSON + plan
```

Every tier converges on the same **oriented point cloud** (metric, +Y up, per-point source camera). Everything downstream is shared, which is how three very different inputs honour one output contract. The photo tier also has a layout-first path that skips the cloud for textureless rooms (§2.1).

**Dimensions come from wall faces, not pixels.** A wall face is a 1D histogram peak of point coordinates in the Manhattan frame, and its position is the mean of thousands of inliers. The 3 cm raster is used only for topology: which room is where. Room polygons are chosen from the arrangement of face lines (cell coverage plus an MDL-style penalty per line), so every edge lies exactly on a measured face (D4, D6).

**Openings** need see-through evidence. A missing wall with hits measured *behind* the plane by a camera *inside* the room is an opening. A missing wall with nothing behind it is occlusion. This is the main phantom-opening guard (D7), and it's covered by a regression test.

## 2. Tier design and device matrix

See `docs/DEVICE_MATRIX.md`. Summary:

| tier | hardware | metric scale from | pose from | drift handling |
|---|---|---|---|---|
| LiDAR | iPhone Pro (12+) | LiDAR depth | ARKit VIO | submap pose graph |
| video | any iPhone 15+ | MoGe-2 ViT-L, self-calibrated FOV, ×1.05 bias fix | KLT + point-map RANSAC | keyframe pose graph |
| photo | any iPhone 15+ | MoGe-2 ViT-L + EXIF FOV, door-head anchor | none (layout) / SIFT | n/a: rooms joined at doors |

### 2.1 Photo tier

The first design (corner photos plus SIFT registration) registered **0 of 15** photo pairs on the stand-in set, because white walls have no texture. A geometric check also showed that 1× corner photos are under-constrained even with perfect matching (D11). The shipped design is **layout-first**: 0.5× ultra-wide, back to the middle of a wall, one photo of the opposite wall, clockwise.
- Each photo measures one wall corner-to-corner on its own (the side-wall faces are the corners), plus the camera-to-wall depth, which estimates the other dimension.
- Opposite photos are independent estimates. Their disagreement inflates the interval (Birge ratio).
- Rooms are stitched through doors: doorway photos registered into the neighbouring room pick the door, with a jigsaw search (no overlap, maximum shared wall) as fallback.

**Metric scale is the photo tier's dominant error.** Measured against LiDAR on 24 frames (D14), MoGe-2 ViT-S is off by a median of 15%. ViT-L is off by 6–7.5% (p90 ~18%) and reads ~5% short. Mitigations:
- ViT-L
- EXIF focal length (cuts scatter by a third)
- a global ×1.05 correction (fitted on the sample captures, not on the benchmark)
- averaging over photos
- a door-head anchor: interior door openings are 2.05 ± 0.04 m, fused by inverse variance with MoGe's scale.

### 2.2 Video tier

Keyframes are chosen by tracking quality: KLT corners are tracked through every frame, and a new keyframe starts when fewer than 30% survive. On the stand-in clip, fixed 1 fps keyframes broke the trajectory into 15 segments. ⏳ adaptive result. Scale is never chained, because each keyframe keeps MoGe's own metric scale, so scale error doesn't random-walk. The protocol ends the walk where it began, which guarantees a loop closure.

## 3. Drift handling

ARKit and visual odometry are gravity-aligned, so the graph is solved in (x, z, yaw).
- **Nodes:** 1.5 m submaps (LiDAR) or keyframes (video).
- **Factors:** relative odometry; ICP (LiDAR) or feature (video) loop closures with a robust loss; a **Manhattan yaw prior** per node, which is the plane-anchored correction that pulls each node's walls to the global 90° grid.

On the 9-room sample apartment: 36 submaps, 3 loop closures, maximum correction 0.55 m / 2.7°. Without ground truth the evidence is internal: wall crispness (occupied 2 cm wall cells) improves 5.5% with correction on (D9). Solver time went from 478 s to 30 s with an unchanged solution (D15). ⏳ footprint ablation vs ground truth (`--no-drift`).

## 4. Error budget

σ² = σ²_stat + n_faces·σ²_face + (scale_rel·L)² + (drift_rel·path)², with a 90% interval of ±1.645·k·σ.

| term | LiDAR | video | photo | source |
|---|---|---|---|---|
| σ_face (per surface bias) | 0.6 cm | 1.5 cm | 3 cm | prior → calibrated |
| scale_rel | 0.3% | 1.2% | per-photo 8% (averages down) + common 4% (door anchor) | D14 |
| drift_rel | 0.2%/m (off when corrected) | 0.6%/m | – | prior |

⏳ measured contribution of each term on the benchmark.

## 5. Calibration analysis

k per tier is fitted conformally: the smallest k such that ≥90% of ground-truth residuals fall inside the interval (`benchmark.evaluate --calibrate`), and the coverage achieved is reported per tier. ⏳ k values, coverage before and after, and reliability per quantity type.

## 6. Benchmark ⏳

Gates at all three tiers, the repeatability table, the head-to-head (app and version) and timing: see `benchmark/out/REPORT.md`.

## 7. Fix loop ⏳

See `docs/FIX_LOOP.md`. Before and after regenerate from tags with `python scripts/fix_loop.py`.

## 8. Known failure modes

- **Mirrors and glass:** LiDAR and MoGe "see" a phantom room behind them. Free-space rays are blocked by a barrier of all wall-height vertical points, and interior components not touched by the walked path are dropped. A mirror spanning a whole wall can still extend a room.
- **Unswept ceiling:** reported as `null` with a warning, never guessed (sample apartment scan).
- **Textureless rooms:** handled in the photo tier by the layout path (rectangles only). An L-shaped textureless room falls back to the feature path, and may fail.
- **Non-Manhattan walls:** everything snaps to 90°. Angled walls become staircases.
- **Sideways images without orientation metadata:** gravity is initialised from image up (D11). iPhone photos and videos always carry it.
- **Damage:** zero-shot CLIP. After the audit, 2 single-view false positives remain on an undamaged room, and recall is unknown until the staged-damage room. Thresholds are a fix-loop candidate.
- **Closed doors** read as wall; the protocol requires open doors.
- **CPU time:** photo and video tiers run ~10 s per image with MoGe-L. A GPU makes this about 20× faster.
- **Low light and wet-look floors:** LiDAR confidence drops (filtered at confidence 2), and MoGe degrades in the dark. ⏳ measured.
