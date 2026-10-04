# Device matrix

Which tier runs on which iPhone, and the accuracy each tier honestly delivers.
Accuracy cells marked *prior* are the error model used for intervals before
benchmark calibration; they are replaced by measured numbers from
`benchmark/out/REPORT.md` once the benchmark captures exist.

## Hardware support

| iPhone (15 or newer) | Photo tier | Video tier | LiDAR tier |
|---|---|---|---|
| 15, 15 Plus, 16, 16 Plus, 17 | ✅ 0.5× ultra-wide | ✅ | ❌ no LiDAR |
| 16e, Air | ✅ 1× landscape (no ultra-wide) | ✅ | ❌ no LiDAR |
| 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max | ✅ | ✅ | ✅ Stray Scanner |

Capture apps: native Camera (photo, video), Stray Scanner (LiDAR, free). All three tiers can be captured on any Pro model.

## Accuracy per tier

| quantity | LiDAR | video | photo | evidence |
|---|---|---|---|---|
| wall length | ±0.6 cm per face + 0.3% *prior* | ±1.5 cm + 1.2% *prior* | ±3 cm + 3.5% *prior* | gates: LiDAR (Round 1), video ±3%, photo ±8% |
| ceiling height | ±1 cm *prior* | ±2 cm *prior* | ±3.5% *prior* | gate ≤1.5 cm (LiDAR) |
| opening width | ±1 cm (jamb-to-jamb) *prior* | ±2 cm *prior* | ±4 cm *prior* | gate ≤2 cm on ≥85% |
| metric scale | ARKit (LiDAR-anchored) | MoGe-2, self-calibrated FOV | MoGe-2 + EXIF focal length | MoGe vs LiDAR on sample frames: see notes D14 |
| multi-room drift | pose graph, 2.7° / 0.55 m max correction on 9-room sample | pose graph + start=end loop | none (rooms independent, joined at doors) | notes D9 |
| processing time (this laptop, CPU only) | ~1 min (1 room) to ~4 min (9 rooms) | **~9.5 min** for a 48 s walkthrough (56 keyframes × ~10 s) | ~10 s per photo | `timing_s` in every result; D23 |

## What thins the data, and how intervals widen

- **Photo:** no poses and no depth. Scale comes from a single-image network, so every dimension carries the MoGe scale term. A wall is measured from one photo (both corners seen), and opposite photos give independent estimates whose disagreement inflates the interval. A corner not seen widens that wall's interval 3×.
- **Video:** scale is averaged over many frames, and drift is held by the pose graph. The path length enters the interval when drift correction is off.
- **LiDAR:** metric depth, so error is dominated by per-surface bias and grazing-angle noise. A ceiling that wasn't swept is reported as `null` with a warning rather than guessed.
