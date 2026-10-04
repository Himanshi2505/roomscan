# Defense guide (study this, tools closed)

Each answer is a claim plus its evidence. Full detail: `notes/decisions.md` (D-numbers).

## 1. Architecture in one breath
Every tier turns its input into **one oriented point cloud**: metric, gravity-aligned with +Y up, and each point knows its source camera. Everything after that is shared:
- floor and ceiling levels
- wall faces
- free-space carving
- room segmentation
- polygons
- openings
- damage
- intervals

That shared core is how three different inputs produce one output contract.

## 2. Likely questions

**Why Route 2 (stock app), not your own iOS app?**
No Mac or Xcode was available. Stray Scanner records depth, confidence, ARKit poses and intrinsics in an open folder format (D1). The protocol is one page and a non-engineer can follow it.

**Where do dimensions actually come from?**
From **wall faces**, not pixels. A wall face is a 1D histogram peak of point coordinates after rotating the cloud to the Manhattan frame, and its position is the mean of thousands of inliers. The 3 cm raster is only used to decide which room is where (D4). Room polygons are chosen from the arrangement of face lines, so every edge lies on a measured face (D6).

**How do you avoid phantom doors?**
An opening needs **see-through evidence**: wall points are missing *and* there are points behind the wall plane, measured by a camera standing inside the room. A gap with nothing seen behind it is occlusion, for example a sofa (D7). There's a regression test for it.

**What do you do about drift?**
Two parts:
- **2D pose graph** over 1.5 m submaps, in x, z and yaw. Its factors are ARKit odometry, ICP loop closures, and a **Manhattan yaw prior**: the "plane-anchored" correction, which pulls each submap's walls back to the global 90° grid (D9).
- **Floor-anchored vertical correction**, added after real data showed ~3 cm of height drift over 48 s (D22). My original assumption that height drift was negligible was wrong, and the fix-loop data proved it.

`--no-drift` is the ablation.

**Why MoGe-2 for photo and video?**
It's MIT-licensed (MASt3R and VGGT are non-commercial). It gives metric depth, intrinsics and normals from a single image, and it runs on CPU (D10).
- **Scale error measured against LiDAR:** ViT-S 15%, ViT-L 6–7.5% (D14).
- **Mitigations:** the large model, EXIF focal length, a ×1.05 global correction fitted on samples (not on the benchmark), averaging over photos, and a door-height anchor (2.05 ± 0.04 m).

**Why is the photo tier "one wall per photo"?**
SIFT registered **0 of 15** photo pairs on white walls. Also, corner photos at 1× don't see enough to constrain the room. So the protocol uses 0.5×, back to the middle of a wall, facing the opposite wall (D11):
- each photo measures one wall corner-to-corner;
- opposite photos are independent estimates;
- their disagreement widens the interval.

**How are intervals made honest?**
σ² = statistical + per-surface bias + scale error × length + drift error × path. The 90% interval is ±1.645·k·σ, and k is fitted **conformally** on ground truth (D8). Until that calibration exists, every result says `calibrated: false`. Video uses a provisional k = 2.1 from a measured stand-in (D18).

**How do you know the numbers are real if you have no benchmark yet?**
They come from **third-party** laser ground truth (Apple ARKitScenes), labelled as such everywhere (`docs/PUBLIC_VALIDATION.md`):
- **wall-to-wall room dimensions:** −0.8 cm and −0.3 cm. They're measured inside our own cloud, so the registration can't flatter them.
- **ceiling height after the fix loop:** −1.29 cm and −0.59 cm.

**Walk me through your fix loop.**
`docs/FIX_LOOP.md`. Every declaration was committed *before* its code.
1. **First target, room segmentation:** three attempts failed. I found the ground truth itself assumed walls behind a 2.4 m opening, so I abandoned that target and reported why (D21).
2. **Second target, ceiling height (+1.51 / +3.01 cm, gate 1.5 cm):**
   - **Iteration 1:** heights from raw horizontal points. My declared mechanism (quantization) was **wrong**; an isolation test showed cross-surface mixing. Result: one room passed.
   - **Iteration 2:** vertical drift. I tested *time vs place*, and the offsets followed time. Result: the **gate passes**.
3. **Missed predictions, reported:** R1 at −1.29 instead of ±0.5 cm (a far-range ceiling bias remains), and the apartment's footprint moved +0.9%.

## 3. Weaknesses to state before they ask
- **Intervals ignore room-layout errors.** On the third-party benchmark, the fitted k is 58 for LiDAR. Measurements are honest, but when a room is split, the polygon walls are a metre off with millimetre intervals. I found this with my own scorer and did *not* paper over it with k = 58. The fix is a layout-uncertainty term.
- **No own benchmark captures, no Polycam head-to-head.** Those need a phone and a property; the harness is built and tested.
- **The video tier is the least reliable.** On a real third-party walkthrough: walls off by 18%, ceiling +52 cm, 21 min on CPU. It fails the 3% gate. Say so plainly: its intervals were wide enough to contain the truth, so it doesn't produce confident garbage, but it isn't accurate yet. Monocular scale and the pose distortion are the causes.
- **Room segmentation** can split rooms at free-standing furniture or merge them through wide openings. The measurement core is accurate; the room layer above it is the weak point.
- **Damage detection is zero-shot CLIP.** False positives went from 11 to 2 after an audit; recall is untested.
- **CPU-only:** about 10 s per image. A 48 s video takes ~9.5 min (was 21 min, before one-inference-per-keyframe). Smaller images were tried and didn't help, because MoGe's cost is set by its token budget. A smaller model would be faster but 2× less accurate on scale, and I chose accuracy. A GPU should help, but that is unmeasured.

## 4. Numbers to remember
| what | number |
|---|---|
| room dims vs laser | −0.8 / −0.3 cm |
| ceiling vs laser | −1.29 / −0.59 cm |
| vertical drift found | up to 3.6 cm |
| MoGe-L scale error | ~6–7.5% median |
| drift solver | 478 s → 30 s, same solution |
| damage false positives | 11 → 2 |
| tests | 21 passing |
