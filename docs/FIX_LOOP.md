# Fix-loop declaration

Committed **before** the fix. Before = tag `fix-before`, after = tag `fix-after`.
Regenerate both with `python tools/fixloop_ceiling.py` (laser ground truth, ARKitScenes visit 421380; third-party data, see docs/PUBLIC_VALIDATION.md).

## 1. Worst-performing gate

- **Ceiling height, LiDAR tier:** gate ≤ 1.5 cm per room.
- **Failing number:** recording 42445021 measures **2.3338 m** against a laser ground truth of **2.3149 m**, an error of **+1.89 cm**.
- **Why this gate:** the room-segmentation attempt (notes D21) turned out to be scored against a ground truth with an assumed wall, so it was the wrong target. Ceiling height is fully observed by the laser, with no assumptions.

## 2. Root-cause hypothesis and evidence

**Hypothesis:** floor and ceiling heights are estimated on the fused cloud after **2 cm voxel averaging**. Averaging the points inside each 2 cm cell quantizes their heights into layers, and the peak-bin + trimmed-mean estimator reads a biased plane height from that quantized distribution.

**Evidence** (`tools/ceiling_diag.py`):

| | floor | ceiling | height | error |
|---|---|---|---|---|
| raw LiDAR points (confidence 2, no voxels) | −1.5220 | 0.7942 | 2.3162 | +0.13 cm |
| pipeline, 2 cm voxel-averaged cloud | −1.5356 | 0.7982 | 2.3338 | +1.89 cm |

- The fused cloud's 1 cm height histogram is a comb (10.8k, 5.0k, 11.8k, 1.7k, 6.7k). That pattern isn't physical; it's voxel quantization.
- The raw points' histogram is smooth.
- **The sensor is not the cause:** raw data agree with the laser to 1.3 mm.
- **Alternatives ruled out:**
  - *Range- or angle-dependent LiDAR bias on the ceiling.* Ceiling plane height varies by ≤ 0.4 cm across 0–3 m range and 40–75° incidence.
  - *A rug pulling the floor estimate.* There is a second floor surface 4 cm up, but the estimator already picks the lower peak.

## 3. Fix and predicted number

- **Fix:** the LiDAR tier keeps a raw, non-voxelized subsample of horizontal-surface points. Plane heights (whole scan and per room) are estimated on those; walls and everything else still use the fused cloud.
- **Predicted:** ceiling error **+1.89 cm → within ±0.5 cm**, so the gate flips **fail → pass**.
- **Predicted side effects:** none on the 9-room sample apartment's plan (same room count and areas).

## 4. Result

*(filled after `fix-after`)*
