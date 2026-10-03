# Capture protocol (one page): follow it literally

**Before you start (all tiers):** turn on every light and open the curtains. Close nothing you want measured: **every door between rooms fully open**. Keep people and pets out of the rooms. Hold the phone in **portrait**, at **chest height**, and **walk slowly** (about one step per second).

---

### Tier A: Photos (any iPhone 15 or newer, native Camera app)

1. Camera app → **Photo** mode → tap **0.5×** (ultra-wide). Flash **off**. Leave all other camera settings at their defaults. Hold the phone **upright (portrait)** at chest height, screen facing you, **level** (don't tilt it up or down). **No 0.5× button** (iPhone 16e, iPhone Air): stay on 1× and hold the phone **sideways (landscape)** instead.
2. For **each room**, in this order:
   1. **Wall photos (one per wall, going clockwise round the room):** stand with your **back touching the middle of a wall** and take one photo of the **opposite wall**. Check on screen that **both ends (corners) of the opposite wall** and **both the floor line and the ceiling line** are visible; if not, step to the side until they are. Then move clockwise to the middle of the next wall and repeat. A rectangular room = **4 wall photos**.
   2. **Doorway photos:** stand **in each doorway** of the room and take one photo **into the next room**.
   3. Total per room: **2 to 8 photos**. If you would go over 8, drop doorway photos first, never wall photos.
3. If a wall is blocked by a wardrobe or a bed, still stand at its middle and photograph the opposite wall; furniture in the photo is fine as long as the corners show.
4. **Hand-off:** on the laptop, create one folder per room named `01_kitchen`, `02_hall`, … and copy that room's photos into it (AirDrop or cable, **original files**, not screenshots, keep the file names; they record the shooting order). Put all room folders inside one folder, e.g. `flat_photos/`.
5. Run: `python -m roomscan run flat_photos`

### Tier B: Video (any iPhone 15 or newer, native Camera app)

1. Camera app → **Video** mode, **1×**, **1080p at 30 fps** (Settings → Camera → Record Video). Turn **Action mode off** and **Cinematic off**.
2. Start recording **standing in the doorway of the first room**, pointing at the floor in front of you, and count to 3.
3. In every room: walk **once along all the walls** at about 1 m from them, pointing the phone **at the wall in front of you**. Then **tilt up to the ceiling for 2 seconds** and back down. Pass slowly through every doorway: **stop in the doorway for 2 seconds** and point at the next room.
4. Visit every room **once** and finish **back where you started**, pointing at the same floor spot, count to 3, stop. Maximum length **6 minutes**.
5. **Hand-off:** AirDrop the `.MOV` to the laptop. Run: `python -m roomscan run IMG_1234.MOV`

### Tier C: LiDAR (iPhone 12 Pro or newer **Pro** model, Stray Scanner app)

1. Install **Stray Scanner** (free, App Store). Leave its settings at their defaults. Tap the red button to start/stop recording.
2. Same walking path as Tier B (start in a doorway, all walls once, **tilt to the ceiling in every room**, stop 2 s in every doorway, end where you started). Keep the phone **0.5–3 m** from surfaces. Maximum length **4 minutes**; for larger homes make one recording per floor.
3. **Hand-off:** open the **Files** app → On My iPhone → **Stray Scanner** → copy the newest recording folder (contains `rgb.mp4`, `depth/`, `odometry.csv`) to the laptop by AirDrop or cable. Run: `python -m roomscan run <folder>`

---

**Avoid, in every tier:** walking backwards; covering the camera with fingers; sudden spins (turn over at least 2 seconds); filming straight into a window or the sun; standing still for more than 10 s; leaving any door half-open.

**Tricky surfaces:** for **mirrors and glass** doors, pass them at an angle, never square on. For **wet/glossy floors**, keep at least 1 m back and tilt the phone more steeply. In **low light**, switch on every light; if the room is still dark, do Tier B/C more slowly (half speed).

**Output:** `out/<capture>/result.json` (all measurements with 90% intervals) and `out/<capture>/plan.png` (whole-property floor plan).
