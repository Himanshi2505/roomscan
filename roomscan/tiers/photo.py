"""Photo tier: 2-8 stills per room, no depth, no poses.

Per photo   MoGe-2 metric point map -> gravity from floor/ceiling/wall normals
            -> camera pose in a gravity-aligned frame (camera at origin).
Per room    SIFT matches between photos lifted to 3D by the point maps ->
            RANSAC 4-DOF + scale (yaw, t, s) per pair -> spanning tree from
            the best-connected photo. Per-photo scale is normalised so the
            geometric mean of the room's photo scales equals MoGe's metric scale.
Output      one OrientedCloud per room (with camera poses, so free-space
            carving and see-through opening tests work as in the LiDAR tier).
Photos that don't register inside their room (typically the doorway photos,
which look into the next room) are returned separately: stitch.py uses them
to link rooms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from roomscan import mono
from roomscan.geometry.cloud import OrientedCloud, voxel_downsample

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
MIN_INLIERS = 25


@dataclass
class View:
    name: str
    mv: mono.MonoView
    R_g: np.ndarray                     # camera -> gravity frame rotation (+Y up)
    kp: np.ndarray = None               # (N,2) keypoints
    desc: np.ndarray = None
    T: np.ndarray | None = None         # 4x4 gravity frame -> room frame (incl. scale)
    scale: float = 1.0


@dataclass
class RoomRecon:
    name: str
    cloud: OrientedCloud | None
    views: list[View]
    unregistered: list[View] = field(default_factory=list)


def exif_fov_x(p: Path) -> float | None:
    """Horizontal FOV (deg) of the EXIF-upright image from FocalLengthIn35mmFilm.
    35 mm equivalence is defined on the diagonal (43.27 mm), so it holds for any
    aspect ratio and either orientation."""
    try:
        im = Image.open(p)
        exif = im.getexif()
        f35 = exif.get_ifd(0x8769).get(0xA405) or exif.get(0xA405)
        if not f35:
            return None
        w, h = ImageOps.exif_transpose(im).size
        f_px = float(f35) * np.hypot(w, h) / 43.27
        return float(np.degrees(2 * np.arctan(w / (2 * f_px))))
    except Exception:
        return None


def load_image(p: Path) -> np.ndarray:
    if p.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise SystemExit("HEIC photo found: pip install pillow-heif (or export JPEG)") from e
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    return np.asarray(im)


def _gravity_rotation(up_cam: np.ndarray) -> np.ndarray:
    """R such that R @ up_cam = +Y and the camera's forward stays in the x-z plane."""
    y = up_cam / np.linalg.norm(up_cam)
    f = np.array([0.0, 0.0, 1.0])
    z = f - (f @ y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.stack([x, y, z])       # rows: world axes expressed in camera coords


def make_view(name: str, rgb: np.ndarray, fov_x: float | None = None) -> View:
    mv = mono.infer(rgb, fov_x)
    R_g = _gravity_rotation(mono.gravity_up(mv))
    sift = cv2.SIFT_create(4000)
    g = cv2.cvtColor(mv.rgb, cv2.COLOR_RGB2GRAY)
    kps, desc = sift.detectAndCompute(g, (mv.mask * 255).astype(np.uint8))
    kp = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
    return View(name, mv, R_g, kp, desc)


def _pts3d(v: View, kp: np.ndarray) -> np.ndarray:
    u = np.clip(kp[:, 0].round().astype(int), 0, v.mv.points.shape[1] - 1)
    w = np.clip(kp[:, 1].round().astype(int), 0, v.mv.points.shape[0] - 1)
    P = v.mv.points[w, u] @ v.R_g.T
    ok = v.mv.mask[w, u] & np.isfinite(P).all(1)
    return P, ok


def _fit_sim4(A, B):
    """Least-squares s, yaw, t with B ~= s*Ry(yaw)*A + t (A,B: (N,3), gravity frames)."""
    ca, cb = A.mean(0), B.mean(0)
    a, b = A - ca, B - cb
    # yaw from the x-z components (rotation about +Y)
    num = np.sum(a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2])
    den = np.sum(a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2])
    th = np.arctan2(num, den)
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    ra = a @ R.T
    s = np.sum(ra * b) / max(np.sum(ra * ra), 1e-9)
    t = cb - s * (R @ ca)
    return s, R, t


def register_pair(vi: View, vj: View, seed: int = 0):
    """Similarity (4-DOF + scale) mapping j's gravity frame into i's. None if weak."""
    if vi.desc is None or vj.desc is None or len(vi.kp) < 20 or len(vj.kp) < 20:
        return None
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(vj.desc, vi.desc, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < MIN_INLIERS:
        return None
    qj = vj.kp[[g.queryIdx for g in good]]
    qi = vi.kp[[g.trainIdx for g in good]]
    return fit_correspondences(vi, vj, qi, qj, seed)


def fit_correspondences(vi: View, vj: View, qi: np.ndarray, qj: np.ndarray, seed: int = 0):
    """Pixel correspondences (qi in i, qj in j) -> similarity mapping j's gravity
    frame into i's, via both views' metric point maps. None if weak.
    Shared by SIFT matching (photos) and KLT tracking (video)."""
    if len(qi) < MIN_INLIERS:
        return None
    # geometric pre-filter with the fundamental matrix
    F, inl = cv2.findFundamentalMat(qj, qi, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None:
        return None
    inl = inl.ravel().astype(bool)
    Pj, okj = _pts3d(vj, qj[inl])
    Pi, oki = _pts3d(vi, qi[inl])
    ok = okj & oki
    Pj, Pi = Pj[ok], Pi[ok]
    if len(Pj) < MIN_INLIERS:
        return None
    rng = np.random.default_rng(seed)
    best, best_in = None, None
    for _ in range(500):
        idx = rng.choice(len(Pj), 3, replace=False)
        s, R, t = _fit_sim4(Pj[idx], Pi[idx])
        if not 0.6 < s < 1.6:
            continue
        r = np.linalg.norm(Pi - (s * Pj @ R.T + t), axis=1)
        tol = 0.05 + 0.03 * np.linalg.norm(Pi, axis=1)       # metric depth noise grows with range
        inn = r < tol
        if best_in is None or inn.sum() > best_in.sum():
            best, best_in = (s, R, t), inn
    if best_in is None or best_in.sum() < MIN_INLIERS:
        return None
    s, R, t = _fit_sim4(Pj[best_in], Pi[best_in])
    return s, R, t, int(best_in.sum())


def _sim_to_T(s, R, t):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def register_room(views: list[View]) -> tuple[list[View], list[View]]:
    n = len(views)
    edges = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                r = register_pair(views[i], views[j], seed=i * 31 + j)
                if r is not None:
                    edges[(i, j)] = r
    if not edges:
        root = max(range(n), key=lambda k: views[k].mv.mask.sum())
        views[root].T = np.eye(4)
        return [views[root]], [v for k, v in enumerate(views) if k != root]
    deg = np.zeros(n)
    for (i, j), r in edges.items():
        deg[i] += r[3]
        deg[j] += r[3]
    root = int(np.argmax(deg))
    views[root].T = np.eye(4)
    placed = {root}
    while True:   # Prim-style: attach the strongest edge from placed to unplaced
        cand = [(r[3], i, j, r) for (i, j), r in edges.items() if i in placed and j not in placed]
        if not cand:
            break
        _, i, j, (s, R, t, _) = max(cand, key=lambda x: x[0])
        views[j].T = views[i].T @ _sim_to_T(s, R, t)
        placed.add(j)
    reg = [views[k] for k in sorted(placed)]
    # normalise scale: geometric mean of per-view scales := 1 (MoGe metric)
    scales = np.array([np.cbrt(np.linalg.det(v.T[:3, :3])) for v in reg])
    g = np.exp(np.mean(np.log(scales)))
    for v in reg:
        v.T = v.T.copy()
        v.T[:3, :] /= g
        v.scale = float(np.cbrt(np.linalg.det(v.T[:3, :3])))
    return reg, [views[k] for k in range(n) if k not in placed]


def view_cloud(v: View, fid: int, stride: int = 2, max_range: float = 8.0) -> OrientedCloud:
    P = v.mv.points[::stride, ::stride]
    N = v.mv.normals[::stride, ::stride]
    m = v.mv.mask[::stride, ::stride] & (np.linalg.norm(P, axis=-1) < max_range)
    A = v.T[:3, :3] @ v.R_g
    sR = A / v.scale
    xyz = P[m] @ A.T + v.T[:3, 3]
    nrm = N[m] @ sR.T
    T_wc = np.eye(4)
    T_wc[:3, :3] = sR
    T_wc[:3, 3] = v.T[:3, 3]
    return OrientedCloud(xyz.astype(np.float64), nrm.astype(np.float64),
                         np.full(int(m.sum()), fid, np.int32), {fid: T_wc})


def reconstruct_room(folder: Path, fid0: int = 0) -> RoomRecon:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    views = [make_view(p.name, load_image(p), exif_fov_x(p)) for p in files]
    reg, unreg = register_room(views)
    clouds = [view_cloud(v, fid0 + k) for k, v in enumerate(reg)]
    c = voxel_downsample(OrientedCloud.concat(clouds), 0.02) if clouds else None
    return RoomRecon(folder.name, c, reg, unreg)
