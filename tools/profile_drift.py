import time, pickle, numpy as np
from roomscan.io import strayscanner as ss
from roomscan import drift as D
cap = ss.load("single_scan_floor_only/1a8384c3f6")
t = time.time(); frames, groups = D._submaps(cap); print("submaps %.1fs" % (time.time() - t), len(groups))
orig_wall, orig_icp, orig_solve = D._wall_pts, D._icp2d, D.solve_pose_graph
acc = {"wall": 0, "icp": 0, "solve": 0}
def wrap(name, f):
    def g(*a, **k):
        t = time.time(); r = f(*a, **k); acc[name] += time.time() - t; return r
    return g
D._wall_pts, D._icp2d, D.solve_pose_graph = wrap("wall", orig_wall), wrap("icp", orig_icp), wrap("solve", orig_solve)
t = time.time(); poses, info = D.correct_drift(cap); print("total %.1fs" % (time.time() - t), {k: round(v, 1) for k, v in acc.items()}, info)
import os
if os.path.exists("out/drift_apt.pkl"):    # written by tools/crisp.py (full-resolution solution)
    old = pickle.load(open("out/drift_apt.pkl", "rb"))[0]
    d = max(np.abs(poses[k][:3, 3] - old[k][:3, 3]).max() for k in poses)
    print("max pose change vs previous solution (m): %.2e" % d)
else:
    print("no out/drift_apt.pkl: run tools/crisp.py first to compare against the full solution")
