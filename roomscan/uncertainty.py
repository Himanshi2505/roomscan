"""Per-tier error model and calibrated 90% intervals.

sigma^2 = sigma_stat^2                (fit statistics, from the data)
        + n_faces * sigma_face^2       (per-surface sensor bias, tier prior)
        + (scale_rel * L)^2            (metric-scale error, tier prior)
        + (drift_rel * path)^2         (accumulated pose drift; multi-room only)

90% interval = value +- 1.645 * k_tier * sigma, where k_tier is fitted on the
benchmark (conformal: smallest k so >= 90% of ground-truth residuals fall
inside). Priors come from published ARKit-LiDAR / monocular-depth error figures
and are replaced by benchmark calibration in config/calibration.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

Z90 = 1.645
CAL_PATH = Path(__file__).resolve().parent.parent / "config" / "calibration.json"

# Priors: sigma_face [m], scale_rel [-], drift_rel [-/m of walked path]
PRIORS = {
    "lidar": {"sigma_face": 0.006, "scale_rel": 0.003, "drift_rel": 0.002, "k": 1.0},
    "video": {"sigma_face": 0.015, "scale_rel": 0.012, "drift_rel": 0.006, "k": 1.0},
    "photo": {"sigma_face": 0.030, "scale_rel": 0.035, "drift_rel": 0.0, "k": 1.0},
}


def load_model(tier: str) -> dict:
    m = dict(PRIORS[tier])
    m["calibrated"] = False
    if CAL_PATH.exists():
        cal = json.loads(CAL_PATH.read_text()).get(tier)
        if cal:
            m.update(cal)
            m["calibrated"] = True
    return m


def measure(value: float, sigma: float, model: dict, digits: int = 3) -> dict:
    half = Z90 * model["k"] * sigma
    return {"value": round(float(value), digits), "sigma": round(float(sigma), 4),
            "ci90": [round(float(value - half), digits), round(float(value + half), digits)]}


def length_sigma(L: float, stat: float, model: dict, n_faces: int = 2,
                 path: float = 0.0) -> float:
    return float(np.sqrt(stat ** 2 + n_faces * model["sigma_face"] ** 2
                         + (model["scale_rel"] * L) ** 2 + (model["drift_rel"] * path) ** 2))


def area_sigma(A: float, perimeter_sigma_len: float, perimeter: float, model: dict) -> float:
    # dA ~ perimeter * d(edge offset) + 2*scale_rel*A (area scales with scale^2)
    return float(np.hypot(perimeter * perimeter_sigma_len / 2, 2 * model["scale_rel"] * A))
