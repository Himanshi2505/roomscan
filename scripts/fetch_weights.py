"""Fetch pretrained weights into ./weights (never committed).

MoGe-2 (Microsoft, MIT licence): monocular metric point map + intrinsics +
normals. Used by the photo and video tiers. ViT-S is the default (CPU-friendly);
pass --large for ViT-L.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {"small": "Ruicheng/moge-2-vits-normal", "large": "Ruicheng/moge-2-vitl-normal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true")
    a = ap.parse_args()
    out = Path(__file__).resolve().parent.parent / "weights"
    for key in (["small", "large"] if a.large else ["small"]):
        p = hf_hub_download(MODELS[key], "model.pt", local_dir=out / key)
        print(key, "->", p)


if __name__ == "__main__":
    main()
