"""Generate a tiny synthetic dataset (4 understanding + 4 imgen samples) used by
the Phase 0 baseline check and the Phase 6 verification suite.

Run once after cloning:

    python internvl_chat/tests/fixtures/make_synth_fixtures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image


FIXTURES_DIR = Path(__file__).resolve().parent
IMG_DIR = FIXTURES_DIR / "synth_images"


def _make_solid_image(path: Path, color: tuple[int, int, int], size: int = 64) -> None:
    img = Image.new("RGB", (size, size), color=color)
    img.save(path)


def write_understanding_meta() -> Path:
    """Tiny meta that contains only understanding samples (old code path)."""
    img_paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        p = IMG_DIR / f"u_{i}.jpg"
        _make_solid_image(p, color)
        img_paths.append(p.name)

    ann_path = FIXTURES_DIR / "synth_understanding.jsonl"
    with ann_path.open("w") as f:
        for i, name in enumerate(img_paths):
            row = {
                "id": i,
                "image": name,
                "conversations": [
                    {"from": "human", "value": f"<image>\nWhat color is the image #{i}?"},
                    {"from": "gpt", "value": "It is a single-color test image."},
                ],
            }
            f.write(json.dumps(row) + "\n")

    meta_path = FIXTURES_DIR / "synth_meta_understanding.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "synth_understanding": {
                    "root": str(IMG_DIR),
                    "annotation": str(ann_path),
                    "data_augment": False,
                    "max_dynamic_patch": 1,
                    "repeat_time": 1,
                    "length": len(img_paths),
                }
            },
            f,
            indent=2,
        )
    return meta_path


def write_unified_meta() -> Path:
    """Meta with 4 understanding + 4 imgen samples used by the new full SFT path."""
    u_img_paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        p = IMG_DIR / f"u_{i}.jpg"
        _make_solid_image(p, color)
        u_img_paths.append(p.name)

    g_img_paths = []
    for i, color in enumerate([(128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]):
        p = IMG_DIR / f"g_{i}.jpg"
        _make_solid_image(p, color, size=128)
        g_img_paths.append(p.name)

    u_ann = FIXTURES_DIR / "synth_understanding.jsonl"
    if not u_ann.exists():
        write_understanding_meta()

    g_ann = FIXTURES_DIR / "synth_imgen.jsonl"
    captions = [
        "A red square on a white background.",
        "A green square on a white background.",
        "A blue square on a white background.",
        "A yellow square on a white background.",
    ]
    with g_ann.open("w") as f:
        for i, (name, cap) in enumerate(zip(g_img_paths, captions)):
            row = {
                "id": i,
                "task_type": "imgen",
                "caption": cap,
                "target_image": name,
            }
            f.write(json.dumps(row) + "\n")

    meta_path = FIXTURES_DIR / "synth_meta_unified.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "synth_understanding": {
                    "root": str(IMG_DIR),
                    "annotation": str(u_ann),
                    "data_augment": False,
                    "max_dynamic_patch": 1,
                    "repeat_time": 1,
                    "length": len(u_img_paths),
                    "task_type": "understanding",
                },
                "synth_imgen": {
                    "root": str(IMG_DIR),
                    "annotation": str(g_ann),
                    "data_augment": False,
                    "max_dynamic_patch": 1,
                    "repeat_time": 1,
                    "length": len(g_img_paths),
                    "task_type": "imgen",
                },
            },
            f,
            indent=2,
        )
    return meta_path


if __name__ == "__main__":
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    u = write_understanding_meta()
    a = write_unified_meta()
    print(f"wrote {u}")
    print(f"wrote {a}")
