#!/usr/bin/env python3
"""
Step 3 – Composite high-resolution original onto upsampled outpainted image.

For each QA-selected outpainted image this script:
  1. Up-samples the 1024×1024 outpainted image to 5120×5120.
  2. Resizes the original clinical photo to a standard high-resolution size.
  3. Pastes the original back into the canvas at the correct offset so that
     the final image has the full-body diffusion context *plus* the original
     clinical detail at high resolution.

Usage
-----
    python -m steps.composite_highres
    python -m steps.composite_highres --patient NIH-000021
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

_POS_RE = re.compile(r"_pos(\d)\.jpg$")


def _parse_position(filename: str) -> int:
    """Extract the position number from a filename like ``*_pos5.jpg``."""
    m = _POS_RE.search(filename)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse position from filename: {filename}")


def composite_patient(patient_id: str):
    """Create high-res composites for every QA-selected image."""
    pdir = cfg.patient_dir(patient_id)
    orig_dir = pdir / cfg.UNMARKED_IMAGES_DIR
    sel_dir = pdir / cfg.QA_SELECTED_DIR
    out_dir = cfg.ensure_dir(pdir / cfg.HIGHRES_COMPOSITE_DIR)

    log_path = sel_dir / "selection_log.json"
    if not log_path.exists():
        print(f"  [skip] No selection_log.json in {sel_dir}")
        return

    with open(log_path) as f:
        selection_log = json.load(f)

    for base_name, info in selection_log.items():
        if info.get("status") != "selected":
            continue

        sel_file = info["file"]
        position = info["position"]
        offset = cfg.position_to_offset(position)

        sel_path = sel_dir / sel_file
        if not sel_path.exists():
            print(f"  [warn] Selected file missing: {sel_path}")
            continue

        orig_path = orig_dir / f"{base_name}.jpg"
        if not orig_path.exists():
            print(f"  [warn] Original missing: {orig_path}")
            continue

        dest_path = out_dir / sel_file
        if dest_path.exists():
            continue

        outpainted = cv2.imread(str(sel_path))
        outpainted = cv2.resize(outpainted,
                                (cfg.HIGHRES_CANVAS_SIZE,
                                 cfg.HIGHRES_CANVAS_SIZE))

        original = cv2.imread(str(orig_path))
        oh, ow = original.shape[:2]
        if oh > ow:
            new_w, new_h = cfg.HIGHRES_PORTRAIT
        else:
            new_w, new_h = cfg.HIGHRES_LANDSCAPE

        original = cv2.resize(original, (new_w, new_h))

        pad_h = max(0, cfg.HIGHRES_CANVAS_SIZE - new_h) \
                + offset[1] * cfg.HIGHRES_SCALE_FACTOR
        pad_w = max(0, cfg.HIGHRES_CANVAS_SIZE - new_w) \
                + offset[0] * cfg.HIGHRES_SCALE_FACTOR

        y_start = int(pad_h // 2)
        x_start = int(pad_w // 2)

        outpainted[y_start:y_start + new_h,
                   x_start:x_start + new_w] = original

        cv2.imwrite(str(dest_path), outpainted)
        print(f"  Composited {dest_path.name}")

    print(f"  Compositing complete for {patient_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: High-res composite")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    patients = [args.patient] if args.patient else cfg.get_patient_ids()
    if not patients:
        print("No patient directories found under", cfg.DATA_DIR)
        sys.exit(1)

    for pid in patients:
        print(f"[Step 3] High-res composite: {pid}")
        composite_patient(pid)


if __name__ == "__main__":
    main()
