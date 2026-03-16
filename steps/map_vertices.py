#!/usr/bin/env python3
"""
Step 5 – Map DensePose CSE embeddings to SMPL mesh vertices.

For each high-res composite image this script:
  1. Loads the DensePose CSE output (``cse_output.pkl``).
  2. Selects the detected person whose bounding box best overlaps the
     region where the original clinical image was placed.
  3. Computes per-pixel closest SMPL vertex assignments.
  4. Records JSON outputs:

     * ``vertex_rgb.json``         – per-vertex median RGB colour
     * ``vertex_parts.json``       – which vertices appear in each image
     * ``vertex_parts_masked.json``– vertices filtered by the clinical
                                     binary mask (only when masks are
                                     available and ``--no-mask`` is not set)

Usage
-----
    python -m steps.map_vertices
    python -m steps.map_vertices --patient NIH-000021
    python -m steps.map_vertices --no-mask              # skip binary-mask filtering
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES

_POS_RE = re.compile(r"_pos(\d)\.jpg$")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_embedder():
    embed_map = np.load(str(cfg.SMPL_EMBED_PATH))
    return torch.tensor(embed_map).float().to(device)


def _parse_position(filename: str) -> int:
    m = _POS_RE.search(filename)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse position from: {filename}")


def _base_name_from_composite(filename: str) -> str:
    """``NIH-000021_007_DSC_3984_pos5.jpg`` → ``NIH-000021_007_DSC_3984``"""
    return _POS_RE.sub("", filename).rstrip(".")


def _iou(box_a, box_b):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_binary_mask(mask_dir: Path, base_name: str):
    """Try to load a binary mask matching *base_name* from *mask_dir*."""
    for ext in (".png", ".jpg", ".bmp"):
        mask_path = mask_dir / f"{base_name}{ext}"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
    return None


def _place_mask_on_canvas(mask: np.ndarray, offset: tuple,
                          is_portrait: bool) -> np.ndarray:
    """
    Resize a clinical binary mask and place it on the high-res canvas at the
    same position the original image would occupy during compositing.
    Returns a boolean canvas of shape (HIGHRES_CANVAS_SIZE, HIGHRES_CANVAS_SIZE).
    """
    if is_portrait:
        new_w, new_h = cfg.HIGHRES_PORTRAIT
    else:
        new_w, new_h = cfg.HIGHRES_LANDSCAPE

    mask_resized = cv2.resize(mask, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((cfg.HIGHRES_CANVAS_SIZE, cfg.HIGHRES_CANVAS_SIZE),
                      dtype=np.uint8)

    pad_h = max(0, cfg.HIGHRES_CANVAS_SIZE - new_h) \
            + offset[1] * cfg.HIGHRES_SCALE_FACTOR
    pad_w = max(0, cfg.HIGHRES_CANVAS_SIZE - new_w) \
            + offset[0] * cfg.HIGHRES_SCALE_FACTOR

    y_start = int(pad_h // 2)
    x_start = int(pad_w // 2)

    canvas[y_start:y_start + new_h, x_start:x_start + new_w] = mask_resized
    return canvas > 127


def map_patient(patient_id: str, *, use_mask: bool = True):
    """Compute vertex mappings for one patient."""
    pdir = cfg.patient_dir(patient_id)
    composite_dir = pdir / cfg.HIGHRES_COMPOSITE_DIR
    cse_pkl = pdir / cfg.DENSEPOSE_OUTPUT_DIR / "cse_output.pkl"
    mask_dir = pdir / cfg.BINARY_MASKS_DIR
    orig_dir = pdir / cfg.UNMARKED_IMAGES_DIR
    out_dir = cfg.ensure_dir(pdir / cfg.MAPPING_OUTPUT_DIR)

    if not cse_pkl.exists():
        print(f"  [skip] No CSE output: {cse_pkl}")
        return

    image_paths = sorted(glob.glob(str(composite_dir / "*.jpg")))
    if not image_paths:
        print(f"  [skip] No composite images in {composite_dir}")
        return

    cse_info = torch.load(str(cse_pkl), weights_only=False)
    cse_by_name = {}
    for entry in cse_info:
        bn = os.path.basename(entry["file_name"])
        cse_by_name[bn] = entry

    embedder = _load_embedder()

    vertices_rgb: dict = {}
    vertices_parts: dict = {}
    vertices_parts_masked: dict = {} if use_mask else None

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        base_name = _base_name_from_composite(fname)
        position = _parse_position(fname)
        offset = cfg.position_to_offset(position)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [warn] Cannot read {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_h, img_w = img_rgb.shape[:2]

        # Determine original image aspect ratio
        orig_path = orig_dir / f"{base_name}.jpg"
        if orig_path.exists():
            orig = cv2.imread(str(orig_path))
            oh, ow = orig.shape[:2]
        else:
            oh, ow = img_h, img_w

        is_portrait = oh > ow
        if is_portrait:
            new_w, new_h = cfg.HIGHRES_PORTRAIT
        else:
            new_w, new_h = cfg.HIGHRES_LANDSCAPE

        # Crop window (where the original image sits in the canvas)
        crop_h = min(img_h, new_h)
        crop_w = min(img_w, new_w)
        start_y = (img_h - crop_h + offset[1] * cfg.HIGHRES_SCALE_FACTOR) // 2
        start_x = (img_w - crop_w + offset[0] * cfg.HIGHRES_SCALE_FACTOR) // 2
        y1 = max(0, start_y)
        x1 = max(0, start_x)
        y2 = min(img_h, start_y + crop_h)
        x2 = min(img_w, start_x + crop_w)
        crop_area = max(0, y2 - y1) * max(0, x2 - x1)

        # CSE data for this image
        cse_entry = cse_by_name.get(fname)
        if cse_entry is None:
            print(f"  [warn] No CSE entry for {fname}")
            continue

        E = cse_entry["pred_densepose"].embedding
        S = cse_entry["pred_densepose"].coarse_segm
        bboxes_xyxy = cse_entry["pred_boxes_XYXY"]

        # Pick the bbox with best IoU to the crop window
        crop_box = (x1, y1, x2, y2)
        best_iou, best_idx = 0.0, 0
        for bi, bbox in enumerate(bboxes_xyxy):
            iou = _iou(bbox.tolist(), crop_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = bi

        bbox = bboxes_xyxy[best_idx]
        E_sel = E[best_idx:best_idx + 1]
        S_sel = S[best_idx:best_idx + 1]

        bx, by, bx2, by2 = bbox.int().tolist()
        bw, bh = bx2 - bx, by2 - by

        canvas = cfg.HIGHRES_CANVAS_SIZE

        # Build full canvas of embeddings & mask, then crop
        emb_resized = F.interpolate(
            E_sel, size=(bh, bw), mode="bilinear", align_corners=False
        )[0]
        emb_canvas = np.zeros((emb_resized.shape[0], canvas, canvas),
                              dtype=np.float32)
        emb_canvas[:, by:by + bh, bx:bx + bw] = \
            emb_resized.detach().cpu().numpy()

        segm_resized = F.interpolate(
            S_sel, size=(bh, bw), mode="bilinear", align_corners=False
        )[0].argmax(dim=0)
        mask_canvas = np.zeros((canvas, canvas), dtype=np.float32)
        mask_canvas[by:by + bh, bx:bx + bw] = \
            segm_resized.squeeze().detach().cpu().numpy()

        img_cropped = img_rgb[y1:y2, x1:x2]
        mask_cropped = mask_canvas[y1:y2, x1:x2]
        emb_cropped = emb_canvas[:, y1:y2, x1:x2]

        emb_tensor = torch.from_numpy(emb_cropped).unsqueeze(0).float().to(device)
        mask_tensor = torch.from_numpy(mask_cropped).unsqueeze(0).unsqueeze(0).float().to(device)

        h_crop, w_crop = mask_cropped.shape
        closest_vertices, return_mask = get_closest_vertices_mask_from_ES(
            emb_tensor, mask_tensor, h_crop, w_crop, embedder, device,
        )

        # Per-vertex RGB colours
        unique_verts = torch.unique(
            closest_vertices[return_mask]
        ).detach().cpu().numpy().tolist()

        cv_np = closest_vertices.detach().cpu().numpy()
        rm_np = return_mask.detach().cpu().numpy()
        image_u8 = (img_cropped * 255).astype(np.uint8)

        vert_rgb: dict = {}
        for v in unique_verts:
            ys, xs = np.where(cv_np == v)
            samples = []
            for k in range(len(ys)):
                yy, xx = ys[k], xs[k]
                if rm_np[yy, xx] == 0:
                    continue
                samples.append(image_u8[yy, xx])
            if samples:
                median = np.median(samples, axis=0).astype(int).tolist()
                vert_rgb[int(v)] = median

        vertices_rgb[base_name] = vert_rgb
        vertices_parts[base_name] = unique_verts

        # Mask-filtered vertex list
        if use_mask:
            bin_mask_raw = _load_binary_mask(mask_dir, base_name)
            if bin_mask_raw is not None:
                bin_canvas = _place_mask_on_canvas(
                    bin_mask_raw, offset, is_portrait)
                bin_cropped = bin_canvas[y1:y2, x1:x2]

                masked_verts = set()
                for v in unique_verts:
                    ys, xs = np.where(cv_np == v)
                    for k in range(len(ys)):
                        yy, xx = ys[k], xs[k]
                        if rm_np[yy, xx] and bin_cropped[yy, xx]:
                            masked_verts.add(int(v))
                            break
                vertices_parts_masked[base_name] = sorted(masked_verts)
            else:
                vertices_parts_masked[base_name] = unique_verts

        if use_mask and base_name in vertices_parts_masked:
            print(f"  Mapped {base_name}: {len(unique_verts)} vertices"
                  f" ({len(vertices_parts_masked[base_name])} masked)")
        else:
            print(f"  Mapped {base_name}: {len(unique_verts)} vertices")

    # Write outputs
    outputs = [
        ("vertex_rgb.json", vertices_rgb),
        ("vertex_parts.json", vertices_parts),
    ]
    if vertices_parts_masked is not None:
        outputs.append(("vertex_parts_masked.json", vertices_parts_masked))

    for name, data in outputs:
        out_path = out_dir / name
        with open(out_path, "w") as f:
            json.dump(dict(sorted(data.items())), f)
        print(f"  Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Map DensePose embeddings to SMPL vertices")
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--no-mask", action="store_true",
                        help="Skip binary-mask filtering (no vertex_parts_masked.json)")
    args = parser.parse_args()

    patients = [args.patient] if args.patient else cfg.get_patient_ids()
    if not patients:
        print("No patient directories found under", cfg.DATA_DIR)
        sys.exit(1)

    use_mask = not args.no_mask
    if not use_mask:
        print("[Step 5] Running without binary masks")

    for pid in patients:
        print(f"[Step 5] Vertex mapping: {pid}")
        map_patient(pid, use_mask=use_mask)


if __name__ == "__main__":
    main()
