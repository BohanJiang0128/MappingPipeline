#!/usr/bin/env python3
"""
Step 1 – Outpaint clinical images using a diffusion inpainting model.

For each input image the script generates 9 outpainted variants by placing
the original image at different offsets within a 1024×1024 canvas and
in-filling the surrounding area with Stable Diffusion XL Inpainting.

Usage
-----
    python -m steps.outpaint                         # all patients
    python -m steps.outpaint --patient NIH-000021    # single patient
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Allow running from MappingPipeline root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def resize_and_pad(img: Image.Image, center_offset: tuple, scale: float = 1.0):
    """
    Resize *img* into a small target size, paste it onto a 1024×1024 noisy
    canvas at the given *center_offset*, and return (padded_image, mask).
    """
    original_width, original_height = img.size

    if original_height < original_width:
        target_h = int(256 * scale)
        target_w = int(384 * scale)
    else:
        target_h = int(384 * scale)
        target_w = int(256 * scale)

    resized = img.resize((target_w, target_h))

    pad_h = max(0, cfg.OUTPAINT_CANVAS_SIZE - target_h) + center_offset[1]
    pad_w = max(0, cfg.OUTPAINT_CANVAS_SIZE - target_w) + center_offset[0]

    noise = np.random.normal(0, 255, (cfg.OUTPAINT_CANVAS_SIZE,
                                       cfg.OUTPAINT_CANVAS_SIZE, 3))
    padded = Image.fromarray(np.uint8(noise))
    padded.paste(resized, (pad_w // 2, pad_h // 2))

    mask_inner = Image.new("RGB", (target_w, target_h), 0)
    mask = Image.new("RGB",
                     (cfg.OUTPAINT_CANVAS_SIZE, cfg.OUTPAINT_CANVAS_SIZE),
                     (255, 255, 255))
    mask.paste(mask_inner, (pad_w // 2, pad_h // 2))

    return padded, mask


def outpaint_patient(patient_id: str):
    """Generate 9 outpainted variants for every image of *patient_id*."""
    from diffusers import AutoPipelineForInpainting

    pdir = cfg.patient_dir(patient_id)
    src_dir = pdir / cfg.UNMARKED_IMAGES_DIR
    out_dir = cfg.ensure_dir(pdir / cfg.OUTPAINTED_DIR)

    image_paths = sorted(glob.glob(str(src_dir / "*.jpg")))
    if not image_paths:
        print(f"  [skip] No .jpg images found in {src_dir}")
        return

    pipe = AutoPipelineForInpainting.from_pretrained(
        cfg.DIFFUSION_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(cfg.DIFFUSION_SEED)

    for pos_idx, offset in enumerate(cfg.OFFSET_GRID):
        position = pos_idx + 1  # 1-indexed
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            save_name = f"{base_name}_pos{position}.jpg"
            save_path = out_dir / save_name
            if save_path.exists():
                continue

            image_orig = Image.open(img_path).convert("RGB")
            image_padded, mask = resize_and_pad(image_orig, center_offset=offset)

            result = pipe(
                prompt=cfg.DIFFUSION_PROMPT,
                negative_prompt=cfg.DIFFUSION_NEGATIVE_PROMPT,
                image=image_padded,
                mask_image=mask,
                guidance_scale=cfg.DIFFUSION_GUIDANCE_SCALE,
                num_inference_steps=cfg.DIFFUSION_NUM_INFERENCE_STEPS,
                strength=cfg.DIFFUSION_STRENGTH,
                generator=generator,
            ).images[0]

            result.save(str(save_path))
            print(f"  Saved {save_path.name}")

    print(f"  Outpainting complete for {patient_id}")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Outpaint clinical images")
    parser.add_argument("--patient", type=str, default=None,
                        help="Process a single patient ID (default: all)")
    args = parser.parse_args()

    patients = [args.patient] if args.patient else cfg.get_patient_ids()
    if not patients:
        print("No patient directories found under", cfg.DATA_DIR)
        sys.exit(1)

    for pid in patients:
        print(f"[Step 1] Outpainting: {pid}")
        outpaint_patient(pid)


if __name__ == "__main__":
    main()
