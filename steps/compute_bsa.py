#!/usr/bin/env python3
"""
Step 6 – Compute Body Surface Area (BSA) percentages.

This script reads vertex mapping outputs and the SMPL mesh to compute:

  * **Photo-level BSA** – the percentage of total mesh surface area
    whose vertices are visible (or masked-visible) in each clinical image.
  * **Patient-level BSA** – the union of all per-image vertex sets,
    giving the cumulative body coverage for the patient.

Both ``vertex_parts.json`` (all DensePose-predicted vertices) and
``vertex_parts_masked.json`` (foreground-only vertices filtered by the
clinical binary mask) are processed.

Outputs are written to ``data/<patient>/bsa_output/``:
  * ``photo_bsa.json``   – per-image BSA values
  * ``patient_bsa.json`` – aggregated patient-level BSA

Usage
-----
    python -m steps.compute_bsa
    python -m steps.compute_bsa --patient NIH-000021
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def _load_mesh():
    """Load the SMPL mesh and return (vertices, faces)."""
    mesh = pv.read(str(cfg.MESH_OBJ_PATH))
    V = mesh.points                              # (N, 3)
    S = mesh.faces.reshape(-1, 4)[:, 1:]         # (M, 3)
    return V, S


def compute_surface_area_percentage(vertices: np.ndarray,
                                    faces: np.ndarray,
                                    selected_vertices) -> float:
    """
    Percentage of total mesh surface area covered by triangles that have
    at least one vertex in *selected_vertices*.
    """
    selected = set(int(v) for v in selected_vertices)
    total_area = 0.0
    affected_area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        total_area += area
        if any(int(vi) in selected for vi in face):
            affected_area += area
    return (affected_area / total_area) * 100.0 if total_area > 0 else 0.0


def compute_bsa_patient(patient_id: str):
    """Compute photo-level and patient-level BSA for one patient."""
    pdir = cfg.patient_dir(patient_id)
    mapping_dir = pdir / cfg.MAPPING_OUTPUT_DIR
    out_dir = cfg.ensure_dir(pdir / cfg.BSA_OUTPUT_DIR)

    parts_path = mapping_dir / "vertex_parts.json"
    masked_path = mapping_dir / "vertex_parts_masked.json"

    if not parts_path.exists():
        print(f"  [skip] No vertex_parts.json in {mapping_dir}")
        return

    V, S = _load_mesh()

    with open(parts_path) as f:
        parts_data = json.load(f)

    masked_data = {}
    if masked_path.exists():
        with open(masked_path) as f:
            masked_data = json.load(f)

    photo_bsa: dict = {}
    all_verts = set()
    all_verts_masked = set()

    for image_key, vert_list in sorted(parts_data.items()):
        bsa = compute_surface_area_percentage(V, S, vert_list)
        entry = {"bsa_percent": round(bsa, 4), "num_vertices": len(vert_list)}

        all_verts.update(int(v) for v in vert_list)

        if image_key in masked_data:
            masked_list = masked_data[image_key]
            bsa_masked = compute_surface_area_percentage(V, S, masked_list)
            entry["bsa_masked_percent"] = round(bsa_masked, 4)
            entry["num_vertices_masked"] = len(masked_list)
            all_verts_masked.update(int(v) for v in masked_list)

        photo_bsa[image_key] = entry
        print(f"  {image_key}: BSA={bsa:.2f}%"
              + (f"  masked={entry.get('bsa_masked_percent', 'N/A')}%"
                 if 'bsa_masked_percent' in entry else ""))

    patient_bsa_val = compute_surface_area_percentage(V, S, all_verts)
    patient_bsa_masked_val = (
        compute_surface_area_percentage(V, S, all_verts_masked)
        if all_verts_masked else None
    )

    patient_bsa = {
        "patient_id": patient_id,
        "total_images": len(parts_data),
        "patient_bsa_percent": round(patient_bsa_val, 4),
        "patient_num_vertices": len(all_verts),
    }
    if patient_bsa_masked_val is not None:
        patient_bsa["patient_bsa_masked_percent"] = round(
            patient_bsa_masked_val, 4)
        patient_bsa["patient_num_vertices_masked"] = len(all_verts_masked)

    print(f"  Patient {patient_id}: "
          f"BSA={patient_bsa_val:.2f}%  "
          f"({len(all_verts)} unique vertices across {len(parts_data)} images)")

    with open(out_dir / "photo_bsa.json", "w") as f:
        json.dump(photo_bsa, f, indent=2)
    with open(out_dir / "patient_bsa.json", "w") as f:
        json.dump(patient_bsa, f, indent=2)
    print(f"  Wrote BSA outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Compute BSA from vertex mappings")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    patients = [args.patient] if args.patient else cfg.get_patient_ids()
    if not patients:
        print("No patient directories found under", cfg.DATA_DIR)
        sys.exit(1)

    for pid in patients:
        print(f"[Step 6] BSA computation: {pid}")
        compute_bsa_patient(pid)


if __name__ == "__main__":
    main()
