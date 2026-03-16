#!/usr/bin/env python3
"""
Step 2 – Manual QA selection of the best outpainted variant.

Presents a fullscreen window showing:
  * the **original clinical image** on the left,
  * the **9 outpainted candidates** in a 3×3 grid on the right,
    each labelled 1–9 matching the spatial offset position.

Keyboard controls
-----------------
  1–9   Select that candidate as the best outpaint for this image.
  0     Skip this image (no selection).
  Q     Quit early (already-saved selections are preserved).

The selected image is copied into ``data/<patient>/qa_selected/`` and a
``selection_log.json`` is written alongside it.

Usage
-----
    python -m steps.qa_select                         # all patients
    python -m steps.qa_select --patient NIH-000021
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def _find_candidates(outpaint_dir: Path, base_name: str):
    """Return a list of 9 candidate paths (index 0 → position 1, etc.)."""
    candidates = [None] * cfg.NUM_POSITIONS
    for pos in range(1, cfg.NUM_POSITIONS + 1):
        fname = f"{base_name}_pos{pos}.jpg"
        p = outpaint_dir / fname
        if p.exists():
            candidates[pos - 1] = str(p)
    return candidates


def _show_qa_window(original_path: str, candidates: list, base_name: str):
    """
    Display a fullscreen window with the original image and 9 candidates.
    Returns the user's choice (1–9), 0 for skip, or None if quit.
    """
    fig = plt.figure(f"QA – {base_name}")
    mng = plt.get_current_fig_manager()
    try:
        mng.window.state("zoomed")          # Windows / some Linux
    except Exception:
        try:
            mng.resize(*mng.window.maxsize())
        except Exception:
            try:
                mng.full_screen_toggle()    # matplotlib ≥ 3.x
            except Exception:
                pass

    # Layout: 3 rows × 4 columns.  Column 0 spans all rows for original.
    gs = fig.add_gridspec(3, 4, wspace=0.08, hspace=0.15)

    # -- Original image (left column, spans all 3 rows) --
    ax_orig = fig.add_subplot(gs[:, 0])
    orig_img = mpimg.imread(original_path)
    ax_orig.imshow(orig_img)
    ax_orig.set_title("Original", fontsize=13, fontweight="bold")
    ax_orig.axis("off")

    # -- 3×3 grid of candidates (columns 1–3) --
    ax_grid = []
    for row in range(3):
        for col in range(3):
            pos = row * 3 + col + 1   # 1-indexed position
            ax = fig.add_subplot(gs[row, col + 1])
            idx = pos - 1
            if candidates[idx] is not None:
                img = mpimg.imread(candidates[idx])
                ax.imshow(img)
            ax.set_title(str(pos), fontsize=12, fontweight="bold",
                         color="white",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="#3366cc", alpha=0.85))
            ax.axis("off")
            ax_grid.append(ax)

    fig.suptitle("Press 1–9 to select  |  0 to skip  |  Q to quit",
                 fontsize=14, y=0.98)

    user_input = {"value": None}

    def on_key(event):
        key = event.key.lower()
        if key in [str(i) for i in range(10)]:
            user_input["value"] = int(key)
            plt.close(fig)
        elif key == "q":
            user_input["value"] = None
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return user_input["value"]


def qa_select_patient(patient_id: str):
    """Run QA selection for a single patient."""
    pdir = cfg.patient_dir(patient_id)
    src_dir = pdir / cfg.UNMARKED_IMAGES_DIR
    outpaint_dir = pdir / cfg.OUTPAINTED_DIR
    sel_dir = cfg.ensure_dir(pdir / cfg.QA_SELECTED_DIR)

    log_path = sel_dir / "selection_log.json"
    if log_path.exists():
        with open(log_path) as f:
            selection_log = json.load(f)
    else:
        selection_log = {}

    originals = sorted(glob.glob(str(src_dir / "*.jpg")))
    if not originals:
        print(f"  [skip] No images in {src_dir}")
        return

    quit_early = False
    for orig_path in originals:
        base_name = os.path.splitext(os.path.basename(orig_path))[0]

        if base_name in selection_log:
            continue

        candidates = _find_candidates(outpaint_dir, base_name)
        if all(c is None for c in candidates):
            print(f"  [skip] No outpainted variants for {base_name}")
            continue

        choice = _show_qa_window(orig_path, candidates, base_name)

        if choice is None:
            quit_early = True
            break
        elif choice == 0:
            selection_log[base_name] = {"position": 0, "status": "skipped"}
        elif 1 <= choice <= 9:
            src = candidates[choice - 1]
            if src is None:
                print(f"  [warn] Position {choice} missing for {base_name}")
                continue
            dest_name = f"{base_name}_pos{choice}.jpg"
            dest_path = sel_dir / dest_name
            shutil.copy2(src, dest_path)
            selection_log[base_name] = {
                "position": choice,
                "file": dest_name,
                "status": "selected",
            }
            print(f"  Selected pos {choice} for {base_name}")

        with open(log_path, "w") as f:
            json.dump(selection_log, f, indent=2)

    if quit_early:
        print("  QA interrupted (progress saved).")


def main():
    parser = argparse.ArgumentParser(description="Step 2: QA selection")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()

    patients = [args.patient] if args.patient else cfg.get_patient_ids()
    if not patients:
        print("No patient directories found under", cfg.DATA_DIR)
        sys.exit(1)

    for pid in patients:
        print(f"[Step 2] QA selection: {pid}")
        qa_select_patient(pid)


if __name__ == "__main__":
    main()
