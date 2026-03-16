"""
Central configuration for the DensePose CSE Mapping Pipeline.

All grid constants, paths, and shared parameters are defined here so every
step in the pipeline uses exactly the same conventions.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout (all relative to PIPELINE_ROOT)
# ---------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = PIPELINE_ROOT / "assets"
CONFIGS_DIR = PIPELINE_ROOT / "configs"
TOOLS_DIR = PIPELINE_ROOT / "tools"
DATA_DIR = PIPELINE_ROOT / "data"

SMPL_EMBED_PATH = ASSETS_DIR / "smpl_27554_embed.npy"
MESH_OBJ_PATH = ASSETS_DIR / "patient.obj"
MODEL_WEIGHTS_PATH = ASSETS_DIR / "model_final_1d3314.pkl"
DENSEPOSE_CONFIG_PATH = CONFIGS_DIR / "cse" / "densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml"
APPLY_NET_PATH = TOOLS_DIR / "apply_net.py"

# ---------------------------------------------------------------------------
# Patient data sub-folder names (inside each data/<patient_id>/)
# ---------------------------------------------------------------------------
UNMARKED_IMAGES_DIR = "UnmarkedImages"
BINARY_MASKS_DIR = "BinaryMasks"

# ---------------------------------------------------------------------------
# Intermediate / output sub-folder names (created under data/<patient_id>/)
# ---------------------------------------------------------------------------
OUTPAINTED_DIR = "outpainted"
QA_SELECTED_DIR = "qa_selected"
HIGHRES_COMPOSITE_DIR = "highres_composite"
DENSEPOSE_OUTPUT_DIR = "densepose_output"
MAPPING_OUTPUT_DIR = "mapping_output"
BSA_OUTPUT_DIR = "bsa_output"

# ---------------------------------------------------------------------------
# Outpaint offset grid  (ROW-FIRST order – matches visual 3×3 layout)
#
#   Position 1  2  3        (-500,-500) ( 0,-500) (500,-500)
#            4  5  6   →    (-500,   0) ( 0,   0) (500,   0)
#            7  8  9        (-500, 500) ( 0, 500) (500, 500)
#
# Each tuple is (center_x, center_y) where positive-x shifts the original
# image rightward inside the 1024×1024 canvas and positive-y shifts it
# downward.
# ---------------------------------------------------------------------------
OFFSET_GRID = [
    (-500, -500), (   0, -500), ( 500, -500),   # top row
    (-500,    0), (   0,    0), ( 500,    0),   # middle row
    (-500,  500), (   0,  500), ( 500,  500),   # bottom row
]

NUM_POSITIONS = len(OFFSET_GRID)

def position_to_offset(position: int):
    """Return the (center_x, center_y) offset for a 1-indexed grid position."""
    if not 1 <= position <= NUM_POSITIONS:
        raise ValueError(f"Position must be 1–{NUM_POSITIONS}, got {position}")
    return OFFSET_GRID[position - 1]


# ---------------------------------------------------------------------------
# Outpaint canvas / image sizing constants
# ---------------------------------------------------------------------------
OUTPAINT_CANVAS_SIZE = 1024          # diffusion model canvas (low-res)
HIGHRES_CANVAS_SIZE = 5120           # = 10240 // 2
HIGHRES_PORTRAIT = (1280, 1920)      # (width, height) = 2560//2, 3840//2
HIGHRES_LANDSCAPE = (1920, 1280)     # (width, height) = 3840//2, 2560//2
HIGHRES_SCALE_FACTOR = 5             # canvas ratio: 5120/1024

# ---------------------------------------------------------------------------
# Diffusion model defaults
# ---------------------------------------------------------------------------
DIFFUSION_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DIFFUSION_PROMPT = (
    "Fill the entire image. Keep only one person in the image. "
    "Given body part image of a person, complete the whole body by generating "
    "a realistic full-body continuation. Keep background simple. "
    "Maintain a smooth transition between the provided image and the generated body."
)
DIFFUSION_NEGATIVE_PROMPT = (
    "holding painting, anime, cartoonish, painting, drawing, art, abstract, "
    "incomplete, text, abstract textures, unrelated objects, t-shirt pattern, "
    "multiple, duo, crowd, grayscale, multihead"
)
DIFFUSION_GUIDANCE_SCALE = 7.5
DIFFUSION_NUM_INFERENCE_STEPS = 40
DIFFUSION_STRENGTH = 0.99
DIFFUSION_SEED = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patient_dir(patient_id: str) -> Path:
    """Absolute path to a patient's data directory."""
    return DATA_DIR / patient_id


def get_patient_ids() -> list:
    """List all patient IDs present under DATA_DIR."""
    if not DATA_DIR.exists():
        return []
    return sorted([
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("NIH-")
    ])


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
