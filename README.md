# DensePose CSE Mapping Pipeline

An end-to-end pipeline for processing clinical dermatology images through
diffusion-based outpainting, DensePose CSE body mapping, and Body Surface
Area (BSA) computation.

---

## Overview

The pipeline takes clinical photographs of patients and:

1. **Outpaints** each image with a diffusion model to recover full-body
   context from partial-body clinical photos.
2. Provides a **manual QA** interface to select the best outpainted variant.
3. **Composites** the original high-resolution image back into the chosen
   outpainted canvas.
4. Runs **DensePose CSE** to obtain dense body-surface embeddings.
5. **Maps** pixel-level embeddings to SMPL mesh vertices, producing
   per-image vertex lists and RGB colour information.
6. **Computes BSA** (Body Surface Area) percentages at both the photo
   and patient level.

---

## Directory structure

```
MappingPipeline/
├── README.md                  ← you are here
├── requirements.txt           ← Python dependencies
├── run_pipeline.sh            ← end-to-end batch runner
├── config.py                  ← central configuration
│
├── steps/                     ← pipeline step scripts
│   ├── outpaint.py            ← Step 1: diffusion outpainting
│   ├── qa_select.py           ← Step 2: interactive QA selection
│   ├── composite_highres.py   ← Step 3: high-res compositing
│   ├── run_densepose.sh       ← Step 4: DensePose CSE inference
│   ├── map_vertices.py        ← Step 5: vertex mapping → JSON
│   └── compute_bsa.py         ← Step 6: BSA computation
│
├── densepose/                 ← DensePose library (with custom modification)
├── tools/
│   └── apply_net.py           ← DensePose inference entry point
├── configs/                   ← DensePose model configs (YAML)
│
├── assets/
│   ├── model_final_1d3314.pkl ← DensePose CSE model weights
│   ├── smpl_27554_embed.npy   ← SMPL vertex embeddings
│   └── patient.obj            ← SMPL mesh for BSA calculation
│
└── data/                      ← patient input data (see data/README.md)
    └── README.md
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.9+ | 3.10 recommended |
| CUDA GPU | Required for Steps 1, 4, 5 |
| PyTorch + CUDA | Install for your driver version: https://pytorch.org |
| Detectron2 | https://detectron2.readthedocs.io/en/latest/tutorials/install.html |
| Tk backend | `sudo apt install python3-tk` (for QA GUI in Step 2) |

### Install Python dependencies

```bash
pip install -r requirements.txt
```

PyTorch, CUDA, and Detectron2 must be installed separately (see links above).

---

## Preparing input data

Place patient data under `data/` following this structure:

```
data/
  NIH-000021/
    UnmarkedImages/
      NIH-000021_001_DSC_1234.jpg
      NIH-000021_002_DSC_1235.jpg
    BinaryMasks/           ← optional, for mask-filtered outputs
      NIH-000021_001_DSC_1234.png
      NIH-000021_002_DSC_1235.png
```

See [`data/README.md`](data/README.md) for full details on naming conventions
and mask format.

---

## Running the pipeline

### Full pipeline (all patients, all steps)

```bash
cd MappingPipeline
bash run_pipeline.sh
```

### Single patient

```bash
bash run_pipeline.sh --patient NIH-000021
```

### Run specific steps

```bash
bash run_pipeline.sh --from 3                  # Steps 3–6
bash run_pipeline.sh --only 2                  # Step 2 only
bash run_pipeline.sh --from 3 --to 5           # Steps 3–5
bash run_pipeline.sh --patient NIH-000021 --from 4
```

### Run steps individually

Each step can also be run as a standalone Python module:

```bash
cd MappingPipeline

python -m steps.outpaint --patient NIH-000021          # Step 1
python -m steps.qa_select --patient NIH-000021         # Step 2
python -m steps.composite_highres --patient NIH-000021 # Step 3
bash steps/run_densepose.sh NIH-000021                 # Step 4
python -m steps.map_vertices --patient NIH-000021      # Step 5
python -m steps.compute_bsa --patient NIH-000021       # Step 6
```

---

## Pipeline steps in detail

### Step 1 – Outpaint (`steps/outpaint.py`)

Uses Stable Diffusion XL Inpainting to generate **9 outpainted variants**
per clinical image.  Each variant places the original image at a different
position on a 1024×1024 canvas (3×3 grid of offsets), then in-fills the
surrounding area.

**Input:** `data/<patient>/UnmarkedImages/*.jpg`
**Output:** `data/<patient>/outpainted/<base>_pos{1..9}.jpg`

### Step 2 – QA Selection (`steps/qa_select.py`)

Opens a **fullscreen interactive window** showing:
- The original clinical image (left panel)
- The 9 outpainted candidates in a labelled 3×3 grid (right panel)

Press **1–9** to select the best variant, **0** to skip, **Q** to quit.
Selections are saved incrementally so you can resume later.

**Input:** Outpainted images + original images
**Output:** `data/<patient>/qa_selected/` + `selection_log.json`

### Step 3 – High-res Composite (`steps/composite_highres.py`)

Upsamples the selected outpainted image to 5120×5120 and pastes the
original clinical image at full resolution back into the correct position.

**Output:** `data/<patient>/highres_composite/<base>_pos{N}.jpg`

### Step 4 – DensePose CSE (`steps/run_densepose.sh`)

Runs the DensePose CSE model on all high-res composites to produce
per-image dense embedding predictions.

**Output:** `data/<patient>/densepose_output/cse_output.pkl`

### Step 5 – Vertex Mapping (`steps/map_vertices.py`)

Maps DensePose embeddings to SMPL mesh vertices and produces three
JSON outputs:

| File | Description |
|------|------------|
| `vertex_rgb.json` | Per-vertex RGB colour samples for each image |
| `vertex_parts.json` | List of vertex indices visible in each image |
| `vertex_parts_masked.json` | Vertex indices filtered by the clinical binary mask |

**Output:** `data/<patient>/mapping_output/`

### Step 6 – BSA Computation (`steps/compute_bsa.py`)

Computes Body Surface Area percentages from the vertex mapping outputs:

| File | Description |
|------|------------|
| `photo_bsa.json` | Per-image BSA percentage (with and without mask) |
| `patient_bsa.json` | Patient-level aggregate BSA (union of all images) |

**Output:** `data/<patient>/bsa_output/`

---

## Outputs summary

After a complete run for patient `NIH-000021`:

```
data/NIH-000021/
├── UnmarkedImages/          ← input (unchanged)
├── BinaryMasks/             ← input (unchanged)
├── outpainted/              ← 9 variants per image
├── qa_selected/             ← chosen variant + selection_log.json
├── highres_composite/       ← high-res composites
├── densepose_output/        ← cse_output.pkl
├── mapping_output/
│   ├── vertex_rgb.json
│   ├── vertex_parts.json
│   └── vertex_parts_masked.json
└── bsa_output/
    ├── photo_bsa.json
    └── patient_bsa.json
```

---

## Configuration

All pipeline constants (offset grid, canvas sizes, diffusion parameters,
folder names, etc.) are defined in [`config.py`](config.py).  Edit this file
to adjust behaviour without modifying individual step scripts.

---

## Custom DensePose modification

The `densepose/modeling/cse/utils.py` file contains a modified version of the
`get_closest_vertices_mask_from_ES` function.  The key changes from the
original Facebook implementation:

- **Removed internal `F.interpolate()`** – the caller pre-sizes the embedding
  and segmentation tensors, so interpolation inside the function would be
  redundant.
- **Added empty-embedding guard** – returns early if no foreground pixels are
  detected, preventing index errors.
- **Changed mask derivation** – uses `coarse_segm[0] > 0` instead of
  `argmax(0) > 0`, matching the pre-processed input format.

These changes are intentional and must be preserved if updating the DensePose
library.

---

## Offset grid convention

The 9 outpaint positions use **row-first** ordering:

```
Position:   1  2  3        Offset (x, y):  (-500,-500) (0,-500) (500,-500)
            4  5  6                        (-500,   0) (0,   0) (500,   0)
            7  8  9                        (-500, 500) (0, 500) (500, 500)
```

Positive x shifts the original image rightward; positive y shifts it downward.
This ordering matches the visual 3×3 grid in the QA interface and is
consistent throughout all pipeline steps.
