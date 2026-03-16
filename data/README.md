# Data Directory

This folder holds patient data for the DensePose CSE Mapping Pipeline.

## Required input structure

Place each patient in its own sub-folder named by patient ID.  
Each patient folder **must** contain at least an `UnmarkedImages/` directory.
A `BinaryMasks/` directory is optional but required for mask-filtered outputs.

```
data/
├── NIH-000021/
│   ├── UnmarkedImages/
│   │   ├── NIH-000021_001_DSC_1234.jpg
│   │   ├── NIH-000021_002_DSC_1235.jpg
│   │   └── ...
│   └── BinaryMasks/
│       ├── NIH-000021_001_DSC_1234.png
│       ├── NIH-000021_002_DSC_1235.png
│       └── ...
├── NIH-000025/
│   ├── UnmarkedImages/
│   │   └── ...
│   └── BinaryMasks/
│       └── ...
└── ...
```

## Naming conventions

| Item | Convention |
|------|-----------|
| Patient folder | `NIH-XXXXXX` (must start with `NIH-`) |
| Unmarked images | Any `.jpg` files inside `UnmarkedImages/` |
| Binary masks | Must share the same base name as the corresponding image, with a `.png` extension, inside `BinaryMasks/` |

### Example

If an unmarked image is named:
```
NIH-000021_007_DSC_3984.jpg
```
then its binary mask should be:
```
NIH-000021_007_DSC_3984.png
```

## Binary mask format

- 8-bit single-channel (grayscale) PNG.
- Foreground (region of interest) pixels should be white (≥128).
- Background pixels should be black (<128).
- The mask should be the same resolution as the corresponding unmarked image.

## Intermediate outputs

The pipeline creates additional sub-folders alongside `UnmarkedImages/` and
`BinaryMasks/`.  These are generated automatically and can be safely deleted
to re-run a step:

| Folder | Created by | Contents |
|--------|-----------|----------|
| `outpainted/` | Step 1 | 9 outpainted variants per image (`*_pos1.jpg` … `*_pos9.jpg`) |
| `qa_selected/` | Step 2 | The chosen variant for each image + `selection_log.json` |
| `highres_composite/` | Step 3 | High-resolution composites (5120×5120) |
| `densepose_output/` | Step 4 | `cse_output.pkl` from DensePose inference |
| `mapping_output/` | Step 5 | `vertex_rgb.json`, `vertex_parts.json`, `vertex_parts_masked.json` |
| `bsa_output/` | Step 6 | `photo_bsa.json`, `patient_bsa.json` |
