#!/usr/bin/env bash
# =============================================================================
# DensePose CSE Mapping Pipeline – end-to-end batch runner
#
# Usage:
#   bash run_pipeline.sh                           # all patients, all steps
#   bash run_pipeline.sh --patient NIH-000021      # single patient
#   bash run_pipeline.sh --from 3                  # resume from step 3
#   bash run_pipeline.sh --only 2                  # run only step 2 (QA)
#   bash run_pipeline.sh --patient NIH-000021 --from 3 --to 5
#   bash run_pipeline.sh --no-mask                     # skip binary-mask filtering in Step 5
#
# Steps:
#   1  Outpaint clinical images (GPU, diffusion model)
#   2  Manual QA selection (interactive – requires display)
#   3  High-resolution compositing
#   4  DensePose CSE inference (GPU)
#   5  Vertex mapping (GPU)
#   6  BSA computation
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PATIENT_ARG=""
FROM_STEP=1
TO_STEP=6
ONLY_STEP=""
NO_MASK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --patient)
            PATIENT_ARG="--patient $2"; shift 2 ;;
        --from)
            FROM_STEP="$2"; shift 2 ;;
        --to)
            TO_STEP="$2"; shift 2 ;;
        --only)
            ONLY_STEP="$2"; FROM_STEP="$2"; TO_STEP="$2"; shift 2 ;;
        --no-mask)
            NO_MASK="--no-mask"; shift ;;
        -h|--help)
            head -n 17 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

should_run() {
    local step=$1
    [[ $step -ge $FROM_STEP && $step -le $TO_STEP ]]
}

echo "========================================"
echo "  DensePose CSE Mapping Pipeline"
echo "  Steps: ${FROM_STEP}–${TO_STEP}"
echo "  Patient: ${PATIENT_ARG:-all}"
echo "  Masks:   ${NO_MASK:-enabled}"
echo "========================================"
echo ""

# Step 1: Outpaint
if should_run 1; then
    echo "──── Step 1: Outpainting ────"
    python -m steps.outpaint $PATIENT_ARG
    echo ""
fi

# Step 2: QA selection (interactive)
if should_run 2; then
    echo "──── Step 2: QA Selection (open browser at localhost:8505) ────"
    python -m steps.qa_select $PATIENT_ARG
    echo ""
fi

# Step 3: High-res composite
if should_run 3; then
    echo "──── Step 3: High-res Compositing ────"
    python -m steps.composite_highres $PATIENT_ARG
    echo ""
fi

# Step 4: DensePose CSE inference
if should_run 4; then
    echo "──── Step 4: DensePose CSE ────"
    if [ -n "$PATIENT_ARG" ]; then
        bash steps/run_densepose.sh ${PATIENT_ARG#--patient }
    else
        bash steps/run_densepose.sh
    fi
    echo ""
fi

# Step 5: Vertex mapping
if should_run 5; then
    echo "──── Step 5: Vertex Mapping ────"
    python -m steps.map_vertices $PATIENT_ARG $NO_MASK
    echo ""
fi

# Step 6: BSA computation
if should_run 6; then
    echo "──── Step 6: BSA Computation ────"
    python -m steps.compute_bsa $PATIENT_ARG
    echo ""
fi

echo "========================================"
echo "  Pipeline complete."
echo "========================================"
