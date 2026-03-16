#!/usr/bin/env bash
# Step 4 – Run DensePose CSE inference on high-res composite images.
#
# Usage:
#   bash steps/run_densepose.sh                     # all patients
#   bash steps/run_densepose.sh NIH-000021          # single patient
#
# Must be run from the MappingPipeline root directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

APPLY_NET="$PIPELINE_ROOT/tools/apply_net.py"
CONFIG="$PIPELINE_ROOT/configs/cse/densepose_rcnn_R_101_FPN_DL_soft_s1x.yaml"
MODEL="$PIPELINE_ROOT/assets/model_final_1d3314.pkl"
DATA_DIR="$PIPELINE_ROOT/data"

run_patient() {
    local patient_id="$1"
    local input_dir="$DATA_DIR/$patient_id/highres_composite"
    local output_dir="$DATA_DIR/$patient_id/densepose_output"
    local output_pkl="$output_dir/cse_output.pkl"

    if [ ! -d "$input_dir" ]; then
        echo "  [skip] No highres_composite directory for $patient_id"
        return
    fi

    mkdir -p "$output_dir"

    if [ -f "$output_pkl" ]; then
        echo "  [skip] CSE output already exists: $output_pkl"
        return
    fi

    echo "  Running DensePose CSE on $patient_id ..."
    cd "$PIPELINE_ROOT"
    python "$APPLY_NET" dump \
        "$CONFIG" \
        "$MODEL" \
        "$input_dir" \
        --output "$output_pkl"

    echo "  DensePose complete for $patient_id"
}

if [ $# -ge 1 ]; then
    echo "[Step 4] DensePose CSE: $1"
    run_patient "$1"
else
    echo "[Step 4] DensePose CSE: all patients"
    for patient_dir in "$DATA_DIR"/NIH-*/; do
        patient_id="$(basename "$patient_dir")"
        echo "[Step 4] DensePose CSE: $patient_id"
        run_patient "$patient_id"
    done
fi
