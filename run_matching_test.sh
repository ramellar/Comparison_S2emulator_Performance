#!/bin/bash

set -euo pipefail

# The matching inputs and outputs live in the shared project area.
PROJECT_DIR=/home/llr/cms/pivato/Comparison_S2emulator_Performance

source /home/llr/cms/pivato/bin/miniconda3/etc/profile.d/conda.sh
conda activate s2_emulator

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "Running: python3 -m scripts.matching_test $*"
python3 -m scripts.matching_test "$@"
