#!/bin/bash
# Helper script to run the required code for:
#   - Q2.7  (LLM training on Grimm corpus)
#   - Q2.8b (generation experiments / accumulated knowledge)
#   - Q3.4d (graph convolution vs graph attention training)
#
# This script is designed to be called from anywhere; it will cd into the
# assignment2 directory where it lives and then run the appropriate commands.
#
# Example SLURM usage (from your home directory):
#   cd $HOME/uvadlc_practicals_2025
#   sbatch your_slurm_job_that_calls:  bash assignment2/run_q2_q3.sh

set -u  # fail on undefined variables; do NOT use -e so later parts still run

# Always work relative to this script's directory (assignment2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

echo "====================================================================="
echo "Q2.7: Train LLM on 'Fairy Tales by Brothers Grimm' for 5 epochs"
echo "====================================================================="
(
  cd part2 || exit 0

  # Command recommended in the assignment README:
  # uses cfg.py defaults (Grimm corpus), enables FlashAttention and compile.
  python train.py \
    --use_flash_attn \
    --compile \
    --num_epochs 5 \
    --num_workers 8
)
if [ $? -ne 0 ]; then
  echo "Q2.7 training FAILED (continuing to Q2.8b)..."
fi

echo
echo "====================================================================="
echo "Q2.8b: Evaluate generations (accumulated knowledge)"
echo "====================================================================="
(
  cd part2 || exit 0

  # Uses default model_weights_folder and prompts/configurations from
  # evaluate_generation.py. Adjust arguments if you want a different run.
  python evaluate_generation.py \
    --output_file q2_8b_generation_results.json
)
if [ $? -ne 0 ]; then
  echo "Q2.8b evaluation FAILED (continuing to Q3.4d)..."
fi

echo
echo "====================================================================="
echo "Q3.4d: Train Graph CNN and Graph Attention (TensorBoard plots)"
echo "====================================================================="
(
  cd part3 || exit 0

  # Uses its own config.py and logs train/val accuracies to TensorBoard.
  python train.py
)
if [ $? -ne 0 ]; then
  echo "Q3.4d training FAILED."
fi

echo
echo "Done. Check TensorBoard logs and JSON outputs for results."


