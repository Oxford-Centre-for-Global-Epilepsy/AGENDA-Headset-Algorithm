#!/bin/bash

# Parameters
MODE=$1                    # "cv" for cross-validation, "final" for full training
EXPERIMENT_NAME=$2         # e.g. electrode_ablation or final_training
DATASET_NAME=$3            # e.g. EEG
POOLING_TYPE=${4:-attention} # default = attention
NUM_FOLDS=${5:-5}          # default = 5 folds (only used in CV)
SEED=${6:-42}              # random seed
OMIT_LIST=$7               # comma-separated: e.g. "Fp1,Fp2" or "" for none

# Usage help
if [ -z "$MODE" ] || [ -z "$EXPERIMENT_NAME" ] || [ -z "$DATASET_NAME" ]; then
  echo "Usage: ./run_experiment.sh <cv|final> <experiment_name> <dataset_name> [pooling_type] [num_folds] [seed] [omit_list]"
  exit 1
fi

if [ "$MODE" == "cv" ]; then
  echo "📊 Running Cross-Validation ($NUM_FOLDS folds)..."
  for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
    bash shell_scripts/experiments/electrode_ablation/submit_fold.sh "$EXPERIMENT_NAME" "$DATASET_NAME" "$FOLD" "$POOLING_TYPE" "$SEED" "$OMIT_LIST"
  done

elif [ "$MODE" == "final" ]; then
  echo "🏁 Running Final Full Training with Multiple Seeds..."
  bash shell_scripts/experiments/electrode_ablation/train_final_model.sh "$EXPERIMENT_NAME" "$DATASET_NAME" "$POOLING_TYPE" "$OMIT_LIST" configs/experiments/electrode_ablation/train_final_model.yaml

else
  echo "❌ Unknown mode: $MODE. Use 'cv' or 'final'."
  exit 1
fi