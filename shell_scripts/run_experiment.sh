#!/bin/bash

# Parameters
EXPERIMENT_NAME=$1           # e.g. electrode_ablation
DATASET_NAME=$2              # e.g. EEG
POOLING_TYPE=${3:-attention} # default = attention
NUM_FOLDS=${4:-5}            # default = 5 folds
SEED=${5:-42}                # random seed
OMIT_LIST=$6                 # comma-separated: e.g. "Fp1,Fp2" or "" for none

# Check required inputs
if [ -z "$EXPERIMENT_NAME" ] || [ -z "$DATASET_NAME" ]; then
  echo "Usage: ./run_experiment.sh <experiment_name> <dataset_name> [pooling_type] [num_folds] [seed] [omit_list]"
  exit 1
fi

# Submit one job per fold
for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
  bash shell_scripts/submit_fold.sh "$EXPERIMENT_NAME" "$DATASET_NAME" "$FOLD" "$POOLING_TYPE" "$SEED" "$OMIT_LIST"
done
