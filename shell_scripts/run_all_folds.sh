#!/bin/bash

# Default values
DEFAULT_NUM_FOLDS=5
DEFAULT_POOLING_TYPE="mean"

# Read input parameters or use defaults
NUM_FOLDS=${5:-$DEFAULT_NUM_FOLDS}
POOLING_TYPE=${2:-$DEFAULT_POOLING_TYPE}

# Create jobs
for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
  bash shell_scripts/run_fold.sh $FOLD $POOLING_TYPE
done
