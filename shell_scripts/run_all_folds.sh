#!/bin/bash

NUM_FOLDS=5
POOLING_TYPE="mean"      # mean, attention or transformer

for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
  bash shell_scripts/run_fold.sh $FOLD $POOLING_TYPE
done
