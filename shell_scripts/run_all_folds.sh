#!/bin/bash

NUM_FOLDS=5

for FOLD in $(seq 0 $((NUM_FOLDS - 1))); do
  bash shell_scripts/run_fold.sh $FOLD
done
