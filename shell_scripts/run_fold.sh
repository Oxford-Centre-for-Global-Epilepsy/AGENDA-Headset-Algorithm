#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Usage: ./run_fold.sh <fold_number>"
  exit 1
fi

FOLD=$1
sbatch --export=ALL,FOLD=$FOLD \
       --job-name=AGENDA_Algorithm_Training_fold_$FOLD \
       --output=logs/fold_${FOLD}_%j.out \
       --error=logs/fold_${FOLD}_%j.err \
       shell_scripts/train.slurm