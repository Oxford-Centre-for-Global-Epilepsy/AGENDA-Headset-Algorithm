#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Usage: ./run_fold.sh <fold_number>"
  exit 1
fi

FOLD=$1
sbatch --export=ALL,FOLD=$FOLD shell_scripts/train.slurm