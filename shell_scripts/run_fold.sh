#!/bin/bash

# Default values
DEFAULT_FOLD=0
DEFAULT_POOLING_TYPE="attention"

# Read input parameters or use defaults
FOLD=${1:-$DEFAULT_FOLD}
POOLING_TYPE=${2:-$DEFAULT_POOLING_TYPE}

echo "📂 Running training for fold $FOLD with pooling type '$POOLING_TYPE'..."

sbatch --export=ALL,FOLD=$FOLD,POOLING_TYPE=$POOLING_TYPE \
       --job-name=AGENDA_Training_fold_${FOLD}_${POOLING_TYPE}_pooling \
       --output=logs/${POOLING_TYPE}_pooling_fold_${FOLD}_%j.out \
       --error=logs/${POOLING_TYPE}_pooling_fold_${FOLD}_%j.err \
       shell_scripts/train.slurm
