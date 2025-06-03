#!/bin/bash

EXPERIMENT_NAME=$1
DATASET_NAME=$2
FOLD=$3
POOLING_TYPE=$4
SEED=$5
OMIT_LIST=$6

# Make safe job name
JOB_NAME="AGENDA_${EXPERIMENT_NAME}_fold_${FOLD}_${POOLING_TYPE}"
if [ ! -z "$OMIT_LIST" ]; then
  OMIT_TAG=$(echo "$OMIT_LIST" | tr ',' '_')
  JOB_NAME="${JOB_NAME}_omit_${OMIT_TAG}"
fi

# Convert OMIT_LIST into a valid JSON/YAML array string for OmegaConf
if [ ! -z "$OMIT_LIST" ]; then
  IFS=',' read -ra CHANNELS <<< "$OMIT_LIST"
  FORMATTED_OMIT_LIST=$(printf '"%s",' "${CHANNELS[@]}")
  FORMATTED_OMIT_LIST="[${FORMATTED_OMIT_LIST%,}]"  # remove trailing comma and wrap in []
else
  FORMATTED_OMIT_LIST="[]"
fi

# Escape the brackets and quotes to pass safely through sbatch
FORMATTED_OMIT_LIST_ESCAPED="'${FORMATTED_OMIT_LIST}'"

# Create log directory
mkdir -p logs

sbatch --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",DATASET_NAME="$DATASET_NAME",FOLD="$FOLD",POOLING_TYPE="$POOLING_TYPE",SEED="$SEED",FORMATTED_OMIT_LIST=$FORMATTED_OMIT_LIST_ESCAPED \
  --job-name=$JOB_NAME \
  --output=logs/${JOB_NAME}_%j.out \
  --error=logs/${JOB_NAME}_%j.err \
  shell_scripts/experiments/electrode_ablation/train.slurm
