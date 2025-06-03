#!/bin/bash

EXPERIMENT_NAME=$1
DATASET_NAME=$2
SEED=$3
POOLING_TYPE=$4
OMIT_LIST=$5
CONFIG_PATH=$6  # typically config/train_final_model.yaml

# Construct a safe job name
JOB_NAME="AGENDA_${EXPERIMENT_NAME}_Final_Model_${POOLING_TYPE}_seed_${SEED}"
if [ ! -z "$OMIT_LIST" ]; then
  OMIT_TAG=$(echo "$OMIT_LIST" | tr ',' '_')
  JOB_NAME="${JOB_NAME}_omit_${OMIT_TAG}"
fi

# Format OMIT_LIST for OmegaConf
if [ ! -z "$OMIT_LIST" ]; then
  IFS=',' read -ra CHANNELS <<< "$OMIT_LIST"
  FORMATTED_OMIT_LIST=$(printf '"%s",' "${CHANNELS[@]}")
  FORMATTED_OMIT_LIST="[${FORMATTED_OMIT_LIST%,}]"
else
  FORMATTED_OMIT_LIST="[]"
fi
FORMATTED_OMIT_LIST_ESCAPED="'${FORMATTED_OMIT_LIST}'"

# Create log directory
mkdir -p logs

sbatch --export=ALL,EXPERIMENT_NAME="$EXPERIMENT_NAME",DATASET_NAME="$DATASET_NAME",SEED="$SEED",POOLING_TYPE="$POOLING_TYPE",FORMATTED_OMIT_LIST=$FORMATTED_OMIT_LIST_ESCAPED,CONFIG_PATH="$CONFIG_PATH" \
  --job-name=$JOB_NAME \
  --output=logs/${JOB_NAME}_%j.out \
  --error=logs/${JOB_NAME}_%j.err \
  shell_scripts/experiments/electrode_ablation/train_final_model.slurm
