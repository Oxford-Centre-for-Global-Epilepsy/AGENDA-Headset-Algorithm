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

# Create log directory
mkdir -p logs

sbatch --export=ALL,EXPERIMENT_NAME=$EXPERIMENT_NAME,DATASET_NAME=$DATASET_NAME,FOLD=$FOLD,POOLING_TYPE=$POOLING_TYPE,SEED=$SEED,OMIT_LIST=$OMIT_LIST \
  --job-name=$JOB_NAME \
  --output=logs/${JOB_NAME}_%j.out \
  --error=logs/${JOB_NAME}_%j.err \
  shell_scripts/train.slurm
