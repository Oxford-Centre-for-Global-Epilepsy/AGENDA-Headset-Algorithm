#!/bin/bash

EXPERIMENT_NAME=$1       # e.g. final_full_training
DATASET_NAME=$2          # e.g. agenda_dataset
POOLING_TYPE=$3          # e.g. attention or mean
OMIT_LIST=$4             # Optional: e.g. Fp1,Fp2 or leave empty
CONFIG_PATH=$5           # Path to the YAML config file
SEEDS=(42 123 999 2024 0)  # Add or remove seeds as desired

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="seed_${SEED}"
  OMIT_ARG=""

  # Format OMIT_LIST for OmegaConf if provided
  if [ ! -z "$OMIT_LIST" ]; then
    OMIT_ARG="dataset.drop_electrodes=['$(echo $OMIT_LIST | sed "s/,/','/g")']"
  fi

  echo "🚀 Training with seed $SEED"

  python train.py \
    --config "$CONFIG_PATH" \
    experiment_name="$EXPERIMENT_NAME" \
    dataset.dataset_name="$DATASET_NAME" \
    model.pooling_type="$POOLING_TYPE" \
    seed="$SEED" \
    run_id="$RUN_NAME" \
    $OMIT_ARG
done
