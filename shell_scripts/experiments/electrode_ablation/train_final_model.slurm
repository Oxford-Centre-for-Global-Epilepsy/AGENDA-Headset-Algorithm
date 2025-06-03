#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=john.fleming@ndcn.ox.ac.uk

# Load environment
source ~/.bashrc
conda activate new_ml_env

# Navigate to project root
cd /data/ndcn-cortico-basal-ganglia-model/ndcn1133/AGENDA-Headset-Algorithm

# Make sure Python can find ml/ as a package
export PYTHONPATH=$PWD:$PYTHONPATH

# Seeds for final model training
SEEDS=(42 123 999 2024 0)  # Add or remove seeds as desired

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="seed_${SEED}"
  OMIT_ARG=""

  # Format OMIT_LIST for OmegaConf if provided
  if [ ! -z "$OMIT_LIST" ]; then
    OMIT_ARG="dataset.drop_electrodes=['$(echo $OMIT_LIST | sed "s/,/','/g")']"
  fi

  echo "🚀 Training with seed $SEED"

  python -m ml.training.train_final_model \
    --config "$CONFIG_PATH" \
    experiment_name="$EXPERIMENT_NAME" \
    dataset.dataset_name="$DATASET_NAME" \
    model.pooling_type="$POOLING_TYPE" \
    seed="$SEED" \
    run_id="$RUN_NAME" \
    $OMIT_ARG
done