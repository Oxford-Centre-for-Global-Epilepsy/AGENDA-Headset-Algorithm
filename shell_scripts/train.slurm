#!/bin/bash
#SBATCH --job-name=AGENDA_Algorithm_Training_fold${FOLD}
#SBATCH --output=logs/fold${FOLD}_%j.out
#SBATCH --error=logs/fold${FOLD}_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=john.fleming@ndcn.ox.ac.uk

# Load environment
source ~/.bashrc
conda activate ml_env

# Navigate to project root
cd $DATA/AGENDA-Headset-Algorithm

# Run training
python -m ml.training.train \
  --config configs/base.yaml \
  fold_index=$FOLD \
  output_dir=outputs/run_fold_$FOLD \
  resume=true