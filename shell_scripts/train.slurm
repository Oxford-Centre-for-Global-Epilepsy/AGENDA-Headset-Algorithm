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

# Run training
python -m ml.training.train \
  --config configs/${POOLING_TYPE}_pooling_model.yaml \
  fold_index=$FOLD \
  pooling_type=$POOLING_TYPE \
  resume=true