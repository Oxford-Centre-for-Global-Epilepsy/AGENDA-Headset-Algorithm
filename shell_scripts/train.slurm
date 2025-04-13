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

# Optional: confirm current dir and Python path
echo \"Working dir: $(pwd)\"
echo \"Python path: $PYTHONPATH\"

# Run training
python -m ml.training.train \
  --config configs/base.yaml \
  fold_index=$FOLD \
  output_dir=outputs/run_fold_$FOLD \
  resume=true