#!/bin/bash
#SBATCH --job-name=snakemake_data_processing_pipeline
#SBATCH --output=logs/snakemake_%j.out
#SBATCH --error=logs/snakemake_%j.err
#SBATCH --clusters=all
#SBATCH --partition=short                  # Change partition if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                  # Default CPUs (overwritten dynamically)
#SBATCH --mem=16G                          # Default Memory (overwritten dynamically)
#SBATCH --time=08:00:00                    # Max execution time
#SBATCH --mail-type=ALL                    # Notify me when the job starts, ends, times out, etc.
#SBATCH --mail-user=john.fleming@ndcn1133@ox.ac.uk

# ✅ Load modules and activate environment
module load Anaconda3/2024.02-1
source activate $HOME/miniforge3/envs/snakemake_env  # Activate my Snakemake environment

# ✅ Change to project directory
cd $DATA/AGENDA-Headset-Algorithm/workflow  # Update to your project path

echo "🔍 Detecting resource needs from Snakemake workflow..."
# Extract rule-specific resource usage
RULE_RESOURCES=$(snakemake --summary | awk '{print $2,$3,$4}' | tail -n +2)

# Default values (if detection fails)
MAX_JOBS=20  # Default max parallel jobs
MAX_CPUS=4   # Default CPU allocation
MAX_MEM="16G"  # Default memory allocation

# Detect resource needs dynamically
if [[ ! -z "$RULE_RESOURCES" ]]; then
    MAX_JOBS=$(echo "$RULE_RESOURCES" | wc -l)
    MAX_CPUS=$(echo "$RULE_RESOURCES" | awk '{print $1}' | sort -nr | head -1)
    MAX_MEM=$(echo "$RULE_RESOURCES" | awk '{print $2}' | sort -nr | head -1)
    echo "✅ Detected: $MAX_JOBS jobs, $MAX_CPUS CPUs, $MAX_MEM RAM"
else
    echo "⚠️ Using default resource allocation: $MAX_JOBS jobs, $MAX_CPUS CPUs, $MAX_MEM RAM"
fi

# ✅ Run Snakemake with dynamic resource allocation
snakemake --cores $MAX_CPUS \
          --use-conda \
          --rerun-incomplete \
          --keep-going \
          --latency-wait 60 \
          --restart-times 2 \
          --cluster "sbatch --job-name={rule} --ntasks=1 --cpus-per-task=$MAX_CPUS --mem=$MAX_MEM --time=08:00:00" \
          --jobs $MAX_JOBS

echo "✅ Snakemake pipeline execution finished."
