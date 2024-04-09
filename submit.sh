#!/bin/bash

#SBATCH -J vqrtree
#SBATCH -N 1
#SBATCH -t 2-0:0:0
#SBATCH --cpus-per-task=4
#SBATCH -G 1
#SBATCH --mem-per-gpu=8G
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

export WANDB_MODE=offline
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HF_DATASETS_OFFLINE=1 
export HF_DATASETS_OFFLINE=1
export HYDRA_FULL_ERROR=1

scontrol show job "$SLURM_JOB_ID"

# shellcheck source=/dev/null
source /data/software/anaconda3/etc/profile.d/conda.sh
while [[ $CONDA_SHLVL -gt 0 ]]; do
    conda deactivate || break
done
conda activate vqtree

sleep $(( ( RANDOM % 10 )  + 1 ))

echo "$@"
echo

srun python "$@"
