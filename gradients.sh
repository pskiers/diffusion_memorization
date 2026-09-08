#!/bin/bash -l
#SBATCH --job-name=elrond-grads
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/slurm/stderr/error_%j.err

module load ML-bundle/24.06a
cd $SCRATCH/LID-project
source env/bin/activate
cd diffusion_memorization 

python token_subspace.py
    