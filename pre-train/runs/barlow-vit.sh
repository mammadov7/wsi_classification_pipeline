#!/bin/sh
#SBATCH -A rnz@v100
#SBATCH --job-name=tcga-barlow-vit
#SBATCH --nodes=2
#SBATCH --constraint=v100-32g
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread 
#SBATCH --array=0-6%1

cd /gpfswork/rech/rnz/uyc98hc/solo-learn
pwd
module load pytorch-gpu/py3/1.11.0
which python

srun python main_pretrain.py \
    --config-path scripts/pretrain/custom/tcga \
    --config-name barlow-vit.yaml 
