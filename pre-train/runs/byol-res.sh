#!/bin/sh
#SBATCH -A rnz@v100
#SBATCH --job-name=tcga-byol-res
#SBATCH --nodes=1
#SBATCH --constraint=v100-32g
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-5%1

cd /gpfswork/rech/rnz/uyc98hc/solo-learn
pwd
module load pytorch-gpu/py3/1.11.0
which python

srun python main_pretrain.py \
    --config-path scripts/pretrain/custom/tcga \
    --config-name byol-res.yaml
