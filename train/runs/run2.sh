#!/bin/bash

cd train
# pwd
# which python

python train.py \
    --feats_size 384 \
    --num_classes 1 \
    --lr 0.001000 \
    --weight_decay 0.000010 \
    --dropout_node 0.200000 \
    --dropout_patch 0.200000 \
    --non_linearity 0 \
    --q_n 128 \
    --average True \
    --dataset tcga_mocov3_vit \
    --model attenmil \
    --num_epochs 100 1>train/runs/tcga/attenmil/logs/out2.txt 2>train/runs/tcga/attenmil/logs/error2.txt 

echo "done"
