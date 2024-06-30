#!/bin/bash

# custom config

# DATA=/path/to/datasets
DATA=/home/yzq/yzq_data
TRAINER=MaPLe

DATASET=$1
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=$2
# 自定义几shot

for SEED in 1 2 3
do
    DIR=OUTPUT/${TRAINER}/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done