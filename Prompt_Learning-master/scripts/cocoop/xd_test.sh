#!/bin/bash

#cd ../..

# custom config
# DATA=/path/to/datasets
TRAINER=CoCoOp

DATASET=$1
SEED=$2
DATA=$3
SOURCE_DATASET=$4

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16


DIR=OUTPUT/evaluation/Domain-Adaptation/${TRAINER}/${SOURCE_DATASET}_${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
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
    --model-dir OUTPUT/Domain-Adaptation/CoCoOp/${SOURCE_DATASET}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 10 \
    --eval-only
fi