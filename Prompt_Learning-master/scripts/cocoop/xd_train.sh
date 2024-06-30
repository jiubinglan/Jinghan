#!/bin/bash

#cd ../..

# 修改人：YinZiQi
# 修改内容：增添了指定数据集的功能，如没有指定数据集则使用默认数据集imagenet，增添了数据集路径的参数

# custom config
TRAINER=CoCoOp

# DATASET=imagenet
SEED=$1
DATASET=${2:-imagenet}
DATA=$3
# 运行示例：

# seed=1
# bash scripts/cocoop/xd_train.sh 1 Vis2017 /path/to/datasets

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16

# DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
DIR=OUTPUT/Domain-Adaptation/CoCoOp/${DATASET}/${CFG}_${SHOTS}shots/seed${SEED}
# 如果按照上面这样命名会存在一个问题，例如这次先训练了AID_NWPU数据集，
# 下次再训练AID_UCM数据集的话，就会发现之前的AID_NWPU已经有了，于是直接用了而不训练，
# 这并不符合我们的预期，所以我们需要修改一下命名方式。

# 暂不修改了，因为和师兄讨论之后我们只做few-shot实验，不涉及迁移学习内容，所以直接train即可。

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