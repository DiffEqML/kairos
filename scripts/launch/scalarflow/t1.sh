#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

for run in 3
do
    seed=$RANDOM
    python3 run.py \
        experiment=scalarflow/t1 \
        seed=$seed \
        logger.wandb.project=sflowhighres \
        datamodule.data_dir="/datasets/scalarflow_full_cam3" \
        datamodule.batch_size=1  \
        train.optimizer.lr="1e-3" \
        train.optimizer.weight_decay=0 \
        train.scheduler.T_0=512 \
        train.integration_order=1 \
        model.modes=224 \
        model.weight_init=1 \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=t1-i1-m224-d1-lr1e-3-t0-512-wd0
done