#!/bin/bash

# NOTE: remember to remove the save cache for machine with low RAM


export CUDA_VISIBLE_DEVICES=3


for run in 1 2 3
do
    seed=$RANDOM
    python3 run.py \
        experiment=scalarflow/t1u \
        seed=$seed \
        logger.wandb.project=sflowhighres \
        datamodule.data_dir="/your/path/here" \
        datamodule.batch_size=1  \
        datamodule.save_cache=False \
        datamodule.max_cache_size=10 \
        train.integration_order=1 \
        train.scheduler.T_0=32 \
        train.optimizer.lr="1e-3" \
        train.optimizer.weight_decay="1e-4" \
        model.modes=384 \
        model.weight_init=1 \
        model.channel_exponent=7 \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=t1u-i1-m384-e7-d1-lr1e-3-wd1e-4-T_032
done