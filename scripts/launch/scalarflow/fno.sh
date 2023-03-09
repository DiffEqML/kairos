#!/bin/bash

seed=$RANDOM

# manual control of device id. This is required so that running jobs to not read 
# different devices id from the config
export CUDA_VISIBLE_DEVICES=1

for run in 1 2 3
do
    seed=$RANDOM
    python3 run.py \
        experiment=scalarflow/fno \
        seed=$seed \
        logger.wandb.project=sflowhighres \
        datamodule.data_dir="/your/path/here" \
        datamodule.batch_size=1 \
        train.optimizer.lr="1e-3" \
        train.optimizer.weight_decay="1e-4" \
        train.scheduler.T_0=32 \
        train.integration_order=0 \
        model.modes1=48 \
        model.modes2=48 \
        model.width=48 \
        model.nlayers=4 \
        model.weight_init=1 \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=fno-i1-m48-l4-w48-d0-lr1e-1-wd1e-4-T_032
done
