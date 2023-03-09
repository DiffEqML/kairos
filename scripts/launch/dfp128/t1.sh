#!/bin/bash

seed=$RANDOM

# manual control of device id. This is required so that running jobs to not read 
# different devices id from the config
export CUDA_VISIBLE_DEVICES=1

for run in 1 2 3
do
    seed=$RANDOM
    python3 run.py \
        experiment=dfp128/t1 \
        seed=$seed \
        model.modes1=100\
        model.modes2=100\
        model.width=24\
        datamodule.dataset_size=10000\
        logger.wandb.project=table2-medium \
        model.weight_init=2 \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=t1-i2-m100
done