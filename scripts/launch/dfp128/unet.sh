#!/bin/bash

seed=$RANDOM

# manual control of device id. This is required so that running jobs to not read 
# different devices id from the config
export CUDA_VISIBLE_DEVICES=3


for run in 1
do
    seed=$RANDOM
    python3 run.py \
        experiment=dfp128/unet \
        seed=$seed \
        datamodule.dataset_size=10000\
        model.channel_exponent=6 \
        logger.wandb.project=table2-medium \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=unet-c6
done

for run in 1
do
    seed=$RANDOM
    python3 run.py \
        experiment=dfp128/unet \
        seed=$seed \
        datamodule.dataset_size=10000\
        model.channel_exponent=7 \
        logger.wandb.project=table2-medium \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=unet-c7
done