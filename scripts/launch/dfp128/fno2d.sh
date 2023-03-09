#!/bin/bash

seed=$RANDOM

# manual control of device id. This is required so that running jobs to not read 
# different devices id from the config
export CUDA_VISIBLE_DEVICES=2

for run in 1 2 3
do
    seed=$RANDOM
    python3 run.py \
        experiment=dfp128/fno2d \
        seed=$seed \
        datamodule.dataset_size=10000\
        model.modes1=64 \
        model.modes2=64 \
        model.width=42 \
        logger.wandb.project=table2-medium \
        logger.wandb.entity="anonymous-research" \
        logger.wandb.name=fno2d-i1-m36-l6-w48-v2
done
