#!/bin/bash

export CUDA_VISIBLE_DEVICES=1


for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/t1 \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    model.modes1=24 \
    model.modes2=24 \
    model.weight_init=1 \
    logger.wandb.project=table1-medium-viscosity  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=t1-i1-m24-l8-w48
done

for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/t1 \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    model.modes1=24 \
    model.modes2=24 \
    model.weight_init=2 \
    logger.wandb.project=table1-medium-viscosity  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=t1-i2-m24-l8-w48
done
