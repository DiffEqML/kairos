#!/bin/bash


export CUDA_VISIBLE_DEVICES=1


for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/t1u \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    datamodule.batch_size=64 \
    model.weight_init=1 \
    model.channel_exponent=6 \
    model.use_operator_layer=True \
    logger.wandb.project=table1-medium-viscosity  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=t1u-i1-m24-c6
done

for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/t1u \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    datamodule.batch_size=64 \
    model.weight_init=2 \
    model.channel_exponent=6 \
    model.use_operator_layer=True \
    logger.wandb.project=table1-medium-viscosity  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=t1u-i2-m24-c6
done