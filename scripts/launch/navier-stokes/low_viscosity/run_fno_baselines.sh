#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/fno \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    model.modes1=24 \
    model.modes2=24 \
    model.nlayers=6 \
    model.width=32 \
    model.weight_init=1 \
    logger.wandb.project=table1-medium-viscosity  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=fno-i1-m24-l6-w32
done

for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/fno \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    model.modes1=24 \
    model.modes2=24 \
    model.nlayers=6 \
    model.width=32 \
    model.weight_init=2 \
    logger.wandb.project=table1-medium-viscosity \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=fno-i2-m24-l6-w32
done


for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/ffno \
    seed=$seed \
    datamodule.viscosity=1e-4 \
    datamodule.target_time=15 \
    model.modes=32 \
    model.width=82 \
    model.n_layers=10 \
    logger.wandb.project=table1-medium-viscosity  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=ffno-m24-l10-w82
done