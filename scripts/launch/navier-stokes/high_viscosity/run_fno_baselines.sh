#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/fno \
    seed=$seed \
    datamodule.viscosity=1e-3 \
    datamodule.target_time=50 \
    model.modes1=24 \
    model.modes2=24 \
    model.nlayers=6 \
    model.width=32 \
    model.weight_init=1 \
    logger.wandb.project=table1-high-viscosity-v2  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=fno-i1-m24-l6-w32
done

for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/fno \
    seed=$seed \
    datamodule.viscosity=1e-3 \
    datamodule.target_time=50 \
    model.modes1=24 \
    model.modes2=24 \
    model.nlayers=6 \
    model.width=32 \
    model.weight_init=2 \
    logger.wandb.project=table1-high-viscosity-v2 \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=fno-i2-m24-l6-w32
done


for run in 1 2 3 4 5 6 7 8
do
seed=$RANDOM
python3 run.py \
    experiment=navier-stokes/ffno \
    seed=$seed \
    datamodule.viscosity=1e-3 \
    datamodule.target_time=50 \
    model.modes=32 \
    model.width=82 \
    model.n_layers=10 \
    logger.wandb.project=table1-high-viscosity-v2  \
    logger.wandb.entity="anonymous-research" \
    logger.wandb.name=ffno-m32-l10-w82
done