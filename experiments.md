
# Transform Once: Efficient Operator Learning in Frequency Domain

To launch the experiments, first install the requirements:

```
pip install -r requirements.txt
```

We provide [wandb logs](https://wandb.ai/diffeqml-research/table2-medium/reports/Transform-Once-Efficient-Operator-Learning-in-Frequency-Domain--VmlldzoyNzYxNjQ0?accessToken=5tnrep1gkt8wlsnblrrw2zm4wp9rjo11m3yqfewof14xpwvood5zigh8z84nuxri) for all experiments in the paper. 

Launch scripts are under `scripts/EXP_NAME`. Example launch for Navier-Stokes high-viscosity

```
./scripts/launch/navier-stokes/high-viscosity/run_t1.sh
```
## Data

All datasets should be downloaded and prepared before running the scripts. The Navier-Stokes incompressible datasets from from [this repository](https://github.com/neuraloperator/neuraloperator).

Instructions for ScalarFlow can be found [here](https://ge.in.tum.de/publications/2019-scalarflow-eckert/).


## Analysis

We also provide self-contained notebooks for quick experimentation on the incompressible Navier-Stokes. 