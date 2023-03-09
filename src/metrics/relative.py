import torch
from torch import tensor
import torch.nn as nn
from src.utils.numerics import dct2d

from torchmetrics.functional.regression.mape import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)

from torchmetrics import Metric
from einops import rearrange


class MeanAbsolutePercentageErrorDCT(Metric):
    r"""Computes `Mean Absolute Percentage Error`_ (MAPE)
    with additional DCT preprocessing step on `targets`. Assumes 
    predictions are already in DCT space."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")   

    def update(self, preds, target):
        target = dct2d(target, norm="ortho")
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self):
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

