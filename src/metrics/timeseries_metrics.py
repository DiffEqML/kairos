import torch
from torch import Tensor, tensor
import torch.nn as nn
from torchmetrics import SymmetricMeanAbsolutePercentageError, MeanAbsoluteError, PeakSignalNoiseRatio

# Use torch.index_select to obtain a loss specific to certain indexes (e.g. time stamps)
# NOTE: device is given to the index each time, there may be a better way

class SMAPEatIndex(SymmetricMeanAbsolutePercentageError):
    r"""Calculate SMAPE at specific index(es). Executes callback to parent function"""

    def __init__(self, *args, index=0, dimension=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(index, torch.Tensor):
            index = torch.LongTensor([index])
        self.index_ = index
        self.dim_ = dimension

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds = torch.index_select(preds, self.dim_, self.index_.to(self.device))
        target = torch.index_select(target, self.dim_, self.index_.to(self.device))
        return super().update(preds, target)


class MAEatIndex(MeanAbsoluteError):
    r"""Calculate MAEs at specific index(es). Executes callback to parent function"""

    def __init__(self, *args, index=0, dimension=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(index, torch.Tensor):
            index = torch.LongTensor([index])
        self.index_ = index
        self.dim_ = dimension

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds = torch.index_select(preds, self.dim_, self.index_.to(self.device))
        target = torch.index_select(target, self.dim_, self.index_.to(self.device))
        return super().update(preds, target)


class PSNRatIndex(PeakSignalNoiseRatio):
    r"""Calculate PSNR at specific index(es). Executes callback to parent function"""

    def __init__(self, *args, index=0, dimension=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(index, torch.Tensor):
            index = torch.LongTensor([index])
        self.index_ = index
        self.dim_ = dimension

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        preds = torch.index_select(preds, self.dim_, self.index_.to(self.device))
        target = torch.index_select(target, self.dim_, self.index_.to(self.device))
        return super().update(preds, target)
