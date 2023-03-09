import torch
import torch.nn as nn
from src.utils.numerics import dct2d

from einops import rearrange


class RelativeL2(nn.Module):

    def forward(self, x, y):
        x = rearrange(x, 'b ... -> b (...)')
        y = rearrange(y, 'b ... -> b (...)')
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()


class MSELossDCT(nn.Module):
    criterion = nn.MSELoss()
    
    def forward(self, x, y):
        y = dct2d(y, norm="ortho")
        return self.criterion(x, y)


class RelativeL2DCT(nn.Module):

    def forward(self, x, y):
        y = dct2d(y, norm="ortho")
        x = rearrange(x, 'b ... -> b (...)')
        y = rearrange(y, 'b ... -> b (...)')
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()
