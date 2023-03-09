from signal import siginterrupt
from typing import List
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from src.models.dfpnet import dfpnet_adaptive
from src.utils.numerics import dct2d, idct2d
from src.utils.init_fn import fdm_init_scale
import hydra

class T1Layer2d(nn.Module):
    """T1 block to be used in k-space FDM models i.e. T1_2d
    
    Performs a spatial and channel mixing via two convolutions"""
    def __init__(self, modes1, modes2, width, weight_init=2, signal_resolution=(128, 128),
                residual=True, act="sin", keep_high=False, transform="dft", *args, **kwargs):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.width = width
        signal_length = signal_resolution[0] * signal_resolution[1]

        self.initialize_weights(weight_init, width, signal_length, modes1, modes2, transform)
        self.residual, self.keep_high = residual, keep_high
        
        if transform == "dft": self.act = torch.sin if act == "sin" else torch.tanh
        else: self.act = torch.sin if act == "sin" else F.gelu
        self.transform = transform

    def initialize_weights(self, weight_init, width, signal_length, modes1, modes2, transform="dct"):
        gain = 1 if transform == "dct" else 1 / 2

        scale_ch = fdm_init_scale(weight_init, fan_in=width, fan_out=width, gain=gain, signal_length=signal_length, modes=modes1 * modes2)
        scale_sp1 = fdm_init_scale(weight_init=4, fan_in=modes1, fan_out=modes1, gain=gain)
        scale_sp2 = fdm_init_scale(weight_init=4, fan_in=modes2, fan_out=modes2, gain=gain)

        weight_sizes_ch = (width, width, modes1, modes2)
        weight_sizes_sp1 = (modes1, modes1, width)
        weight_sizes_sp1 = (modes2, modes2, width)

        if transform == "dft": 
            weight_sizes_ch += (2,) # add two dimensions for real and complex components
            weight_sizes_sp1 += (2,)
            weight_sizes_sp2 += (2,)

        self.ch_mixing_1 = nn.Parameter(scale_ch * torch.randn(*weight_sizes_ch))
        self.spatial_mixing_1 = nn.Parameter(scale_sp1 * torch.randn(*weight_sizes_sp1))
        self.spatial_mixing_2 = nn.Parameter(scale_sp2 * torch.randn(*weight_sizes_sp1))
    

    def mul2d_ch_1(self, input):
        weights = self.ch_mixing_1 if self.transform == "dct" else torch.view_as_complex(self.ch_mixing_1)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def mul2d_sp_1(self, input):
        weights = self.spatial_mixing_1 if self.transform == "dct" else torch.view_as_complex(self.spatial_mixing_1)
        return torch.einsum("bcix,ioc->bcox", input, weights)

    def mul2d_sp_2(self, input):
        weights = self.spatial_mixing_2 if self.transform == "dct" else torch.view_as_complex(self.spatial_mixing_2)
        return torch.einsum("bcyi,ioc->bcyo", input, weights)

    def forward(self, x):
        out = torch.zeros_like(x) 

        # channel mixing
        z1 = self.mul2d_ch_1(x[..., :self.modes1:, :self.modes2])

        z1 = self.act(z1)

        # spatial mixing
        z1 = self.mul2d_sp_1(z1)
        z1 = self.mul2d_sp_2(z1)

        if self.residual:
             z1 = z1 + x[..., :self.modes1:, :self.modes2]

        out[..., :self.modes1:, :self.modes2] = z1

        if self.keep_high: 
            out[..., self.modes1:-self.modes1, :] = x[..., self.modes1:-self.modes1, :]
            out[..., self.modes2:] = x[..., self.modes2:]

        return out


class DFTConv2d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes1, 
            modes2,
            keep_high=True,
            weight_init=2, 
            signal_resolution=(128, 128)
        ):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        weight_init: [1: "naive", 2: "vp", 3: "half-naive", 4: "zero"]: vp (variance preserving) or naive as per Li et al.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  
        self.modes2 = modes2
        tot_signal_resolution = signal_resolution[0] * signal_resolution[1]
        if weight_init == 1: # naive
           self.scale = (1 / (in_channels * out_channels))
        elif weight_init == 2: # vp
            self.scale = math.sqrt((1 / (in_channels)) * tot_signal_resolution / (4 * modes1 * modes2 + 4))
        elif weight_init == 3: # half-naive
            self.scale = 1 / in_channels
        elif weight_init == 4: # zero 
            self.scale = 1e-8

        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, 2))

        self.keep_high = keep_high


    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Multiply relevant Fourier modes
        weights1, weights2 = torch.view_as_complex(self.weights1), torch.view_as_complex(self.weights2)

        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, x_ft.shape[-2], x_ft.shape[-1]).to(x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        if self.keep_high:  
            out_ft[..., self.modes1:-self.modes1, self.modes2:] = x[..., self.modes1:-self.modes1, :self.modes2:]

        #Return to n-space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x


class DCTConv2d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes1, 
            modes2,
            keep_high=True,
            weight_init=2, 
            signal_resolution=(128, 128),
            identity=False
        ):
        super().__init__()

        """
        2D DCT layer. DCT, linear transform, and Inverse DCT.

        weight_init: [1: "naive", 2: "vp", 3: "half-naive", 4: "zero"]: vp (variance preserving) or naive as per Li et al.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        tot_signal_resolution = signal_resolution[0] * signal_resolution[1]


        if weight_init == 1: # naive
           self.scale = (1 / (in_channels * out_channels))
        elif weight_init == 2: # vp
            self.scale = math.sqrt((1 / (in_channels)) * tot_signal_resolution / (2 * modes1 * modes2))
        elif weight_init == 3: # half-naive
            self.scale = 1 / in_channels
        elif weight_init == 4: # zero 
            self.scale = 1e-8

        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2))
        self.identity = identity
        self.keep_high = keep_high


    def forward_transform(self, x):
        z = dct2d(x, norm="ortho")
        return z

    def inverse_transform(self, z):
        x = idct2d(z, norm='ortho')
        return x

    def forward(self, x):
        Nx, Ny = x.shape[-1], x.shape[-2]
        z = self.forward_transform(x)
        
        if not self.identity:
            z_out = torch.zeros_like(z)
            z_in = z[..., :self.modes1, :self.modes2]
            z_out[..., :self.modes1, :self.modes2] = torch.einsum("bixy,oixy->boxy", z_in, self.weights1)

            z_in = z[..., -self.modes1:, :self.modes2]
            z_out[..., -self.modes1:, :self.modes2] = torch.einsum("bixy,oixy->boxy", z_in, self.weights1)

            if self.keep_high: 
                z_out[..., self.modes1:-self.modes1, :] = z[..., self.modes1:-self.modes1, :]
                z_out[..., self.modes2:] = z[..., self.modes2:]
            
        else: 
            z_out = z

        x = self.inverse_transform(z_out)
        return x
