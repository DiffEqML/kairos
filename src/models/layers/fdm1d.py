from signal import siginterrupt
from typing import List
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from src.models.dfpnet import dfpnet_adaptive
from src.utils.numerics import dct1d, idct1d
from src.utils.init_fn import fdm_init_scale
import hydra


class T1Layer1d(nn.Module):
    """Operator block to be used in k-space FDM models i.e. T1
    
    Performs a spatial and channel mixing via two convolutions"""

    def __init__(self, modes, width, weight_init=2, signal_resolution=1024,
                residual=True, act="sin", keep_high=False, transform="dft", *args, **kwargs):
        super().__init__()
        self.modes = modes
        self.width = width
        signal_length = signal_resolution

        self.initialize_weights(weight_init, width, signal_length, modes, transform)
        self.residual, self.keep_high = residual, keep_high
        
        if transform == "dft": self.act = torch.sin if act == "sin" else torch.tanh
        else: self.act = torch.sin if act == "sin" else F.gelu
        self.transform = transform

    def initialize_weights(self, weight_init, width, signal_length, modes, transform="dct"):
        gain = 1 if transform == "dct" else 1 / 2

        scale_ch = fdm_init_scale(weight_init, fan_in=width, fan_out=width, gain=gain, signal_length=signal_length)
        scale_sp = fdm_init_scale(weight_init=4, fan_in=modes, fan_out=modes, gain=gain)

        weight_sizes_ch = (width, width, modes)
        weight_sizes_sp = (modes, modes, width)

        if transform == "dft": 
            weight_sizes_ch += (2,) # add two dimensions for real and complex components
            weight_sizes_sp += (2,)

        self.ch_mixing_1 = nn.Parameter(scale_ch * torch.randn(*weight_sizes_ch))
        self.spatial_mixing_1 = nn.Parameter(scale_sp * torch.randn(*weight_sizes_sp))

    def mul1d_ch(self, input, weights):
        weights = self.ch_mixing if self.transform == "dct" else torch.view_as_complex(self.ch_mixing)
        return torch.einsum("bix,iox->box", input, weights)

    def mul1d_sp(self, input, weights):
        weights = self.spatial_mixing if self.transform == "dct" else torch.view_as_complex(self.spatial_mixing)
        return torch.einsum("bix,xyi->biy", input, weights)

    def forward(self, x):
        out = torch.zeros_like(x) 
        z = self.mul1d_ch(x[..., :self.modes], self.ch_mixing)
        z = self.act(z)
        z = self.mul1d_sp(z, self.spatial_mixing)

        if self.residual:
             z = z + x[..., :self.modes]

        out[..., :self.modes] = z
        if self.keep_high: out[..., self.modes:] = x[..., self.modes:]
        
        return out


class DFTConv1d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes1, 
            keep_high=True,
            weight_init=2, 
            signal_resolution=1024,
            use_rfft=True
        ):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        weight_init: [1: "naive", 2: "vp", 3: "half-naive", 4: "zero"]: vp (variance preserving) or naive as per Li et al.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        if weight_init == 1: # naive
           self.scale = self.bias_scale = (1 / (in_channels * out_channels))
        elif weight_init == 2: # vp
            self.scale = math.sqrt((1 / (in_channels)) * signal_resolution / (2 * modes1))
        elif weight_init == 3: # half-naive
            self.scale = self.bias_scale = 1 / in_channels
        elif weight_init == 4: # zero 
            self.scale = self.bias_scale = 1e-8
        # self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        # weight is stored as real to avoid issue with Adam not working on complex parameters
        # FNO code initializes with rand but we initializes with randn as that seems more natural.
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, 2))

        #dc_bias = self.bias_scale * torch.randn(in_channels, out_channels, 1, 2)
        #dc_bias[..., 1:] = 0 # set to real
        #self.dc_bias = nn.Parameter(dc_bias)

        self.keep_high = keep_high
        self.use_rfft = use_rfft

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        transform = torch.fft.rfft if self.use_rfft else torch.fft.fft
        x_ft = transform(x, norm='ortho')

        # Multiply relevant Fourier modes
        weights1 = torch.view_as_complex(self.weights1)
        #dc_bias = torch.view_as_complex(self.dc_bias)


        out_ft = torch.zeros(x_ft.shape[0], self.out_channels, x_ft.shape[-1]).to(x.device)
        out_ft[..., :self.modes1] = self.compl_mul1d(x_ft[..., :self.modes1], weights1) 
        if self.keep_high:
            out_ft[..., self.modes1:] = x_ft[..., self.modes1:]

            # only for variance preservation experiments
            # does not support combination with `keep_high`
            # if not self.use_rfft:
            #     out_ft[..., -self.modes1:] = self.compl_mul1d(x_ft[..., -self.modes1:], weights1)
        
        # dc term is real
        #out_ft[..., :1] = self.compl_mul1d(x_ft[..., :1], dc_bias) 

        itransform = torch.fft.irfft if self.use_rfft else torch.fft.ifft
        x = itransform(out_ft, n=x.size(-1), norm='ortho')
        return x


class DCTConv1d(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes1, 
            keep_high=True,
            weight_init=2, 
            signal_resolution=1024,
            identity=False
        ):
        super().__init__()

        """
        1D DCT layer. DCT, linear transform, and Inverse DCT.

        weight_init: [1: "naive", 2: "vp", 3: "half-naive", 4: "zero"]: vp (variance preserving) or naive as per Li et al.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        if weight_init == 1: # naive
           self.scale = (1 / (in_channels * out_channels))
        elif weight_init == 2: # vp
            self.scale = math.sqrt((1 / (in_channels)) * signal_resolution / (modes1))
        elif weight_init == 3: # half-naive
            self.scale = 1 / in_channels
        elif weight_init == 4: # zero 
            self.scale = 1e-8

        self.identity = identity
        self.keep_high = keep_high
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1))

    def forward_transform(self, x):
        z = dct1d(x, norm="ortho")
        return z

    def inverse_transform(self, z):
        x = idct1d(z, norm='ortho')
        return x

    def forward(self, x):
        N = x.shape[-1]
        z = self.forward_transform(x)
        
        if not self.identity:
            if self.keep_high:
                z_out = torch.zeros_like(z)
                z_out[..., :self.modes1] = torch.einsum("bix,oix->box", z[..., :self.modes1], self.weights1)
                z_out[..., self.modes1:] = z[..., self.modes1:]
            else:
                z = torch.einsum("bix,oix->box", z[..., :self.modes1], self.weights1)
                z_out = F.pad(z, (0, N - self.modes1), mode="constant")
        
        else: z_out = z
        x = self.inverse_transform(z_out)
        return x

