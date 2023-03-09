from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from src.models.layers.fdm1d import DCTConv1d, DFTConv1d
from src.models.layers.fdm2d import DCTConv2d, DFTConv2d

import hydra

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class SpectralOperator1d(nn.Module):
    """Operator combining a spectral convolution with a residual or regular convolution.
    """
    def __init__(self, modes, width, weight_init, signal_resolution=1024, use_spectral=True, 
                residual=True, conv_residual=False, act="sin", spectral_layer="src.models.layers.spectral_conv.DCTConv1d",
                keep_high=False):
        super().__init__()
        self.modes = modes
        self.width = width

        layer_cfg = {
            "in_channels": self.width, 
            "out_channels": self.width, 
            "weight_init": weight_init,
            "modes1": self.modes,
            "keep_high": keep_high,
            "signal_resolution": signal_resolution,
            "_target_": spectral_layer
        }

        self.conv = hydra.utils.instantiate(layer_cfg)

        self.w = nn.Conv1d(self.width, self.width, 1)
        self.use_spectral = use_spectral
        self.residual = residual
        self.conv_residual = conv_residual
        self.act = torch.sin if act == "sin" else F.gelu

    def forward(self, x):
        if self.use_spectral: out = self.conv(x)
        else: out = self.w(x)
        if self.residual: out = out + x
        if self.conv_residual: out = out + self.w(x)
        return self.act(out)


class NeuralOperator1d(nn.Module):
    def __init__(
            self, 
            modes, 
            width, 
            signal_resolution,
            in_channels=1, 
            out_channels=1, 
            nlayers=4, 
            padding=0, 
            use_spectral=True,
            residual=True,
            conv_residual=True,
            keep_high=False,
            spectral_layer=DCTConv1d,
            act=torch.sin,
            weight_init=2
        ):
        super(NeuralOperator1d, self).__init__()

        """
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1 + self.in_channels, self.width)  # input channel is: (a(x), x)

        self.layers = nn.Sequential(*[SpectralOperator1d(
            self.modes1, 
            self.width, 
            signal_resolution=signal_resolution,
            use_spectral=use_spectral, 
            weight_init=weight_init,
            residual=residual,
            conv_residual=conv_residual,
            keep_high=keep_high,
            act=act,
            spectral_layer=spectral_layer
            )
            for _ in range(self.nlayers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.use_spectral = use_spectral

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device).unsqueeze(-1)
        if self.in_channels == 1: 
            x = rearrange(x, 'b x -> b x 1')
        else:
            x = rearrange(x, 'b c x -> b x c')

        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = rearrange(x, 'b x c -> b c x')
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        # FNO code doesn't apply activation on the last block, but we do for code's simplicity.
        # Performance seems about the same.
        x = self.layers(x)

        if self.padding != 0:
            x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = rearrange(x, 'b c x -> b x c')
        x = self.fc2(F.gelu(self.fc1(x)))

        if self.out_channels == 1:
             out = rearrange(x, 'b x 1 -> b x')
        else:
            out = rearrange(x, 'b x c -> b c x')
        return out 

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[-1]
        return repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                      'x -> b x', b=batchsize)


class SpectralOperator2d(nn.Module):
    def __init__(self, modes1, modes2, width, weight_init, signal_resolution=1024, use_spectral=True, 
                residual=True, conv_residual=False, uconv_residual=False, act="gelu", spectral_layer=DFTConv2d,
                keep_high=False, use_context=False):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # we assume the context to have 3 channels ### TO MODIFY
        if use_context: input_width = self.width + 3
        else: input_width = self.width

        layer_cfg = {
            "in_channels": input_width, 
            "out_channels": self.width, 
            "weight_init": weight_init,
            "modes1": self.modes1,
            "modes2": self.modes2,
            "keep_high": keep_high,
            "signal_resolution": signal_resolution,
            "_target_": spectral_layer
        }

        self.conv = hydra.utils.instantiate(layer_cfg)
        if conv_residual: 
            self.w = nn.Conv2d(input_width, self.width, 1)

        self.use_spectral, self.use_context = use_spectral, use_context
        self.context = None
        self.act = torch.sin if act == "sin" else F.gelu
        self.residual = residual
        self.conv_residual = conv_residual
        self.uconv_residual = uconv_residual

    def reset_context(self, context):
        self.context = context 

    def forward(self, x):

        if self.use_context:
            x = torch.cat([x, self.context], 1)

        if self.use_spectral: out = self.conv(x)
        else: out = self.w(x)
        if self.residual: out = out + x
        if self.conv_residual: out = out + self.w(x)
        if self.uconv_residual: out = out + self.uw(x)
        return self.act(out)


class NeuralOperator2d(nn.Module):
    def __init__(
        self, 
        modes1,
        modes2, 
        width, 
        signal_resolution,
        in_channels=1, 
        out_channels=1, 
        nlayers=4, 
        padding=0, 
        use_spectral=True,
        residual=True,
        conv_residual=True,
        uconv_residual=False,
        keep_high=False,
        spectral_layer=DFTConv2d,
        act=torch.sin,
        weight_init=2,
        use_context=False
        ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + self.in_channels, self.width)  # input channel: (z(x, y), x, y)

        self.layers = nn.Sequential(*[SpectralOperator2d(
            self.modes1,
            self.modes2, 
            self.width, 
            signal_resolution=signal_resolution,
            use_spectral=use_spectral, 
            weight_init=weight_init,
            residual=residual,
            conv_residual=conv_residual,
            uconv_residual=uconv_residual,
            keep_high=keep_high,
            act=act,
            spectral_layer=spectral_layer
            )
            for _ in range(self.nlayers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.use_fourier = use_spectral
        # whether to use "Founder" context embeddings produced by a ViT
        self.use_context = use_context 

    def reset_context(self, context):

        for layer in self.layers:
            layer.reset_context(context)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        if self.in_channels == 1: 
            x = rearrange(x, 'b x y -> b x y 1')
        else:
            x = rearrange(x, 'b c x y -> b x y c')
        x = torch.cat([x, grid], dim=-1)

        x = self.fc0(x)
        x = rearrange(x, 'b x y c -> b c x y')
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # FNO code doesn't apply activation on the last block, but we do for code's simplicity.
        # Performance seems about the same
        x = self.layers(x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]
        x = rearrange(x, 'b c x y -> b x y c')
        x = self.fc2(F.gelu(self.fc1(x)))
        if self.out_channels == 1:
             out = rearrange(x, 'b x y 1 -> b x y')
        else:
            out = rearrange(x, 'b x y c -> b c x y')
        return out 

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                       'x -> b x y', b=batchsize, y=size_y)
        gridy = repeat(torch.linspace(0, 1, size_y, dtype=torch.float, device=device),
                       'y -> b x y', b=batchsize, x=size_x)
        return torch.stack([gridx, gridy], dim=-1)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0., upsample_factor=2, in_place=True):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=in_place))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=in_place))

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=upsample_factor, mode='bilinear')) # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=(size-1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout>0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d( dropout, inplace=True))
    return block


class MiniUNet64(nn.Module):
    def __init__(self, channelExponent=8, dropout=0., in_channels=1, out_channels=1):
        super(MiniUNet64, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(in_channels, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels*2, 'layer2', transposed=False, bn=True,   relu=False, dropout=dropout )
        self.layer3= blockUNet(channels*2, channels*2, 'layer2b',transposed=False, bn=True,  relu=False, dropout=dropout )

        self.dlayer4= blockUNet(channels*2, channels*2, 'dlayer2b',transposed=True, bn=True, relu=True, dropout=dropout )
        self.dlayer5 = blockUNet(channels*4, channels  , 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout )

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels*2, out_channels, 4, 2, 1, bias=True))

    def forward(self, x):
        x_in = x
        if len(x.shape) < 4: 
            x_in = x[:, None]

        out1 = self.layer1(x_in) 
        out2 = self.layer2(out1)

        out3 = self.layer3(out2)
        dout4 = self.dlayer4(out3)

        dout4 = torch.cat([dout4, out2], 1)
        dout5 = self.dlayer5(dout4)

        dout5 = torch.cat([dout5, out1], 1)
        dout = self.dlayer1(dout5)


        if len(x.shape) < 4: 
            dout = dout.squeeze(1)
        return dout
