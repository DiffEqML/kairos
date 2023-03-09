import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, repeat

from src.models.layers.fdm1d import T1Layer1d
from src.models.layers.fdm2d import T1Layer2d
from src.utils.numerics import dct1d, idct1d, dct2d, idct2d

from src.models.dfpnet import dfpnet_adaptive


class T1_1d(nn.Module):
    def __init__(
            self, 
            modes, 
            width, 
            signal_resolution,
            in_channels=1, 
            out_channels=1, 
            nlayers=4, 
            padding=0, 
            perform_inverse=True,
            use_spectral=True,
            residual=True,
            keep_high=False,
            act="sin",
            transform="dct",
            weight_init=2
        ):
        super().__init__()

        self.modes1 = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(1 + self.in_channels, self.width)  # input channel is: (a(x), x)
        self.transform = transform # ["dct", "dft"]

        self.layers = nn.Sequential(*[T1Layer1d(
            self.modes1, 
            self.width, 
            signal_resolution=signal_resolution,
            use_spectral=use_spectral, 
            weight_init=weight_init,
            residual=residual,
            keep_high=keep_high,
            act=act,
            transform=transform
            )
            for _ in range(self.nlayers)])

        if transform == "dct": # real weights
            self.fc1 = nn.Parameter(1e-8 * torch.randn(self.width, 128))
            self.fc2 = nn.Parameter(1e-8 * torch.randn(128, out_channels))
        elif transform == "dft": # complex weights   
            self.fc1 = nn.Parameter(1e-8 * torch.randn(self.width, 128, 2))
            self.fc2 = nn.Parameter(1e-8 * torch.randn(128, out_channels, 2))

        self.use_spectral = use_spectral
        self.perform_inverse = perform_inverse

        signal_resolution = signal_resolution + padding

    def forward_transform(self, x):
        if self.transform == "dct":
            z = dct1d(x, norm='ortho')
        elif self.transform == "dft":
            z = torch.fft.rfft(x, norm='ortho')
        return z

    def inverse_transform(self, z):
        if self.transform == "dct":
            x = idct1d(z, norm='ortho')
        elif self.transform == "dft":
            x = torch.fft.irfft(z, norm='ortho')
        return x

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

        # transform 
        N = x.shape[-1]
        #half_shift = self.half_shift.to(x.device) if self.transform == "dct" else None
        x = self.forward_transform(x)
        
        x = self.layers(x)

        x = rearrange(x, 'b c x -> b x c')

        ### out head needs to account cases where `x` is complex
        ### i.e. with transform="dft"
        weights_fc1 = self.fc1 if self.transform == "dct" else torch.torch.view_as_complex(self.fc1)
        x = torch.einsum("bxi,io->bxo", x, weights_fc1)

        x = torch.sin(x)

        weights_fc2 = self.fc2 if self.transform == "dct" else torch.torch.view_as_complex(self.fc2)
        x = torch.einsum("bxi,io->bxo", x, weights_fc2)

        if self.out_channels == 1:
             out = rearrange(x, 'b x 1 -> b x')
        else:
            out = rearrange(x, 'b x c -> b c x')

 
        if self.perform_inverse:
            out = self.inverse_transform(out)

        return out 

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[-1]
        return repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                      'x -> b x', b=batchsize)


class T1_2d(nn.Module):
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
            perform_forward=True,
            perform_inverse=True,
            use_spectral=True,
            residual=True,
            keep_high=False,
            act="sin",
            transform="dct",
            weight_init=2,
            first_layer_init_only=False
        ):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nlayers = nlayers
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + self.in_channels, self.width)  # input channel is: (a(x), x)
        self.transform = transform # ["dct", "dft"]

        self.layers = nn.Sequential()
        for k in range(self.nlayers):
            weight_init_ = weight_init
            if first_layer_init_only: weight_init_ = weight_init if k == 0 else 4
            
            self.layers.append(T1Layer2d(
                self.modes1, 
                self.modes1,
                self.width, 
                signal_resolution=signal_resolution,
                use_spectral=use_spectral, 
                weight_init=weight_init_,
                residual=residual,
                keep_high=keep_high,
                act=act,
                transform=transform
                )
            )

        if transform == "dct": # real weights
            self.fc1 = nn.Parameter(1e-8 * torch.randn(self.width, 128))
            self.fc2 = nn.Parameter(1e-8 * torch.randn(128, out_channels))
        elif transform == "dft": # complex weights   
            self.fc1 = nn.Parameter(1e-8 * torch.randn(self.width, 128, 2))
            self.fc2 = nn.Parameter(1e-8 * torch.randn(128, out_channels, 2))

        self.use_spectral = use_spectral
        self.perform_inverse = perform_inverse
        self.perform_forward = perform_forward

    def forward_transform(self, x):
        if self.transform == "dct":
            z = dct2d(x, norm="ortho")
        elif self.transform == "dft":
            z = torch.fft.rfft2(x, norm='ortho')
        return z

    def inverse_transform(self, z):
        if self.transform == "dct":
            x = idct2d(z, norm='ortho')
        elif self.transform == "dft":
            x = torch.fft.irfft(z, norm='ortho')
        return x

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        if self.in_channels == 1 and len(x.shape) == 3: 
            x = rearrange(x, 'b x y -> b x y 1')
        else:
            x = rearrange(x, 'b c x y -> b x y c')
        x = torch.cat([x, grid], dim=-1)

        x = self.fc0(x)
        x = rearrange(x, 'b x y c -> b c x y')

        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding]) # pad the domain if input is non-periodic

        # transform 
        if self.perform_forward:
            x = self.forward_transform(x)
        
    
        x = self.layers(x)
        
        x = rearrange(x, 'b c x y -> b x y c')

        ### out head needs to account cases where `x` is complex
        ### i.e. with transform="dft"
        weights_fc1 = self.fc1 if self.transform == "dct" else torch.torch.view_as_complex(self.fc1)
        x = torch.einsum("bxyi,io->bxyo", x, weights_fc1)

        x = torch.sin(x)

        weights_fc2 = self.fc2 if self.transform == "dct" else torch.torch.view_as_complex(self.fc2)
        x = torch.einsum("bxyi,io->bxyo", x, weights_fc2)

        if self.out_channels == 1:
             out = rearrange(x, 'b x y 1 -> b x y')
        else:
            out = rearrange(x, 'b x y c -> b c x y')

 
        if self.perform_inverse:
            out = self.inverse_transform(out)

        return out 

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-1], shape[-2]
        gridx = repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                       'x -> b x y', b=batchsize, y=size_y)
        gridy = repeat(torch.linspace(0, 1, size_y, dtype=torch.float, device=device),
                       'y -> b x y', b=batchsize, x=size_x)
        return torch.stack([gridx, gridy], dim=-1)


class T1U_2d(nn.Module):
    def __init__(
            self, 
            modes,
            signal_resolution,
            in_channels=1, 
            out_channels=1, 
            channel_exponent=4,
            padding=0, 
            perform_forward=True,
            perform_inverse=True,
            use_spectral=True,
            residual=True,
            keep_high=False,
            use_operator_layer=True,
            act="sin",
            transform="dct",
            *args,
            **kwargs
        ):
        super().__init__()

        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding  
        self.fc0 = nn.Linear(2 + in_channels, 2 + in_channels)  # input channel is: (a(x), x)
        self.transform = transform # ["dct", "dft"]

        self.op_layer = T1Layer2d(
            self.modes, 
            self.modes,
            width=2 + in_channels, 
            signal_resolution=signal_resolution,
            use_spectral=use_spectral, 
            weight_init=2,
            residual=residual,
            keep_high=keep_high,
            act=act,
            transform=transform
            )

        self.dfpnet = dfpnet_adaptive(
            res=modes, 
            channel_exponent=channel_exponent,
            in_channels=2 + in_channels,
            out_channels=out_channels)

        self.use_spectral = use_spectral
        self.perform_inverse = perform_inverse
        self.perform_forward = perform_forward
        self.use_operator_layer = use_operator_layer
        self.padding_x = signal_resolution[0] - modes
        self.padding_y = signal_resolution[1] - modes

    def forward_transform(self, x):
        if self.transform == "dct":
            z = dct2d(x, norm="ortho")
        elif self.transform == "dft":
            z = torch.fft.rfft2(x, norm='ortho')
        return z

    def inverse_transform(self, z):
        if self.transform == "dct":
            x = idct2d(z, norm='ortho')
        elif self.transform == "dft":
            x = torch.fft.irfft(z, norm='ortho')
        return x

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        if self.in_channels == 1: 
            x = rearrange(x, 'b x y -> b x y 1')
        else:
            x = rearrange(x, 'b c x y -> b x y c')
        x = torch.cat([x, grid], dim=-1)

        x = self.fc0(x)

        x = rearrange(x, 'b x y c -> b c x y')

        # transform 
        if self.perform_forward:
            x = self.forward_transform(x)

        if self.use_operator_layer:
            x = self.op_layer(x)

        # keep only modes
        x = x[..., :self.modes, :self.modes]

        x = self.dfpnet(x)
        
        x = F.pad(x, [0, self.padding_y, 0, self.padding_x])

        if self.perform_inverse:
            out = self.inverse_transform(out)

        if self.out_channels == 1:
             x = rearrange(x, 'b 1 x y -> b x y')
 
        return x 

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = repeat(torch.linspace(0, 1, size_x, dtype=torch.float, device=device),
                       'x -> b x y', b=batchsize, y=size_y)
        gridy = repeat(torch.linspace(0, 1, size_y, dtype=torch.float, device=device),
                       'y -> b x y', b=batchsize, x=size_x)
        return torch.stack([gridx, gridy], dim=-1)