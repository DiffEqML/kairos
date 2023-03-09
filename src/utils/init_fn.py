import torch
import torch.nn as nn
import math

def fdm_init_scale(weight_init, fan_in, fan_out, gain=1, signal_length=None, modes=None, *args, **kwargs):
    """Returns scale (stdev) for weight initialization of 1D FDMs

    Args:
        weight_init: {1: fno, 2: vp, 3: approx. zero, 4: kaiming}
    """
    if weight_init == 1:
        scale = 1 / (fan_in * fan_out)
    elif weight_init == 2:
        scale = math.sqrt((1 / fan_in) * signal_length / (modes))
    elif weight_init == 3:
        scale = 1e-8
    elif weight_init == 4:
        scale = math.sqrt(1 / (fan_in))
    return gain * scale