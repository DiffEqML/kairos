import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


# Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "train",
        "eval",
        "callbacks",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger,
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''
    return x * torch.sigmoid(1.702 * x)


def swish(x):
    '''Swish activation (https://arxiv.org/abs/1710.05941).'''
    return x * torch.sigmoid(x)


def concat_relu(x):
    '''Concatenated ReLU (http://arxiv.org/abs/1603.05201).'''
    return F.relu(torch.cat([x, -x], dim=1))


def concat_elu(x):
    '''Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead.'''
    return F.elu(torch.cat([x, -x], dim=1))


def gated_tanh(x, dim):
    '''Gated Tanh activation.'''
    x_tanh, x_sigmoid = torch.chunk(x, 2, dim=dim)
    return torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)


class GELU(nn.Module):
    '''GELU activation (https://arxiv.org/abs/1606.08415) as used in Sparse Transformers (https://arxiv.org/abs/1904.10509).'''

    def forward(self, input):
        return gelu(input)


class Swish(nn.Module):
    '''Swish activation (https://arxiv.org/abs/1710.05941).'''

    def forward(self, input):
        return swish(input)


class ConcatReLU(nn.Module):
    '''Concatenated ReLU (http://arxiv.org/abs/1603.05201).'''

    def forward(self, input):
        return concat_relu(input)


class ConcatELU(nn.Module):
    '''Like concatenated ReLU (http://arxiv.org/abs/1603.05201), but with ELU instead.'''

    def forward(self, input):
        return concat_elu(input)


class GatedTanhUnit(nn.Module):
    '''Gated Tanh activation.'''

    def __init__(self, dim=-1):
        super(GatedTanhUnit, self).__init__()
        self.dim = dim

    def forward(self, x):
        return gated_tanh(x, dim=self.dim)

act_strs = {'elu', 'relu', 'gelu', 'swish'}
concat_act_strs = {'concat_elu', 'concat_relu'}


def act_module(act_str, allow_concat=False):
    if allow_concat: assert act_str in act_strs + concat_act_strs, 'Got invalid activation {}'.format(act_str)
    else:            assert act_str in act_strs, 'Got invalid activation {}'.format(act_str)
    if act_str == 'relu': return nn.ReLU()
    elif act_str == 'elu': return nn.ELU()
    elif act_str == 'gelu': return GELU()
    elif act_str == 'swish': return Swish()
    elif act_str == 'concat_relu': return ConcatReLU()
    elif act_str == 'concat_elu': return ConcatELU()


def act_factor(act_str, allow_concat=False):
    if allow_concat: assert act_str in act_strs + concat_act_strs, 'Got invalid activation {}'.format(act_str)
    else:            assert act_str in act_strs, 'Got invalid activation {}'.format(act_str)
    if act_str == 'relu': return 1
    elif act_str == 'elu': return 1
    elif act_str == 'gelu': return 1
    elif act_str == 'swish': return 1
    elif act_str == 'concat_relu': return 2
    elif act_str == 'concat_elu': return 2

class UnitGaussianNormalizer(nn.Module):

    def __init__(self, x, eps=0.00001):
        super().__init__()
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)
