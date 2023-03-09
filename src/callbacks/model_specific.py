from itertools import filterfalse
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize as NormalizeCmap

import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.callbacks.wandb_callbacks import get_wandb_logger
from src.utils.hooks import vitdyn_hooked_forward
from src.models.vitdyn import PatchUnembed
import PIL


class LogViTDynActivations(Callback):
    """Logs all activations of a vitdyn-fno stem (blocks)"""

    def __init__(self, log_every_n = 30):
        super().__init__()
        self.unembed_layer = PatchUnembed(
            img_size=224, 
            patch_size=16, 
            out_chans=3, 
            embed_dim=768, 
            norm_layer=None, 
            flatten=True,
            proj_layer=None)

        self.ready = True
        self.log_every_n = log_every_n

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready and (trainer.current_epoch + 1) % self.log_every_n == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            step = experiment.step

            # get a validation batch from the validation dat loader
            val_imgs, _ = next(iter(trainer.datamodule.val_dataloader()))

            # keep only a single image to avoid OOM
            val_imgs = val_imgs[:1]

            # run the batch through the hooked forward to extract activations
            val_imgs = val_imgs.to(device=pl_module.device)

            _, val_activations = vitdyn_hooked_forward(pl_module.model, val_imgs)

            for c in range(3):
                fig, axs = plt.subplots(2, 6, figsize=(16, 8))

                for k, act in enumerate(val_activations):
                    reshaped_act = self.unembed_layer(act)
                    axs[k//6, k%6].imshow(reshaped_act[0,c].cpu().detach())

                plt.tight_layout()
            
                experiment.log(
                    {
                        f"activations_block_{k}_channel_{c}/{experiment.name}": 
                            wandb.Image(fig)          
                    }, 
                )