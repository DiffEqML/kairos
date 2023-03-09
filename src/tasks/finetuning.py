
from typing import Any, List
from itertools import chain
from matplotlib.pyplot import yticks

import torch
import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from omegaconf import OmegaConf

from src.utils.utils import get_logger
from src.optim.param_grouping import group_parameters_for_optimizer
from src.utils.checkpoint import load_checkpoint
from src.tasks.default import DefaultModel
from src.tasks.sequence_default import SequenceDefaultModel

logger = get_logger(__name__)


class FinetuningTask(DefaultModel):

    def __init__(self, cfg, model_cfg=None):
        super().__init__(cfg, model_cfg)
        self.automatic_optimization=False

    def training_step(self, batch: Any, batch_idx: int):
        opt_stem, opt_pstem = self.optimizers(use_pl_optimizer=True)

        opt_stem.zero_grad()
        opt_pstem.zero_grad()        
        loss, output, targets = self.step(batch, is_train=True)
        self.manual_backward(loss)
        opt_stem.step()
        opt_pstem.step()
        self.manual_step_schedulers()

        with torch.no_grad():
            metrics = getattr(self, f'train_metrics')(output, targets)
        self.log(f"train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "output": output, "targets": targets}


    def manual_step_schedulers(self):
        sched_stem, sched_pstem = self.lr_schedulers()
        lrs = {
            "stem_lr": sched_stem.__dict__['_last_lr'][0] ,
            "post_stem_lr": sched_pstem.__dict__['_last_lr'][0] ,
        }
        self.log(f"learning rates", lrs, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        sched_stem.step()
        sched_pstem.step()


    def configure_optimizers(self):
        
        stem_parameters = self.model.stem.parameters()
        # manual assignment of "initial_lr" key for 'pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR'
        # param_groups = tuple(stem_parameters.param_groups)[0]
        # param_groups["initial_lr"] = param_groups["lr"]


        pstem_parameters = chain(
            self.model.pre_stem.parameters(), 
            self.model.unembed_layer.parameters(), 
            self.model.post_stem.parameters()
        )
        
        stem_optimizer = hydra.utils.instantiate(self.cfg.train.stem_optimizer, stem_parameters)
        pstem_optimizer = hydra.utils.instantiate(self.cfg.train.post_stem_optimizer, pstem_parameters)

        # TODO: it would be nice to use the code below to debug/log sizes of param groups
        # in order to ensure that stem and pstem param groups correspond to the desired ones
        # self.log_optimizer(stem_optimizer)
        # self.log_optimizer(pstem_optimizer)

        if 'stem_scheduler' in self.cfg.train:
            stem_lr_scheduler = hydra.utils.instantiate(self.cfg.train.stem_scheduler, stem_optimizer)
            pstem_lr_scheduler = hydra.utils.instantiate(self.cfg.train.post_stem_scheduler, pstem_optimizer)
            stem_lr_scheduler = {'scheduler': stem_lr_scheduler,
                                'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}
            pstem_lr_scheduler = {'scheduler': pstem_lr_scheduler,
                                'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}
                             
            return [stem_optimizer, pstem_optimizer], [stem_lr_scheduler, pstem_lr_scheduler]
        else:
            return [stem_optimizer, pstem_optimizer]


    def log_optimizer(self, optimizer):
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            logger.info(f'Stem Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')



class SequenceFinetuningTask(SequenceDefaultModel):

    def __init__(self, cfg, model_cfg=None):
        super().__init__(cfg, model_cfg)
        self.automatic_optimization=False

    def training_step(self, batch: Any, batch_idx: int):
        opt_stem, opt_pstem = self.optimizers(use_pl_optimizer=True)

        opt_stem.zero_grad()
        opt_pstem.zero_grad()   

        loss, output, targets = self.step(batch, is_train=True)

        self.manual_backward(loss)
        opt_stem.step()
        opt_pstem.step()
        self.manual_step_schedulers()

        with torch.no_grad():
            metrics = getattr(self, f'train_metrics')(output, targets)
        self.log(f"train/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"loss": loss, "output": output, "targets": targets}


    def manual_step_schedulers(self):
        sched_stem, sched_pstem = self.lr_schedulers()
        lrs = {
            "stem_lr": sched_stem.__dict__['_last_lr'][0] ,
            "post_stem_lr": sched_pstem.__dict__['_last_lr'][0] ,
        }
        self.log(f"learning rates", lrs, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        sched_stem.step()
        sched_pstem.step()


    def configure_optimizers(self):
        
        stem_parameters = self.model.stem.parameters()
        # manual assignment of "initial_lr" key for 'pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR'
        # param_groups = tuple(stem_parameters.param_groups)[0]
        # param_groups["initial_lr"] = param_groups["lr"]


        pstem_parameters = chain(
            self.model.pre_stem.parameters(), 
            self.model.unembed_layer.parameters(), 
            self.model.post_stem.parameters()
        )
        
        stem_optimizer = hydra.utils.instantiate(self.cfg.train.stem_optimizer, stem_parameters)
        pstem_optimizer = hydra.utils.instantiate(self.cfg.train.post_stem_optimizer, pstem_parameters)

        # TODO: it would be nice to use the code below to debug/log sizes of param groups
        # in order to ensure that stem and pstem param groups correspond to the desired ones
        # self.log_optimizer(stem_optimizer)
        # self.log_optimizer(pstem_optimizer)

        if 'stem_scheduler' in self.cfg.train:
            stem_lr_scheduler = hydra.utils.instantiate(self.cfg.train.stem_scheduler, stem_optimizer)
            pstem_lr_scheduler = hydra.utils.instantiate(self.cfg.train.post_stem_scheduler, pstem_optimizer)
            stem_lr_scheduler = {'scheduler': stem_lr_scheduler,
                                'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}
            pstem_lr_scheduler = {'scheduler': pstem_lr_scheduler,
                                'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}
                             
            return [stem_optimizer, pstem_optimizer], [stem_lr_scheduler, pstem_lr_scheduler]
        else:
            return [stem_optimizer, pstem_optimizer]


    def log_optimizer(self, optimizer):
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            logger.info(f'Stem Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')