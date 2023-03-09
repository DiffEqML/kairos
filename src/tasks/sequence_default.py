from typing import Any, List
from matplotlib.pyplot import yticks

import torch
import hydra
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection
import wandb

from einops import rearrange

from omegaconf import OmegaConf

from src.utils.utils import get_logger
from src.optim.param_grouping import group_parameters_for_optimizer
from src.utils.checkpoint import load_checkpoint

logger = get_logger(__name__)


class SequenceDefaultModel(LightningModule):

    def __init__(self, cfg, model_cfg=None):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model_cfg = model_cfg or self.cfg.model

        self.instantiate_datamodule()
        self.instantiate_model()
        self.warmstart()
        self.instantiate_loss()
        self.instantiate_metrics()

    def instantiate_datamodule(self):
        logger.info(f"Instantiating datamodule <{self.cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()

    def instantiate_model(self):
        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        self.model = hydra.utils.instantiate(self.model_cfg, _recursive_=False)

    def instantiate_loss(self):
        loss_fn_cfg = self.cfg.train.get('loss_fn', {'_target_': 'torch.nn.CrossEntropyLoss'})
        self.loss_fn = hydra.utils.instantiate(loss_fn_cfg)
        loss_fn_val_cfg = self.cfg.train.get('loss_fn_val', loss_fn_cfg)
        self.loss_fn_val = hydra.utils.instantiate(loss_fn_val_cfg)

    def instantiate_metrics(self):
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # NOTE: eval only affects validation and testing!
        if 'eval' in self.cfg and 'metrics' in self.cfg.eval:
            metrics_cfg = self.cfg.metrics
            metrics_cfg_eval = self.cfg.eval.metrics
        else:
            metrics_cfg = {'acc': {'_target_': 'torchmetrics.Accuracy'}}
            metrics_cfg_eval = metrics_cfg

        metrics_train = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg.items()})
        metrics_eval  = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg_eval.items()})
        self.train_metrics = metrics_train.clone(prefix='train/')
        self.val_metrics = metrics_eval.clone(prefix='val/')
        self.test_metrics = metrics_eval.clone(prefix='test/')

    def warmstart(self):
        if self.cfg.train.get('warmstart', None) is not None:
            logger.info(f"Warm-starting with weights from {self.cfg.train.warmstart.path}")
            strict = self.cfg.train.warmstart.get('strict', True)
            state_dict = load_checkpoint(self.cfg.train.warmstart.path)
            if self.cfg.train.warmstart.get('post_process', None) is not None:
                state_dict = hydra.utils.instantiate(self.cfg.train.warmstart.post_process,
                                                     state_dict)
            load_return = self.model.load_state_dict(state_dict, strict=False)
            logger.info(load_return)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        # batch x channels x height x width
        # channels = camera views x time steps. TODO: need to change this if camera views > 1
        # NOTE: DO NOT CHANGE THIS FLAG (is_train)! If you, for some reason segmentation dump will await :p
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None

        # Rollout number changes based on the number of target steps
        if is_train: rollout_n = self.cfg.datamodule.target_steps
        else: rollout_n = self.cfg.datamodule.target_steps_val_test
        
        # Perform rollout
        output = self._rollout(x, rollout_n=rollout_n, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets = self.step(batch, is_train=(phase == 'train'))
        with torch.no_grad():
            metrics = getattr(self, f'{phase}_metrics')(output, targets)
        
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # TODO log single losses
        # if loss_steps: self.log(f"{phase}/loss_steps", wandb.plot.line(wandb.Table(data=loss_steps.T.tolist(), columns=['step', 'loss']), 'step', 'loss'), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if phase in ('val', 'test'): pass
            # log images
        return {"loss": loss, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):    
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg.train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
        else:
            parameters = self.parameters() # this will train task specific parameters such as Retrieval head for AAN

        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)

        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}

    def _rollout(self, x, rollout_n=1, lengths=None):
        """Perform rollout iteratively depending rollout_n of times"""
        past_context_n = self.cfg.datamodule.context_steps - 1

        # Rollout
        if rollout_n == 1:
            output = self._one_step_rollout(x, lengths)
        else: 
            output = []
            for i in range(rollout_n):
                new_x = self._one_step_rollout(x, lengths)
                output.append(new_x)

                # Slide window forward by including past states as new context
                if new_x.ndim == 3: new_x = new_x.unsqueeze(-3) # cast
                x = torch.cat([x[:, -past_context_n:, ...], new_x], dim=-3)
            output = torch.stack(output, dim=1) # stack on channel dimension
        return output

    def _one_step_rollout(self, x, lengths=None):
            """
            Perform stepping in the following ways based on integration_order:
            - 0: x(t+1) = model(x) (predict next state)
            - 1: x(t+1) = x(t) + model(x) (predict velocity)
            - 2: x(t+1) = 2*x(t) - x(t-1) + model(x) (predict acceleration)
            """
            # Model Inference
            model_output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)

            # Integration
            order = self.cfg.train.integration_order        				
            if order == 0:
                next_x = model_output
            elif order == 1:
                x = x[:, -1]
                next_x = x + model_output
            elif order == 2:
                cur_x  = x[:, -1]
                prev_x = x[:, -2]
                next_x = 2*cur_x - prev_x + model_output
            else:
                raise ValueError("integration_order has to be in [0,1,2]")
            return next_x


class ModelwNormalizer(SequenceDefaultModel):

    def instantiate_datamodule(self):
        super().instantiate_datamodule()
        # We need to register the datamodule's y_normalizer as sub-module
        # so that it gets moved to the current device.
        self.y_normalizer = self._datamodule.y_normalizer

    def step(self, batch: Any, is_train=True):
        x, y = batch
        output = self.forward(x)
        output = self.y_normalizer.decode(output)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y

