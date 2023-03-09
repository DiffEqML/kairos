# Fix the bug where Deepspeed store all the checkpoints, wasting disk space
# https://github.com/PyTorchLightning/pytorch-lightning/issues/11687
from typing import Dict

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import _METRIC


class ModelCheckpointMine(pl.callbacks.model_checkpoint.ModelCheckpoint):

    def _save_last_checkpoint(self, trainer: pl.Trainer, monitor_candidates: Dict[str, _METRIC]) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        #if previous:
        #    trainer.strategy.remove_checkpoint(previous)
        trainer.save_checkpoint(filepath, self.save_weights_only)
