from pathlib import Path
from typing import Any, List, Union, Optional

import scipy.io

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.scalarflow import ScalarFlowDataset, ScalarFlowSampler

class ScalarFlow(LightningDataModule):

    def __init__(self, data_dir, batch_size=32, context_steps=2, target_steps=1, target_steps_val_test=None,
                pin_memory=False, num_workers=0, shuffle=True, res=None, **dataset_kwargs):
                
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.context_steps = context_steps
        
        # Allow for different target steps to test rollout generalization
        self.target_steps_train = target_steps
        target_steps_val_test = target_steps_val_test if target_steps_val_test else target_steps
        self.target_steps_val = target_steps_val_test
        self.target_steps_test = target_steps_val_test

    def prepare_data(self):
        pass

    def normalize_fn(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_train = ScalarFlowDataset(self.data_dir, split='train', **self.dataset_kwargs)
        self.dataset_val = ScalarFlowDataset(self.data_dir, split='val', **self.dataset_kwargs)
        self.dataset_test = ScalarFlowDataset(self.data_dir, split='test', **self.dataset_kwargs)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> ScalarFlowSampler:
        """ Train dataloader """
        return self._dataloader(self.dataset_train, shuffle=self.shuffle, target_steps=self.target_steps_train)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[ScalarFlowSampler, List[ScalarFlowSampler]]:
        """ Val dataloader """
        return self._dataloader(self.dataset_val, target_steps=self.target_steps_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[ScalarFlowSampler, List[ScalarFlowSampler]]:
        """ Test dataloader """
        return self._dataloader(self.dataset_test, target_steps=self.target_steps_test)

    def _dataloader(self, dataset: ScalarFlowDataset, shuffle: bool = True, target_steps: int = 1):
        sampler = ScalarFlowSampler(
            dataset,
            target_steps=target_steps,
            context_steps=self.context_steps,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=None, # batch size is included in the sampler
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='/datasets/scalarflow/m1521788',
                             help='root path of dataset')
    parser.add_argument('--name', type=str, default="scalarflow",
                             help='custom dataset name')
    parser.add_argument('--preprocessed', type=bool, default=False,
                             help='preprocessed flag')
    args = parser.parse_args()

    scalarflow_path = os.path.join(args.datadir, args.name)
    scalarflow = ScalarFlow(scalarflow_path, num_workers=8, is_preprocessed=args.preprocessed)
    scalarflow.setup()

    x, y = next(iter(scalarflow.train_dataloader()))

    print(f"Input shape: {x.shape}\nOutput shape: {y.shape}")
    print("Dataloader lenght: {}".format(len(scalarflow.train_dataloader())))