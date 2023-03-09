from typing import Any, List, Union, Optional
import numpy as np
from os.path import join

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.dfp512 import DFP512Dataset, DFP512Sampler
from src.datamodules.datasets.utils import split_fnames, read_splits, create_file_splits

class DFP512(LightningDataModule):

    def __init__(self, data_dir, dataset_size=30000, batch_size=32, shuffle=True, pin_memory=False, drop_last=False, num_workers=0, is_preprocessed=True, splits=None):
        super().__init__()
        self.data_dir = data_dir
        if not splits:
            try:
                splits = read_splits(self.data_dir)
            except:
                print("Training split not found. Generating random splits...")
                splits = create_file_splits(self.data_dir)

        self.train_files, self.val_files, self.test_files = splits
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.shuffle, self.pin_memory, self.drop_last, self.num_workers= shuffle, pin_memory, drop_last, num_workers
        self.is_preprocessed = is_preprocessed
        
    def prepare_data(self):
        pass

    def normalize_fn(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = None
        # The dataset is always the same, we change train/val/test via the sampler
        self.dataset = DFP512Dataset(self.data_dir, transform=transforms, is_preprocessed=self.is_preprocessed)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ Train dataloader """
        return self._data_loader(self.dataset, self.train_files, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ Val dataloader """
        return self._data_loader(self.dataset, self.val_files)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ Test dataloader """
        return self._data_loader(self.dataset, self.test_files)

    def _data_loader(self, dataset: Dataset, files: list, shuffle: bool = False) -> DataLoader:
        sampler = DFP512Sampler(
            dataset,
            files,
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
    parser.add_argument('--datadir', type=str, default='/datasets',
                             help='root path of dataset')
    parser.add_argument('--name', type=str, default="dfp512",
                             help='custom dataset name')
    parser.add_argument('--preprocessed', type=bool, default=False,
                             help='preprocessed flag')
    args = parser.parse_args()

    data_path = join(args.datadir, args.name)
    dataset = DFP512(data_path, num_workers=8, is_preprocessed=args.preprocessed)
    dataset.setup()

    x, y = next(iter(dataset.train_dataloader()))

    print(f"Input shape: {x.shape}\nOutput shape: {y.shape}")
    print("Train Dataloader lenght: {}".format(len(dataset.train_dataloader())))
