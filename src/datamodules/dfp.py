from typing import Any, List, Union, Optional
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
import glob
import random

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning import LightningDataModule
import numpy as np

from src.datamodules.datasets.dfp import DFP6Dataset, DFPFullDataset
from src.datamodules.dfp512 import DFP512
from src.datamodules.datasets.utils import read_splits, create_file_splits


class DFP6(LightningDataModule):

    def __init__(self, data_dir, dataset_size=30000, batch_size=32, shuffle=True, pin_memory=False, drop_last=False, res=None, preprocess=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.shuffle, self.pin_memory, self.drop_last = shuffle, pin_memory, drop_last
        assert type(res) == int or res is None, "Resolution must be an integer or None."
        self.res = 128 if res == None else res
        self.preprocess = True
        
    def prepare_data(self):
        pass

    def normalize_fn(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = None
        if self.res > 128:
            transforms = torchvision.transforms.Resize((self.res, self.res), interpolation=InterpolationMode.BICUBIC)

        self.dataset_train = DFP6Dataset(self.data_dir, transform=transforms, train=True, res=self.res, preprocess=self.preprocess)
        self.dataset_test = DFP6Dataset(self.data_dir, transform=transforms, train=False, res=self.res, preprocess=self.preprocess)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ Train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ Val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ Test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


class DFPFull(DFP6):

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        transforms = None
        if self.res > 128:
            transforms = torchvision.transforms.Resize((self.res, self.res), interpolation=InterpolationMode.BICUBIC)

        self.dataset_train = DFPFullDataset(self.data_dir, transform=transforms, train=True, res=self.res, preprocess=self.preprocess)
        self.dataset_test = DFPFullDataset(self.data_dir, transform=transforms, train=False, res=self.res, preprocess=self.preprocess)

        
        min_val_size = 256
        n_data = min(self.dataset_train.data.shape[0], self.dataset_size)
        n_train, n_val = int(0.8 * n_data), max(int(0.2 * n_data), min_val_size)
        train, val = self.dataset_train.data[: n_train], self.dataset_train.data[n_train : n_train + n_val]
        self.dataset_train.data = train

        # instantiate dummy val and replace with validation data
        self.dataset_val = DFPFullDataset(self.data_dir, transform=transforms, train=True, res=self.res, preprocess=self.preprocess)
        self.dataset_val.data = val


class DFPFullOnline(DFP512):
    """
    Version of the DFPFull with sampling
    """
    def __init__(self, data_dir, dataset_size=69420, batch_size=32, shuffle=True, pin_memory=False, drop_last=False, num_workers=0, is_preprocessed=True, splits=None):
        super().__init__(data_dir)
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



if __name__ == '__main__':
    import argparse, os
    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='/data',
                             help='root path of dataset')
    parser.add_argument('--dataset', type=str, default='dfpfull',
                             help='dataset')
    args = parser.parse_args()
    
    dfpfull_path = os.path.join(args.datadir, args.dataset)

    # dm_full = DFPFull(dfpfull_path, res=224)
    # dm_full.setup()

    dm_full = DFPFullOnline(dfpfull_path)
    dm_full.setup()

    x, y = next(iter(dm_full.train_dataloader()))

    print(x.dtype)
    
    print(x.shape, y.shape)

    x, y = next(iter(dm_full.val_dataloader()))
    print(x.shape, y.shape)

    x, y = next(iter(dm_full.test_dataloader()))
    print(x.shape, y.shape)


    # dfp6_path = os.path.join(args.datadir, "dfp6")
    # dm_dfp6 = DFP6(dfp6_path, res=224)
    # dm_dfp6.setup()
    # x, y = next(iter(dm_dfp6.train_dataloader()))
    # print(x.shape, y.shape)
