from os.path import basename, exists, join
import glob
import random
from dataclasses import dataclass
from more_itertools import chunked
from typing import Tuple, Optional, List
from pandas import describe_option
from tqdm import tqdm
from pathlib import Path
import math
from sys import getsizeof

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

from src.datamodules.datasets.utils import bytes_to_gigabytes


class DFP512Manager:
    def __init__(self):
        raise NotImplementedError


class DFP512Normalizer:
    def __init__(self):
        raise NotImplementedError("Needs to be reimplemented")
    

@dataclass(frozen=True)
class DFP512BatchKey:
    """
    Custom key for querying the dataset
    """
    data_files: List[str]
    description_files: List[Optional[str]] = None # for future use
    image_files: List[Optional[str]] = None # for future use


class DFP512Dataset(Dataset):
    """
        DFP airfoil dataset with size 512x512
    """
    def __init__(self, root_dir, is_preprocessed=False, transform=None, normalize=False, f32=True, save_cache=False, max_cache_size=16, download=False):
        """_summary_

        Args:
            root_dir (_type_): _description_
            is_preprocessed (bool, optional): if True, skip (=override) preprocessing since data is already preprocessed. Defaults to False.
            transform (_type_, optional): _description_. Defaults to None.
            normalize (bool, optional): _description_. Defaults to True.
            f32 (bool, optional): _description_. Defaults to True.
            save_cache (bool, optional): If True, save loaded data into a cache to reduce loading time. Defaults to True.
            max_cache_size (float, optional): cache size in GB to save loading times. Defaults to 16.
            download (bool, optional): download dataset. Defaults to False.
        """
        # download_manager = DFP512Manager(root_dir=Path(root_dir), download=download)
        # self.download(download_manager, root_dir)
        
        # Path
        self.root_dir = root_dir
        
        # Preprocessing
        self.is_preprocessed = is_preprocessed # if True, skip the rest
        self.norm_fn = DFP512Normalizer() if normalize else None
        self.transform =  transform
        self.f32 = f32

        self.cache = {} 
        self.save_cache = save_cache
        self.max_cache_size = max_cache_size

    def download(self, download_manager, root_dir):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: DFP512BatchKey) -> Tuple[Tensor, Tensor]:
        """_summary_"""
        data_full = []
        
        # Try loading from cache
        for file in key.data_files:
            data = self.cache.get(file) 
            if data is None:
                data = torch.tensor(np.load(join(self.root_dir, file))["a"])
            if self.save_cache and bytes_to_gigabytes(getsizeof(self.cache)) < self.max_cache_size: # cache saving
                self.cache[file] = data # save in cache
            if not self.is_preprocessed: 
                if self.f32: data = data.float()
                if self.transform is not None:
                    data = self.transform(data)
            data_full.append(data)
        data_full = torch.stack(data_full, dim=0)
        inputs, targets = data_full[:, :3], data_full[:, 3:]
        return inputs.float(), targets.float()


class DFP512Sampler(Sampler):
    def __init__(
        self,
        dataset: DFP512Dataset,
        file_names: list,
        batch_size: int,
        shuffle: bool,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_names = file_names

    def __iter__(self):
        file_names = self.file_names
        if self.shuffle:
            file_names = file_names.copy()
            random.shuffle(file_names)
        for chunk in chunked(file_names, self.batch_size):
            yield DFP512BatchKey(data_files=chunk)

    def __len__(self):
        return math.ceil(len(self.file_names) / self.batch_size)


