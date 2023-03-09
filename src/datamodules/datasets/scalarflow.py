import os
from os.path import exists
from pathlib import Path
from typing import Optional, List, Tuple
import glob
import tarfile
import shutil 
from sys import getsizeof

import gdown
from more_itertools import chunked
import math
import random
from dataclasses import dataclass
from tqdm.auto import tqdm
import re
import PIL

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
import scipy.ndimage
from torchvision import transforms



class ScalarFlowManager:
    id = "1B_XJUvpmT7dAY0XTsq7o-FFxg1rTX1a1"

    def __init__(self, root_dir, target_dir=None, split_idxs={'train': 63, 'val': 83}, download=True, transform=None, normalizer=None, crop_images=True, camera_views='center', f32=True):
        self.root_dir = Path(root_dir)
        target_dir = Path(target_dir) if target_dir is not None else root_dir
        self.train_data_path = target_dir / "train"
        self.val_data_path = target_dir /"val"
        self.test_data_path = target_dir / "test"
        download_exists = not self.download_exists()

        self.file_re = re.compile("^.*/imgsTarget_[0-9]{6}.npz$")
        self.split_idxs = split_idxs
        self.to_download = download and download_exists
        self.transform = transform
        self.normalizer = normalizer
        self.crop_images = crop_images
        self.camera_views = self._setup_camera_views(camera_views)
        self.f32 = f32

    def download(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        output_path = os.path.join(root_dir, "data.tar.xz")
     
    def move_individual(self, folder, data_path, sim_idx):
        """Extract data from archive and save it in corresponding split"""
        file_paths = list(glob.glob(os.path.join(folder, "*.npz")))
        new_sim_path = os.path.join(data_path, f"sim_{sim_idx:06d}")
        os.makedirs(new_sim_path, exist_ok=True)

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            new_file_path = os.path.join(new_sim_path, file_name)
            shutil.copy(file_path, new_file_path)


    def extract(self, data_folder):
        """Move ScalarFlow individual simulations (one for each folder) into train/ and test/"""
        # Extract only if first-level folder in downloaded tar does not exist
        if not exists(os.path.join(data_folder, "m1521788")):
            data_path = os.path.join(data_folder, 'data.tar.xz')
            with tarfile.open(data_path) as tf:
                tf.extractall(path=data_folder, numeric_owner=False)

        sim_folder = os.path.join(data_folder, "m1521788")
        sim_folders = glob.glob(os.path.join(sim_folder, "sim_*"))
        idx = lambda archive: int(os.path.basename(archive)[4:])

        train_folders = [arch for arch in sim_folders if idx(arch) <= self.split_idxs['train']]
        val_folders = [arch for arch  in sim_folders if self.split_idxs['train'] < idx(arch) <= self.split_idxs['val']]
        test_folders = [arch for arch in sim_folders if self.split_idxs['val'] < idx(arch)]

        for folder in tqdm(train_folders, desc='Moving training data'):   
            os.makedirs(self.train_data_path, exist_ok=True)
            self.move_individual(folder, self.train_data_path, idx(folder))
        for folder in tqdm(val_folders, desc='Moving validation data'):
            os.makedirs(self.val_data_path, exist_ok=True)
            self.move_individual(folder, self.val_data_path, idx(folder))
        for folder in tqdm(test_folders, desc='Moving test data'): 
            os.makedirs(self.test_data_path, exist_ok=True)
            self.move_individual(folder, self.test_data_path, idx(folder))

    def download_exists(self):
        return exists(self.train_data_path) and exists(self.test_data_path)

    def _get_camera_views(self, image, camera_views):
        """
        Get specific camera views of the 5 cameras
        Input: [batch] x 5 x 1062 x 600
        """
        return np.take(image, camera_views, axis=-3)
        
    def _setup_camera_views(self, camera_views):
        if camera_views == 'all': return np.array([0,1,2,3,4])
        elif camera_views == 'center': return np.array([2])
        elif camera_views == 'sides_external': return np.array([0,2,4])
        elif camera_views == 'sides_internal': return np.array([1,2,3])
        else: return np.array(camera_views)


class ScalarFlowNormalizer:
    """Normalizes ScalarFlow dataset to [0, 1].

    Notes:
        Empirical statistics from training data (recordings from 0 to 63)
    """
    def __init__(self):
        self.maxes = torch.Tensor([
            4.3574872,
            4.3590126,
            4.4561429,
            4.42352962,
            4.29803944]
        )[None, None, :, None, None]
        self.mins = torch.Tensor([
            0,
            0,
            0,
            0,
            0]  
        )[None, None, :, None, None]

    def __call__(self, data):
        data = (data - self.mins) / (self.maxes - self.mins) 
        return data


@dataclass(frozen=True)
class ScalarFlowBatchKey:
    """
    Custom key for class for querying the dataset
    seq_ranges: (simulation folder,  (sequence start, sequence end)
    """
    seq_ranges: List[Tuple[int, Tuple[Optional[int], Optional[int]]]]
    context_steps: int


class ScalarFlowDataset(Dataset):
    """
    Denoised recordings from the ScalarFlow dataset consisting of 5 camera views 
    at different angles of rising smoke plumes collected by Eckert et al. [1]

    Here we consider only the original data (images) without reconstructions. 
    Data composition:
    - 104 real smoke plumes with 150 frames each (2s)
    - each frame is made of 2D Input images with size 600 × 1062 x 5 (camera angles)
    
    We crop the interesting portion in the middle to 360 x 840 for each image

    [1] Marie-Lena Eckert, Kiwon Um, Nils Thuerey, "ScalarFlow: A Large-Scale Volumetric
        Data Set of Real-world Scalar Transport Flows for Computer Animation and Machine
        Learning"
    """
    def __init__(self, root_dir, split='train', is_preprocessed=False, transform=None, normalize=False, f32=True, crop=True, camera_views='all', stack_on_channels=True, save_cache=True, max_cache_size=16, download=False):
        """_summary_

        Args:
            root_dir (_type_): _description_
            split (_type_): _description_. Defaults to train.
            is_preprocessed (bool, optional): if True, skip (=override) preprocessing since data is already preprocessed. Defaults to False.
            transform (_type_, optional): _description_. Defaults to None.
            normalize (bool, optional): _description_. Defaults to True.
            f32 (bool, optional): _description_. Defaults to True.
            crop (bool, optional): If True, crops image to interesting part to 360x840. Defaults to True.
            camera_views (str or List, optional): get views from certain camera on the total of 5. Defaults to all (all the cameras).
            stack_on_channels (bool, optional): If True, stack all context steps and camera views on the channel dimension. Defaults to True.
            save_cache (bool, optional): If True, save loaded data into a cache to reduce loading time. Defaults to True.
            max_cache_size (float, optional): cache size in GB to save loading times. Defaults to 16.
        """
        download_manager = ScalarFlowManager(root_dir=Path(root_dir), download=download)
        self.download(download_manager, root_dir)
        
        # Path
        assert split in ['train', 'val', 'test'], "Split should be one of train, val or test"
        root = Path(root_dir) / split
        self.root = root
        
        # Preprocessing
        self.is_preprocessed = is_preprocessed # if True, skip the rest
        self.norm_fn = ScalarFlowNormalizer() if normalize else None
        self.transform =  transform
        self.f32 = f32
        self.crop = crop
        self.camera_views = self._setup_camera_views(camera_views)

        # Temporal and context dimension
        self.stack_on_channels = stack_on_channels

        self.cache = {} 
        self.save_cache = save_cache
        self.max_cache_size = max_cache_size

        # All recordings are 2 seconds long and have 150 steps
        self.n_steps = 150
        self.t = torch.linspace(0.0, 2.0, self.n_steps, dtype=torch.float32)
        self.sim_idxs = [int(f.stem[4:]) for f in self.root.glob("sim_*/")]
        
    def download(self, download_manager, root_dir):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: ScalarFlowBatchKey) -> Tuple[Tensor, Tensor]:
        """Sizes: batch x timesteps x camera views x height x width"""
        data = []

        # Get simulation from index and elements in a range of context and target steps
        for sim_idxs, start_end in key.seq_ranges:
            seq = []
            for time_idx in range(*start_end):
                frame = self.cache.get((sim_idxs, time_idx)) # load from cache to improve loading time
                if frame is None:
                    frame = np.load(self.root / f"sim_{sim_idxs:06d}" / f"imgsTarget_{time_idx:06d}.npz")["data"]
                    frame = frame.squeeze(-1) if frame.shape[-1] == 1 else frame # squeeze last dim (which is useless in dataset)
                    # Save to cache: we do it here because seq variable may contain repeated frames which increases memory consumption
                    if self.save_cache and bytes_to_gigabytes(getsizeof(self.cache)) < self.max_cache_size: # cache saving
                        self.cache[sim_idxs, time_idx] = frame
                seq.append(frame)
            seq = torch.from_numpy(np.stack(seq, axis=0))
            
            # Processing (NOTE: better build the transformed dataset beforehand to reduce loading times)
            # We do this on one sequence only because tranforms don't work on bigger batches
            if not self.is_preprocessed:
                if self.f32: seq = seq.float()
                if self.crop: seq = crop_frame(seq)
                if self.transform is not None: seq = self.transform(seq)
                if self.norm_fn: seq = self.norm_fn(seq)
                if self.camera_views is not None: seq = self._get_camera_views(seq)
            data.append(seq)
        data = torch.stack(data, dim=0)
        
        # Inputs and outputs
        inputs, targets = data[:, : key.context_steps, ...], data[:, key.context_steps :, ...]
        if self.stack_on_channels: inputs, targets = self._channel_reshape(inputs), self._channel_reshape(targets)
        return inputs, targets

    def download(self, download_manager, root_dir):
      if download_manager.to_download:
            download_manager.download(root_dir)
            download_manager.extract(root_dir)

    def _get_camera_views(self, image):
        """
        Get specific camera views of the 5 cameras
        Input: [batch] x 5 x 1062 x 600
        """
        return torch.index_select(image, dim=-3, index=self.camera_views)
        
    def _setup_camera_views(self, camera_views):
        if camera_views == 'all': return torch.tensor([0,1,2,3,4])
        elif camera_views == 'center': return torch.tensor([2])
        elif camera_views == 'sides_external': return torch.tensor([0,2,4])
        elif camera_views == 'sides_internal': return torch.tensor([1,2,3])
        elif camera_views == None: return None
        else: return torch.tensor(camera_views)

    def _channel_reshape(self, x):
        """
        Reshape dimensions into image channel (c x )
        Input dims: batch x timesteps x camera views x height x width  
        Output dims: batch x channels x height x widht
        """
        bdim, tdim, cdim, hdim, wdim = x.shape
        return x.reshape(bdim, tdim * cdim, hdim, wdim).squeeze(1)


class ScalarFlowSampler(Sampler):
    def __init__(
        self,
        dataset: ScalarFlowDataset,
        target_steps: int,
        context_steps: int,
        batch_size: int,
        shuffle: bool,
    ):
        super().__init__(dataset)

        self.dataset = dataset
        self.target_steps = target_steps # prediction target
        self.context_steps = context_steps # current + past: e.g. 1 is current only
        self.batch_size = batch_size
        self.shuffle = shuffle
        seq_len = target_steps + context_steps
        self.indices = [
            (sim_idx, (start, start + seq_len))
            for sim_idx in self.dataset.sim_idxs
            for start in range(self.dataset.n_steps - seq_len)
        ]

    def __iter__(self):
        indices = self.indices
        if self.shuffle:
            indices = indices.copy()
            random.shuffle(indices)
        for chunk in chunked(indices, self.batch_size):
            yield ScalarFlowBatchKey(seq_ranges=chunk, context_steps=self.context_steps)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)


def crop_frame(image, coords={'x_min': 120, 'x_max': 480, 'y_min': 0, 'y_max': 840}):
    """
    Crop ScalarFlow image input according to interesting coordinates
    Input dimensions: [batch] x 5 x 1062 x 600
    Output: [batch] x 5 x (y_max-y_min) x (x_max-x_min)
    Defaults to [batch] x 5 x 840 x 360
    """
    image = image[..., coords['y_min']:coords['y_max'], coords['x_min']:coords['x_max']]
    return image


def bytes_to_gigabytes(bytes):
    return bytes / (1024**3)