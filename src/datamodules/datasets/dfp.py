import os
from os.path import exists
import glob
import tarfile
import gdown
from typing import Tuple
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import scipy.ndimage
from torchvision import transforms

from src.datamodules.datasets.utils import read_splits, create_file_splits, list_to_txt_file, split_fnames

def dfp6_norm_fn(data, train=True):
    """Normalizes dfp6 dataset to [0, 1].

    Notes:
        Empirical statistics:
        df6_train_data.max(axis=(0,2,3))
        >>> array([9.99706317e+01, 3.78558354e+01, 1.00000000e+00, 2.91473000e+04,                                                         │
            1.98526000e+02, 2.20345000e+02]) 
        df6_train_data.min(axis=(0,2,3))
        >>> array([ 9.28791839e+00, -3.81157590e+01,  0.00000000e+00, -4.04938000e+04,                                                     │
            -1.63852000e+02, -2.15777000e+02]) 


    """
    raise NotImplementedError("Needs to be reimplemented")


def dfpfull_norm_fn(data, train=True):
    """Normalizes dfpfull dataset to [0, 1].

    Notes:
        Empirical statistics:
        dfpfull_train_data.max(axis=(0,2,3))
        >>> array([9.99706317e+01, 3.78558354e+01, 1.00000000e+00, 2.91473000e+04,                                                         │
            1.98526000e+02, 2.20345000e+02]) 
        dfpfull_train_data.min(axis=(0,2,3))
        >>> array([ 9.28791839e+00, -3.81157590e+01,  0.00000000e+00, -4.04938000e+04,                                                     │
            -1.63852000e+02, -2.15777000e+02]) 


    """
    maxes = np.array([
        9.99706317e+01, 
        3.78558354e+01, 
        1.00000000e+00, 
        2.91473000e+04,
        1.98526000e+02,
        2.20345000e+02
        ]
    )[None, :, None, None]
    mins = np.array([
        9.28791839e+00, 
        -3.81157590e+01, 
        0.00000000e+00, 
        -4.04938000e+04,
        -1.63852000e+02, 
        -2.15777000e+02
        ]
    )[None, :, None, None]

    data = (data - mins) / (maxes - mins) 

    if train:
        err1 = abs(data.max((0,2,3)) -  np.ones(data.shape[1]))
        err0 = abs(data.min((0,2,3)) -  np.zeros(data.shape[1]))
        assert (err1 <= 1e-4).all(), f"{err1}"
        assert (err0 <= 1e-4).all(), f"{err0}"  

    return data, (mins, maxes)


class DFP6Manager:
    id = "1SsC1Fy1ijHzNm0AYsF44K8w1QNOaGLg3"

    def __init__(self, root_dir, download=True):
        self.root_dir = root_dir
        self.data_path = os.path.join(root_dir, "data")
        self.train_data_path = os.path.join(self.data_path, "train")
        self.test_data_path = os.path.join(self.data_path, "test")
        download_exists = not self.download_exists()

        self.to_download = download and download_exists

    def download(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        output_path = os.path.join(root_dir, "data.gz")
        gdown.download(id=self.id, quiet=False, output=output_path, resume=True)
    
    def extract(self, root_dir):
        
        token_folder_path = os.path.join(root_dir, '*.gz')
        matched_data_path = glob.glob(token_folder_path)
        assert len(matched_data_path) == 1, "Wrong (or no) folder downloaded from GDrive, check ID."

        with tarfile.open(matched_data_path[0]) as tf:
            tf.extractall(path=root_dir, numeric_owner=False)

    def download_exists(self):
        return exists(self.data_path)


class DFPFullManager(DFP6Manager):
    id = "1h8ICSxiHr01YV6_cZNKGLdbrzsMvJVlT"


class DFP6Dataset(Dataset):
    """Deep Flow Prediction reduced dataset."""

    def __init__(self, root_dir, train=True, transform=None, normalize=True, f32=True, res=128, preprocess=True):
        """_summary_

        Args:
            root_dir (_type_): _description_
            train (bool, optional): _description_. Defaults to True.
            transform (_type_, optional): _description_. Defaults to None.
            normalize (bool, optional): _description_. Defaults to True.
            f32 (bool, optional): _description_. Defaults to True.
            preprocess (bool, optional): If True, applies `transform` only once during setup. Avoids high cpu utilization due to online (sampled) transforms.
        """
        download_manager = DFP6Manager(root_dir=root_dir, download=True)
        self.download(download_manager, root_dir)
        norm_fn = dfp6_norm_fn if normalize else None
        self.transform =  transform
        self.f32 = f32
        self.preprocess = preprocess 
        self.res = res

        data, self.norm_stats = self.setup(root_dir, train, norm_fn)

    def setup(self, root_dir, train, norm_fn=None):
        split = "train" if train else "test"
        path = os.path.join(root_dir, 'data', f'{split}')

        path_to_stacked_file =  os.path.join(path, f'data_stack_{self.res}.npy')

        # attempt loading of stacked data chunk
        if exists(path_to_stacked_file): 
            data = np.load(path_to_stacked_file)

        # slow loading of individual data files
        else:
            data = []
            if split == "train":
                data_filepaths = os.path.join(path, '*.npz')
                data_filepaths = sorted(glob.glob(data_filepaths))
                
                for k, path in enumerate(data_filepaths):
                    with np.load(path) as data_file:

                        if type(data_file["a"]) is not None:
                            data.append(data_file["a"])
            else:
                data_filepaths = os.path.join(path, '*.npz')
                data_filepaths = sorted(glob.glob(data_filepaths))
                
                for k, path in enumerate(data_filepaths):
                    with np.load(path) as data_file:
                        # import pdb; pdb.set_trace()
                        if type(data_file["a"]) is not None:
                            data.append(data_file["a"])               

            # save as stacked for later use
            data = np.stack(data)
            data = self.remove_mask_from_xy(data)

            if norm_fn is not None: 
                data, norm_stats = norm_fn(data, train=train)

            if self.preprocess and self.transform is not None:
                data = torch.from_numpy(data)
                data = self.transform(data)

            self.data = data
            np.save(path_to_stacked_file, data)
             
        return data, None # second return should be `norm_stats``, for now unused

    def remove_mask_from_xy(self, x):
        "Remove mask from xvel and yvel (first two channels) by replacing it with the corresponding uniform velocity field"
        xvel = x[:,0,0,0].copy()
        yvel = x[:,1,0,0].copy() 

        x[:,0] = x[:,0] * 0 + xvel[:,None,None]
        x[:,1] = x[:,1] * 0 + yvel[:,None,None]
        return x


    def download(self, download_manager, root_dir):
      if download_manager.to_download:
            download_manager.download(root_dir)
            download_manager.extract(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        data = torch.tensor(self.data[idx])
        if self.f32: data = data.float()
        if self.transform is not None and self.preprocess == False:
            data = self.transform(data)
        inputs, targets = data[:3], data[3:]
        return inputs, targets


class DFPFullDataset(DFP6Dataset):
    """Deep Flow Prediction full dataset."""

    def __init__(self, root_dir, train=True, transform=None, normalize=True, f32=True, res=128, preprocess=True):
        download_manager = DFPFullManager(root_dir=root_dir, download=True)
        self.download(download_manager, root_dir)
        norm_fn = dfpfull_norm_fn if normalize else None 
        self.transform =  transform
        self.f32 = f32
        self.preprocess = preprocess
        self.res = res

        self.data, self.norm_stats = self.setup(root_dir, train, norm_fn)

    def setup(self, root_dir, train, norm_fn=None):
        "Note: normalization is only done during first setup"
        split = "train" if train else "test"

        path = os.path.join(root_dir, 'data', f'{split}')
        path_to_stacked_file =  os.path.join(path, f'data_stack_{self.res}.npy')

        # attempt loading of stacked data chunk
        if exists(path_to_stacked_file): 
            data = np.load(path_to_stacked_file)

        # slow loading of individual data files
        else:
            data = []
            if split == "train":
                for airfoil_class in ["reg", "shear"]:
                    full_path = os.path.join(path, airfoil_class)
                    data_filepaths = os.path.join(full_path, '*.npz')
                    data_filepaths = sorted(glob.glob(data_filepaths))
                    
                    for k, path in enumerate(data_filepaths):
                        with np.load(path) as data_file:
                            if type(data_file["a"]) is not None:
                                data.append(data_file["a"])
            else:
                data_filepaths = os.path.join(path, '*.npz')
                data_filepaths = sorted(glob.glob(data_filepaths))
                
                for k, path in enumerate(data_filepaths):
                    with np.load(path) as data_file:
                        if type(data_file["a"]) is not None:
                            data.append(data_file["a"])               

            # save as stacked for later use
            data = np.stack(data)
            data = self.remove_mask_from_xy(data)

            if norm_fn is not None: 
                data, norm_stats = norm_fn(data, train=train)

            if self.preprocess and self.transform is not None:
                data = torch.from_numpy(data)
                data = self.transform(data)

            np.save(path_to_stacked_file, data)
        
        return data, None # second return should be `norm_stats``, for now unused



def remove_mask_from_xy(x):
    "Remove mask from xvel and yvel (first two channels) by replacing it with the corresponding uniform velocity field"
    xvel = x[:,0,0,0].copy()
    yvel = x[:,1,0,0].copy() 

    x[:,0] = x[:,0] * 0 + xvel[:,None,None]
    x[:,1] = x[:,1] * 0 + yvel[:,None,None]
    return x


def preprocess(x):    
    x = remove_mask_from_xy(x[None]).squeeze()
    x, _ = dfpfull_norm_fn(x, train=False) # train false not to check for assertion
    return x.squeeze()


def read_data(path):
    with np.load(path) as data_file:
        if type(data_file["a"]) is not None:
            return data_file["a"]


def preprocess_file_save(origin_path, dest_folder='.'):
    x = read_data(origin_path)
    x = preprocess(x)
    filename = os.path.basename(origin_path)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    data_filepath = os.path.join(dest_folder, filename)
    np.savez(data_filepath, a=x)


def get_data_filepaths(path, split, airfoil_classes=["reg", "shear"], preprocessed=False):
    if split == "train" and not preprocessed:
        data_filepaths = []
        for airfoil_class in airfoil_classes:
            full_path = os.path.join(path, split, airfoil_class)
            data_filepaths_ = os.path.join(full_path, '*.npz')
            data_filepaths.extend(sorted(glob.glob(data_filepaths_)))
    else:
        data_filepaths = os.path.join(path, split, '*.npz')
        data_filepaths = sorted(glob.glob(data_filepaths))
    return data_filepaths


def generate_dataset_from_raw(orig_path, dest_path, splits=['train', 'test']):
    for split in splits:
        print(f"Generating {split} data...")
        data_filepaths = get_data_filepaths(orig_path, split=split)
        save_path = os.path.join(dest_path, split)
        with Pool() as p:
            _ = list(tqdm(p.imap(partial(preprocess_file_save, dest_folder=save_path), data_filepaths)))


def generate_splits_dfpfull(orig_path, dest_path, val_num=400, save_txt=True, parent_number=1):
    data_filepaths = get_data_filepaths(orig_path, split='train', preprocessed=True)
    train_num = len(data_filepaths) - val_num
    train, val , _ = create_file_splits(os.path.join(dest_path, 'train'), save_txt=False, train_size=train_num, valid_size=val_num, test_size=0)
    _, _ , test = create_file_splits(os.path.join(dest_path, 'test'), save_txt=False, train_size=0, valid_size=0, test_size=1)
    if save_txt:
        list_to_txt_file(os.path.join(dest_path, split_fnames['train']), train, parent_number=parent_number)
        list_to_txt_file(os.path.join(dest_path, split_fnames['val']), val, parent_number=parent_number)
        list_to_txt_file(os.path.join(dest_path, split_fnames['test']), test, parent_number=parent_number)
    return (train, val, test)


if __name__ == "__main__":
    d = DFPFullDataset('./data/dfpfull')
