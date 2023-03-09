from pathlib import Path
from typing import Any, List, Union, Optional
import os
import mat73
import gdown
import zipfile

import scipy.io
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pytorch_lightning import LightningDataModule
from einops import rearrange

from src.utils.numerics import dct2d


# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):

    def __init__(self, x, eps=0.00001):
        super().__init__()
        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean


class BasicPDEManager:
    def __init__(self, root_dir, 
                id="1SsC1Fy1ijHzNm0AYsF44K8w1QNOaGLg3",
                filename='lasagne.zip',
                download=True):
        """
        Download manager for basic PDEs from the Fourier Neural Operators paper
        https://github.com/zongyi-li/fourier_neural_operator

        Args:
            root_dir (_type_): root directory to load data
            id (str, optional): Google Drive file ID. Defaults to "1SsC1Fy1ijHzNm0AYsF44K8w1QNOaGLg3".
            filename (str, optional): filename for archive downloading. Defaults to 'lasagne.zip'.
            download (bool, optional): download flag. Defaults to True.
        """
        self.id, self.filename = id, filename
        self.root_dir = root_dir
        self.output_path = os.path.join(root_dir, self.filename)
        download_exists = not self.download_exists()
        self.to_download = download and download_exists

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        gdown.download(id=self.id, output=self.output_path, quiet=False, resume=True)

    def download_exists(self):
        return os.path.exists(self.output_path)

    def extract(self):
        with zipfile.ZipFile(self.output_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)


class Burgers(LightningDataModule):

    file_name = 'burgers_data_R10.mat'

    def __init__(self, data_dir, ntrain=1000, ntest=200, subsampling_rate=1,
                 batch_size=32, shuffle=False, pin_memory=False, drop_last=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.ntrain = ntrain
        self.ntest = ntest
        self.subsampling_rate = subsampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if not (self.data_dir / self.file_name).is_file():
            if '10.' in self.file_name:
                download_manager = BasicPDEManager(self.data_dir, '16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe', 'burgersv10.zip')
            elif '100.' in self.file_name:
                download_manager = BasicPDEManager(self.data_dir, '1nzT0-Tu-LS2SoMUCcmO1qyjQd6WC9OdJ', 'burgersv100.zip')
            elif '1000.' in self.file_name:
                download_manager = BasicPDEManager(self.data_dir, '1G9IW_2shmfgprPYISYt_YS8xa87p4atu', 'burgersv1000.zip')
            else: 
                raise NameError("Dataset should include 10, 100 or 1000 in the name")
            if download_manager.to_download:
                download_manager.download()
                download_manager.extract()  

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        data = scipy.io.loadmat(self.data_dir / self.file_name)
        x_data = torch.tensor(data['a'], dtype=torch.float)[:, ::self.subsampling_rate]
        y_data = torch.tensor(data['u'], dtype=torch.float)[:, ::self.subsampling_rate]
        x_train, y_train = x_data[:self.ntrain], y_data[:self.ntrain]
        x_test, y_test = x_data[-self.ntest:], y_data[-self.ntest:]
        self.dataset_train = TensorDataset(x_train, y_train)
        self.dataset_test = TensorDataset(x_test, y_test)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


class Darcy(LightningDataModule):

    file_name_train = 'piececonst_r421_N1024_smooth1.mat'
    file_name_test = 'piececonst_r421_N1024_smooth2.mat'

    def __init__(self, data_dir, ntrain=1000, ntest=100, subsampling_rate=1,
                 batch_size=32, shuffle=False, pin_memory=False, drop_last=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.ntrain = ntrain
        self.ntest = ntest
        self.subsampling_rate = subsampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def prepare_data(self):
        if not (self.data_dir / self.file_name_train).is_file() or not (self.data_dir / self.file_name_test).is_file():
            if '241' in self.file_name_train:
                download_manager = BasicPDEManager(self.data_dir, '1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV', 'darcy241.zip')
            elif '421' in self.file_name_train:
                download_manager = BasicPDEManager(self.data_dir, '1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf', 'darcy421.zip')
            else: 
                raise NameError("Dataset should include 241 or 421 in the name")
            if download_manager.to_download:
                download_manager.download()
                download_manager.extract()  

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        data_train = scipy.io.loadmat(self.data_dir / self.file_name_train)
        data_test = scipy.io.loadmat(self.data_dir / self.file_name_test)
        rate = self.subsampling_rate
        x_train = torch.tensor(data_train['coeff'], dtype=torch.float)[:self.ntrain, ::rate, ::rate]
        y_train = torch.tensor(data_train['sol'], dtype=torch.float)[:self.ntrain, ::rate, ::rate]

        x_test = torch.tensor(data_test['coeff'], dtype=torch.float)[:self.ntest, ::rate, ::rate]
        y_test = torch.tensor(data_test['sol'], dtype=torch.float)[:self.ntest, ::rate, ::rate]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)

        self.dataset_train = TensorDataset(x_train, y_train)
        self.dataset_test = TensorDataset(x_test, y_test)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


class NavierStokes(LightningDataModule):

    def __init__(self, data_dir, ntrain=1000, ntest=200, subsampling_rate=1, viscosity=1e-3,
                 batch_size=32, target_time=40, shuffle=False, pin_memory=False, drop_last=False,
                 apply_x_dct=False, apply_y_dct=False, normalize=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        assert viscosity in [1e-3, 1e-4, 1e-5], f"Viscosity setting: {viscosity} not available."
        self.viscosity = viscosity
        self.set_file_names()
        self.viscosity = viscosity
        self.ntrain = ntrain
        self.ntest = ntest
        self.subsampling_rate = subsampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.target_time = target_time
        self.apply_x_dct = apply_x_dct
        self.apply_y_dct = apply_y_dct
        self.normalize = normalize

    def set_file_names(self):
        if self.viscosity == 1e-3:
            self.file_name = f'ns_V1e-3_N5000_T50.mat'
            self.sol_file_name = f'ns_V1e-3_N5000_T50_sol'
            self.t_span_file_name = f'ns_V1e-3_N5000_T50_t_span'
            self.ic_file_name = f'ns_V1e-3_N5000_T50_ic'
        elif self.viscosity == 1e-4:
            self.file_name = f'ns_V1e-4_N10000_T30.mat'
            self.sol_file_name = f'ns_V1e-4_N10000_T30_sol'
            self.t_span_file_name = f'ns_V1e-4_N10000_T30_t_span'
            self.ic_file_name = f'ns_V1e-4_N10000_T30_ic'   
        elif self.viscosity == 1e-5:
            self.file_name = f'NavierStokes_V1e-5_N1200_T20.mat'
            self.sol_file_name = f'ns_V1e-5_N1200_T20_sol'
            self.t_span_file_name = f'ns_V1e-5_N1200_T20_t_span'
            self.ic_file_name = f'ns_V1e-5_N1200_T20_ic'           

    def prepare_data(self):
        if not (self.data_dir / self.file_name).is_file():
            if self.viscosity == 1e-3:
                download_manager = BasicPDEManager(self.data_dir, '1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d', 'ns1e-3.zip')
            elif self.viscosity == 1e-4:
                download_manager = BasicPDEManager(self.data_dir, '1RmDQQ-lNdAceLXrTGY_5ErvtINIXnpl3', 'ns1e-3.zip')
            elif self.viscosity == 1e-5:
                download_manager = BasicPDEManager(self.data_dir, '1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5', 'ns1e-5.zip')
            if download_manager.to_download:
                download_manager.download()
                download_manager.extract()   

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        if not (self.data_dir / self.sol_file_name).is_file(): # preprocess .mat file only the first time
            if self.viscosity in [1e-3, 1e-4]:
                data = mat73.loadmat(self.data_dir / self.file_name)
            else:
               data = scipy.io.loadmat(self.data_dir / self.file_name)

            rate = self.subsampling_rate
            t_span = np.array(data['t'], dtype=int).flatten()
            x = data['a'][:, ::rate, ::rate]   
            sol = data['u'][:, ::rate, ::rate]   

            np.save(self.data_dir / self.t_span_file_name, t_span)
            np.save(self.data_dir / self.ic_file_name, x)
            np.save(self.data_dir / self.sol_file_name, sol)

        else:
            t_span = np.load(self.data_dir / self.t_span_file_name)
            x = np.load(self.data_dir / self.ic_file_name)
            sol = np.load(self.data_dir / self.sol_file_name)  

        # downsample
        x = x[:, ::rate, ::rate]   
        sol = sol[:, ::rate, ::rate]  

        # select solution at desired time (`target_time`) as `y`
        sol_idx = np.where(t_span == self.target_time)[0][0]
        y = sol[..., sol_idx]

        x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

        if self.apply_x_dct: x = dct2d(x, norm="ortho")
        if self.apply_y_dct: y = dct2d(y, norm="ortho")

        x_train, x_test = x[:self.ntrain], x[-self.ntest:]
        y_train, y_test = y[:self.ntrain], y[-self.ntest:]

        if self.normalize:
            x_normalizer = UnitGaussianNormalizer(x_train)
            x_train = x_normalizer.encode(x_train)
            x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)

        self.dataset_train = TensorDataset(x_train, y_train)
        self.dataset_test = TensorDataset(x_test, y_test)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )


if __name__ == '__main__':
    dmodule = NavierStokes('~/datasets', viscosity=1e-5, target_time=20)
    dmodule.prepare_data()
    dmodule.setup()
    # import pdb; pdb.set_trace()
    x, y = next(iter(dmodule.train_dataloader()))
    print(x.shape, y.shape)
    x, y = next(iter(dmodule.test_dataloader()))
    print(x.shape, y.shape)