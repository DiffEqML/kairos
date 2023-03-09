
from os.path import basename, exists, join
import glob
import random
from pathlib import Path

split_fnames = {'train': 'train.txt',
                'val': 'val.txt',
                'test': 'test.txt'}


def bytes_to_gigabytes(bytes):
    return bytes / (1024**3)


def list_to_txt_file(path, list_, include_root=False, parent_number=None):
    textfile = open(path, "w")
    for element in list_:
        if not include_root and not parent_number: 
            element = basename(element)
        elif parent_number:
            element = join(*Path(element).parts[-1-parent_number:]) # get name of parent_number directories above
        textfile.writelines(element + "\n")
    textfile.close()


def txt_file_to_list(file):
    file = open(file, 'r')
    data = file.read().splitlines()
    return data


def split_train_val_test_files(files, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=True):
    """Split data intro train, validation and test splits. Supports size args in fractions or any integer"""
    if random_state:
        random.shuffle(files)
    total_len = len(files)
    splits_fractions = train_size + valid_size + test_size
    train_idx = int(total_len * train_size / splits_fractions)
    val_idx = int(train_idx + total_len * valid_size / splits_fractions)
    return files[:train_idx], files[train_idx:val_idx], files[val_idx:]


def create_file_splits(path, file_extension='*.npz', save_txt=True, include_root=False, **split_kwargs):
    """Get train, val, test splits from files in path"""
    data_filepaths = join(path, file_extension)
    data_filepaths = sorted(glob.glob(data_filepaths))
    train, val, test = split_train_val_test_files(data_filepaths, **split_kwargs)
    if save_txt:
        list_to_txt_file(join(path, split_fnames['train']), train, include_root)
        list_to_txt_file(join(path, split_fnames['val']), val, include_root)
        list_to_txt_file(join(path, split_fnames['test']), test, include_root)
    return (train, val, test)


def read_splits(path):
    train = txt_file_to_list(join(path, split_fnames['train']))
    val   = txt_file_to_list(join(path, split_fnames['val']))
    test  = txt_file_to_list(join(path, split_fnames['test']))
    return (train, val, test)