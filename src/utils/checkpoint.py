import re
from pathlib import Path

import torch


def load_checkpoint(path, device='cpu'):
    path = Path(path).expanduser()
    is_deepspeed = False
    if path.is_dir():  # DeepSpeed checkpoint
        is_deepspeed = True
        latest_path = path / 'latest'
        if latest_path.is_file():
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        path /= f'{tag}/mp_rank_00_model_states.pt'
    state_dict = torch.load(path, map_location=device)
    if is_deepspeed:
        state_dict = state_dict['module']

        # Replace the names of some of the submodules
        def key_mapping(key):
            return re.sub(r'^module.model.', '', key)

        state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    return state_dict