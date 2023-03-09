from typing import Callable

import dotenv
import hydra
from omegaconf import OmegaConf, DictConfig

import matplotlib

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def dictconfig_filter_key(d: DictConfig, fn: Callable) -> DictConfig:
    """Only keep keys where fn(key) is True. Support nested DictConfig.
    """
    return DictConfig({k: dictconfig_filter_key(v, fn) if isinstance(v, DictConfig) else v
                       for k, v in d.items() if fn(k)})


@hydra.main(config_path="configs/", config_name="default_config")
def main(config: DictConfig):

    # fix for: _tkinter.TclError: no display name and no $DISPLAY environment variable
    import matplotlib
    matplotlib.use('Agg')

    # Remove config keys that start with '__'. These are meant to be used only in computing
    # other entries in the config.
    config = dictconfig_filter_key(config, lambda k: not k.startswith('__'))
    
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.eval import evaluate
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    mode = config.get('mode', 'train')
    if mode not in ['train', 'eval']:
        raise NotImplementedError(f'mode {mode} not supported')
    if mode == 'train':
            return train(config)
    elif mode == 'eval':
        return evaluate(config)


if __name__ == "__main__":
    main()
