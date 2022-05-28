import os

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

config_files = [f.split(".")[0] for f in os.listdir("config") if "yaml" in f]


@pytest.mark.parametrize("config_name", config_files)
def test_cfg(config_name: str) -> None:
    with initialize(config_path="../config"):
        cfg = compose(config_name=config_name)
        assert isinstance(cfg, DictConfig)
