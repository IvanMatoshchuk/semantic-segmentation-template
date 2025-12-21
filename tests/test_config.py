import os

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

config_files = [f.split(".")[0] for f in os.listdir("config") if "yaml" in f]


@pytest.mark.parametrize("config_name", config_files)
def test_cfg(config_name: str) -> None:
    GlobalHydra.instance().clear()

    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name=config_name)
        assert isinstance(cfg, DictConfig)

    GlobalHydra.instance().clear()
