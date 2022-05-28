import os

import pytest
import torch
import hydra
from hydra import compose, initialize


@pytest.mark.parametrize("opt_name", os.listdir("config/optimizer"))
def test_optimizers(opt_name: str) -> None:
    optimizer_name = opt_name.split(".")[0]
    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=[f"optimizer={optimizer_name}"])
        hydra.utils.instantiate(cfg.optimizer, params=torch.nn.Linear(1, 1).parameters())
