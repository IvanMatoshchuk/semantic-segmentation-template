import os

import pytest
import torch
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.mark.parametrize("schd_name", os.listdir("config/scheduler"))
def test_scheduler(schd_name: str) -> None:
    scheduler_name = schd_name.split(".")[0]
    GlobalHydra.instance().clear()

    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"scheduler={scheduler_name}"])
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=torch.nn.Linear(1, 1).parameters())
        hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    GlobalHydra.instance().clear()
