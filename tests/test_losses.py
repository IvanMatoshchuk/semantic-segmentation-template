import os

import pytest
import hydra
from hydra import compose, initialize


@pytest.mark.parametrize("loss_name", os.listdir("config/loss"))
def test_losses(loss_name: str) -> None:
    loss_name_full = loss_name.split(".")[0]
    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=[f"loss={loss_name_full}"])
        hydra.utils.instantiate(cfg.loss)
