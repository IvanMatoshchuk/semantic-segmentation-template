import os

import pytest
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.mark.parametrize("logger_name", os.listdir("config/logger"))
def test_loggers(logger_name: str) -> None:
    logger_name_full = logger_name.split(".")[0]

    # Skip optional loggers that require additional dependencies
    if logger_name_full in ["wandb", "tensorboard", "multiple_loggers"]:
        pytest.skip(f"Skipping {logger_name_full} logger test (requires optional dependencies)")

    GlobalHydra.instance().clear()

    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"logger={logger_name_full}"])
        hydra.utils.instantiate(cfg.logger)

    GlobalHydra.instance().clear()
