import os

import pytest
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.mark.parametrize("callback_name", os.listdir("config/callbacks"))
def test_callbacks(callback_name: str) -> None:
    callback_name_full = callback_name.split(".")[0]

    # Skip wandb callback test as it requires HydraConfig to be fully initialized
    if callback_name_full == "wandb":
        pytest.skip("Skipping wandb callback test (requires full Hydra initialization)")

    GlobalHydra.instance().clear()

    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"callbacks={callback_name_full}"])
        hydra.utils.instantiate(cfg.callbacks)

    GlobalHydra.instance().clear()
