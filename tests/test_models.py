import os

import pytest
import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.mark.parametrize("model_name", os.listdir("config/model"))
def test_models(model_name: str) -> None:
    model_name_full = model_name.split(".")[0]
    GlobalHydra.instance().clear()

    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config", overrides=[f"model={model_name_full}"], return_hydra_config=True)
        hydra.utils.instantiate(cfg.model.model_cfg)

    GlobalHydra.instance().clear()
