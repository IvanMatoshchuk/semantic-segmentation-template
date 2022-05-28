import os

import pytest
import hydra
from hydra import compose, initialize


@pytest.mark.parametrize("model_name", os.listdir("config/model"))
def test_models(model_name: str) -> None:
    model_name_full = model_name.split(".")[0]
    with initialize(config_path="../config"):
        cfg = compose(config_name="config", overrides=[f"model={model_name_full}"], return_hydra_config=True)
        hydra.utils.instantiate(cfg.model.model_cfg)
