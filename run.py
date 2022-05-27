import hydra
from omegaconf import DictConfig

from src.train import train
from src.utils import utils


@hydra.main(config_path="config", config_name="config.yaml")
def run(cfg: DictConfig) -> None:

    # optional utilities. Currently only disables warnings
    utils.extras(cfg)

    if cfg.get("print_config"):
        utils.print_config(cfg, resolve=True)

    return train(cfg)


if __name__ == "__main__":
    run()
