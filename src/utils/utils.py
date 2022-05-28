import json
import warnings
import rich.syntax
import rich.tree

import logging
from typing import List, Sequence
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import loggers


def read_label_classes_config(path_to_label_classes: str) -> dict:
    """
    Read json-config with label-classes located in "data" folder.
    """
    with open(path_to_label_classes, "r") as f:
        label_classes = json.load(f)
    return label_classes


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "general",
        "trainer",
        "model",
        "loss",
        "datamodule",
        "callbacks",
        "logger",
        "optimizer",
        "scheduler",
        "test_after_training",
        "ignore_warnings",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def finish(logger: List[loggers.LightningLoggerBase]) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
