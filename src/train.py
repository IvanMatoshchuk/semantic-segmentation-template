from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)


def train(cfg: DictConfig) -> None:
    """
    Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Parameters
    ----------
        config: DictConfig
            DictConfig composed by Hydra.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.general.get("seed"):
        log.info("setting seed")
        seed_everything(cfg.general.seed, workers=True)

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    log.info(f"Instantiating train dataset <{cfg.datamodule.dataset_args.train._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    log.info(f"Instantiating model Architecture <{cfg.model.model_cfg._target_}>")
    log.info(f"Instantiating model encoder <{cfg.model.model_cfg.encoder_name}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, cfg=cfg)

    # define Trainer
    log.info("Defining Trainer")
    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    # TODO: not implemented
    if cfg.get("test_after_training"):
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(logger=logger)

    # Print path to best checkpoint
    try:
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")
    except Exception as e:
        log.error(e)
