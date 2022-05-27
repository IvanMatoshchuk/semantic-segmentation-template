import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from monai.inferers import SlidingWindowInferer
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from src.utils.utils import read_label_classes_config

project_path = Path(__file__).parent.parent.parent
path_to_label_classes = os.path.join(project_path, "data", "label_classes.json")


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Parameters
        ----------
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1)  # noqa: W503
                ):
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """
    Generate confusion matrix and send it to wandb.
    Apply sliding window inference on the validation image to get the inferred mask.

    Parameters
    ----------
        log_freq: int
            Frequency of logging
    """

    def __init__(self, log_freq: int):
        self.preds = []
        self.targets = []
        self.ready = True

        self.log_freq = log_freq
        self.infer = SlidingWindowInferer(roi_size=(224, 224), sw_batch_size=16, overlap=0.3, mode="gaussian")

    @rank_zero_only
    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    @rank_zero_only
    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Use validation dataloader to get the batch.
        Apply sliding window inference.
        Generate confusion matrix.

        Parameters
        ----------
            log_freq: int
                Frequency of logging
        """
        if self.ready & (trainer.current_epoch % self.log_freq == 0):
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # Currently uploading only the first image in batch!
            val_imgs = val_imgs[0, ...].unsqueeze(0)
            val_labels = val_labels[0, ...].unsqueeze(0)

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            with torch.no_grad():
                logits = self.infer(val_imgs, pl_module)
            print("\nlogits:", logits.size())
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

            val_labels = val_labels.squeeze().cpu().numpy()

            class_labels = read_label_classes_config(path_to_label_classes)

            confusion_matrix = metrics.confusion_matrix(
                y_true=val_labels.flatten(), y_pred=preds.flatten(), labels=[k for k in class_labels.values()]
            )

            ground_truth = confusion_matrix.sum(axis=1)

            cm_df = pd.DataFrame(
                confusion_matrix, index=[k for k in class_labels.keys()], columns=[k for k in class_labels.keys()]
            )
            cm_df.index.name = "ground truth ➡"

            # which percentage of predictions correspond to which ground-truth class
            cm_df_normalized = cm_df / cm_df.sum(axis=0)
            cm_df_normalized["Accuracy"] = np.round(confusion_matrix.diagonal() / ground_truth, 3)

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sns.set(font_scale=1.0)

            # set font size
            sns.heatmap(cm_df_normalized, annot=True, annot_kws={"size": 6}, fmt=".2%")

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log({f"confusion_matrix/{experiment.name}": wandb.Image(plt)}, commit=False)

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """
    Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY

    Apply sliding window inference on the validation image to get the inferred mask.
    Currently works on the first image in the batch!

    Parameters
    ----------
        log_freq: int
            Frequency of logging
    """

    def __init__(self, log_freq: int = 10):
        super().__init__()
        self.ready = True
        self.log_freq = log_freq

        self.infer = SlidingWindowInferer(roi_size=(224, 224), sw_batch_size=16, overlap=0.3, mode="gaussian")

    @rank_zero_only
    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    @rank_zero_only
    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if self.ready & (trainer.current_epoch % self.log_freq == 0):
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # Currently uploading only the first image in batch!
            val_imgs = val_imgs[0, ...].unsqueeze(0)
            val_labels = val_labels[0, ...].unsqueeze(0)

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            with torch.no_grad():
                preds = self.infer(val_imgs, pl_module)
            preds = torch.argmax(preds.squeeze(), dim=0)

            # TODO: read from data folder
            label_classes = read_label_classes_config(path_to_label_classes)
            class_labels = {v: k for k, v in label_classes.items()}

            # TODO: make for loop
            experiment.log(
                {
                    "my_image_key": wandb.Image(
                        val_imgs,
                        masks={
                            "predictions": {
                                "mask_data": preds.detach().cpu().numpy(),
                                "class_labels": class_labels,
                            },
                            "ground_truth": {
                                "mask_data": val_labels.squeeze().detach().cpu().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    )
                }
            )
