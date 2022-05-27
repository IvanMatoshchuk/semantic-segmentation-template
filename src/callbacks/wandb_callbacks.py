import subprocess
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from monai.inferers import SlidingWindowInferer


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

        Args:
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
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
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

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     """Gather data from single batch."""
    #     if self.ready:
    #         self.preds.append(outputs["preds"])
    #         self.targets.append(outputs["targets"])

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate confusion matrix."""
        if self.ready & (trainer.current_epoch % self.log_freq == 0):
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_targets = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)

            logits = self.infer(val_imgs, pl_module)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

            val_targets = val_targets.squeeze().cpu().numpy()

            # TODO: read from data folder
            class_labels = {
                0: "background",
                1: "bees",
                2: "empty_cell",
                3: "open_brood",
                4: "open_honey",
                5: "capped_honey",
                6: "capped_brood",
                7: "pollen",
                8: "bee_in_cell",
            }

            confusion_matrix = metrics.confusion_matrix(
                y_true=val_targets.flatten(), y_pred=preds.flatten(), labels=[k for k in class_labels.keys()]
            )

            ground_truth = confusion_matrix.sum(axis=1)

            cm_df = pd.DataFrame(
                confusion_matrix, index=[k for k in class_labels.values()], columns=[k for k in class_labels.values()]
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
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
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
        # TODO: Currently works only on 1 image!
        # TODO: is there a better way to specify log frequency?

        if self.ready & (trainer.current_epoch % self.log_freq == 0):
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)

            # val_imgs, val_labels = self.adjust_image_and_mask(val_imgs.squeeze(), val_labels.squeeze())

            preds = self.infer(val_imgs, pl_module)
            preds = torch.argmax(preds.squeeze(), dim=0)

            # val_imgs_blocks = self.create_blocks(val_imgs)
            # preds = self.generate_prediction(val_imgs_blocks, val_imgs, pl_module)

            # logits = torch.sigmoid(pl_module(val_imgs))
            # # preds = torch.argmax(logits, dim=-1)
            # preds = torch.argmax(logits.squeeze(), dim=0)

            # print("GLOBAL STEP: ", trainer.global_step)
            # print("Preds:", preds.size())
            # print("Logits: ", logits.size())
            # print("Val_imgs: ", val_imgs.size())
            # print("val_labels:", val_labels.size())

            # TODO: read from data folder
            class_labels = {
                0: "background",
                1: "bees",
                2: "empty_cell",
                3: "open_brood",
                4: "open_honey",
                5: "capped_honey",
                6: "capped_brood",
                7: "pollen",
                8: "bee_in_cell",
            }

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

    def adjust_image_and_mask(
        self, image: torch.tensor, mask: torch.tensor, block_size: int = 256
    ) -> Tuple[torch.tensor, torch.tensor]:

        image = image.squeeze()
        height = image.size()[0] // block_size * block_size // 32 * 32
        width = image.size()[1] // block_size * block_size // 32 * 32
        image = image[:height, :width]
        mask = mask[:height, :width]
        image = image[:height, :width]

        return image, mask

    def create_blocks(self, image: torch.tensor, block_size: int = 256) -> List[torch.tensor]:

        y_steps = int(image.shape[0] / block_size)
        x_steps = int(image.shape[1] / block_size)

        blocks = []
        for x in range(0, x_steps):
            for y in range(0, y_steps):
                block = image[y * block_size : (y + 1) * block_size, x * block_size : (x + 1) * block_size]
                blocks.append(block)

        return blocks

    def generate_prediction(
        self, blocks: List[torch.tensor], adjusted_image: torch.tensor, pl_module: Trainer, block_size: int = 256
    ) -> torch.tensor:

        # gather into batch
        blocks_stacked = torch.stack(blocks)

        # predict batch
        blocks_stacked_pred = torch.sigmoid(pl_module(blocks_stacked.unsqueeze(1)))
        blocks_stacked_pred = torch.argmax(blocks_stacked_pred, dim=1).detach()  # .cpu().numpy()
        # print("Stacked batch pred size: ", blocks_stacked_pred.size())

        # predict
        reconstructed_prediction = torch.zeros_like(adjusted_image)

        y_steps = int(adjusted_image.shape[0] / block_size)
        x_steps = int(adjusted_image.shape[1] / block_size)

        j = 0
        for x in range(0, x_steps):
            for y in range(0, y_steps):
                reconstructed_prediction[
                    y * block_size : (y + 1) * block_size, x * block_size : (x + 1) * block_size
                ] = blocks_stacked_pred[j, ...]
                j += 1

        return reconstructed_prediction
