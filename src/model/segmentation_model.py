import torch
from torch import Tensor
from torchmetrics import Accuracy
import pytorch_lightning as pl
from sklearn.metrics import balanced_accuracy_score

import hydra
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss


class SegmentationModel(pl.LightningModule):
    """
    Pytorch-Lightning module for handling training.
    Due to functionality of hydra, passing 2 configs: 1 for initiating model and 1 for loss, optimizer and scheduler.

    For evaluating validation and testing the Dice Loss and the weighted-Accuracy are used by default.
    These two functions require input about number of classes, which is taken from 'model_cfg'.

    Parameters
    ----------
        model_cfg: DictConfig
            hydar config for initiating model
        cfg: DictConfig
            main hydra config with all 'sub-configs'. Used for initiating loss, optimizer and scheduler.
    """

    def __init__(self, model_cfg: DictConfig, cfg: DictConfig):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        self.save_hyperparameters()

        self.cfg = cfg

        self.model = hydra.utils.instantiate(model_cfg)

        self.criterion = hydra.utils.instantiate(cfg.loss)

        self.dice_loss = DiceLoss(mode="multiclass", classes=list(range(model_cfg.classes)), from_logits=True)
        self.accuracy = Accuracy(num_classes=model_cfg.classes, average="weighted")

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        pred_masks = self.model(images)

        loss = self.criterion(pred_masks, masks.long().to(self.device))
        self.log("train/loss", loss)

        accuracy = self.get_weighted_accuracy(pred_masks, masks)
        self.log("train/balanced_accuracy", accuracy)

        return {"loss": loss, "preds": pred_masks.detach(), "targets": masks.detach()}

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        pred_masks = self.model(images)

        dice_loss_value = self.dice_loss(pred_masks, masks.long().to(self.device))
        self.log("val/loss", dice_loss_value)

        accuracy = self.get_weighted_accuracy(pred_masks, masks)
        self.log("val/balanced_accuracy", accuracy)

        return {"loss": dice_loss_value, "preds": pred_masks.detach(), "targets": masks.detach()}

    def test_step(self, batch, batch_idx):
        images, masks = batch

        pred_masks = self.model(images)

        dice_loss_value = self.dice_loss(pred_masks, masks.long().to(self.device))
        self.log("test/loss", dice_loss_value)

        accuracy = self.get_weighted_accuracy(pred_masks, masks)
        self.log("test/balanced_accuracy", accuracy)

        return {"loss": dice_loss_value, "preds": pred_masks.detach(), "targets": masks.detach()}

    def configure_optimizers(self):

        params = [x for x in self.model.parameters() if x.requires_grad]

        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=params)

        if self.cfg.get("scheduler", None) is not None:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def get_weighted_accuracy(self, pred: Tensor, target: Tensor) -> float:

        pred = torch.argmax(pred, dim=1).detach()
        accuracy = self.accuracy(pred, target)

        return accuracy

    def get_balanced_accuracy(self, pred: Tensor, target: Tensor) -> float:

        pred = torch.argmax(pred, dim=1).detach()

        accuracy = balanced_accuracy_score(target.detach().cpu().numpy().flatten(), pred.cpu().numpy().flatten())

        return accuracy
