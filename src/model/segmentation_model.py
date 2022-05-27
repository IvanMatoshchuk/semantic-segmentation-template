import torch
from torch import Tensor
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss
from monai.inferers import SlidingWindowInferer


class HoneyBeeModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, cfg: DictConfig):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        self.save_hyperparameters()

        self.cfg = cfg

        self.model = hydra.utils.instantiate(model_cfg)

        self.criterion = hydra.utils.instantiate(cfg.loss)
        self.dice_loss = DiceLoss(mode="multiclass", classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], from_logits=True)

        self.inferer = SlidingWindowInferer(
            roi_size=(416, 416), overlap=0, sw_batch_size=16, mode="gaussian", sigma_scale=0.125
        )

        # print("\n **** Criterion: ", self.criterion, "\n ****")

    def forward(self, images: Tensor) -> Tensor:
        # print("\n **** Images size: ", images.size(), "\n ****")
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        # print("\n **** Images train size: ", images.size(), "\n ****")

        pred_masks = self.model(images)
        # pred_masks = self.inferer(images, self.model)

        loss = self.criterion(pred_masks, masks.long().to(self.device))
        self.log("train/loss", loss)

        accuracy = self.get_accuracy(pred_masks, masks)
        self.log("train/accuracy", accuracy)
        # print("\n Train Accuracy: ", accuracy, "\n")

        return {"loss": loss, "preds": pred_masks.detach(), "targets": masks.detach()}

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        pred_masks = self.model(images)
        # pred_masks = self.inferer(images, self.model)

        dice_loss_value = self.dice_loss(pred_masks, masks.long().to(self.device))
        self.log("val/loss", dice_loss_value)

        accuracy = self.get_accuracy(pred_masks, masks)
        self.log("val/accuracy", accuracy)

        return {"loss": dice_loss_value, "preds": pred_masks.detach(), "targets": masks.detach()}

    def test_step(self, batch, batch_idx):
        images, masks = batch

        pred_masks = self.model(images)
        # pred_masks = self.inferer(images, self.model)

        dice_loss_value = self.dice_loss(pred_masks, masks.long().to(self.device))
        self.log("test/loss", dice_loss_value)

        accuracy = self.get_accuracy(pred_masks, masks)
        self.log("test/accuracy", accuracy)

        return {"loss": dice_loss_value, "preds": pred_masks.detach(), "targets": masks.detach()}

    def configure_optimizers(self):

        params = [x for x in self.model.parameters() if x.requires_grad]

        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params=params)

        if self.cfg.get("scheduler", None) is not None:
            scheduler = hydra.utils.instantiate(self.cfg.scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def get_accuracy(self, pred: Tensor, target: Tensor) -> float:

        pred = torch.argmax(pred, dim=1).detach()  # .cpu()
        accuracy = torch.mean((pred == target).float())

        return accuracy
