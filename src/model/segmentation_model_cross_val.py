import torch
from torch import Tensor

import pytorch_lightning as pl

from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch import Unet
from src.losses.losses import DiceWithCE


class HoneyBeeModelCrossVal(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = Unet(
            encoder_name="efficientnet-b0", encoder_weights="imagenet", encoder_depth=5, classes=9, in_channels=1
        )

        self.dice_loss = DiceLoss(mode="multiclass", classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], from_logits=True)
        self.criterion = DiceWithCE(
            self.dice_loss, class_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dice_weight=0.3, ce_weight=0.7
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

    def configure_optimizers(self):

        params = [x for x in self.model.parameters() if x.requires_grad]

        optimizer = torch.optim.Adam(params=params, lr=3e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            T_0=150, T_mult=3, eta_min=0.000001, optimizer=optimizer
        )

        return [optimizer], [scheduler]

    def get_accuracy(self, pred: Tensor, target: Tensor) -> float:

        pred = torch.argmax(pred, dim=1).detach()  # .cpu()
        accuracy = torch.mean((pred == target).float())

        return accuracy
