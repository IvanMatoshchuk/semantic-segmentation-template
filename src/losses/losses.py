from typing import List

import numpy as np
from scipy import signal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from monai.losses import GeneralizedDiceLoss


class DiceWithCE(nn.Module):
    """
    Dice with Cross-Entropy Loss.
    L = dice_weight * Dice Loss + ce_weight * Cross-Entropy Loss

    Parameters
    ----------
        dice_loss: DiceLoss
            initiated DiceLoss (e. g. from segmentation_models_pytorch.DiceLoss).
        class_weights: List[float]
            weights used in Cross-Entropy Loss (per class).
        dice_weight: float
            weight of the Dice Loss.
        ce_weight: float
            weight of the Cross-Entropy Loss.
    """

    def __init__(self, dice_loss: DiceLoss, class_weights: List[float], dice_weight: float, ce_weight: float):

        super().__init__()

        self.criterion_dice = dice_loss
        self.criterion_ce = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: Tensor, target: Tensor) -> float:

        loss = self.dice_weight * self.criterion_dice(pred, target) + self.ce_weight * self.criterion_ce(pred, target)

        return loss


class FocalWithCE(nn.Module):
    """
    Focal with Cross-Entropy Loss.
    L = focal_weight * Focal Loss + ce_weight * Cross-Entropy Loss

    Parameters
    ----------
        focal_loss: FocalLoss
            initiated FocalLoss (e. g. from segmentation_models_pytorch.FocalLoss).
        class_weights: List[float]
            weights used in Cross-Entropy Loss (per class).
        focal_weight: float
            weight of the Focal Loss.
        ce_weight: float
            weight of the Cross-Entropy Loss.
    """

    def __init__(self, focal_loss: FocalLoss, class_weights: List[float], focal_weight: float, ce_weight: float):

        super().__init__()

        self.criterion_focal = focal_loss
        self.criterion_ce = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

    def forward(self, pred: Tensor, target: Tensor) -> float:

        loss = self.focal_weight * self.criterion_focal(pred, target) + self.ce_weight * self.criterion_ce(
            pred, target
        )

        return loss


class DiceCEGausWeighted(nn.Module):
    """
    Dice with Cross-Entropy Loss and possibility for the Gaussian weighting of the input's center.
    Gaussian weighting is applied on Cross-Entropy Loss.
    For this, it is necessary to set 'reduciton="none"' for Cross-Entropy Loss.

    Parameters
    ----------
        dice_loss: DiceLoss
            initiated DiceLoss (e. g. from segmentation_models_pytorch.DiceLoss).
        class_weights_ce: List[float]
            weights used in Cross-Entropy Loss (per class).
        kernel_size: int
            size of the Gaussian kernel in pixels. Kernel will be equal to kernel_size x kernel_size.
            Standard deviation of the kernel will be equal to kernel_size // 4.
        dice_weight: float
            weight of the Dice Loss.
        ce_weight: float
            weight of the Cross-Entropy Loss.
    """

    def __init__(
        self,
        dice_loss: DiceLoss,
        class_weights_ce: List[float],
        kernel_size: int = 256,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
    ):

        super().__init__()
        self.criterion_dice = dice_loss

        self.criterion_ce_no_reduction = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights_ce), reduction="none"
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # create gaussian kernel
        gaus_kernel = self.create_gaus_kernel(kernel_size=kernel_size, std=kernel_size // 4)
        self.gaus_kernel = torch.cuda.FloatTensor(gaus_kernel).unsqueeze(0)

    def forward(self, pred: Tensor, target: Tensor) -> float:

        loss_ce = self.criterion_ce_no_reduction(pred, target)
        loss_ce = loss_ce * self.gaus_kernel.to(pred.device)

        # calculate total loss
        loss = self.dice_weight * self.criterion_dice(pred, target) + self.ce_weight * torch.mean(loss_ce)

        return loss

    def create_gaus_kernel(self, kernel_size=20, std=10) -> np.ndarray:
        """Returns a 2D Gaussian kernel array."""
        gkern1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d


class DiceLogCosh(nn.Module):
    """
    log-cosh dice loss function reproduced from https://arxiv.org/pdf/2006.14822.pdf
    """

    def __init__(self, dice_loss: DiceLoss):
        super().__init__()
        self.dice_loss = dice_loss

    def forward(self, pred: Tensor, target: Tensor) -> float:
        return torch.log(torch.cosh(self.dice_loss(pred, target)))


class GenDiceLoss(nn.Module):
    """
    Generalized Dice Loss. Implementation from 'monai' library.
    Generalized Dice loss controls the contribution that each class makes to the loss by
    weighting classes by the inverse size of the expected region.
    """

    def __init__(self, n_classes: int, to_onehot_y=False):
        super().__init__()
        self.loss = GeneralizedDiceLoss(to_onehot_y=to_onehot_y)
        self.n_classes = n_classes

    def forward(self, pred: Tensor, target: Tensor):
        target = F.one_hot(torch.tensor(target).long(), self.n_classes).permute(0, 3, 1, 2)
        return self.loss(pred, target)


class MyFocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)**gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray, omegaconf.listconfig.ListConfig)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            # alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def dice_loss(true, logits, eps=1e-7):
    """
    Custom Dice Loss

    https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py#L54

    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2.0 * intersection / (cardinality + eps)).mean()
    return 1 - dice_loss
