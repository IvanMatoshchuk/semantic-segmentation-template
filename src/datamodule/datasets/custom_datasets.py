import os
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
from torch import Tensor
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

project_path = Path(__file__).parent.parent.parent.parent


# TODO: add augmentations into config


class CustomDataset(Dataset):
    """
    Class for reading images for training and (or) validation.
    Since the number of images is small, they will be uploaded in the '__init__' method.
    Make uploading inside '__getitem__' method if problem with memory occurs.

    Parameters
    ----------
        iamges_path: str
            path to the location of images
        labels_path: str
            path to the location of labels corresponding to training images
        phase: str
            name of the process; if == "train", then an extensive augmentations are applied
    """

    def __init__(self, images_path: str, labels_path: str, phase: str):
        self.images_folder = os.path.join(project_path, images_path)
        self.masks_folder = os.path.join(project_path, labels_path)

        image_names = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        mask_names = [i for i in os.listdir(self.masks_folder) if not i.startswith(".")]

        img_paths = [os.path.join(self.images_folder, image_name) for image_name in image_names]
        mask_paths = [os.path.join(self.masks_folder, mask_name) for mask_name in mask_names]

        self.images = [cv2.imread(img_path, 0) for img_path in img_paths]
        self.masks = [cv2.imread(mask_path, 0) for mask_path in mask_paths]

        self.transforms = self.get_transforms(phase)

    def get_transforms(self, phase: str) -> A.core.composition.Compose:

        list_trans = []
        if phase == "train":
            list_trans.extend(
                [
                    A.CLAHE(),
                    A.Rotate(limit=(-90, 90)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(scale_limit=0.5, p=0.5),
                    A.RandomBrightnessContrast(),
                    A.SmallestMaxSize(max_size=1024, p=0.5),
                    A.RandomGamma(p=0.5),
                ]
            )

        list_trans.extend([A.Normalize(mean=0, std=1), ToTensorV2()])
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        augmentation = self.transforms(image=self.images[index], mask=self.masks[index])
        img_aug = augmentation["image"]
        mask_aug = augmentation["mask"]

        # make compatible with U-Net architecture
        width = img_aug.size()[1] // 32 * 32
        height = img_aug.size()[2] // 32 * 32

        return img_aug[:, :width, :height], mask_aug[:width, :height]

    def __len__(self) -> int:
        return len(self.images)


class CustomDatasetArtificialBatchCrops(Dataset):
    """
    This class artifically creates batches consisting of crops from the small number of training images.
    This is done by adjusting the "__len__" method.

    Since the number of images is small, they will be uploaded in the '__init__' method.
    Make uploading inside '__getitem__' method if problem with memory occurs.

    Parameters
    ----------
        iamges_path: str
            path to the location of images.
        labels_path: str
            path to the location of labels corresponding to training images.
        phase: str
            name of the process; if == "train", then an extensive augmentations are applied.
        crop_size: int
            size of the random crop that will be applied with the probability=1.
            If using U-Net-like architecture, the crop-size should be divisble by 32.
        n_crops_in_batch: int
            number of crops that will be included in 1 batch.
            A suggestions is to use the number >= 16 and divisible by 2.
    """

    def __init__(
        self,
        images_path: str,
        labels_path: str,
        phase: str,
        crop_size: int,
        n_crops_in_batch: str,
    ):
        self.images_folder = os.path.join(project_path, images_path)
        self.masks_folder = os.path.join(project_path, labels_path)

        self.images = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        self.masks = [i for i in os.listdir(self.masks_folder) if not i.startswith(".")]

        self.images_paths = [os.path.join(self.images_folder, i) for i in self.images]
        self.masks_paths = [os.path.join(self.masks_folder, i) for i in self.masks]

        # combine into one big image
        self.image = np.concatenate([cv2.imread(os.path.join(self.images_folder, i), 0) for i in self.images])
        self.mask = np.concatenate([cv2.imread(os.path.join(self.masks_folder, i), 0) for i in self.masks])

        self.transforms = self.get_transforms(phase, crop_size)

        self.n_crops_in_batch = n_crops_in_batch

    def get_transforms(self, phase: str, crop_size: int) -> A.core.composition.Compose:

        list_trans = []
        if phase == "train":
            list_trans.extend(
                [
                    A.CLAHE(),
                    A.Rotate(limit=(-90, 90)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(scale_limit=(0.5, 2.0), p=0.5),
                    A.RandomBrightnessContrast(),
                ]
            )

        list_trans.extend(
            [
                A.RandomCrop(height=crop_size, width=crop_size, always_apply=True, p=1),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        augmentation = self.transforms(image=self.image, mask=self.mask)
        img_aug = augmentation["image"]
        mask_aug = augmentation["mask"]

        return img_aug, mask_aug

    def __len__(self) -> int:
        return self.n_crops_in_batch * len(self.images)


class CustomTestDataset(Dataset):
    """
    Class used for uploading images for testing. Doesn't include extensive augmentation.
    Images will be read in '__getitem__' method.

    Parameters
    ----------
        images_path: str
            path to images for inference.

    Output of __getitem__
    ---------------------
        img_aug: Tensor
            preprocessed image, which will be used for inference
        img_path: str
            path to the used image
    """

    def __init__(self, images_path: str):
        self.images_folder = os.path.join(project_path, images_path)

        self.images = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        self.transforms = self.get_transforms()

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = []
        list_trans.extend([A.Normalize(mean=0, std=1), ToTensorV2()])
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:

        img_path = os.path.join(self.images_folder, self.images[index])

        image = cv2.imread(img_path, 0)

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32
        image = image[:height, :width]

        augmentation = self.transforms(image=image)
        img_aug = augmentation["image"]

        return img_aug, img_path

    def __len__(self) -> int:
        return len(self.images)
