import os
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

project_path = Path(__file__).parent.parent.parent.parent


class CustomDataset(Dataset):
    """
    TODO: maybe add tranforms into config
    TODO: maybe read images in the __init__

    Uploads and augments images.
    Oversamples "open-brood"
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
                    A.RandomGamma(p=0.5),  # changes image's luminance
                ]
            )

        list_trans.extend([A.Normalize(mean=0, std=1), ToTensorV2()])
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int):

        augmentation = self.transforms(image=self.images[index], mask=self.masks[index])
        img_aug = augmentation["image"]
        mask_aug = augmentation["mask"]

        width = img_aug.size()[1] // 32 * 32
        height = img_aug.size()[2] // 32 * 32

        return img_aug[:, :width, :height], mask_aug[:width, :height]

    def __len__(self) -> int:
        return len(self.images)


class CustomDatasetBatch(Dataset):
    """
    TODO: currently only for 2 specific images, generalize

    Creates batches from crops. Oversamples "open-brood" based on selected coordinates.

    """

    def __init__(self, images_path: str, labels_path: str, img_size: int, phase: str, n_crops_in_batch: str):
        self.images_folder = os.path.join(project_path, images_path)
        self.masks_folder = os.path.join(project_path, labels_path)

        self.images = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        self.masks = [i for i in os.listdir(self.masks_folder) if not i.startswith(".")]

        # combine into one big image
        self.image = np.concatenate([cv2.imread(os.path.join(self.images_folder, i), 0) for i in self.images])
        self.mask = np.concatenate([cv2.imread(os.path.join(self.masks_folder, i), 0) for i in self.masks])

        self.transforms = self.get_transforms(phase, img_size)

        self.n_crops_from_center = n_crops_in_batch // 8
        self.n_crops_in_batch = n_crops_in_batch - self.n_crops_from_center

        # self.n_crops_in_batch = n_crops_in_batch

    def get_transforms(self, phase: str, img_size: int) -> A.core.composition.Compose:

        list_trans = []
        if phase == "train":
            list_trans.extend(
                [
                    A.Rotate(limit=(-90, 90)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5),
                    A.RandomGamma(p=0.5),  # changes image's luminance
                ]
            )

        list_trans.extend(
            [
                A.RandomCrop(height=img_size, width=img_size, always_apply=True, p=1),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int):

        img_aug_batch = []
        mask_aug_batch = []

        for _ in range(self.n_crops_from_center):
            augmentation = self.transforms(
                image=self.image[3250:4750, 1000:2100], mask=self.mask[3250:4750, 1000:2100]
            )
            img_aug = augmentation["image"]
            mask_aug = augmentation["mask"]
            img_aug_batch.append(img_aug)
            mask_aug_batch.append(mask_aug)

        for _ in range(self.n_crops_in_batch):
            augmentation = self.transforms(image=self.image, mask=self.mask)
            img_aug = augmentation["image"]
            mask_aug = augmentation["mask"]
            img_aug_batch.append(img_aug)
            mask_aug_batch.append(mask_aug)

        batch = torch.stack(img_aug_batch), torch.stack(mask_aug_batch)
        batch = (i.squeeze(0) for i in batch)

        return batch

    def __len__(self) -> int:
        return 2  # len(self.images)


class CustomDatasetBatchNew(Dataset):
    """
    Creates batches by adjusting "__len__" method.
    """

    def __init__(
        self,
        images_path: str,
        labels_path: str,
        img_size: int or str,
        crop_size: int,
        phase: str,
        n_crops_in_batch: str,
    ):
        self.images_folder = os.path.join(project_path, images_path)
        self.masks_folder = os.path.join(project_path, labels_path)

        self.images = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        self.masks = [i for i in os.listdir(self.masks_folder) if not i.startswith(".")]

        self.images_paths = [os.path.join(self.images_folder, i) for i in self.images]
        self.masks_paths = [os.path.join(self.masks_folder, i) for i in self.masks]

        # combine into one big image
        self.image = np.concatenate(
            [cv2.imread(os.path.join(self.images_folder, i), 0)[300:2300, :] for i in self.images]
        )
        self.mask = np.concatenate(
            [cv2.imread(os.path.join(self.masks_folder, i), 0)[300:2300, :] for i in self.masks]
        )

        self.transforms = self.get_transforms(phase, crop_size)

        self.n_crops_in_batch = n_crops_in_batch

    def get_transforms(self, phase: str, crop_size: int) -> A.core.composition.Compose:

        list_trans = []
        if phase == "train":
            list_trans.extend(
                [
                    A.RandomCrop(height=2024, width=1024, always_apply=True, p=1),
                    A.CLAHE(),
                    A.Rotate(limit=(-90, 90)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(scale_limit=(0.5, 2.0), p=0.5),
                    A.RandomBrightness()
                    # A.RandomBrightnessContrast(),
                    # A.RandomGamma(p=0.5),  # changes image's luminance
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

    def __getitem__(self, index: int):

        augmentation = self.transforms(image=self.image, mask=self.mask)
        img_aug = augmentation["image"]
        mask_aug = augmentation["mask"]

        return img_aug, mask_aug

    def __len__(self) -> int:
        return self.n_crops_in_batch * len(self.images)


class CustomTestDataset(Dataset):
    """
    Used for testing. Doesn't include extensive augmentation.
    """

    def __init__(self, images_path: str):
        self.images_folder = os.path.join(project_path, images_path)

        self.images = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        self.transforms = self.get_transforms()

    def get_transforms(self) -> A.core.composition.Compose:

        # list_trans = [A.RandomCrop(height = 512, width = 512, always_apply = True, p = 1)]
        list_trans = []
        list_trans.extend([A.Normalize(mean=0, std=1), ToTensorV2()])
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int):

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


class CustomDatasetBatchCrossVal(Dataset):
    """
    Creates batches by adjusting "__len__" method.
    """

    def __init__(
        self,
        images_paths: List[str],
        masks_paths: List[str],
        crop_size: int,
        phase: str,
        n_crops_in_batch: str,
    ):

        self.images_paths = images_paths
        self.masks_paths = masks_paths

        # combine into one big image
        self.image = np.concatenate([cv2.imread(i, 0)[300:2300, :] for i in self.images_paths])
        self.mask = np.concatenate([cv2.imread(i, 0)[300:2300, :] for i in self.masks_paths])

        self.transforms = self.get_transforms(phase, crop_size)

        self.n_crops_in_batch = n_crops_in_batch

    def get_transforms(self, phase: str, crop_size: int) -> A.core.composition.Compose:

        list_trans = []
        if phase == "train":
            list_trans.extend(
                [
                    A.RandomCrop(height=2024, width=1024, always_apply=True, p=1),
                    A.CLAHE(),
                    A.Rotate(limit=(-90, 90)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(scale_limit=(0.5, 2.0), p=0.5),
                    A.RandomBrightnessContrast(),
                    # A.SmallestMaxSize(max_size=1024, p=0.5),
                    A.RandomGamma(p=0.5),  # changes image's luminance
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

    def __getitem__(self, index: int):

        augmentation = self.transforms(image=self.image, mask=self.mask)
        img_aug = augmentation["image"]
        mask_aug = augmentation["mask"]

        return img_aug, mask_aug

    def __len__(self) -> int:
        return self.n_crops_in_batch * len(self.images_paths)


class CustomDatasetCrossVal(Dataset):
    """
    TODO: maybe add tranforms into config
    TODO: maybe read images in the __init__

    Uploads and augments images.
    Oversamples "open-brood"
    """

    def __init__(self, images_paths: List[str], masks_paths: List[str], phase: str):

        self.images = [cv2.imread(img_path, 0) for img_path in images_paths]
        self.masks = [cv2.imread(mask_path, 0) for mask_path in masks_paths]

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
                    A.RandomGamma(p=0.5),  # changes image's luminance
                ]
            )

        list_trans.extend([A.Normalize(mean=0, std=1), ToTensorV2()])
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int):
        augmentation = self.transforms(image=self.images[index], mask=self.masks[index])
        img_aug = augmentation["image"]
        mask_aug = augmentation["mask"]

        width = img_aug.size()[1] // 32 * 32
        height = img_aug.size()[2] // 32 * 32

        return img_aug[:, :width, :height], mask_aug[:width, :height]

    def __len__(self) -> int:
        return len(self.images)
