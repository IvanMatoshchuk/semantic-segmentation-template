import os
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import hydra
from omegaconf import DictConfig

project_path = Path(__file__).parent.parent.parent


class HoneyBeesDataModule(LightningDataModule):
    def __init__(self, dataset_args: DictConfig, dataloader_args: DictConfig, fold: int):
        super().__init__()

        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args
        self.fold = fold

        path_images = os.path.join(project_path, "data", "train", "images")
        path_labels = os.path.join(project_path, "data", "train", "labels")

        img_names = [i for i in os.listdir(path_images) if not i.startswith(".")]
        self.images_paths = [os.path.join(path_images, i) for i in img_names]
        self.masks_paths = [os.path.join(path_labels, i) for i in img_names]

        self.images_paths_val = [self.images_paths.pop(fold)]
        self.masks_paths_val = [self.masks_paths.pop(fold)]

    def train_dataset(self) -> Dataset:

        return hydra.utils.instantiate(
            self.dataset_args.train, images_paths=self.images_paths, masks_paths=self.masks_paths, phase="train"
        )

    def val_dataset(self) -> Dataset:

        return hydra.utils.instantiate(
            self.dataset_args.val, images_paths=self.images_paths_val, masks_paths=self.masks_paths_val, phase="val"
        )

    def test_dataset(self) -> Dataset:

        return hydra.utils.instantiate(self.dataset_args.test, phase="val")

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        return DataLoader(dataset=train_dataset, **self.dataloader_args.train)

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        return DataLoader(dataset=val_dataset, **self.dataloader_args.val)

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.test_dataset()
        return DataLoader(dataset=test_dataset, **self.dataloader_args.test)
