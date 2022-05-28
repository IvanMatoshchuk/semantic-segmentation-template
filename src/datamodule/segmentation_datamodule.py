from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

import hydra
from omegaconf import DictConfig


class SegmentationDataModule(LightningDataModule):
    def __init__(self, dataset_args: DictConfig, dataloader_args: DictConfig):
        super().__init__()

        self.dataset_args = dataset_args
        self.dataloader_args = dataloader_args

    def train_dataset(self) -> Dataset:

        return hydra.utils.instantiate(self.dataset_args.train, phase="train")

    def val_dataset(self) -> Dataset:

        return hydra.utils.instantiate(self.dataset_args.val, phase="val")

    def test_dataset(self) -> Dataset:

        return hydra.utils.instantiate(self.dataset_args.test, phase="test")

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        return DataLoader(dataset=train_dataset, **self.dataloader_args.train)

    def val_dataloader(self) -> DataLoader:
        val_dataset = self.val_dataset()
        return DataLoader(dataset=val_dataset, **self.dataloader_args.val)

    def test_dataloader(self) -> DataLoader:
        test_dataset = self.test_dataset()
        return DataLoader(dataset=test_dataset, **self.dataloader_args.test)
