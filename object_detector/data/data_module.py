import argparse

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from object_detector.utils.utils import worker_seed_set, load_classes
from object_detector.utils.datasets import ListDataset
from object_detector.utils.parse_cfg import parse_data_config
from object_detector.utils.augmentations import AUGMENTATION_TRANSFORMS


class LitDataModule(LightningDataModule):
    def __init__(self, args: argparse.Namespace = None, mini_batch_size=64, img_size:int = 416):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.mini_batch_size = mini_batch_size

        self.data_config = parse_data_config(self.args.get("data"))
        self.train_list_path = self.data_config["train"]
        self.valid_list_path = self.data_config["valid"]
        self.class_names = load_classes(self.data_config["names"])

        self.transform = AUGMENTATION_TRANSFORMS
        self.img_size = img_size
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--data", type=str, default="config/obj.data", help="Path to data config file (.data)")
    
    def setup(self, stage : str = None):
        
        if stage == 'fit' or stage is None:
            self.dataset = ListDataset(
                self.train_list_path,
                img_size=self.img_size,
                transform=self.transform
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ListDataset(
                self.valid_list_path,
                img_size=self.img_size,
                transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
            worker_init_fn=worker_seed_set
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.mini_batch_size,
            pin_memory=True,
            collate_fn=self.dataset.collate_fn,
            worker_init_fn=worker_seed_set
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.mini_batch_size,
            pin_memory=True,
            collate_fn=self.test_dataset.collate_fn
        )