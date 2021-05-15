import torch
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from torchvision import transforms

from object_detector.utils.datasets import ListDataset


DATA_DIR = 'Data/data'

class LitDataModule(LightningDataModule):
    def __init__(self, batch_size=64, img_size:int = 416, data_dir:str = DATA_DIR):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.img_size = img_size
    
    def setup(self, stage : str = None):
        
        if stage == 'fit' or stage is None:
            self.train_data = ListDataset(
                self.data_dir,
                img_size=self.img_size,
                transform=self.transform
            )
        
        if stage == 'test' or stage is None:
            self.test_data = ListDataset(
                self.data_dir,
                img_size=self.img_size,
                transform=self.transform
            )
    
    