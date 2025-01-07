import torch
import random
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader

class DonutDataPLModuleCustom(pl.LightningDataModule):
    def __init__(self, config, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.train_batch_size = config['train_batch_sizes']
        self.val_batch_size = config['val_batch_sizes']
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.g = torch.Generator()
        self.g.manual_seed(config['seed'])

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            self.train_batch_size,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
            shuffle=True,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            self.val_batch_size,
            pin_memory=True,
            shuffle=False,
        )
        
        return val_loader

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)