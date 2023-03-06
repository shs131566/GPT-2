import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np
import random


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size):
        super().__init__()
        # TODO
        #! self.seed

        self.seed = 42
        self.num_workers = 4
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

    def train_dataloader(self):
        return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    worker_init_fn=self.seed_worker,
                    generator=self.g,
                    shuffle=True,
                )

    def val_dataloader(self):
        return DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    shuffle=False,
                )

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
