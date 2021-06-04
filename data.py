#!/usr/bin/env python3

##########################################################
#### data.py ### COLD: Concurrent Loads Disaggregator ####
##########################################################

import os
import torch
import numpy as np
import pytorch_lightning as pl
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


def collate(batch):
    """
    Wrap the spectrograms and binary labels into two tensors-batches
    ---
    batch :: list
    """
    data = list(zip(*batch))
    # Batch of spectrograms
    x0 = torch.stack(data[0], 0).to(dtype=torch.float32)
    y = torch.stack(data[1], 0)

    return x0, y


class SyntheticDataset(Dataset):
    """
    Configure data access
    """

    def __init__(self, labels, dataset_path):
        """
        labels :: list -- names of appliances/loads (e.g. kettle, laptop etc.)
        dataset_path :: path to the data generated via the SNS algorithm
        """
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        self.num_labels = len(labels)
        # Path to the spectrograms
        self.specgrams_path = os.path.join(dataset_path, "specgrams")
        # List of filenames inside spectrograms directory
        self.filenames = os.listdir(self.specgrams_path)

    def __getitem__(self, idx):
        """
        idx :: int
        """
        filename = self.filenames[idx]
        filepath = os.path.join(self.specgrams_path, filename)
        # Read the spectrogram and wrap by torch.Tensor
        data = np.load(filepath, allow_pickle=True).item()
        x0 = torch.tensor(data["specgram"], dtype=torch.float32)
        # Encode concurrent labels into binary vector
        labels_w = self.label_encoder.transform(list(data["labels"]))
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        y[labels_w] = 1

        return x0, y

    def __len__(self):
        return len(self.filenames)


class DataModule(pl.LightningDataModule):
    """
    Assigns the data loaders for train/validation/test procedures
    """

    def __init__(self, labels, train_dataset_path, val_dataset_path, test_dataset_path,
                 batch_size=32, collate_fn=None, shuffle=True, pin_memory=True,
                 prefetch=4, n_jobs=1, drop_last=False):
        """
        Connects datasets and specifies hyperparameters of the data loaders
        """

        super().__init__()

        # Parameters
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.drop_last = drop_last

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        # Linking datasets
        self.train_dataset = SyntheticDataset(labels, train_dataset_path)
        self.validation_dataset = SyntheticDataset(labels, val_dataset_path)
        self.test_dataset = SyntheticDataset(labels, test_dataset_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, shuffle=self.shuffle,
                          pin_memory=self.pin_memory, num_workers=self.n_jobs,
                          prefetch_factor=self.prefetch, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, pin_memory=self.pin_memory,
                          num_workers=self.n_jobs, prefetch_factor=self.prefetch,
                          drop_last=False, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=self.n_jobs,
                          prefetch_factor=self.prefetch, drop_last=False, shuffle=False)
