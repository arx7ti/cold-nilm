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

# Types
from beartype import beartype
from typing import Optional, Union, List, Tuple, Callable, Dict


@beartype
def collate(
    mini_batch: List[Tuple[torch.Tensor, ...]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrap the waveforms and binary labels into two mini-batches

    Arguments:
        mini_batch: List[Tuple[torch.Tensor, ...]]
    Returns:
        Tuple[torch.Tensor, torch.Tensor]
    """
    data = list(zip(*mini_batch))
    # Mini-batches
    x = torch.stack(data[0], 0).to(dtype=torch.float32)
    y = torch.stack(data[1], 0)
    return x, y


class SyntheticDataset(Dataset):
    """
    Configure an data access
    """

    label_encoder = LabelEncoder()

    @beartype
    def __init__(self, labels: List[str], w_max: int, dataset_path: str) -> None:
        """
        Arguments:
            labels: List[str] - names of appliances/loads (e.g. kettle, laptop etc.)
            w_max: int - maximum number of loads working simultaneously to be considered
            dataset_path: str - path to the data generated via the SNS algorithm
        Returns:
            None
        """
        self.label_encoder.fit(labels)
        self.num_labels = len(labels)
        self.dataset_path = dataset_path
        # List of filenames inside a dataset directory
        self.filenames = [
            filename
            for filename in os.listdir(dataset_path)
            if int(filename.split("-")[0]) <= w_max
        ]
        return None

    @beartype
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx: int
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        filename = self.filenames[idx]
        filepath = os.path.join(self.dataset_path, filename)
        # Read the waveform and wrap by torch.Tensor
        data = np.load(filepath, allow_pickle=True).item()
        x = torch.tensor(data["signal"], dtype=torch.float32)
        # Encode concurrent labels into binary vector
        labels_w = torch.tensor(self.label_encoder.transform(list(data["labels"])))
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        y[labels_w] = 1.0
        return x, y

    @beartype
    def __len__(self) -> int:
        return len(self.filenames)


class DataModule(pl.LightningDataModule):
    """
    Assigns the data loaders for train/validation/test procedures
    """

    @beartype
    def __init__(
        self,
        labels: List[str],
        w_max: int,
        train_dataset_path: Union[str, None],
        val_dataset_path: Union[str, None],
        test_dataset_path: Union[str, None],
        mini_batch_size: int,
        shuffle: bool = True,
        n_jobs: int = 1,
        pin_memory: bool = False,
        prefetch: int = 4,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        """
        Connects datasets and specifies hyperparameters of the data loaders

        Arguments:
            labels: List[str] - names of appliances/loads (e.g. kettle, laptop etc.)
            w_max: int - maximum number of loads working simultaneously to be considered
            train_dataset_path: Union[str, None] - path to the data generated via the SNS algorithm
            val_dataset_path: Union[str, None] - ...
            test_dataset_path: Union[str, None] - ...
            mini_batch_size: int
            shuffle: bool
            n_jobs: int
            pin_memory: bool
            prefetch: int - number of mini_batches to fetch at a time
            drop_last: bool - drop the last mini-batch
            collate_fn: Optional[Callable] - method to wrap the readings into the mini-batch
        Returns:
            None
        """
        super().__init__()
        # Parameters
        self.mini_batch_size = mini_batch_size
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
        if train_dataset_path is not None:
            self.train_dataset = SyntheticDataset(labels, w_max, train_dataset_path)
        else:
            self.train_dataset = None
        if val_dataset_path is not None:
            self.validation_dataset = SyntheticDataset(labels, w_max, val_dataset_path)
        else:
            self.validation_dataset = None
        if test_dataset_path is not None:
            self.test_dataset = SyntheticDataset(labels, w_max, test_dataset_path)
        else:
            self.test_dataset = None
        return None

    @beartype
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.mini_batch_size,
                collate_fn=self.collate_fn,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                num_workers=self.n_jobs,
                prefetch_factor=self.prefetch,
                drop_last=self.drop_last,
            )
        else:
            raise ValueError("Train_dataset is not defined.")

    @beartype
    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is not None:
            return DataLoader(
                self.validation_dataset,
                batch_size=self.mini_batch_size,
                collate_fn=self.collate_fn,
                pin_memory=self.pin_memory,
                num_workers=self.n_jobs,
                prefetch_factor=self.prefetch,
                drop_last=False,
                shuffle=False,
            )
        else:
            raise ValueError("Validation dataset is not defined.")

    @beartype
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.mini_batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.n_jobs,
                prefetch_factor=self.prefetch,
                drop_last=False,
                shuffle=False,
            )
        else:
            raise ValueError("Test dataset is not defined.")
