#!/usr/bin/env python3

###########################################################
#### model.py ### COLD: Concurrent Loads Disaggregator ####
###########################################################

import os

# Prototyping
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# Evaluation
from sklearn.metrics import f1_score

# Plotting
import matplotlib
import seaborn as sb
import matplotlib.pyplot as plt

# Types
from beartype import beartype
from typing import Any, Optional, Union, List, Tuple, Dict


sb.set()
# Do not use GUI preview
matplotlib.use("Agg")


class AugmentAbstract(nn.Module):
    """
    Abstract layer for any augmentation class in this module
    """

    # Probability of each example to be augmented
    p: float

    @beartype
    def __init__(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Arguments:
            *args: Any
            **kwargs: Dict[str, Any]
        Returns:
            None
        """
        super(AugmentAbstract, self).__init__(*args, **kwargs)
        return None

    @beartype
    def _get_mini_batch_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes waveform and computes a mask for examples in the mini-batch to be augmented

        Arguments:
            x: torch.Tensor - waveforms of size [Batch x Time]
        Returns:
            torch.Tensor - indices of examples from the mini-batch to be augmented
        """
        probas = torch.rand(x.size(0), device=x.device)
        mini_batch_mask = (probas > (1 - self.p)).nonzero().squeeze()
        return mini_batch_mask


class RandomGaussianNoise(AugmentAbstract):
    """
    Simulates Gaussian noise for some examples in the mini-batch
    """

    @beartype
    def __init__(self, variance: float = 0.01, p: float = 0.5) -> None:
        """
        Arguments:
            variance: float
            p: float
        Returns:
            None
        """
        super(RandomGaussianNoise, self).__init__()
        self.std = np.sqrt(variance)
        self.p = p
        return None

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor - waveforms of size [Batch x Time]
        Returns:
            torch.Tensor - augmented waveforms
        """
        mini_batch_mask = self._get_mini_batch_mask(x)
        noise = torch.zeros_like(x, device=x.device)
        noise[mini_batch_mask] = self.std * torch.randn(
            *x[mini_batch_mask].size(), device=x.device
        )
        return x + noise


class RandomMeasurementSpike(AugmentAbstract):
    """
    Simulates sensor corrupted readings (spikes in this case). On of the reasons for that
    is non-synchronization between ADC and CPU
    """

    @beartype
    def __init__(self, max: float = 100.0, n: int = 2, p: float = 0.1) -> None:
        """
        Arguments:
            max: float - spike magnitude
            n: int - duration of a spike
            p: float
        Returns:
            None
        """
        super(RandomMeasurementSpike, self).__init__()
        self.max = max
        self.n = n
        self.p = p
        return None

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor - waveform of size [Batch x Time]
        Returns:
            torch.Tensor - augmented waveforms
        """
        mini_batch_mask = self._get_mini_batch_mask(x)
        spikes_mask = torch.randint(0, x.size(1) - 2, (x.size(0),), device=x.device)
        spikes = torch.zeros_like(x, device=x.device)
        sign = torch.tensor([-1, 1])[torch.randperm(2)[0]].item()
        spikes[mini_batch_mask, spikes_mask[mini_batch_mask]] = sign * self.max
        return x + spikes


class RandomZeroFrequencyComponent(AugmentAbstract):
    """
    Simulates DC component in the observed signal
    """

    @beartype
    def __init__(self, variance: float = 0.1, p: float = 0.1) -> None:
        """
        Arguments:
            variance: float
            p: float
        Returns:
            None
        """
        super(RandomZeroFrequencyComponent, self).__init__()
        self.std = np.sqrt(variance)
        self.p = p
        return None

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor - waveform of size [Batch x Time]
        Returns:
            torch.Tensor - augmented waveforms
        """
        mini_batch_mask = self._get_mini_batch_mask(x)
        dc = torch.zeros_like(x, device=x.device)
        dc[mini_batch_mask] += self.std * torch.randn(
            *mini_batch_mask.unsqueeze(-1).size(), device=x.device
        )
        return x + dc


class Spectrum(nn.Module):
    """
    Computes the Short-Time Fourier Transform of a given mini-batch of waveforms
    """

    @beartype
    def __init__(self, window_size: int, hop_size: int) -> None:
        """
        Arguments:
            window_size: int
            hop_size: int - distance between the overlapped windows
        Returns:
            None
        """
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        return None

    @beartype
    def forward(self, xn: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            xn: torch.Tensor - mini-batch of normalized waveforms
        Returns:
            torch.Tensor
        """
        x0 = torch.stft(
            xn,
            self.window_size,
            hop_length=self.hop_size,
            window=torch.hann_window(self.window_size, device=xn.device),
            return_complex=True,
            normalized=True,
        )
        return torch.abs(x0)


@beartype
def init_weights(layer: nn.Module) -> None:
    """
    Initializes weights of the layer

    Arguments:
        layer: nn.Module
    Returns:
        None
    """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer.bias, "data"):
        nn.init.constant_(layer.bias, 0.0)
    return None


class ResidualPoswiseSubNetwork(nn.Module):
    """
    Torch Module for Residual Position-wise Sub Network
    """

    @beartype
    def __init__(self, q: int, dropout: float) -> None:
        """
        Arguments:
            q: int - number of input neurons
            dropout: float - probability of a dropout
        Returns:
            None
        """
        super().__init__()
        # Constants
        self.dropout = dropout
        # Layers
        self.h1 = nn.Linear(q, q)
        self.h2 = nn.Linear(q, q)
        self.norm = nn.LayerNorm(q)
        # Initialization
        init_weights(self.h1)
        init_weights(self.h2)

    @beartype
    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x0: torch.Tensor - spectrum of size [Batch x Time x Frequency]
        Returns:
            torch.Tensor
        """
        # Following composition will be computed:
        # x2' = (ReLU∘Norm∘(r+Dropout∘h2∘ReLU∘Dropout∘h1))(x0) | r(x)=x
        # 1. x1 = (ReLU∘Dropout∘h1)(x0)
        x1 = nn.functional.relu_(
            nn.functional.dropout(self.h1(x0), p=self.dropout, training=self.training)
        )
        # 2. x2 = (Dropout∘h2)(x1)
        x2 = nn.functional.dropout(self.h2(x1), p=self.dropout, training=self.training)
        # 3. x2' = (ReLU∘Norm∘(r+x0))(x2) | r(x)=x
        x2_prime = nn.functional.relu_(self.norm(x2 + x0))
        return x2_prime


class ModelCOLD(pl.LightningModule):
    """
    Pytorch Lightning module for COLD architecture and its training/testing pipeline
    """

    @beartype
    def __init__(
        self,
        config: Dict[str, Union[int, float]],
        n_labels: int,
        w_max: int,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, np.ndarray]] = None,
        variance_gaussian: float = 0.01,
        p_gaussian: float = 0.5,
        max_spike: float = 100.0,
        n_spike: int = 2,
        p_spike: float = 0.1,
        variance_dc: float = 0.1,
        p_dc: float = 0.1,
        eps: float = 1e-8,
        max_thresholds: int = 30,
        threshold: Optional[float] = None,
        plots_dir: Optional[str] = None,
    ) -> None:
        """
        Arguments:
            config: Dict[str, Union[int, float]] - dict of hyperparameters
            n_labels: int - number of output units (i.e. #L)
            w_max: int - maximum number of concurrent loads in the dataset
            lr_scheduler_config: Optional[Dict[str, Any]] - parameters for the `ReduceLROnPlateau`
            stats: Optional[Dict[str, np.ndarray]] - mean and std of the training dataset
            variance_gaussian: float - variance for the `RandomGaussianNoise` augmentation
            p_gaussian: float - probability for the `RandomGaussianNoise` augmentation
            max_spike: float - magnitude of a spike for `RandomMeasurementSpike` augmentation
            n_spike: int - duration of a spike for the `RandomMeasurementSpike` augmentation
            p_spike: float - probability of a spike for the `RandomMeasurementSpike` augmentation
            variance_dc: float - variance of a DC component magnitude for the `RandomZeroFrequencyComponent` augmentation
            p_dc: float - probability for the `RandomZeroFrequencyComponent` augmentation
            eps: float - very small constant
            max_thresholds: int - maximum number of iterations for threshold tuning
            threshold: Optional[float] - a optimal threshold obtained from validation procedure
            plots_dir: Optional[str] - path to the figures to be plotted
        Returns:
            None
        """
        super().__init__()
        assert n_labels > 0 and w_max > 0
        self.save_hyperparameters()
        # Assign constants
        self.w_max = w_max
        self.lr_scheduler_config = lr_scheduler_config
        self.eps = eps
        self.max_thresholds = max_thresholds
        self.threshold = threshold
        # Spectrum parameters
        window_size = config.get("window_size", 400)
        hop_size = config.get("hop_size", 80)
        # COLD layout:
        # RPSN width, number of RPSNs and number of heads for MHSA function
        q = config.get("q", 256)
        k = config.get("k", 15)
        n_head = config.get("n_head", 8)
        assert q % n_head == 0
        # Dropout probability
        dropout = config.get("dropout", 0.1)
        # Training parameters:
        # Learning rate
        self.lr = config.get("learning_rate", 5e-4)
        # Weight decay
        self.weight_decay = config.get("weight_decay", 0.01)
        # Input dimension
        v = window_size // 2 + 1
        # Normalization statistics
        if stats is not None:
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
            self.std = torch.tensor(stats["std"], dtype=torch.float32)
        else:
            self.mean = torch.zeros(1)
            self.std = torch.ones(1)
        # Non-learnable parameters
        self.augment = nn.Sequential(
            RandomGaussianNoise(variance=variance_gaussian, p=p_gaussian),
            RandomMeasurementSpike(max=max_spike, n=n_spike, p=p_spike),
            RandomZeroFrequencyComponent(variance=variance_dc, p=p_dc),
        )
        self.spectrum = Spectrum(window_size, hop_size)
        # Network learnable parameters
        self.affine = nn.Linear(v, q)
        self.sequence = nn.Sequential()
        # Add k Residual Position-wise Sub-Networks
        for i in range(1, k + 1):
            self.sequence.add_module(
                "RPSN@%d" % (2 * i + 1), ResidualPoswiseSubNetwork(q, dropout=dropout)
            )
        # Multi-Head Self Attention
        self.mhsa = nn.MultiheadAttention(
            embed_dim=q, num_heads=n_head, dropout=dropout
        )
        self.norm = nn.LayerNorm(q)
        # Classification layer
        self.classifier = nn.Linear(q, n_labels)
        # Learnable parameter alpha for sigmoid activation
        self.a = nn.Parameter(torch.ones(1, n_labels), requires_grad=True)
        # Initialization
        init_weights(self.classifier)
        # Extra
        self.plots_dir = plots_dir
        return None

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor - mini-batch of waveforms [Batch x Time]
        Returns:
            torch.Tensor - logits for each appliance being detected
        """
        # Augmentation
        if self.training:
            x = self.augment(x)
        # Normalize
        xn = (x - self.mean.to(x.device)) / self.std.to(x.device)
        # Transform to the spectrum
        x0 = self.spectrum(xn)
        # Transpose to get the new order of dimensions [Batch x Time x Freqs]
        x0 = x0.transpose(1, 2)
        # Affine projection to match the dimensions with the network's width `q`
        x1 = self.affine(x0)
        # Pass the projected spectrograms x1 through the sequence of RPSNs
        # That is, compute x'_(2k+1) = (RPSN_(2k+1)∘...∘RPSN_3)(x1)
        x_prime = self.sequence(x1)
        # MHSA freq
        # Compute MHSA to resolve time-dependency
        x_prime = x_prime.transpose(0, 1)
        x_att, _ = self.mhsa(x_prime, x_prime, x_prime)
        x_att = x_att.transpose(0, 1)
        x_prime = x_prime.transpose(0, 1)
        # Global pooling across time dimension
        x_global = self.norm(x_att + x_prime).mean(dim=1)
        # Output of classification layer
        x_class = self.classifier(x_global)
        # Class y_hat (predictions)
        y_hat = 1 / (1 + self.a * torch.exp(-x_class))
        return y_hat

    @beartype
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes Binary Cross Entropy loss

        Arguments:
            y_hat: torch.Tensor
            y: torch.Tensor
        Returns:
            torch.Tensor
        """
        return nn.functional.binary_cross_entropy(y_hat, y)

    @beartype
    def training_step(
        self, mini_batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]], *args: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Performs 1 gradient update

        Arguments:
            mini_batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            *args: Any
        Returns:
            Dict[str, torch.Tensor]
        """
        x0, y = mini_batch
        y_hat = self.forward(x0)
        loss = self.loss(y_hat, y)
        # Track results
        self.log("train_loss", loss, on_epoch=True)
        return {"loss": loss}

    @beartype
    def validation_step(
        self, mini_batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]], *args: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Performs 1 validation step

        Arguments:
            mini_batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            *args: Any
        Returns:
            Dict[str, torch.Tensor]
        """
        x0, y = mini_batch
        y_hat = self.forward(x0)
        loss = self.loss(y_hat, y)
        # Track results
        self.log("validation_loss", loss, on_epoch=True)
        return {"y_hat": y_hat, "y": y}

    @beartype
    def test_step(
        self, mini_batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]], *args: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Performs 1 test step

        Arguments:
            mini_batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            *args: Any
        Returns:
            Dict[str, torch.Tensor]
        """
        x0, y = mini_batch
        y_hat = self.forward(x0)
        return {"y_hat": y_hat, "y": y}

    @beartype
    def validation_step_end(
        self, parallel_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return parallel_batch

    @beartype
    def test_step_end(
        self, parallel_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return parallel_batch

    @beartype
    def _compute_f1(
        self, y: np.ndarray, y_hat: np.ndarray, t: float, average: Optional[str] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Arguments:
            y: np.ndarray
            y_hat: np.ndarray
            t: float
            average: Optional[str]
        Returns:
            Tuple[np.ndarray, List[float]]
        """
        # Keep f1-scores and relative cardinalities of each w-subset of S
        f1_values = []
        weights = []
        for w in range(1, self.w_max + 1):
            mask = y.sum(1) == w
            f1_w = f1_score(
                y[mask], np.where(y_hat[mask] > t, 1.0, 0.0), average=average
            )
            weights.append(len(mask) / len(y))
            f1_values.append(f1_w)
        return np.array(f1_values).astype(np.float32), weights

    @beartype
    def _compute_metrics(
        self,
        y: np.ndarray,
        y_hat: np.ndarray,
        threshold: float,
        average: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Arguments:
            y: np.ndarray,
            y_hat: np.ndarray,
            threshold: float,
            average: Optional[str] = None,
        Returns:
            Tuple[np.ndarray, float]
        """
        # Compute mean F1-score values across w
        mean_f1_values, weights = self._compute_f1(
            y,
            y_hat,
            threshold,
            average=average,
        )
        # Compute weighted mean F1-score
        weighted_mean_f1 = np.average(mean_f1_values, weights=weights)
        return mean_f1_values, weighted_mean_f1

    @beartype
    def _evaluate(
        self, y: np.ndarray, y_hat: np.ndarray, prefix: str
    ) -> Tuple[float, float, Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        Computes optimal threshold for decision making

        Arguments:
            y: np.ndarray
            y_hat: np.ndarray
            prefix: str
        Returns:
            Tuple[float, float, Union[np.ndarray, None], Union[np.ndarray, None]]
        """
        if self.threshold is not None:
            t_opt = self.threshold
            mean_f1_values, weighted_mean_f1 = self._compute_metrics(
                y, y_hat, t_opt, average="samples"
            )
        else:
            # Initial value for the threshold
            t_opt = 1e-2
            thresholds = np.linspace(0, 1, self.max_thresholds)
            # We will use weighted mean F1 as a reference metric
            # for finding the optimal threshold
            weighted_mean_f1 = 0
            mean_f1_values = None
            for t in thresholds:
                # Compute temporary mean F1-score values across w and
                # temporary weighted mean F1-score
                tmp_mean_f1_values, tmp_weighted_mean_f1 = self._compute_metrics(
                    y, y_hat, t, average="samples"
                )
                if tmp_weighted_mean_f1 > weighted_mean_f1:
                    weighted_mean_f1 = tmp_weighted_mean_f1
                    mean_f1_values = tmp_mean_f1_values
                    t_opt = t
        if prefix == "test":
            f1_values, weights = self._compute_f1(y, y_hat, t_opt, average=None)
            weighted_f1_values = np.average(f1_values, axis=0, weights=weights)
        else:
            weighted_f1_values = None
        return t_opt, weighted_mean_f1, mean_f1_values, weighted_f1_values

    @beartype
    def _shared_epoch_end(
        self, parallel_data: List[Dict[str, torch.Tensor]], prefix: str
    ) -> None:
        """
        Summarizes validation/test procedure

        Arguments:
            parallel data: List[Dict[str, torch.Tensor]]
            prefix: str - either "val" or "test", depending on the procedure
        Returns:
            None
        """
        y_hat = torch.cat([var["y_hat"] for var in parallel_data])
        y = torch.cat([var["y"] for var in parallel_data])
        # Bind matrices to CPU and transform into np.ndarray for computations in sklearn
        # The size is [Batch x Labels]
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        # Weighted mean F1-score and w-subset-wise F1 metrics
        eval_pack = self._evaluate(y, y_hat, prefix)
        t_opt, weighted_mean_f1, mean_f1_values, weighted_f1_values = eval_pack
        # Track results
        self.log("%s_t_opt" % prefix, t_opt)
        self.log("%s_weighted_mean_f1" % prefix, weighted_mean_f1)
        for w, mean_f1_w in zip(range(1, self.w_max + 1), mean_f1_values):
            self.log("%s_mean_f1_w=%d" % (prefix, w), mean_f1_w)
        # Data visualization
        if prefix == "test":
            labels = self.trainer.datamodule.test_dataset.label_encoder.classes_
            if self.plots_dir is not None:
                plt.figure(figsize=(8, 5))
                plt.bar(np.arange(1, self.w_max + 1), mean_f1_values, alpha=0.66)
                plt.xlabel("Number of concurrent loads (w)", fontsize=14)
                plt.ylabel("Mean F1-score", fontsize=14)
                plt.xticks(np.arange(1, self.w_max + 1), fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(True)
                plt.ylim((0.9 * np.min(weighted_f1_values), 1))
                plot_path = os.path.join(self.plots_dir, "per_w.png")
                plt.savefig(plot_path, bbox_inches="tight")
                plt.show()
                # Sort the labels with accordance to their f1-scores
                args = np.argsort(weighted_f1_values)
                plt.figure(figsize=(12, 5))
                plt.bar(labels[args], weighted_f1_values[args], alpha=0.66)
                plt.xlabel("", fontsize=14)
                plt.ylabel("F1-score", fontsize=14)
                plt.ylim((0.9 * np.min(weighted_f1_values), 1))
                plt.xticks(rotation=90, fontsize=14)
                plt.yticks(fontsize=14)
                # plt.tick_params(axis="x", labelsize=14)
                plt.grid(True)
                plot_path = os.path.join(self.plots_dir, "per_label.png")
                plt.savefig(plot_path, bbox_inches="tight")
                plt.show()
            for label, f1 in zip(labels[args], weighted_f1_values[args]):
                self.log("f1_%s" % label, f1)
        return None

    @beartype
    def validation_epoch_end(self, parallel_data: List[Dict[str, torch.Tensor]]):
        self._shared_epoch_end(parallel_data, "validation")

    @beartype
    def test_epoch_end(self, parallel_data: List[Dict[str, torch.Tensor]]):
        self._shared_epoch_end(parallel_data, "test")

    @beartype
    def _setup_lr_scheduler(self, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """
        Arguments:
            optimizer: optim.Optimizer
        Returns:
            Dict[str, Any]
        """
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.lr_scheduler_config["kwargs"]
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": True,
            "monitor": self.lr_scheduler_config["monitor"],
        }
        return scheduler

    @beartype
    def configure_optimizers(
        self, *args: Any, **kwargs: Dict[str, Any]
    ) -> Union[
        Tuple[
            List[optim.Optimizer],
            List[Dict[str, Any]],
        ],
        List[optim.Optimizer],
    ]:
        """
        Setup the optimizer and the learning rate scheduler

        Arguments:
            *args: Any
            **kwargs: Dict[str, Any]
        Returns:
            Union[
                Tuple[
                    List[optim.Optimizer],
                    List[Dict[str, Any]],
                ],
                List[optim.Optimizer],
            ]
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
        )
        if self.lr_scheduler_config is not None:
            scheduler = self._setup_lr_scheduler(optimizer)
            return [optimizer], [scheduler]
        else:
            return [optimizer]
