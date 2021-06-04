#!/usr/bin/env python3

############################################################
#### neural.py ### COLD: Concurrent Loads Disaggregator ####
############################################################

# Prototyping 
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# Evaluation
from sklearn.metrics import f1_score
# Plotting
import seaborn as sb
import matplotlib.pyplot as plt

sb.set()


def init_weights(layer):
    """
    Initializes weights of the layer
    """

    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer.bias, "data"):
        nn.init.constant_(layer.bias, 0.)

    return None


class ResidualPoswiseSubNetwork(nn.Module):
    """
    Torch Module for Residual Position-wise Sub Network
    """

    def __init__(self, q, dropout=0.):
        """
        q :: int -- number of hidden units for each layer (i.e. width)
        dropout :: float -- probability of neuron dropout
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

    def forward(self, x0):
        """
        x :: torch.Tensor -- input tensor of size [Batch x Time x Freqs]
        """

        # Following composition will be computed:
        # x2' = (ReLU∘Norm∘(r+Dropout∘h2∘ReLU∘Dropout∘h1))(x0) | r(x)=(x)
        # 1. x1 = (ReLU∘Dropout∘h1)(x0)
        x1 = nn.functional.relu_(nn.functional.dropout(
            self.h1(x0), p=self.dropout, training=self.training))
        # 2. x2 = (Dropout∘h2)(x1)
        x2 = nn.functional.dropout(
            self.h2(x1), p=self.dropout, training=self.training)
        # 3. x2' = (ReLU∘Norm∘(r+x0))(x2) | r(x)=(x)
        x2_prime = nn.functional.relu_(self.norm(x2 + x0))

        return x2_prime


class ModelCOLD(pl.LightningModule):
    """
    Pytorch Lightning module for COLD architecture and its training/testing pipeline
    """

    def __init__(self, config, v, n_labels, w_max=10, max_thresholds=50, eps=1e-6):
        """
        config :: dict -- dict of hyperparameters
        v :: int -- number of input units (i.e. number of frequency bins)
        n_labels :: int -- number of output units (number of y i.e. #L)
        w_max :: int -- maximum number of concurrent loads in dataset
        max_thresholds :: int -- maximum number of iterations for threshold tuning
        eps :: float -- very small constant 
        """

        super().__init__()

        assert v > 0 and n_labels > 0 and w_max > 0
        self.save_hyperparameters()

        # Assign constants
        self.eps = eps 
        self.w_max = w_max
        self.max_thresholds = max_thresholds
        # COLD layout:
        # RPSN width, number of RPSNs and number of heads for MHSA function
        q, k, n_head = config.get("layout", (256, 14, 8))
        assert q % n_head == 0
        # Dropout probability
        dropout = config.get("dropout", 0.2)
        # Training parameters:
        # Learning rate
        self.lr = config.get("lr", 3e-4)
        # Weight decay
        self.weight_decay = config.get("weight_decay", 0.028)
        # Network learnable parameters
        self.affine = nn.Linear(v, q)
        self.sequence = nn.Sequential()
        # Add k Residual Position-wise Sub-Networks
        for i in range(1, k+1):
            self.sequence.add_module(
                "RPSN@%d" % (2*i+1), ResidualPoswiseSubNetwork(q, dropout=dropout))
        # Multi-Head Self Attention
        self.mhsa = nn.MultiheadAttention(
            embed_dim=q, num_heads=n_head, dropout=dropout)
        # Classification layer
        self.classifier = nn.Linear(q, n_labels)
        # Learnable parameter alpha for sigmoid activation
        self.a = nn.Parameter(torch.ones(1, n_labels), requires_grad=True)
        # Initialization
        init_weights(self.classifier)

    def forward(self, x0):
        """
        x0 :: torch.Tensor -- batch of spectrograms [Batch x Freqs x Time]
        """
        # Transpose to get the right dimensions [Batch x Time x Freqs]
        x0 = x0.transpose(1, 2)
        # Affine projection to match the dimensions with network width `q`
        x1 = self.affine(x0)
        # Pass the projected spectrograms x1 through the sequence of RPSNs
        # That is, compute x'_(2k+1) = (RPSN_(2k+1)∘...∘RPSN_3)(x1)
        x_prime = self.sequence(x1)
        # Compute MHSA
        x_prime = x_prime.transpose(0, 1)
        x_att, _ = self.mhsa(x_prime, x_prime, x_prime)
        x_att = x_att.transpose(0, 1)
        # Global pooling across time dimension
        x_global = x_att.mean(dim=1)
        # Output of classification layer
        x_class = self.classifier(x_global)
        # Class y_hat (predictions)
        y_hat = 1/(1+self.a*torch.exp(-x_class))
        # To avoid NaN in loss function
        y_hat = y_hat.clip(min=self.eps)

        return y_hat

    def loss(self, y_hat, y):
        """
        Computes Binary Cross Entropy loss
        ---
        y_hat :: torch.Tensor
        y :: torch.Tensor
        """
        return nn.functional.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, *args):
        """
        Performs 1 gradient update
        ---
        batch :: torch.Tensor
        """
        x0, y = batch
        y_hat = self.forward(x0)
        loss = self.loss(y_hat, y)
        # Track results
        self.log("train_loss", loss)
        self.log("train_loss", loss, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, *args):
        """
        Performs 1 validation step
        ---
        batch :: torch.Tensor
        """
        x0, y = batch
        y_hat = self.forward(x0)
        loss = self.loss(y_hat, y)
        # Track results
        self.log("validation_loss", loss, on_epoch=True)

        return {"y_hat": y_hat, "y": y}

    def test_step(self, batch, *args):
        """
        Performs 1 test step
        ---
        batch :: torch.Tensor
        """
        x0, y = batch
        y_hat = self.forward(x0)

        return {"y_hat": y_hat, "y": y}

    def validation_step_end(self, parallel_batch):
        return parallel_batch

    def test_step_end(self, parallel_batch):
        return parallel_batch

    def _compute_f1(self, y, y_hat, t, average=None):
        """
        y :: np.ndarray
        y_hat :: np.ndarray
        """
        # Keep f1-scores and relative cardinalities of each w-subset of S
        f1_values = []
        weights = []
        for w in range(1, self.w_max+1):
            mask = y.sum(1) == w
            f1_w = f1_score(y[mask], np.where(
                y_hat[mask] > t, 1., 0.), average=average)
            weights.append(len(mask)/len(y))
            f1_values.append(f1_w)

        return np.array(f1_values).astype(np.float32), weights

    def _evaluate(self, y, y_hat, prefix):
        """
        Computes optimal threshold for decision making
        ---
        y :: np.ndarray
        y_hat :: np.ndarray
        prefix :: str
        """
        # Initial value for threshold
        t_opt = 1e-2
        # We will use weighted mean F1 as a reference metric
        # for finding the optimal threshold
        weighted_mean_f1 = 0
        mean_f1_values = None
        for t in np.linspace(0, 1, self.max_thresholds):
            # Compute temporary mean F1-score values across w
            tmp_mean_f1_values, weights = self._compute_f1(
                y, y_hat, t, average="samples")
            # Compute temporary weighted mean F1-score
            tmp_weighted_mean_f1 = np.average(
                tmp_mean_f1_values, weights=weights)
            if tmp_weighted_mean_f1 > weighted_mean_f1:
                weighted_mean_f1 = tmp_weighted_mean_f1
                mean_f1_values = tmp_mean_f1_values
                t_opt = t

        if prefix == "test":
            f1_values, weights = self._compute_f1(
                y, y_hat, t_opt, average=None)
            weighted_f1_values = np.average(
                f1_values, axis=0, weights=weights)
        else:
            weighted_f1_values = None

        return t_opt, weighted_mean_f1, mean_f1_values, weighted_f1_values

    def _shared_epoch_end(self, parallel_data, prefix):
        """
        Summarizes validation/test procedure
        parallel data :: list
        prefix :: str -- either "val" or "test", depending on procedure
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
        for w, mean_f1_w in zip(range(1, self.w_max+1), mean_f1_values):
            self.log("%s_mean_f1_w=%d" % (prefix, w), mean_f1_w)

        # Data visualization
        if prefix == "test":
            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(1, self.w_max+1), mean_f1_values, alpha=0.66)
            plt.xlabel("Number of concurrent loads (w)", fontsize=14)
            plt.ylabel("Mean F1-score", fontsize=14)
            plt.xticks(np.arange(1, self.w_max+1))
            plt.grid(True)
            plt.ylim((0.9*np.min(weighted_f1_values), 1))
            plt.savefig("./plots/per_w.svg", bbox_inches="tight")
            plt.show()

            labels = self.trainer.datamodule.test_dataset.label_encoder.classes_
            args = np.argsort(weighted_f1_values)
            plt.figure(figsize=(12, 5))
            plt.bar(labels[args], weighted_f1_values[args], alpha=0.66)
            plt.xlabel("", fontsize=14)
            plt.ylabel("F1-score", fontsize=14)
            plt.ylim((0.9*np.min(weighted_f1_values), 1))
            plt.xticks(rotation=90)
            plt.tick_params(axis="x", labelsize=14)
            plt.grid(True)
            plt.savefig("./plots/per_label.svg", bbox_inches="tight")
            plt.show()
            for label, f1 in zip(labels[args], weighted_f1_values[args]):
                self.log("f1_%s" % label, f1)

    def validation_epoch_end(self, parallel_data):
        self._shared_epoch_end(parallel_data, "validation")

    def test_epoch_end(self, parallel_data):
        self._shared_epoch_end(parallel_data, "test")

    def configure_optimizers(self):
        """
        Choice of the optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)

        return [optimizer]
