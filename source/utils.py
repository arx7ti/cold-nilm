#!/usr/bin/env python3

###########################################################
#### utils.py ### COLD: Concurrent Loads Disaggregator ####
###########################################################

# System
import re
import os
import numpy as np
from tqdm import tqdm

# Model and the data
from .model import ModelCOLD
from .data import collate, DataModule

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

# Hyperparameters search
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Types
from beartype import beartype
from typing import Union, List, Dict, Tuple, Any, Optional


@beartype
def collect_running_statistics(
    dataset_path: str,
    w_max: int,
) -> Tuple[Any, Any]:
    """
    Runs over the training dataset and collects time-distributed global statistics of magnitudes
    of a signals i.e. mean and std at each time-point

    Arguments:
        dataset_path: str
        w_max: int
    Returns:
        Tuple[Any, Any]
    """
    filenames = [
        filename
        for filename in os.listdir(dataset_path)
        if int(filename.split("-")[0]) <= w_max
    ]
    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        filepath = os.path.join(dataset_path, filename)
        x = np.load(filepath, allow_pickle=True).item()["signal"]
        if i == 0:
            mean = x
            std = np.zeros_like(x)
        else:
            mean = i * mean / (i + 1) + 1 / (i + 1) * x
            # Clip the min value by 0 since due to the floating point problem some small values
            # can have negative sign (e.g. -1e-15)
            std = (
                i / (i + 1) * (std ** 2 + mean_prev ** 2)
                + 1 / (i + 1) * x ** 2
                - mean ** 2
            ).clip(min=0.0)
            std = np.sqrt(std)
        mean_prev = mean
    return mean, std


@beartype
def parse_checkpoint_name(query: str, name: str) -> float:
    return float(re.findall(r"%s=([0]\.\d{4})" % query, name)[0])


@beartype
def get_best_checkpoint(dir_path: str) -> Union[str, None]:
    """
    Arguments:
        dir_path : str - path to the checkpoints dir
    Returns:
        Union[str, None] - path to the checkpoint
    """
    paths = list(map(lambda f: os.path.join(dir_path, f), os.listdir(dir_path)))
    try:
        checkpoint_path = max(paths, key=lambda x: parse_checkpoint_name("f1", x))
    except ValueError:
        checkpoint_path = None
    return checkpoint_path


@beartype
def run_pipeline(
    config: Dict[str, Union[int, float]],
    labels: List[str],
    w_max: int,
    stats: Union[Dict[str, np.ndarray], None],
    variance_gaussian: float,
    p_gaussian: float,
    max_spike: float,
    n_spike: int,
    p_spike: float,
    variance_dc: float,
    p_dc: float,
    lr_scheduler_config: Union[Dict[str, Any], None],
    train_path: str,
    val_path: str,
    max_epochs: int,
    early_stop: int,
    min_delta: float,
    save_top_k: int,
    clip_grad: float,
    pin_memory: bool,
    n_jobs: int,
    n_gpus: float,
    prefetch_factor: int,
    drop_last: bool,
    random_state: int,
    trainer_logger: str,
    neptune_api_key: Optional[str] = None,
    neptune_project: Optional[str] = None,
) -> None:
    pl.seed_everything(random_state, workers=True)
    model = ModelCOLD(
        config,
        len(labels),
        w_max,
        lr_scheduler_config=lr_scheduler_config,
        stats=stats,
        variance_gaussian=variance_gaussian,
        p_gaussian=p_gaussian,
        max_spike=max_spike,
        n_spike=n_spike,
        p_spike=p_spike,
        variance_dc=variance_dc,
        p_dc=p_dc,
    )
    # `shuffle` -> True for training purpose
    data_module = DataModule(
        labels,
        w_max,
        train_path,
        val_path,
        None,
        mini_batch_size=config["mini_batch_size"],
        shuffle=True,
        n_jobs=n_jobs,
        pin_memory=pin_memory,
        prefetch=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate,
    )
    callbacks = []
    if early_stop > 0:
        early_stop_callback = EarlyStopping(
            monitor="validation_loss",
            min_delta=min_delta,
            patience=early_stop,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stop_callback)
    ray_report_callback = TuneReportCallback(
        {
            "loss": "validation_loss",
            "f1": "validation_weighted_mean_f1",
            "t_opt": "validation_t_opt",
        },
        on="validation_end",
    )
    callbacks.append(ray_report_callback)
    checkpoint = ModelCheckpoint(
        monitor="validation_weighted_mean_f1",
        mode="max",
        dirpath="checkpoints",
        filename="{epoch:d}-{validation_weighted_mean_f1:.4f}-{validation_t_opt:.4f}",
        save_top_k=save_top_k,
    )
    callbacks.append(checkpoint)
    # Logger selection
    if trainer_logger == "neptune":
        logger = NeptuneLogger(
            api_key=neptune_api_key,
            project=neptune_project,
            name=tune.get_trial_name(),
        )
    elif trainer_logger == "tensorboard":
        logger = TensorBoardLogger(tune.get_trial_dir(), name="", version=".")
    else:
        raise NotImplementedError("Only Neptune and Tensorboard loggers are supported")
    if isinstance(n_gpus, int):
        pl_gpus = n_gpus
    else:
        pl_gpus = 1
    pipeline = pl.Trainer(
        max_epochs=max_epochs,
        gpus=pl_gpus,
        accelerator="dp",
        callbacks=callbacks,
        num_sanity_val_steps=0,
        val_check_interval=0.9,
        progress_bar_refresh_rate=0,
        gradient_clip_val=clip_grad,
        logger=logger,
        deterministic=True,
    )
    pipeline.fit(model, datamodule=data_module)
    return None


@beartype
def run_tuner(
    config: Dict[str, Union[int, float]],
    labels: List[str],
    w_max: int,
    stats: Union[Dict[str, np.ndarray], None],
    variance_gaussian: float,
    p_gaussian: float,
    max_spike: float,
    n_spike: int,
    p_spike: float,
    variance_dc: float,
    p_dc: float,
    lr_scheduler_config: Union[Dict[str, Any], None],
    train_path: str,
    val_path: str,
    tuner_dir: str,
    name: str,
    trials: int,
    max_epochs: int,
    early_stop: int,
    min_delta: float,
    save_top_k: int,
    grace_period: int,
    clip_grad: float,
    reduction_factor: int,
    pin_memory: bool,
    cpus_per_trial: int,
    gpus_per_trial: float,
    prefetch_factor: int,
    drop_last: bool,
    random_state: int,
    resume: bool,
    trainer_logger: str,
    neptune_api_key: Optional[str] = None,
    neptune_project: Optional[str] = None,
) -> tune.analysis.experiment_analysis.ExperimentAnalysis:
    scheduler = ASHAScheduler(
        max_t=max_epochs, grace_period=grace_period, reduction_factor=reduction_factor
    )
    reporter = CLIReporter(
        parameter_columns=["q", "k", "window_size", "dropout", "learning_rate"],
        metric_columns=["loss", "f1"],
    )
    tuner = tune.run(
        tune.with_parameters(
            run_pipeline,
            labels=labels,
            w_max=w_max,
            stats=stats,
            variance_gaussian=variance_gaussian,
            p_gaussian=p_gaussian,
            max_spike=max_spike,
            n_spike=n_spike,
            p_spike=p_spike,
            variance_dc=variance_dc,
            p_dc=p_dc,
            lr_scheduler_config=lr_scheduler_config,
            train_path=train_path,
            val_path=val_path,
            max_epochs=max_epochs,
            early_stop=early_stop,
            min_delta=min_delta,
            save_top_k=save_top_k,
            clip_grad=clip_grad,
            pin_memory=pin_memory,
            n_jobs=cpus_per_trial,
            n_gpus=gpus_per_trial,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            random_state=random_state,
            trainer_logger=trainer_logger,
            neptune_api_key=neptune_api_key,
            neptune_project=neptune_project,
        ),
        resources_per_trial={"gpu": gpus_per_trial},
        metric="f1",
        mode="max",
        config=config,
        num_samples=trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=name,
        resume=resume,
        local_dir=tuner_dir,
        raise_on_failed_trial=False,
    )
    return tuner
