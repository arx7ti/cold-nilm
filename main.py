#!/usr/bin/env python3

# System
import os
import sys
import logging
import warnings
from json import dumps
from hashlib import md5
from argparse import ArgumentParser
from distutils.util import strtobool
from source.wrappers import CustomConfigParser

# Types
from beartype import beartype
from typing import List, Dict, Optional

# Calculus
import numpy as np

# Paper modules
import source.utils as utils
import source.synthesizer as synthesizer
from source.model import ModelCOLD
from source.data import collate, DataModule

# Ray Tune
# will be used during eval()
from ray.tune import randint, choice, loguniform

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


warnings.filterwarnings("ignore")


@beartype
def check_config_sections(config: CustomConfigParser, sections: List[str]) -> None:
    for section in sections:
        if not config.has_section(section):
            raise Exception("ConfigError", 'Section "%s" is required' % section)


@beartype
def check_config_options(
    config: CustomConfigParser, section: str, options: List[str]
) -> None:
    for option in options:
        if not config.has_option(section, option):
            raise Exception(
                "ConfigError",
                'Option "%s" for section "%s" is required' % (option, section),
            )


@beartype
def request_running_stats(
    running_stats_path: str,
    train_dataset_path: Optional[str] = None,
    w_max: Optional[int] = None,
    make_logs: bool = False,
):
    stats_exist = os.path.exists(running_stats_path)
    # Collect train dataset statistics for the further normalization
    if stats_exist:
        logging.info('Statistics of "train" dataset for `w_max`=%d found.' % w_max)
    else:
        assert train_dataset_path is not None and w_max is not None
        if make_logs:
            logging.info(
                'Statistics of "train" dataset for `w_max`=%d not found. Collecting...'
                % w_max
            )
        mean, std = utils.collect_running_statistics(train_dataset_path, w_max)
        if make_logs:
            logging.info("Statistics collected successfully. Saving...")
        np.save(running_stats_path, {"mean": mean, "std": std})
        if make_logs:
            logging.info('Statistics saved to "%s"' % running_stats_path)
    stats = np.load(running_stats_path, allow_pickle=True).item()
    return stats


@beartype
def run_sns(config: CustomConfigParser, kwargs: Dict[str, str]) -> None:
    make_logs = True if kwargs["log"] == "y" else False
    min_cardinality = int(kwargs["min_cardinality"])
    n_jobs = int(kwargs["n_jobs"])
    # Config sections verification
    check_config_sections(config, ["signal", "filter", "paths", "task", "extra"])
    # Matching map section
    matching_map = (
        dict(config["matching_map"]) if config.has_section("matching_map") else {}
    )
    # Signal section / check and parse
    check_config_options(
        config,
        "signal",
        ["duration", "forbidden_interval", "voltage", "frequency", "sampling_rate"],
    )
    duration = int(config["signal"].get("duration", None))
    forbidden_interval = float(config["signal"].get("forbidden_interval", None))
    voltage = float(config["signal"].get("voltage", None))
    frequency = float(config["signal"].get("frequency", None))
    sampling_rate = int(config["signal"].get("sampling_rate", None))
    # Filter section / check and parse
    check_config_options(config, "filter", ["thd_threshold", "activation_threshold"])
    thd_threshold = float(config["filter"].get("thd_threshold", None))
    activation_threshold = float(config["filter"].get("activation_threshold", None))
    # Paths section / check and parse
    check_config_options(config, "paths", ["whited_dir", "plaid_dir", "save_dir"])
    whited_path = config["paths"].get("whited_dir", None)
    plaid_path = config["paths"].get("plaid_dir", None)
    save_dir = config["paths"].get("save_dir", None)
    # Task section / check and parse
    check_config_options(config, "task", ["datasets", "split_ratios"])
    datasets = config["task"].get("datasets", None).split(",")
    for dataset_name in datasets:
        if dataset_name not in ["train", "validation", "test"]:
            raise NotImplementedError
    split_ratios = list(map(float, config["task"].get("split_ratios", None).split(",")))
    labels = [label for label in config["task"].get("labels", "").split(",") if label]
    # Extra section / check and parse
    check_config_options(config, "extra", ["vicinity", "limit", "random_state"])
    vicinity = int(config["extra"].get("vicinity", None))
    limit = int(config["extra"].get("limit", None))
    random_state = int(config["extra"].get("random_state", None))
    # Load normalized signatures or create them from the WHITED and the PLAID
    if make_logs:
        logging.info("Starting the SNS algorithm...")
    labels_path = os.path.join(save_dir, "labels")
    patterns_hash = md5(
        dumps(
            {**config["matching_map"], **config["signal"], **config["filter"]},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    patterns_path = os.path.join(save_dir, "patterns-%s.npy" % patterns_hash)
    os.makedirs(save_dir, exist_ok=True)
    try:
        patterns = np.load(patterns_path, allow_pickle=True).item()
        if make_logs:
            logging.info("Normalized signatures found and loaded successfully.")
    except FileNotFoundError:
        if make_logs:
            logging.info("Normalized signatures not found. Processing from raw data...")
        whited_dataset = synthesizer.read_whited(
            whited_path,
            downsampling_rate=sampling_rate,
            voltage_standard=voltage,
            frequency_standard=frequency,
            thd_thresh=thd_threshold,
            max_duration=duration,
            vicinity=vicinity,
            limit=limit,
            make_logs=make_logs,
        )
        plaid_dataset = synthesizer.read_plaid(
            plaid_path,
            downsampling_rate=sampling_rate,
            voltage_standard=voltage,
            frequency_standard=frequency,
            activation_thresh=activation_threshold,
            thd_thresh=thd_threshold,
            max_duration=duration,
            vicinity=vicinity,
            limit=limit,
            matching_map=matching_map,
            make_logs=make_logs,
        )
        patterns = synthesizer.merge_datasets(
            (whited_dataset, plaid_dataset),
            make_logs,
        )
        if make_logs:
            logging.info("Signatures normalized successfully. Saving...")
        np.save(patterns_path, patterns)
        if make_logs:
            logging.info(
                'Normalized signatures saved successfully to "%s"' % patterns_path
            )
    synthesizer.drop_empty_categories(patterns, min_cardinality, True, make_logs)
    # Drop categories which are not in `labels`
    if len(labels) > 0:
        labels_to_write = labels
        synthesizer.filter_categories(patterns, labels, True, make_logs)
    else:
        labels_to_write = list(patterns.keys())
    with open(labels_path, "w+") as labels_file:
        labels_file.write(",".join(labels_to_write))
    # Train, validation, test splits
    if make_logs:
        logging.info(
            "Splitting set of normalized signatures into %d subsets..." % len(datasets)
        )
    subsets = synthesizer.split(
        patterns, ratios=split_ratios, random_state=random_state, make_logs=make_logs
    )
    # Build datasets
    for dataset_name in datasets:
        if make_logs:
            logging.info('Building "%s" dataset...' % dataset_name)
        check_config_sections(config, [dataset_name + "_limits"])
        subset_path = os.path.join(save_dir, dataset_name)
        patterns_subset = subsets[dataset_name]
        # Will be used during eval()
        median, size = synthesizer.get_stats(subsets[dataset_name])
        limits = {}
        for k, v in config[dataset_name + "_limits"].items():
            vals = v.split(",")
            limits.update({int(k): (eval(vals[0]), eval(vals[1]))})
        synthesizer.build_dataset(
            dataset_name,
            patterns_subset,
            limits,
            dataset_path=subset_path,
            signal_duration=duration,
            forbidden_interval=forbidden_interval,
            voltage_standard=voltage,
            fundamental=frequency,
            sampling_rate=sampling_rate,
            make_logs=make_logs,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        if make_logs:
            logging.info('"%s" dataset is ready.' % dataset_name)
    if make_logs:
        logging.info("The SNS algorithm terminated successfully.")
    return None


@beartype
def run_tuner(config: CustomConfigParser, kwargs: Dict[str, str]) -> None:
    trainer_logger = kwargs["trainer_logger"]
    n_trials = int(kwargs["n_trials"])
    n_epochs = int(kwargs["n_epochs"])
    early_stop = int(kwargs["early_stop"])
    grace_period = int(kwargs["grace_period"])
    min_delta = float(kwargs["min_delta"])
    save_top_k = int(kwargs["save_top_k"])
    reduction_factor = int(kwargs["reduction_factor"])
    pin_memory = True if kwargs["pin_memory"] == "y" else False
    n_jobs = int(kwargs["n_jobs"])
    n_gpus = float(kwargs["n_gpus"])
    prefetch_factor = int(kwargs["prefetch_factor"])
    drop_last = True if kwargs["drop_last"] == "y" else False
    resume = True if kwargs["resume"] == "y" else False
    config_path = kwargs["config"]
    make_logs = True if kwargs["log"] == "y" else False
    # Config sections verification
    check_config_sections(config, ["hyperparameters", "model", "paths", "extra"])
    # Hyperparameters section / check and parse
    check_config_options(
        config,
        "hyperparameters",
        [
            "learning_rate",
            "weight_decay",
            "mini_batch_size",
            "dropout",
            "q",
            "k",
            "n_head",
            "window_size",
            "hop_size",
        ],
    )
    hyperparameters = {k: eval(v) for k, v in dict(config["hyperparameters"]).items()}
    # Model section / check and parse
    check_config_options(
        config,
        "model",
        [
            "w_max",
            "variance_gaussian",
            "p_gaussian",
            "max_spike",
            "n_spike",
            "p_spike",
            "variance_dc",
            "p_dc",
        ],
    )
    w_max = int(config["model"].get("w_max", None))
    variance_gaussian = float(config["model"].get("variance_gaussian", None))
    p_gaussian = float(config["model"].get("p_gaussian", None))
    max_spike = float(config["model"].get("max_spike", None))
    n_spike = int(config["model"].get("n_spike", None))
    p_spike = float(config["model"].get("p_spike", None))
    variance_dc = float(config["model"].get("variance_dc", None))
    p_dc = float(config["model"].get("p_dc", None))
    # Reduce lr on plateau
    if config.has_section("reduce_lr_on_plateau"):
        scheduler_mode = config["reduce_lr_on_plateau"].get("mode", None)
        scheduler_patience = float(config["reduce_lr_on_plateau"].get("patience", None))
        scheduler_factor = float(config["reduce_lr_on_plateau"].get("factor", None))
        scheduler_min_lr = float(config["reduce_lr_on_plateau"].get("min_lr", None))
        scheduler_monitor_value = config["reduce_lr_on_plateau"].get("monitor", None)
        lr_scheduler_config = {
            "kwargs": {
                "mode": scheduler_mode,
                "patience": scheduler_patience,
                "factor": scheduler_factor,
                "min_lr": scheduler_min_lr,
            },
            "monitor": scheduler_monitor_value,
        }
    else:
        lr_scheduler_config = None
    # Paths section / check and parse
    check_config_options(config, "paths", ["tuner_dir", "synthetic_dir"])
    tuner_dir = config["paths"].get("tuner_dir", None)
    os.makedirs(tuner_dir, exist_ok=True)
    synthetic_dir = config["paths"].get("synthetic_dir", None)
    # Extra section / check and parse
    check_config_options(
        config,
        "extra",
        ["use_normalized_data", "random_state", "clip_grad"],
    )
    use_normalized_data = bool(
        strtobool(config["extra"].get("use_normalized_data", None))
    )
    random_state = int(config["extra"].get("random_state", None))
    clip_grad = float(config["extra"].get("clip_grad", None))
    # Paths
    labels_path = os.path.join(synthetic_dir, "labels")
    if os.path.isabs(synthetic_dir):
        train_dataset_path = os.path.join(synthetic_dir, "train")
        validation_dataset_path = os.path.join(synthetic_dir, "validation")
    else:
        train_dataset_path = os.path.join(os.getcwd(), synthetic_dir, "train")
        validation_dataset_path = os.path.join(os.getcwd(), synthetic_dir, "validation")
    # Labels available
    with open(labels_path, "r") as labels_file:
        labels = sorted(labels_file.readline().split(","))
    if resume:
        if make_logs:
            logging.info("Resuming the tuner...")
        else:
            pass
    else:
        if make_logs:
            logging.info("Starting the tuner...")
        else:
            pass
    if use_normalized_data:
        # Path to the running statistics
        running_stats_path = os.path.join(tuner_dir, "stats-w_max=%d.npy" % w_max)
        stats = request_running_stats(
            running_stats_path, train_dataset_path, w_max, make_logs
        )
    else:
        stats = None
    if trainer_logger == "neptune":
        check_config_options(config, "extra", ["neptune_api_key", "neptune_project"])
    neptune_api_key = config["extra"].get("neptune_api_key", None)
    neptune_project = config["extra"].get("neptune_project", None)
    tuner = utils.run_tuner(
        hyperparameters,
        labels,
        w_max,
        stats,
        variance_gaussian,
        p_gaussian,
        max_spike,
        n_spike,
        p_spike,
        variance_dc,
        p_dc,
        lr_scheduler_config,
        train_dataset_path,
        validation_dataset_path,
        tuner_dir,
        name="cold",
        trials=n_trials,
        max_epochs=n_epochs,
        early_stop=early_stop,
        min_delta=min_delta,
        save_top_k=save_top_k,
        grace_period=grace_period,
        clip_grad=clip_grad,
        reduction_factor=reduction_factor,
        pin_memory=pin_memory,
        cpus_per_trial=n_jobs,
        gpus_per_trial=n_gpus,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        random_state=random_state,
        resume=resume,
        trainer_logger=trainer_logger,
        neptune_api_key=neptune_api_key,
        neptune_project=neptune_project,
    )
    if make_logs:
        logging.info("Tuner terminated successfully.")
    # Get best trial
    best_trial = tuner.get_best_trial(metric="f1", mode="max", scope="all")
    trial_dir = best_trial.logdir
    checkpoints_dir = os.path.join(trial_dir, "checkpoints")
    checkpoint_path = utils.get_best_checkpoint(checkpoints_dir)
    checkpoint_name = os.path.basename(checkpoint_path)
    checkpoint_t_opt = utils.parse_checkpoint_name("t_opt", checkpoint_name)
    # Create new config for the best model
    models_path = os.path.join(os.path.split(tuner_dir)[0], "models")
    model_config = CustomConfigParser()
    model_config["hyperparameters"] = tuner.best_config
    model_config["model"] = {
        "w_max": w_max,
        "threshold": checkpoint_t_opt,
        "variance_gaussian": variance_gaussian,
        "p_gaussian": p_gaussian,
        "max_spike": max_spike,
        "n_spike": n_spike,
        "p_spike": p_spike,
        "variance_dc": variance_dc,
        "p_dc": p_dc,
    }
    if config.has_section("reduce_lr_on_plateau"):
        model_config["reduce_lr_on_plateau"] = config["reduce_lr_on_plateau"]
    model_config["paths"] = {
        "synthetic_dir": synthetic_dir,
        "models_dir": models_path,
        "weights": checkpoint_path,
    }
    model_config["extra"] = {
        "use_normalized_data": str(use_normalized_data).lower(),
        "clip_grad": clip_grad,
        "random_state": random_state,
    }
    model_config_path = os.path.join(os.path.dirname(config_path), "model.ini")
    if make_logs:
        logging.info(
            "Best hyperparameters found (%s)\nSaving into the config file ..."
            % " / ".join(["%s = %f" % (k, v) for k, v in tuner.best_config.items()])
        )
    with open(model_config_path, "w+") as config_file:
        model_config.write(config_file)
    if make_logs:
        logging.info('Config file saved to "%s"' % model_config_path)
    return None


@beartype
def run_model(config: CustomConfigParser, kwargs: Dict[str, str]) -> Dict[str, float]:
    name = kwargs["name"]
    trainer_logger = kwargs["trainer_logger"]
    train = True if kwargs["train"] == "y" else False
    override_w_max = int(kwargs["override_w_max"])
    n_epochs = int(kwargs["n_epochs"])
    early_stop = int(kwargs["early_stop"])
    min_delta = float(kwargs["min_delta"])
    save_top_k = int(kwargs["save_top_k"])
    pin_memory = True if kwargs["pin_memory"] == "y" else False
    n_jobs = int(kwargs["n_jobs"])
    n_gpus = int(kwargs["n_gpus"])
    prefetch_factor = int(kwargs["prefetch_factor"])
    drop_last = True if kwargs["drop_last"] == "y" else False
    resume = True if kwargs["resume"] == "y" else False
    make_logs = True if kwargs["log"] == "y" else False
    # Config sections verification
    check_config_sections(config, ["hyperparameters", "model", "paths", "extra"])
    # Hyperparameters section / check and parse
    check_config_options(
        config,
        "hyperparameters",
        [
            "learning_rate",
            "weight_decay",
            "mini_batch_size",
            "dropout",
            "q",
            "k",
            "n_head",
            "window_size",
            "hop_size",
        ],
    )
    hyperparameters = {k: eval(v) for k, v in dict(config["hyperparameters"]).items()}
    # Model section / check and parse
    check_config_options(
        config,
        "model",
        [
            "w_max",
            "variance_gaussian",
            "p_gaussian",
            "max_spike",
            "n_spike",
            "p_spike",
            "variance_dc",
            "p_dc",
        ],
    )
    w_max = int(config["model"].get("w_max", None))
    # Threshold is required for the test procedure only
    if "threshold" in config["model"].keys():
        threshold = float(config["model"].pop("threshold"))
    else:
        threshold = None
    variance_gaussian = float(config["model"].get("variance_gaussian", None))
    p_gaussian = float(config["model"].get("p_gaussian", None))
    max_spike = float(config["model"].get("max_spike", None))
    n_spike = int(config["model"].get("n_spike", None))
    p_spike = float(config["model"].get("p_spike", None))
    variance_dc = float(config["model"].get("variance_dc", None))
    p_dc = float(config["model"].get("p_dc", None))
    # Reduce lr on plateau
    if config.has_section("reduce_lr_on_plateau"):
        scheduler_mode = config["reduce_lr_on_plateau"].get("mode", None)
        scheduler_patience = float(config["reduce_lr_on_plateau"].get("patience", None))
        scheduler_factor = float(config["reduce_lr_on_plateau"].get("factor", None))
        scheduler_min_lr = float(config["reduce_lr_on_plateau"].get("min_lr", None))
        scheduler_monitor_value = config["reduce_lr_on_plateau"].get("monitor", None)
        lr_scheduler_config = {
            "kwargs": {
                "mode": scheduler_mode,
                "patience": scheduler_patience,
                "factor": scheduler_factor,
                "min_lr": scheduler_min_lr,
            },
            "monitor": scheduler_monitor_value,
        }
    else:
        lr_scheduler_config = None
    # Paths section / check and parse
    paths_options_to_check = ["synthetic_dir", "models_dir"]
    check_config_options(config, "paths", paths_options_to_check)
    synthetic_dir = config["paths"].get("synthetic_dir", None)
    models_dir = config["paths"].get("models_dir", None)
    # Extra section / check and parse
    check_config_options(
        config,
        "extra",
        ["use_normalized_data", "random_state", "clip_grad"],
    )
    use_normalized_data = bool(
        strtobool(config["extra"].get("use_normalized_data", None))
    )
    random_state = int(config["extra"].get("random_state", None))
    clip_grad = float(config["extra"].get("clip_grad", None))
    # Labels available
    labels_path = os.path.join(synthetic_dir, "labels")
    with open(labels_path, "r") as labels_file:
        labels = sorted(labels_file.readline().split(","))
    # Freeze the environment
    pl.seed_everything(random_state, workers=True)
    # Version control
    if config.has_section("reduce_lr_on_plateau"):
        reduce_lr_on_plateau_as_dict = config["reduce_lr_on_plateau"]
    else:
        reduce_lr_on_plateau_as_dict = {}
    config_hash = md5(
        dumps(
            {
                **config["hyperparameters"],
                **config["model"],
                **reduce_lr_on_plateau_as_dict,
                "use_normalized_data": use_normalized_data,
                "clip_grad": clip_grad,
                "random_state": random_state,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    # Paths
    log_dir = os.path.join(models_dir, config_hash, "logs")
    checkpoints_dir = os.path.join(models_dir, config_hash, "checkpoints")
    plots_dir = os.path.join(models_dir, config_hash, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    # Callbacks
    callbacks = []
    if train:
        checkpoints_callback = ModelCheckpoint(
            monitor="validation_weighted_mean_f1",
            dirpath=checkpoints_dir,
            filename="{epoch:d}-{validation_weighted_mean_f1:.4f}-{validation_t_opt:.4f}",
            save_top_k=save_top_k,
            mode="max",
        )
        callbacks.append(checkpoints_callback)
    if train and early_stop > 0:
        early_stop_callback = EarlyStopping(
            monitor="validation_loss",
            min_delta=min_delta,
            patience=early_stop,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stop_callback)
    # Logger selection
    if trainer_logger == "neptune" and train:
        check_config_options(config, "extra", ["neptune_api_key", "neptune_project"])
        neptune_api_key = config["extra"].get("neptune_api_key", None)
        neptune_project = config["extra"].get("neptune_project", None)
        logger = NeptuneLogger(
            api_key=neptune_api_key,
            project=neptune_project,
            name=name,
        )
    elif trainer_logger == "tensorboard" and train:
        logger = TensorBoardLogger(log_dir, name=name)
    elif not train:
        logger = None
    else:
        raise NotImplementedError("Only Neptune and Tensorboard loggers are supported")
    if train and resume:
        if make_logs:
            logging.info("Resuming the training...")
        checkpoint_path = utils.get_best_checkpoint(checkpoints_dir)
    else:
        if make_logs:
            logging.info("Starting the model training...")
        checkpoint_path = None
    pipeline = pl.Trainer(
        max_epochs=n_epochs,
        gpus=n_gpus,
        accelerator="dp",
        num_sanity_val_steps=0,
        callbacks=callbacks,
        val_check_interval=0.9,
        logger=logger,
        gradient_clip_val=clip_grad,
        resume_from_checkpoint=checkpoint_path,
        deterministic=True,
    )
    test_dataset_path = os.path.join(synthetic_dir, "test")
    # Path to the running statistics
    running_stats_path = os.path.join(models_dir, "stats-w_max=%d.npy" % w_max)
    stats_exist = os.path.exists(running_stats_path)
    if train:
        validation_dataset_path = os.path.join(synthetic_dir, "validation")
    else:
        validation_dataset_path = None
    if train or (use_normalized_data and not stats_exist):
        train_dataset_path = os.path.join(synthetic_dir, "train")
    else:
        train_dataset_path = None
    if use_normalized_data:
        stats = request_running_stats(
            running_stats_path, train_dataset_path, w_max, make_logs
        )
    else:
        stats = None
    data_module = DataModule(
        labels,
        w_max,
        train_dataset_path,
        validation_dataset_path,
        test_dataset_path,
        mini_batch_size=hyperparameters["mini_batch_size"],
        n_jobs=n_jobs,
        pin_memory=pin_memory,
        prefetch=prefetch_factor,
        drop_last=drop_last,
        collate_fn=collate,
    )
    # Train/test pipeline
    if train:
        model = ModelCOLD(
            hyperparameters,
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
            plots_dir=plots_dir,
        )
        pipeline.fit(model, datamodule=data_module)
        if make_logs:
            logging.info("Model trained successfully.")
        # Weights path after a training
        weights_path = utils.get_best_checkpoint(checkpoints_dir)
        checkpoint_name = os.path.basename(weights_path)
        # Override the optimal threshold value
        threshold = utils.parse_checkpoint_name("t_opt", checkpoint_name)
        # Drop the logger from the Trainer
        pipeline.logger = None
        # Update the config with the optimal threshold
        config.set("model", "threshold", str(threshold))
        with open(config.path, "w+") as config_file:
            config.write(config_file)
        del model
    else:
        weights_path = utils.get_best_checkpoint(checkpoints_dir)
        override_weights = config.has_option("paths", "weights")
        if weights_path is None and not override_weights:
            if make_logs:
                logging.error(
                    "Weights not found. Train the model or define `weights` option in the `paths` section of the configuration file."
                )
            check_config_options(config, "paths", ["weights"])
        elif override_weights:
            # Override the weights path if one set
            weights_path = config["paths"]["weights"]
    if make_logs:
        logging.info('Loading the weights from "%s"' % weights_path)
    model = ModelCOLD.load_from_checkpoint(weights_path)
    if make_logs:
        logging.info("Weights loaded successfully.")
    # TODO add hyper-parameters match validation between config file and the loaded model
    # it is needed due to the overrided `weights`
    assert w_max == model.w_max
    # Override `w_max`
    if override_w_max != 0:
        assert override_w_max > 0
        model.w_max = override_w_max
    model.threshold = threshold
    model.plots_dir = plots_dir
    if make_logs:
        logging.info("Starting the model testing...")
    results = pipeline.test(model, datamodule=data_module, verbose=False)[0]
    results_copy = results.copy()
    optimal_theshold = results.pop("test_t_opt")
    if make_logs:
        logging.info(
            "Model tested successfully: \n%s\nUnder the optimal threshold = %.4f"
            % (
                "\n".join(["%s = %.2f%%" % (k, 100 * v) for k, v in results.items()]),
                optimal_theshold,
            )
        )
    return results_copy


if __name__ == "__main__":
    SNS_PARSER_CMD = "run-sns"
    TUNER_PARSER_CMD = "run-tuner"
    MODEL_PARSER_CMD = "run-model"
    parser = ArgumentParser(
        description="Welcome to the Concurrent Loads Disaggregator (COLD) pipeline"
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        choices=["y", "n"],
        default="y",
        help="Print and save logs",
    )
    parser.add_argument(
        "-d", "--logdir", type=str, default="./", help="Directory to save logs file"
    )
    subparsers = parser.add_subparsers(
        help="For the first time you are supposed to run each section one by one"
    )
    sns_parser = subparsers.add_parser(
        SNS_PARSER_CMD,
        help="Synthesizer of Normalized Signatures algorithm",
    )
    sns_parser.set_defaults(task=SNS_PARSER_CMD)
    tuner_parser = subparsers.add_parser(
        TUNER_PARSER_CMD,
        help="Hyperparameters tuning pipeline for the COLD architecture",
    )
    tuner_parser.set_defaults(task=TUNER_PARSER_CMD)
    model_parser = subparsers.add_parser(
        MODEL_PARSER_CMD, help="Train and test the COLD model"
    )
    model_parser.set_defaults(task=MODEL_PARSER_CMD)
    # SNS commands
    sns_parser.add_argument(
        "-mc",
        "--min-cardinality",
        type=int,
        default=5,
        help="Minimal number of signatures within a category of appliances",
    )
    sns_parser.add_argument(
        "-j", "--n-jobs", type=int, default=1, help="Number of CPU's threads to utilize"
    )
    sns_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/sns.ini",
        help="Path to the configuration file",
    )
    # Tuner commands
    tuner_parser.add_argument(
        "-tl",
        "--trainer-logger",
        type=str,
        choices=["tensorboard", "neptune"],
        default="tensorboard",
        help="Trainer logger (tensorboard, neptune)",
    )
    tuner_parser.add_argument(
        "-t", "--n-trials", type=int, default=1, help="Number of trials"
    )
    tuner_parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=50,
        help="Max number of epochs per 1 trial",
    )
    tuner_parser.add_argument(
        "-es",
        "--early-stop",
        type=int,
        default=0,
        help="Number of epochs without improvements. 0 to disable",
    )
    tuner_parser.add_argument(
        "-gp", "--grace-period", type=int, default=5, help="Grace period in epochs"
    )
    tuner_parser.add_argument(
        "-d",
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute change in the loss to qualify as an improvement",
    )
    tuner_parser.add_argument(
        "-s",
        "--save-top-k",
        type=int,
        default=3,
        help="Save checkpoints of top k models",
    )
    tuner_parser.add_argument(
        "-rf",
        "--reduction-factor",
        type=int,
        default=4,
        help="Used to set halving rate and amount for ASHA",
    )
    tuner_parser.add_argument(
        "-m",
        "--pin-memory",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Pin memory to speed up training",
    )
    tuner_parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help="Number of CPU's to utilize per 1 trial",
    )
    tuner_parser.add_argument(
        "-g",
        "--n-gpus",
        type=float,
        default=1.0,
        help="Number of GPUs to utilize per 1 trial",
    )
    tuner_parser.add_argument(
        "-pf",
        "--prefetch-factor",
        type=int,
        default=4,
        help="Prefetch factor for data loader",
    )
    tuner_parser.add_argument(
        "-dl",
        "--drop-last",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Drop last batch",
    )
    tuner_parser.add_argument(
        "-r",
        "--resume",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Resume tuner if interrupted",
    )
    tuner_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/tuner.ini",
        help="Path to the configuration file",
    )
    # Model commands
    model_parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="cold",
        help="Name of the training attempt",
    )
    model_parser.add_argument(
        "-tl",
        "--trainer-logger",
        type=str,
        choices=["tensorboard", "neptune"],
        default="tensorboard",
        help="Trainer logger (tensorboard, neptune)",
    )
    model_parser.add_argument(
        "-t",
        "--train",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Train model from scratch",
    )
    model_parser.add_argument(
        "-w",
        "--override-w-max",
        type=int,
        default=0,
        help="Override `w_max` attribute for the model testing; if 0 no override will be provided",
    )
    model_parser.add_argument(
        "-e", "--n-epochs", type=int, default=50, help="Max number of epochs"
    )
    model_parser.add_argument(
        "-es",
        "--early-stop",
        type=int,
        default=0,
        help="Number of epochs without improvements. 0 to disable",
    )
    model_parser.add_argument(
        "-d",
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute change in the loss to qualify as an improvement",
    )
    model_parser.add_argument(
        "-s",
        "--save-top-k",
        type=int,
        default=3,
        help="Save checkpoints of top k models",
    )
    model_parser.add_argument(
        "-m",
        "--pin-memory",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Pin memory to speed up training",
    )
    model_parser.add_argument(
        "-g", "--n-gpus", type=int, default=0, help="Number of GPUs to utilize"
    )
    model_parser.add_argument(
        "-j", "--n-jobs", type=int, default=1, help="Number of CPU's threads to utilize"
    )
    model_parser.add_argument(
        "-pf",
        "--prefetch-factor",
        type=int,
        default=4,
        help="Prefetch factor for data loader",
    )
    model_parser.add_argument(
        "-dl",
        "--drop-last",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Drop last batch",
    )
    model_parser.add_argument(
        "-r",
        "--resume",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Resume training if interrupted",
    )
    model_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/model.ini",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    task = args.task
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    config_path = args.config
    if not os.path.exists(config_path):
        print("Config file is required")
    config = CustomConfigParser(path=config_path)
    # Logger setup
    log_dir = vars(args)["logdir"]
    logs_path = os.path.join(log_dir, ".log")
    logging.basicConfig(
        filename=logs_path,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
    if task == "run-sns":
        run_sns(config, vars(args))
    elif task == "run-tuner":
        run_tuner(config, vars(args))
    elif task == "run-model":
        run_model(config, vars(args))
    sys.exit(0)
