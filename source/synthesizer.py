#!/usr/bin/env python3

#################################################################
#### synthesizer.py ### COLD: Concurrent Loads Disaggregator ####
#################################################################

# Signal processing
import scipy
import librosa
import torchaudio

# Calculus
import numpy as np
import pandas as pd
from math import comb
from lmfit import Model, Parameters

# System
import os
import json
import logging
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from .wrappers import wrap_generator
from pathos.multiprocessing import cpu_count

# Types
from beartype import beartype
from typing import Union, List, Tuple, Dict, Callable


@beartype
def remove_fluctuating_frequency(
    voltage: np.ndarray, current: np.ndarray, reference_semiperiod: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes fluctuating frequency behavior from the given waveforms.
    A voltage waveform is used as a reference to align the current waveform

    Arguments:
        voltage: np.ndarray
        current: np.ndarray
        reference_semiperiod: int
    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    voltage_windows = []
    current_windows = []
    zero_crossings = np.argwhere(np.diff(np.sign(voltage)) != 0.0).ravel()
    semiperiods = np.diff(zero_crossings)
    for idx, semiperiod in zip(zero_crossings, semiperiods):
        idx_next = idx + semiperiod
        voltage_window = voltage[idx:idx_next]
        current_window = current[idx:idx_next]
        if semiperiod != reference_semiperiod:
            scaler = reference_semiperiod / semiperiod
            voltage_window = scipy.signal.resample(
                voltage_window, int(scaler * len(voltage_window))
            )
            current_window = scipy.signal.resample(
                current_window, int(scaler * len(current_window))
            )
        voltage_windows.append(voltage_window)
        current_windows.append(current_window)
    shift = semiperiods[0]
    voltage = np.concatenate(voltage_windows)
    current = np.concatenate(current_windows)
    voltage = np.roll(voltage, shift)[shift:-shift]
    current = np.roll(current, shift)[shift:-shift]
    return voltage, current


@beartype
def check_anomalies(signal: np.ndarray, semiperiod: int, threshold: int = 2) -> bool:
    """
    Detects anomalies in the measurement by analyzing zero-crossings

    Arguments:
        signal: np.ndarray
        semiperiod: int
        threshold: int
    Returns:
        bool
    """
    zero_crossings = np.argwhere(np.diff(np.sign(signal)) != 0.0).ravel()
    semiperiods = np.diff(zero_crossings)
    condition = (semiperiods < semiperiod - threshold) | (
        semiperiods > semiperiod + threshold
    )
    anomalies = np.argwhere(condition)
    if len(anomalies) > 0:
        return True
    else:
        return False


@beartype
def parse_spectrum_bins(
    coefs: np.ndarray,
    fundamental: int,
    sampling_rate: int,
    vicinity: int,
    limit: int,
) -> np.ndarray:
    """
    Arguments:
        coefs: np.ndarray - rfft absolute values
        fundamental: int - fundamental harmonic, Hz
        sampling_rate: int - number of points within 1 second interval
        vicinity: int - frequencies around fundamental to capture, Hz
        limit: int - number of harmonics to parse starting from constant term
    Return:
        np.ndarray
    """
    # Convert natural units to points (unitless)
    fundamental_rescaled = int(2 * len(coefs) / sampling_rate * fundamental)
    vicinity_rescaled = int(2 * len(coefs) / sampling_rate * vicinity)
    assert fundamental_rescaled > 0
    # Vectorized operations
    shifts = np.arange(
        0, fundamental_rescaled * limit, fundamental_rescaled, dtype=np.int32
    )
    windows = coefs[
        shifts[:, None]
        + np.arange(-vicinity_rescaled, vicinity_rescaled, dtype=np.int32)
    ]
    max_values = np.max(windows, axis=1)
    return max_values


@beartype
def parse_magnitude(
    signal: np.ndarray,
    fundamental: int,
    sampling_rate: int,
) -> float:
    """
    Arguments:
        signal: np.ndarray - waveform
        fundamental: int - fundamental harmonic, Hz
        sampling_rate: int - number of points within 1 second interval
    Return:
        float - median value of a magnitude
    """
    period = sampling_rate // fundamental
    shifts = np.arange(period, len(signal) - period, dtype=np.int32)
    windows = signal[shifts[:, None] + np.arange(-period, period, dtype=np.int32)]
    max_values = np.max(np.abs(windows), axis=1)
    return float(np.median(max_values))


@beartype
def extract_signal_mask(
    signal: np.ndarray, window_length: int, thresh: float
) -> np.ndarray:
    """
    Arguments:
        signal: np.ndarray - waveform
        window_length: int - size of a RMS window
        thresh: float - a value above which an appliance is considered as switched on
    Return:
        np.ndarray - indices sequence of non-zero signal
    """
    weights = np.ones(window_length, dtype=np.float32) / window_length
    conv = np.sqrt(np.convolve(signal ** 2, weights, "same"))
    mask = np.argwhere(conv > thresh)
    return mask


@beartype
def get_thd(
    signal: np.ndarray, sampling_rate: int, vicinity: int, limit: int
) -> Tuple[float, float]:
    """
    Arguments:
        signal: np.ndarray - waveform
        sampling_rate: int - number of points within 1 second interval
        vicinity: int - frequencies around fundamental to capture, Hz
        limit: int - number of harmonics to parse starting from constant term
    Return:
        Tuple[float, float]
    """
    coefs = np.abs(np.fft.rfft(signal)).astype(np.float32)
    freqs = np.fft.rfftfreq(len(signal), 1 / sampling_rate).astype(np.float32)
    # Assuming the fundamental harmonic has highest magnitude
    fundamental = float(freqs[np.argmax(coefs)])
    assert fundamental > 0
    spectrum_bins = parse_spectrum_bins(
        coefs, round(fundamental), sampling_rate, vicinity, limit
    )
    thd = (np.sum(spectrum_bins ** 2) - spectrum_bins[1] ** 2) ** 0.5 / spectrum_bins[1]
    return thd, fundamental


@beartype
def normalize_frequencies(
    voltage: np.ndarray,
    current: np.ndarray,
    fundamental: float,
    frequency_standard: float,
    sampling_rate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arguments:
        voltage: np.ndarray - voltage waveform
        current: np.ndarray - current waveform
        fundamental: float - fundamental harmonic, Hz
        frequency_standard: float - desired frequency, Hz
        sampling_rate: int - number of points within 1 second interval
    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    freq_scaler = fundamental / frequency_standard
    voltage = librosa.resample(voltage, sampling_rate, int(sampling_rate * freq_scaler))
    current = librosa.resample(current, sampling_rate, int(sampling_rate * freq_scaler))
    assert len(voltage) == len(current)
    return voltage, current


@beartype
def normalize_durations(
    voltage: np.ndarray, current: np.ndarray, fundamental_period
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arguments:
        voltage: np.ndarray - voltage waveform
        current: np.ndarray - current waveform
        fundamental_period - number of points within 1 period
    Returns:
        Tuple[np.ndarray, np.ndarray, int]
    """
    min_duration = len(voltage) // fundamental_period * fundamental_period
    voltage = voltage[:min_duration]
    current = current[:min_duration]
    return voltage, current


@beartype
def downsample(
    voltage: np.ndarray, current: np.ndarray, sampling_rate: int, downsampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arguments:
        voltage: np.ndarray - voltage waveform
        current: np.ndarray - current waveform
        sampling_rate: int - number of points within 1 second interval
        downsampling_rate: int - desired number of points within 1 second interval
    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    downsampled_length = int(len(voltage) / sampling_rate * downsampling_rate)
    voltage = scipy.signal.resample(voltage, downsampled_length)
    current = scipy.signal.resample(current, downsampled_length)
    return voltage, current


@beartype
def normalize_levels(
    voltage: np.ndarray,
    current: np.ndarray,
    downsampling_rate: int,
    fundamental: float,
    voltage_standard: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Arguments:
        voltage: np.ndarray - voltage waveform
        current: np.ndarray - current waveform
        downsampling_rate: int - desired number of points within 1 second interval
        fundamental: float - fundamental harmonic, Hz
        voltage_standard: float - desired voltage magnitude, V
    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    voltage = voltage - voltage.mean()
    power_scaler = (
        parse_magnitude(
            voltage, sampling_rate=downsampling_rate, fundamental=round(fundamental)
        )
        / voltage_standard
    )
    voltage /= power_scaler
    current *= power_scaler
    return voltage, current


@beartype
def read_whited(
    dataset_path: str,
    downsampling_rate: int,
    voltage_standard: float,
    frequency_standard: float,
    thd_thresh: float,
    max_duration: int,
    vicinity: int,
    limit: int,
    make_logs: bool,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Arguments:
        dataset_path: str - path to the WHITED dataset
        downsampling_rate: int - desired number of points within 1 second interval
        voltage_standard: float - desired voltage magnitude, V
        frequency_standard: float - desired frequency, Hz
        thd_thresh: float - threshold for the maximum THD value (0..1)
        max_duration: int - maximal length of a signal in seconds
        vicinity: int - frequencies around fundamental to capture, Hz
        limit: int - number of harmonics to parse starting from constant term
        make_logs: bool
    Returns:
        Dict[str, Dict[str, List[np.ndarray]]]
    """
    dataset = {}
    # Scaling factors for .flac conversion (as in `readme`)
    mk_factors = {
        "MK1": {"voltage": 1033.64, "current": 61.4835},
        "MK2": {"voltage": 861.15, "current": 60.200},
        "MK3": {"voltage": 988.926, "current": 60.9562},
    }
    # Common duration of each sample in WHITED
    base_duration = 5
    skips = 0
    counter = 0
    filenames = os.listdir(dataset_path)
    for _, filename in tqdm(
        enumerate(filenames), position=0, leave=False, total=len(filenames)
    ):
        name, extension = os.path.splitext(filename)
        if extension != ".flac":
            continue
        # Save category
        name_params = name.split("_")
        category = name_params[0].lower()
        if not category in dataset.keys():
            dataset[category] = {"current": [], "voltage": []}
        # Load audio file
        mk_type = name_params[-2]
        filepath = os.path.join(dataset_path, filename)
        waveforms, sampling_rate = torchaudio.load(filepath)
        waveforms = waveforms.numpy().astype(np.float32)
        # Get first points without useful signal
        # Raw waveform
        current = (
            waveforms[1][-base_duration * sampling_rate :]
            * mk_factors[mk_type]["current"]
        )
        voltage = (
            waveforms[0][-base_duration * sampling_rate :]
            * mk_factors[mk_type]["voltage"]
        )
        # Voltage quality control
        thd, fundamental = get_thd(voltage, sampling_rate, vicinity, limit)
        if thd > thd_thresh:
            skips += 1
            continue
        # Frequency normalization
        if fundamental != frequency_standard:
            voltage, current = normalize_frequencies(
                voltage, current, fundamental, frequency_standard, sampling_rate
            )
        if len(voltage) < sampling_rate:
            skips += 1
            continue
        # Downsampling
        voltage, current = downsample(
            voltage, current, sampling_rate, downsampling_rate
        )
        # Remove fluctuating frequency
        period_standard = downsampling_rate // round(frequency_standard)
        voltage, current = remove_fluctuating_frequency(
            voltage, current, period_standard // 2
        )
        # Signal duration control
        voltage, current = normalize_durations(voltage, current, period_standard)
        if check_anomalies(voltage, period_standard // 2):
            skips += 1
            continue
        # European format
        voltage, current = normalize_levels(
            voltage, current, downsampling_rate, fundamental, voltage_standard
        )
        maximal_length = max_duration * downsampling_rate
        dataset[category]["current"].append(current[:maximal_length])
        dataset[category]["voltage"].append(voltage[:maximal_length])
        counter += 1
    if make_logs:
        if skips > 0:
            logging.warning("%d signatures were skipped" % skips)
        logging.info(
            "Dataset contains %d signatures from %d categories (%s)"
            % (counter, len(dataset.keys()), ", ".join(list(dataset.keys())))
        )
    return dataset


@beartype
def read_plaid(
    dataset_path: str,
    downsampling_rate: int,
    voltage_standard: float,
    frequency_standard: float,
    activation_thresh: float,
    thd_thresh: float,
    max_duration: int,
    vicinity: int,
    limit: int,
    matching_map: Dict[str, str],
    make_logs: bool,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Arguments:
        dataset_path: str - path to the PLAID dataset.
                            NOTE: "meta.json" file must be in the same directory
        downsampling_rate: int - desired number of points within 1 second interval
        voltage_standard: float - desired voltage magnitude, V
        frequency_standard: float - desired frequency, Hz
        activation_thresh: float - a value above which an appliance is considered as switched on
        thd_thresh: float - threshold for the maximum THD value (0..1)
        max_duration: int - maximal length of a signal in seconds
        vicinity: int - frequencies around fundamental to capture, Hz
        limit: int - number of harmonics to parse starting from constant term
        matching_map: Dict[str, str] - map PLAID labels to WHITED labels if possible
        make_logs: bool
    Returns:
        Dict[str, Dict[str, List[np.ndarray]]]
    """
    dataset = {}
    # Assuming the "meta.json" is among the "*.csv" files
    with open(os.path.join(dataset_path, "meta.json"), "r") as fbuffer:
        meta = json.load(fbuffer)
    skips = 0
    counter = 0
    for idx, data in tqdm(meta.items(), total=len(meta)):
        primary_category = data["appliance"]["type"].lower().replace(" ", "")
        category = matching_map.get(primary_category, primary_category)
        sampling_rate = int(data["header"]["sampling_frequency"].replace("Hz", ""))
        if not category in dataset.keys():
            dataset[category] = {"current": [], "voltage": []}
        filepath = os.path.join(dataset_path, idx + ".csv")
        # Load file with raw waveforms
        waveforms = pd.read_csv(filepath, names=["current", "voltage"]).astype(
            np.float32
        )
        current = waveforms.current.to_numpy()
        voltage = waveforms.voltage.to_numpy()
        # Voltage quality control
        thd, fundamental = get_thd(voltage, sampling_rate, vicinity, limit)
        if thd > thd_thresh:
            skips += 1
            continue
        # Extract non-zero signal
        signal_mask = extract_signal_mask(
            current, int(sampling_rate // fundamental), thresh=activation_thresh
        )
        # Current sensitivity control
        if len(signal_mask) == 0:
            skips += 1
            continue
        signal_mask_continuous = np.arange(
            signal_mask[0], signal_mask[-1], dtype=np.int32
        )
        voltage = voltage[signal_mask_continuous]
        current = current[signal_mask_continuous]
        # Frequency normalization
        if fundamental != frequency_standard:
            voltage, current = normalize_frequencies(
                voltage, current, fundamental, frequency_standard, sampling_rate
            )
        if len(voltage) < sampling_rate:
            skips += 1
            continue
        # Downsampling
        voltage, current = downsample(
            voltage, current, sampling_rate, downsampling_rate
        )
        # Remove fluctuating frequency
        period_standard = downsampling_rate // round(frequency_standard)
        voltage, current = remove_fluctuating_frequency(
            voltage, current, period_standard // 2
        )
        # Signal duration control
        voltage, current = normalize_durations(voltage, current, period_standard)
        if check_anomalies(voltage, period_standard // 2):
            skips += 1
            continue
        # European format
        voltage, current = normalize_levels(
            voltage, current, downsampling_rate, fundamental, voltage_standard
        )
        maximal_length = max_duration * downsampling_rate
        dataset[category]["current"].append(current[:maximal_length])
        dataset[category]["voltage"].append(voltage[:maximal_length])
        counter += 1

    if make_logs:
        if skips > 0:
            logging.warning("%d signatures were skipped" % skips)
        logging.info(
            "Dataset contains %d signatures from %d categories (%s)"
            % (counter, len(dataset.keys()), ", ".join(list(dataset.keys())))
        )
    return dataset


@beartype
def merge_datasets(
    datasets: Tuple[Dict[str, Dict[str, List[np.ndarray]]], ...], make_logs: bool
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Arguments:
        datasets: Tuple[Dict[str, Dict[str, List[np.ndarray]]], ...]
        make_logs: bool
    Returns:
        Dict[str, Dict[str, List[np.ndarray]]]
    """
    assert len(datasets) > 1, "At least two datasets are required"
    intersections = []
    patterns = deepcopy(datasets[0])
    for dataset in datasets[1:]:
        for category, channels in dataset.items():
            current, voltage = channels["current"], channels["voltage"]
            _channels = patterns.get(category, {})
            _current, _voltage = _channels.get("current", []), _channels.get(
                "voltage", []
            )
            if len(_channels) != 0:
                intersections.append(category)
            else:
                patterns[category] = {"current": [], "voltage": []}
            patterns[category]["current"] = _current + current
            patterns[category]["voltage"] = _voltage + voltage
    if make_logs:
        intersections = set(intersections)
        logging.info(
            "%d categories intersect (%s)"
            % (len(intersections), ", ".join(list(intersections)))
        )
        logging.info(
            "%d categories in merged dataset (%s)"
            % (len(patterns), ", ".join(list(patterns.keys())))
        )
    return patterns


@beartype
def drop_empty_categories(
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    min_cardinality: int,
    inplace: bool,
    make_logs: bool,
) -> Union[Dict[str, Dict[str, List[np.ndarray]]], None]:
    """
    Arguments:
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        min_cardinality: int - minimal number of signatures in the category of appliances
        inplace: bool - update the `patterns` directly as an object
        make_logs: bool
    Returns:
        Union[Dict[str, Dict[str, List[np.ndarray]]], None]
    """
    blacklist = []
    if not inplace:
        patterns = deepcopy(patterns)
    for category, channels in patterns.items():
        voltage = channels["voltage"]
        if len(voltage) < min_cardinality:
            blacklist.append(category)
    if make_logs:
        logging.info(
            "%d categories will be dropped (%s)"
            % (len(blacklist), ", ".join(blacklist))
        )
    for category in blacklist:
        patterns.pop(category)
    if not inplace:
        return patterns
    else:
        return None


@beartype
def filter_categories(
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    categories: List[str],
    inplace: bool,
    make_logs: bool,
) -> Union[Dict[str, Dict[str, List[np.ndarray]]], None]:
    """
    Arguments:
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        categories: List[str] - all the labels known
        inplace: bool - update the `patterns` directly as an object
        make_logs: bool
    Returns:
        Union[Dict[str, Dict[str, List[np.ndarray]]], None]
    """
    blacklist = []
    if not inplace:
        patterns = deepcopy(patterns)
    for category in categories:
        assert category in patterns.keys(), '"%s" is not a valid category' % category
    for category in patterns.keys():
        if category not in categories:
            blacklist.append(category)
    if make_logs:
        logging.info(
            "%d categories will be dropped (%s)"
            % (len(blacklist), ", ".join(blacklist))
        )
    for category in blacklist:
        patterns.pop(category)
    if not inplace:
        return patterns
    else:
        return None


@beartype
def split(
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    ratios: List[float],
    random_state: int,
    make_logs: bool,
) -> Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]:
    """
    Arguments:
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        ratios: List[float] - if only two values passed, the ratio for test subset will be computed
                              as 1-train-validation. Otherwise, you can pass all the three values
                              for train, validation and test relatively
        random_state: int
        make_logs: bool
    Returns:
        Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]
    """
    assert sum(ratios) <= 1.0 and len(ratios) == 2
    patterns = deepcopy(patterns)
    train_patterns = {}
    val_patterns = {}
    test_patterns = {}
    np.random.seed(random_state)
    for category, channels in patterns.items():
        train_patterns[category] = {"current": [], "voltage": []}
        val_patterns[category] = {"current": [], "voltage": []}
        if sum(ratios) != 1.0:
            test_patterns[category] = {"current": [], "voltage": []}
        current, voltage = channels["current"], channels["voltage"]
        common = list(zip(current, voltage))
        np.random.shuffle(common)
        current, voltage = zip(*common)
        train_size = round(ratios[0] * len(common))
        val_size = round(ratios[1] * len(common))
        test_size = len(common) - val_size - train_size
        assert train_size > 0
        assert val_size > 0
        assert test_size > 0
        train_patterns[category]["current"] = current[:train_size]
        train_patterns[category]["voltage"] = voltage[:train_size]
        val_patterns[category]["current"] = current[train_size : train_size + val_size]
        val_patterns[category]["voltage"] = voltage[train_size : train_size + val_size]
        # The format of ratios is [train, validation]
        # If train+validation=1 - no test data will be provided
        if sum(ratios) != 1.0:
            test_patterns[category]["current"] = current[-test_size:]
            test_patterns[category]["voltage"] = voltage[-test_size:]
        if make_logs:
            logging.info(
                "%s (train = %d / validation = %d / test = %d)"
                % (category, train_size, val_size, test_size)
            )
    return {"train": train_patterns, "validation": val_patterns, "test": test_patterns}


@beartype
def get_random_labels(
    categories: List[str],
    w: int,
    marginals: Dict[str, float],
    random_state: int,
) -> List[str]:
    """
    Arguments:
        categories: List[str] - all the labels known
        w: int - maximum number of loads working simultaneously
        marginals: Union[Dict[str, float], None] - probabilities for each label to be selected
        random_state: int
    Returns:
        List[str]
    """
    np.random.seed(random_state)
    return list(
        np.random.choice(categories, p=list(marginals.values()), size=w, replace=False)
    )


@beartype
def distribute(
    name: str,
    categories: List[str],
    w: int,
    combination_limit: int,
    representation_limit: int,
    counts: Dict[str, int],
    marginals: Union[Dict[str, float], None],
    random_state: int,
    make_logs: bool,
) -> Dict[Tuple[str, ...], int]:
    """
    Arguments:
        name: str - train, validation or test
        categories: List[str] - all the labels known
        w: int - maximum number of loads working simultaneously
        combination_limit: int - user bound for the number of combinations
        representation_limit: int - user bound for the number of representations of particular combination
        counts: Dict[str, int] - number of normalized signatures for each label in train/validation/test subset
        marginals: Union[Dict[str, float], None] - probabilities for each label to be selected
        random_state: int
        make_logs: bool
    Returns:
        Dict[Tuple[str, ...], int]
    """
    if representation_limit < combination_limit:
        raise NotImplementedError
    if not marginals and w > 1:
        # Compute marginals as a frequency of category occurrence
        marginals = {k: 1 / len(counts) for k, _ in counts.items()}
    elif not marginals and w == 1:
        total = sum(counts.values())
        marginals = {k: v / total for k, v in counts.items()}
    assert len(marginals) == len(categories), "Each category must present"
    assert np.round(sum(marginals.values()), 4) == 1.0, "Invalid distribution"
    counter = 0
    joints = []
    combinations = []
    # Prevent duplicates for different w-subsets of different datasets (train, validation, test)
    hash_base = {
        "categories": tuple(categories),
        "w": w,
        "marginals": tuple(marginals),
        "name": name,
        "combination_limit": combination_limit,
        "representation_limit": representation_limit,
        "random_state": random_state,
    }
    random_state = hash(frozenset(hash_base.items())) % 10 ** 9
    labels_representation_bounds = []
    limit = min(combination_limit, comb(len(categories), w))
    # Unique labels generation
    while counter < limit:
        labels = tuple(
            sorted(get_random_labels(categories, w, marginals, random_state))
        )
        random_state += 1
        # Store only unique labels
        if labels in combinations:
            continue
        joint = 1
        labels_representation_bound = 1
        for category in labels:
            joint *= marginals[category]
            labels_representation_bound *= counts[category]
        joints.append(joint)
        combinations.append(labels)
        labels_representation_bounds.append(labels_representation_bound)
        counter += 1
    # Labels multiplicity
    denominator = sum(joints)
    distribution = {}
    for labels, joint, labels_representation_bound in zip(
        combinations, joints, labels_representation_bounds
    ):
        # Multiply labels as a minimum number between its maximum possible
        # number of combinations and a number obtained through the weight
        # from marginal distribution
        representations = min(
            round(joint / denominator * representation_limit),
            labels_representation_bound,
        )
        # If this number is positive, then keep these labels
        if representations > 0:
            distribution[labels] = representations
    if make_logs:
        logging.info(
            "(w = %d) %d combinations with %d representations will be generated."
            % (w, len(distribution), sum(distribution.values()))
        )
    return distribution


@beartype
def pure_sine_function(
    x: np.ndarray, magnitude: float, phase: float, fundamental: float
) -> np.ndarray:
    """
    Arguments:
        x: np.ndarray - waveform
        magnitude: float
        phase: float
        fundamental: float - fundamental harmonic, Hz
    Returns:
        np.ndarray
    """
    return magnitude * np.sin(x * fundamental * np.pi * 2 + phase)


@beartype
def extrapolate_voltage(
    points: np.ndarray,
    voltage: np.ndarray,
    start_point: int,
    magnitude: float,
    fundamental: float,
    sampling_rate: int,
) -> np.ndarray:
    """
    Extrapolates voltage in both directions on new timeline with preservation
    of phase shifts.
    NOTE: new voltage signal will be sinusoidal

    Arguments:
        points: np.ndarray - points where to place a new voltage
        voltage: np.ndarray - voltage waveform
        start_point: int - point (index) where the known voltage starts on `points` axis
        magnitude: float
        fundamental: float - fundamental harmonic, Hz
        sampling_rate: int - number of points within 1 second interval
    Returns:
        np.ndarray
    """
    assert len(points) >= len(voltage)
    parameters = Parameters()
    parameters.add(
        "magnitude", value=magnitude, min=0.9 * magnitude, max=1.1 * magnitude
    )
    parameters.add("phase", value=0.0)
    timeline = np.linspace(
        np.min(points) / sampling_rate, np.max(points) / sampling_rate, len(points)
    )
    start_time = start_point / sampling_rate
    subtimeline = timeline[: len(voltage)]
    subtimeline = subtimeline + start_time
    reference_function = partial(pure_sine_function, fundamental=fundamental)
    reference_function.__name__ = pure_sine_function.__name__
    model = Model(reference_function, param_names=["magnitude", "phase"])
    fit = model.fit(voltage, parameters, x=subtimeline)
    extrapolated_voltage = model.func(timeline, *list(fit.best_values.values()))
    return extrapolated_voltage


@beartype
def pad_signal(signal: np.ndarray, width: int, start_point: int) -> np.ndarray:
    """
    Arguments:
        signal: np.ndarray - waveform
        width: int - length of the timeline
        start_point: int - starting point of a signal
    Returns:
        np.ndarray - padded signal
    """
    padded_signal = np.zeros(width, dtype=np.float32)
    if start_point >= 0:
        signal = signal[: width - start_point]
    else:
        signal = signal[-start_point:]
    padded_signal[max(0, start_point) : len(signal) + max(0, start_point)] = signal
    return padded_signal


@beartype
def align_phases(
    voltages: List[np.ndarray], currents: List[np.ndarray], period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    In fact, all the current signals from the same grid are carrying
    under the same voltage phase. This method takes first voltage signal
    as a reference, then aligns other voltages by phase with it and applies
    corresponding shifts to current signals. The result is a sum of currents.

    Arguments:
        voltages: List[np.ndarray] - list of voltage waveforms
        currents: List[np.ndarray] - list of current waveforms
        period: int - number of points in the one period
    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    assert len(voltages) == len(currents)
    if len(voltages) == len(currents) == 1:
        return currents[0], voltages[0]
    reference_voltage = voltages.pop(0)
    reference_current = currents.pop(0)
    # Get phase shifts for each waveform
    shifts = []
    rolls = np.arange(-period, period, 1, dtype=np.int32)
    for voltage, current in zip(voltages, currents):
        scores = []
        for roll in rolls:
            scores.append(np.sum(np.abs(np.roll(voltage, roll) - reference_voltage)))
        shifts.append(rolls[np.argmin(np.array(scores, dtype=np.float32))])
    # If all the waveforms are with same phase
    if max(np.abs(shifts)) == 0:
        return reference_current + sum(currents), reference_voltage
    for shift, current, voltage in zip(shifts, currents, voltages):
        reference_current += np.roll(current, shift)
    return reference_current, reference_voltage


@beartype
def save_example(
    combination_idx: int,
    representation_idx: int,
    example: Tuple[np.ndarray, Tuple[str, ...]],
    path: str,
) -> None:
    """
    Writes down the aggregated signal into .npy format file

    Arguments:
        combination_idx : int - index number of a combination
        representation_idx: int - index number of a representation for particular combination
        example: Tuple[np.ndarray, Tuple[str, ...]] - aggregated waveform
        path: str - path to save to
    Returns:
        None
    """
    merged_signal, labels = example
    labels = list(labels)
    w = len(labels)
    os.makedirs(path, exist_ok=True)
    data = {"signal": merged_signal.astype(np.float32), "labels": labels}
    file_name = "%d-%d-%d" % (w, combination_idx, representation_idx)
    file_path = os.path.join(path, file_name)
    np.save(file_path, data)
    return None


@beartype
def get_random_indices(
    labels: Tuple[str, ...],
    representations: int,
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    make_logs: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Shuffles indices of patterns for specific label

    Arguments:
        labels: Tuple[str, ...]
        representations: int - number of representations for particular combination of `w` labels
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        make_logs: bool
    Returns:
        Tuple[Dict[str, np.ndarray], int]
    """
    indices = {}
    for label in labels:
        # To prevent duplicated distributions of patterns the random state was
        # introduced as a combination of the hash of labels and the hash of a particular label
        hash_base = {"labels": labels, "label": label}
        random_state = hash(frozenset(hash_base.items())) % 10 ** 9
        np.random.seed(random_state)
        channels = patterns[label]
        num_patterns = len(channels["current"])
        uniques = np.arange(num_patterns, dtype=np.int32)
        if representations <= num_patterns:
            random_vector = np.random.choice(
                uniques, size=representations, replace=False
            )
        else:
            # If label requested more times than corresponding
            # unique patterns exist, then allow repetitions of these patterns
            random_base_vector = np.random.permutation(uniques)
            redundancy = np.random.choice(
                uniques, size=representations - num_patterns, replace=True
            )
            np.random.seed(random_state)
            random_vector = np.random.permutation(
                np.concatenate([random_base_vector, redundancy])
            )
        indices[label] = random_vector
    matrix = np.stack([indices[label] for label in labels])
    # Some combinations of individual patterns for one particular combination of appliances
    # may overlap with each other due to the random sampling at small `w` (usually, related to the w=2).
    # We skip these overlaps by taking only unique combinations
    unique = np.unique(matrix, axis=1)
    indices_without_collisions = {label: vec for label, vec in zip(labels, unique)}
    # TODO fix logger for multiprocessing
    # collisions = matrix.shape[-1] - unique.shape[-1]
    # if collisions > 0 and make_logs:
    #     logging.warning(
    #         "%d collisions will be skipped (%s)" % (collisions, ", ".join(labels))
    #     )
    return indices_without_collisions, unique.shape[-1]


@beartype
def select_patterns(
    labels: Tuple[str, ...],
    representation_idx: int,
    indices: Dict[str, np.ndarray],
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    num_forbidden_points: int,
    num_signal_points: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Selects waveforms for given representation from distribution obtained earlier

    Arguments:
        labels: Tuple[str, ...]
        representation_idx: int - index number of a representation for particular combination
        indices: Dict[str, np.ndarray] - indices of random normalized signatures of a particular
                                         category of appliances
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        num_forbidden_points: int - similar to `forbidden_interval` but in number of points
        num_signal_points: int - length of a aggregated waveform
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[int]]
    """
    currents = []
    voltages = []
    start_points = []
    patterns_by_id = np.array(
        [indices[label][representation_idx] for label in labels]
    ).astype(np.int32)
    for label, pattern_id in zip(labels, patterns_by_id):
        channels = patterns[label]
        current = channels["current"][pattern_id]
        voltage = channels["voltage"][pattern_id]
        # Random state calculation for reproducability and preventing same
        # start points. Here hash of current label and pattern_id are accounted
        # with support of hash of all patterns appeared for given representation.
        # The last term allows avoid repetitions of start time for given label-pattern_id
        hash_base = {
            "label": label,
            "pattern_id": pattern_id,
            "labels": labels,
            "patterns_by_id": patterns_by_id.tobytes(),
        }
        random_state = hash(frozenset(hash_base.items())) % 10 ** 9
        np.random.seed(random_state)
        if len(current) <= num_forbidden_points:
            points_to_start = np.arange(
                0, num_signal_points - num_forbidden_points, dtype=np.int32
            )
        else:
            points_to_start = np.arange(
                num_forbidden_points - len(current),
                num_signal_points - num_forbidden_points,
                dtype=np.int32,
            )
        # Randomly choose the point to place signal
        start_point = int(np.random.choice(points_to_start))
        start_points.append(start_point)
        currents.append(current)
        voltages.append(voltage)
    return voltages, currents, start_points


@beartype
def generate_example(
    labels: Tuple[str, ...],
    voltages: List[np.ndarray],
    currents: List[np.ndarray],
    start_points: List[int],
    num_signal_points: int,
    voltage_standard: float,
    fundamental: float,
    sampling_rate: int,
    period: int,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Takes signals corresponding to each label, places randomly on common
    timeline, pads currents, extrapolates voltage, aligns phases and summarizes

    Arguments:
        labels: Tuple[str, ...]
        voltages: List[np.ndarray]
        currents: List[np.ndarray]
        start_points: List[int]
        num_signal_points: int - length of a aggregated waveform
        voltage_standard: float - desired voltage magnitude, V
        fundamental: float - fundamental harmonic, Hz
        sampling_rate: int - number of points within 1 second interval
        period: int - number of points in the one period
    Returns:
        Tuple[np.ndarray, Tuple[str, ...]]
    """
    time_as_points = np.arange(0, num_signal_points + 2 * period, dtype=np.int32)
    for i in range(len(labels)):
        currents[i] = pad_signal(currents[i], len(time_as_points), start_points[i])
        voltages[i] = extrapolate_voltage(
            time_as_points,
            voltages[i],
            start_points[i],
            magnitude=voltage_standard,
            fundamental=fundamental,
            sampling_rate=sampling_rate,
        )
    merged_signal, voltage = align_phases(
        deepcopy(voltages), deepcopy(currents), period
    )
    time_as_points = time_as_points[period:-period]
    merged_signal = merged_signal[period:-period]
    voltage = voltage[period:-period]
    example = (merged_signal, labels)
    return example


@beartype
def create_multiple_examples(
    combination_idx: int,
    config: Tuple[Tuple[str, ...], int],
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    signal_duration: int,
    forbidden_interval: float,
    voltage_standard: float,
    fundamental: float,
    sampling_rate: int,
    period: int,
    saver: Callable,
    make_logs: bool,
) -> None:
    """
    Creates examples one by one. Validates all examples on uniqueness

    Arguments:
        combination_idx : int - index number of a combination
        config: Tuple[Tuple[str, ...], int] - w-labels and number of corresponding representations
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        signal_duration: int - duration of a signal in number of points
        forbidden_interval: float - number of seconds from the left AND from the right sides
                                    of a timeline to prohibit a start/end of the signal
        voltage_standard: float - desired voltage magnitude, V
        fundamental: float - fundamental harmonic, Hz
        sampling_rate: int - number of points within 1 second interval
        period: int - number of points in the one period
        saver: Callable - method to save an example
        make_logs: bool
    Returns:
        None
    """
    labels, representations = config
    indices, unique_representations = get_random_indices(
        labels, representations, patterns, make_logs
    )
    # Imagine that generation space is represented as a matrix, where
    # rows are labels, and columns are indices of corresponding patterns,
    # the number of columns equals to number of representations
    for representation_idx in range(unique_representations):
        num_forbidden_points = int(sampling_rate * forbidden_interval) + period
        num_signal_points = int(signal_duration * sampling_rate)
        voltages, currents, start_points = select_patterns(
            labels,
            representation_idx,
            indices,
            patterns,
            num_forbidden_points,
            num_signal_points,
        )
        example = generate_example(
            labels,
            voltages,
            currents,
            start_points,
            num_signal_points,
            voltage_standard=voltage_standard,
            fundamental=fundamental,
            sampling_rate=sampling_rate,
            period=period,
        )
        saver(combination_idx, representation_idx, example)
    return None


@beartype
def build_dataset(
    name: str,
    patterns: Dict[str, Dict[str, List[np.ndarray]]],
    limits: Dict[int, Tuple[int, ...]],
    dataset_path: str,
    signal_duration: int,
    forbidden_interval: float,
    voltage_standard: float,
    fundamental: float,
    sampling_rate: int,
    random_state: int,
    make_logs: bool,
    n_jobs: int = 1,
) -> None:
    """
    Arguments:
        name: str
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
        limits: Dict[int, Tuple[int, ...]] - a table of requested combinations/representation
                                             for each particular `w` to consider as in the paper
        dataset_path: str - path to save the dataset to
        signal_duration: int - duration of a signal in number of points
        forbidden_interval: float - number of seconds from the left AND from the right sides
                                    of a timeline to prohibit a start/end of the signal
        voltage_standard: float - desired voltage magnitude, V
        fundamental: float - fundamental harmonic, Hz
        sampling_rate: int - number of points within 1 second interval
        random_state: int
        make_logs: bool
        n_jobs: int = 1 - number of threads to utilize
    Returns:
        None
    """
    period = sampling_rate // round(fundamental)
    saver = partial(save_example, path=dataset_path)
    counts = {
        category: len(channels["current"]) for category, channels in patterns.items()
    }
    for w in limits.keys():
        if make_logs:
            logging.info("Subset for w = %d is in progress..." % w)
        distribution = distribute(
            name,
            list(patterns.keys()),
            w,
            combination_limit=limits[w][0],
            representation_limit=limits[w][1],
            counts=counts,
            marginals=None,
            random_state=random_state,
            make_logs=make_logs,
        )
        generator = partial(
            create_multiple_examples,
            patterns=patterns,
            signal_duration=signal_duration,
            forbidden_interval=forbidden_interval,
            voltage_standard=voltage_standard,
            fundamental=fundamental,
            sampling_rate=sampling_rate,
            period=period,
            saver=saver,
            make_logs=make_logs,
        )
        if n_jobs == -1:
            n_jobs = cpu_count()
        wrap_generator(generator, distribution, n_jobs)
        del distribution
        if make_logs:
            logging.info("Subset for w = %d is ready." % w)
    return None


@beartype
def get_stats(patterns: Dict[str, Dict[str, List[np.ndarray]]]) -> Tuple[int, int]:
    """
    Median value of class-cardinalities and total size of subset

    Arguments:
        patterns: Dict[str, Dict[str, List[np.ndarray]]] - collection of normalized signatures
    Returns:
        Tuple[int, int]
    """
    median = int(np.median([len(v["voltage"]) for _, v in patterns.items()]))
    size = sum([len(v["voltage"]) for _, v in patterns.items()])
    return median, size
