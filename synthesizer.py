#!/usr/bin/env python3

#################################################################
#### synthesizer.py ### COLD: Concurrent Loads Disaggregator ####
#################################################################

# Signal processing
import scipy
import librosa
import torchaudio
# Calculus 
import random
import numpy as np
import pandas as pd
from math import comb
from lmfit import Model, Parameters
# System
import os
import json
import shutil
import hashlib
import warnings
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from functools import partial
from pathos import multiprocessing


def parse_spectrum_bins(coefs, freqs, fundamental=50, vicinity=5, limit=10,
                        sampling_rate=4000):
    """
    coefs :: np.ndarray -- rfft absolute values
    freqs :: np.ndarray -- correspondig frequencies
    fundamental :: int -- fundamental harmonic, Hz
    vicinity :: int -- frequencies around fundamental to capture, Hz
    limit :: int -- number of harmonics to parse starting from constant term 
    sampling_rate :: int
    ---
    -> np.ndarray -- magnitudes of each harmonic frequency
    """
    assert isinstance(coefs, np.ndarray)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(fundamental, int)
    assert isinstance(vicinity, int)
    assert isinstance(limit, int)
    assert isinstance(sampling_rate, int)

    # Convert natural units to points (unitless)
    fundamental_rescaled = int(2*len(coefs)/sampling_rate*fundamental)
    vicinity_rescaled = int(2*len(coefs)/sampling_rate*vicinity)
    assert fundamental_rescaled > 0

    # Vectorized operations
    shifts = np.arange(0, fundamental_rescaled * limit,
                       fundamental_rescaled, dtype=np.int32)
    windows = coefs[shifts[:, None] +
                    np.arange(-vicinity_rescaled, vicinity_rescaled, dtype=np.int32)]
    max_values = np.max(windows, axis=1)

    return max_values


def parse_magnitude(signal,  fundamental=50, sampling_rate=4000):
    """
    signal :: np.ndarray -- waveform
    fundamental :: int -- fundamental harmonic, Hz
    sampling_rate :: int
    ---
    -> float -- median value of magnitude 
    """
    assert isinstance(signal, np.ndarray)
    assert isinstance(fundamental, int)
    assert isinstance(sampling_rate, int)

    period = sampling_rate//fundamental

    shifts = np.arange(period, len(signal)-period, dtype=np.int32)
    windows = signal[shifts[:, None] +
                     np.arange(-period, period, dtype=np.int32)]
    max_values = np.max(np.abs(windows), axis=1)

    return np.median(max_values)


def extract_signal_mask(signal, window_length, thresh):
    """
    signal :: np.ndarray -- waveform
    window_length :: int -- size of RMS window 
    thresh :: float -- above which an appliance is considered as switched on 
    ---
    -> np.ndarray -- indices sequence of non-zero signal 
    """
    assert isinstance(signal, np.ndarray)
    assert isinstance(window_length, int)
    assert isinstance(thresh, float)

    weights = np.ones(window_length, dtype=np.float32) / window_length
    conv = np.sqrt(np.convolve(signal ** 2, weights, "same"))
    mask = np.argwhere(conv > thresh)

    return mask


def read_whited(dataset_path, downsampling=4000, voltage_standard=311.0,
                frequency_standard=50, thd_thresh=0.1, max_duration=5,
                verbose=True):
    """
    dataset_path :: str -- path to directory with .flac files of WHITED
    downampling :: int -- sampling rate after downsampling
    voltage_standard :: float 
    frequency_standard :: int 
    thd_thresh :: float -- THD allowed (from 0 to 1) 
    max_duration :: int -- non-zero signals will be restricted by this time 
    verbose :: bool
    ---
    -> dict -- category: list of patterns (waveforms)
    """
    assert isinstance(dataset_path, str)
    assert isinstance(downsampling, int)
    assert isinstance(voltage_standard, float)
    assert isinstance(frequency_standard, int)
    assert isinstance(thd_thresh, float)
    assert isinstance(max_duration, int)
    assert isinstance(verbose, bool)

    dataset = {}
    # Scaling factors for .flac conversion (as in `readme`)
    mk_factors = {"MK1": {"voltage": 1033.64, "current": 61.4835}, "MK2": {
        "voltage": 861.15, "current": 60.200}, "MK3": {
        "voltage": 988.926, "current": 60.9562}}
    # Common duration of each sample in WHITED
    base_duration = 5

    skips = 0
    counter = 0
    filenames = os.listdir(dataset_path)
    for i, filename in tqdm(enumerate(filenames), position=0, leave=False,
                            total=len(filenames)):

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
        silence_interval = waveforms.shape[-1] - base_duration * sampling_rate
        # Raw waveform
        current = waveforms[1][silence_interval:] * \
            mk_factors[mk_type]["current"]
        voltage = waveforms[0][silence_interval:] * \
            mk_factors[mk_type]["voltage"]

        # Voltage quality control
        coefs = np.abs(np.fft.rfft(voltage)).astype(np.float32)
        freqs = np.fft.rfftfreq(
            len(voltage), 1/sampling_rate).astype(np.float32)
        # Assuming the fundamental harmonic has highest magnitude
        # Otherwise, this example will be dropped
        fundamental = freqs[np.argmax(coefs)]

        assert fundamental > 0

        spectrum_bins = parse_spectrum_bins(
            coefs, freqs, fundamental=round(fundamental),
            sampling_rate=sampling_rate)
        thd = (np.sum(spectrum_bins**2)-spectrum_bins[1]
               ** 2)**0.5/spectrum_bins[1]
        if thd > thd_thresh:
            skips += 1
            continue

        # Frequency normalization
        if fundamental != frequency_standard:
            freq_scaler = fundamental/frequency_standard
            voltage = librosa.resample(
                voltage, sampling_rate, int(sampling_rate*freq_scaler))
            current = librosa.resample(
                current, sampling_rate, int(sampling_rate*freq_scaler))
            assert len(voltage) == len(current)

        # Signal duration control
        min_duration = len(voltage)//sampling_rate*sampling_rate
        voltage = voltage[:min_duration]
        current = current[:min_duration]
        if min_duration == 0:
            skips += 1
            continue

        # Downsampling
        downsampled_length = int(len(voltage)/sampling_rate*downsampling)
        voltage = scipy.signal.resample(voltage, downsampled_length)
        current = scipy.signal.resample(current, downsampled_length)

        # European format
        voltage = voltage-voltage.mean()
        power_scaler = parse_magnitude(
            voltage, sampling_rate=downsampling, fundamental=frequency_standard)\
            / voltage_standard

        voltage /= power_scaler
        current *= power_scaler

        maximal_length = max_duration*downsampling
        dataset[category]["current"].append(current[:maximal_length])
        dataset[category]["voltage"].append(voltage[:maximal_length])
        counter += 1

    if skips > 0:
        warnings.warn("%d patterns were skipped" % skips)

    if verbose:
        print("Dataset contains", counter, "patterns from", len(
            dataset.keys()), "categories\n", list(dataset.keys()))

    return dataset


def read_plaid(dataset_path, downsampling=4000, voltage_standard=311.0,
               frequency_standard=50, activation_thresh=0.1, thd_thresh=0.1,
               max_duration=5, matching_map={}, verbose=True):
    """
    dataset_path :: str -- path to directory with .csv files of PLAID 
    downampling :: int -- sampling rate after downsampling
    voltage_standard :: float 
    frequency_standard :: int 
    activation_thresh :: float -- to consider device as switched on 
    thd_thresh :: float -- THD allowed (from 0 to 1) 
    max_duration :: int -- non-zero signals will be restricted by this time 
    matching_map :: dict -- actual category: whited category analog (if any) 
    verbose :: bool
    ---
    -> dict -- category: list of patterns (waveforms)
    """
    assert isinstance(dataset_path, str)
    assert isinstance(downsampling, int)
    assert isinstance(voltage_standard, float)
    assert isinstance(frequency_standard, int)
    assert isinstance(activation_thresh, float)
    assert isinstance(thd_thresh, float)
    assert isinstance(max_duration, int)
    assert isinstance(matching_map, dict)
    assert isinstance(verbose, bool)

    dataset = {}
    # Assuming the "meta.json" is among the "*.csv" files
    with open(os.path.join(dataset_path, "meta.json"), "r") as fbuffer:
        meta = json.load(fbuffer)

    skips = 0
    counter = 0
    for idx, data in tqdm(meta.items(), total=len(meta)):
        primary_category = data["appliance"]["type"].lower().replace(" ", "")
        category = matching_map.get(primary_category, primary_category)
        sampling_rate = int(
            data["header"]["sampling_frequency"].replace("Hz", ""))

        if not category in dataset.keys():
            dataset[category] = {"current": [], "voltage": []}

        filepath = os.path.join(dataset_path, idx + ".csv")
        # Load file with raw waveforms
        waveforms = pd.read_csv(
            filepath, names=["current", "voltage"]).astype(np.float32)
        current = waveforms.current.to_numpy()
        voltage = waveforms.voltage.to_numpy()

        # Voltage quality control
        coefs = np.abs(np.fft.rfft(voltage)).astype(np.float32)
        freqs = np.fft.rfftfreq(
            len(voltage), 1/sampling_rate).astype(np.float32)
        # Assuming the fundamental harmonic has highest magnitude
        # Otherwise, this sample will be dropped
        fundamental = freqs[np.argmax(coefs)]

        assert fundamental > 0

        spectrum_bins = parse_spectrum_bins(
            coefs, freqs, fundamental=round(fundamental),
            sampling_rate=sampling_rate)
        thd = (np.sum(spectrum_bins**2)-spectrum_bins[1]
               ** 2)**0.5/spectrum_bins[1]
        if thd > thd_thresh:
            skips += 1
            continue

        # Extract non-zero signal
        signal_mask = extract_signal_mask(current, int(
            sampling_rate//fundamental), thresh=activation_thresh)

        # Current sensitivity control
        if len(signal_mask) == 0:
            skips += 1
            continue

        signal_mask_continuous = np.arange(
            signal_mask[0], signal_mask[-1], dtype=np.int32)
        voltage = voltage[signal_mask_continuous]
        current = current[signal_mask_continuous]

        # Frequency normalization
        # If signal length < 1 sec, then drop it
        min_duration = int(len(voltage)/sampling_rate)*sampling_rate
        voltage = voltage[:min_duration]
        current = current[:min_duration]

        # Signal duration control
        if min_duration == 0:
            skips += 1
            continue

        if fundamental != frequency_standard:
            freq_scaler = fundamental/frequency_standard
            voltage = librosa.resample(
                voltage, sampling_rate, int(sampling_rate*freq_scaler))
            current = librosa.resample(
                current, sampling_rate, int(sampling_rate*freq_scaler))
            assert len(voltage) == len(current)

        # Signal duration control
        min_duration = int(len(voltage)/sampling_rate)*sampling_rate
        voltage = voltage[:min_duration]
        current = current[:min_duration]
        if min_duration == 0:
            skips += 1
            continue

        # Downsampling
        downsampled_length = int(len(voltage)/sampling_rate*downsampling)
        voltage = scipy.signal.resample(voltage, downsampled_length)
        current = scipy.signal.resample(current, downsampled_length)

        # European format
        voltage -= voltage.mean()
        power_scaler = parse_magnitude(
            voltage, sampling_rate=downsampling,
            fundamental=frequency_standard) \
            / voltage_standard

        voltage /= power_scaler
        current *= power_scaler

        maximal_length = max_duration*downsampling
        dataset[category]["current"].append(current[:maximal_length])
        dataset[category]["voltage"].append(voltage[:maximal_length])
        counter += 1

    if skips > 0:
        warnings.warn("%d patterns were skipped" % skips)

    if verbose:
        print("Dataset contains", counter, "patterns from", len(
            dataset.keys()), "categories\n", list(dataset.keys()))

    return dataset


def merge_datasets(*datasets, verbose=True):
    """
    datasets :: dict -- category: list of waveforms
    verbose :: bool
    ---
    -> dict -- category: list of waveforms
    """
    assert len(datasets) > 1, "At least two datasets are required"
    assert all([isinstance(dataset, dict) for dataset in datasets])
    assert isinstance(verbose, bool)

    intersections = []
    patterns = deepcopy(datasets[0])
    for dataset in datasets[1:]:
        for category, channels in dataset.items():
            current, voltage = channels["current"], channels["voltage"]
            _channels = patterns.get(category, {})
            _current, _voltage = _channels.get(
                "current", []), _channels.get("voltage", [])

            if len(_channels) != 0:
                intersections.append(category)
            else:
                patterns[category] = {"current": [], "voltage": []}

            patterns[category]["current"] = _current + current
            patterns[category]["voltage"] = _voltage + voltage

    if verbose:
        intersections = set(intersections)
        print(len(intersections), "categories intersects\n", list(intersections))
        print(len(patterns), "categories in merged dataset\n",
              list(patterns.keys()))

    return patterns


def drop_empty_categories(patterns, inplace=False, verbose=True):
    """
    patterns :: dict -- category: list of waveforms
    inplace :: bool -- updates `patterns` without returning it
    verbose :: bool 
    ---
    -> dict/None -- if not inplace: category: list of waveforms
    """
    assert isinstance(patterns, dict)
    assert isinstance(inplace, bool)
    assert isinstance(verbose, bool)

    blacklist = []
    if not inplace:
        patterns = deepcopy(patterns)
    for category, channels in patterns.items():
        current, voltage = channels["current"], channels["voltage"]
        if len(voltage) == 0 or len(voltage) == 0:
            blacklist.append(category)
    if verbose:
        print(len(blacklist), "categories will be dropped.")
        if len(blacklist) > 0:
            print(blacklist)
    for category in blacklist:
        patterns.pop(category)
    if not inplace:
        return patterns
    else:
        return None


def filter_categories(patterns, categories, inplace=False, verbose=True):
    """
    patterns :: dict -- category: list of waveforms
    categories :: list -- categories to keep 
    inplace :: bool -- updates `patterns` without returning it
    verbose :: bool 
    ---
    -> dict/None -- if not inplace: category: list of waveforms
    """
    assert isinstance(patterns, dict)
    assert isinstance(categories, list)
    assert isinstance(inplace, bool)
    assert isinstance(verbose, bool)

    blacklist = []
    if not inplace:
        patterns = deepcopy(patterns)

    for category, channels in patterns.items():
        if category not in categories:
            blacklist.append(category)

    if verbose:
        print(len(blacklist), "categories will be dropped.")

    for category in blacklist:
        patterns.pop(category)

    if not inplace:
        return patterns
    else:
        return None


def split(patterns, ratios=[0.6, 0.1], random_state=0, verbose=True):
    """
    patterns :: dict -- category: list of waveforms
    ratios :: list of floats, train/validation/the rest for the test
    random_state :: int
    verbose :: bool
    ---
    -> (dict,dict,dict) -- (category: list of waveforms,)
    """
    assert isinstance(patterns, dict)
    assert isinstance(ratios, list)
    assert isinstance(random_state, int)
    assert isinstance(verbose, bool)

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

        train_size = int(ratios[0] * len(common))
        val_size = int(ratios[1] * len(common))

        train_patterns[category]["current"] = current[:train_size]
        train_patterns[category]["voltage"] = voltage[:train_size]
        val_patterns[category]["current"] = current[train_size: train_size + val_size]
        val_patterns[category]["voltage"] = voltage[train_size: train_size + val_size]
        # The format of ratios is [train, validation]
        # If train+validation=1 - no test data will be provided
        if sum(ratios) != 1.0:
            test_patterns[category]["current"] = current[train_size + val_size:]
            test_patterns[category]["voltage"] = voltage[train_size + val_size:]

        if verbose:
            print(category, "train size =", train_size, "/ validation size =",
                  val_size, "/ test size =", len(common)-val_size-train_size)

    return train_patterns, val_patterns, test_patterns


def get_random_labels(categories, w, marginals, random_state=0):
    """
    categories :: list of str -- all the categories known
    w :: int -- number of simultaneously working devices
    marginals :: probabilities for each category being selected
    random_state :: int 
    ---
    -> list of str -- labels of simultaneously working devices
    """
    assert isinstance(categories, list)
    assert isinstance(w, int)
    assert isinstance(marginals, dict)
    assert isinstance(random_state, int)

    np.random.seed(random_state)

    return np.random.choice(categories, p=list(marginals.values()),
                            size=w, replace=False)


def distribute(name, categories, w, combination_limit, representation_limit, counts,
               marginals={}, random_state=0, verbose=True):
    """
    name :: str
    categories :: list of str -- all categories known
    w :: int -- number of simultaneously working devices
    combination_limit :: int -- limit on choosing categories by groups of size `w` 
    representation_limit :: int -- limit on combinations of patterns being selected
                        for `w` number of categories (details are in paper) 
    counts :: dict -- str: int, number of signatures per each category  
    marginals :: dict -- str: float, marginal probabilities per each category
    random_state :: int
    verbose :: bool
    ---
    -> dict -- (str,): int, representations per labels of simultaneously working devices 
    """
    assert isinstance(name, str)
    assert isinstance(categories, list)
    assert isinstance(w, int)
    assert isinstance(combination_limit, int)
    assert isinstance(representation_limit, int)
    assert isinstance(counts, dict)
    assert isinstance(marginals, dict)
    assert isinstance(random_state, int)
    assert isinstance(verbose, bool)

    if representation_limit < combination_limit:
        raise NotImplementedError

    if not marginals and w > 1:
        # Compute marginals as a frequency of category occurrence
        marginals = {k: 1 / len(counts) for k, _ in counts.items()}
    elif not marginals and w == 1:
        total = sum([v for _, v in counts.items()])
        marginals = {k: v / total for k, v in counts.items()}

    assert len(marginals) == len(categories), "Each category must present"
    assert np.round(sum(marginals.values()), 4) == 1., "Invalid distribution"

    counter = 0
    joints = []
    combinations = []

    if name == "val":
        # TODO do it in a smart way 
        random_state += 10**5
    elif name == "test":
        random_state += 10**6

    labels_representation_bounds = []
    limit = min(combination_limit, comb(len(categories), w))

    # Unique labels generation
    while counter < limit:

        labels = tuple(sorted(get_random_labels(
            categories, w, marginals, random_state=random_state)))

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
    for labels, joint, labels_representation_bound in zip(combinations, joints,
                                                      labels_representation_bounds):
        # Multiply labels as a minimum number between its maximum possible
        # number of combinations and a number obtained through the weight
        # from marginal distribution
        representations = min(round(joint / denominator *
                                representation_limit), labels_representation_bound)
        # If this number is positive, then keep these labels
        if representations > 0:
            distribution[labels] = representations

    if verbose:
        print("[DISTRIBUTION] %d combinations from %d labels; %d examples." %
              (len(distribution), w, sum(distribution.values())))

    return distribution


def pure_sine_function(x, magnitude, phase, fundamental=50):
    """
    x :: np.ndarray
    magnitude :: float
    phase :: float
    fundamental :: int  
    ---
    -> np.ndarray
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(magnitude, float)
    assert isinstance(phase, float)
    assert isinstance(fundamental, int)

    return magnitude * np.sin(x * fundamental * np.pi * 2+phase)


def extrapolate_voltage(points, voltage, start_point, magnitude=311.0,
                        fundamental=50, sampling_rate=4000):
    """
    Extrapolates voltage in both directions on new timeline with preservation
    of phase shifts. 
    NOTE: new voltage signal will be sinusoidal 
    ---
    points :: np.ndarray -- points where to place new voltage 
    voltage :: np.ndarray -- values of voltage which is known
    start_point :: int -- point where known voltage starts on `points` axis
    magnitude :: float 
    sampling_rate :: int 
    --- 
    -> np.ndarray
    """
    assert isinstance(points, np.ndarray)
    assert isinstance(voltage, np.ndarray)
    assert isinstance(start_point, int)
    assert isinstance(magnitude, float)
    assert isinstance(fundamental, int)
    assert isinstance(sampling_rate, int)

    assert len(points) >= len(voltage)

    parameters = Parameters()
    parameters.add("magnitude", value=magnitude,
                   min=.9*magnitude, max=1.1*magnitude)
    parameters.add("phase", value=0.0)

    timeline = np.linspace(np.min(points) / sampling_rate,
                           np.max(points) / sampling_rate, len(points))
    start_time = start_point/sampling_rate
    subtimeline = timeline[:len(voltage)]
    subtimeline = subtimeline+start_time

    reference_function = partial(pure_sine_function, fundamental=fundamental)
    reference_function.__name__ = pure_sine_function.__name__

    model = Model(reference_function, param_names=["magnitude", "phase"])
    fit = model.fit(voltage, parameters, x=subtimeline)

    return model.func(timeline, *list(fit.best_values.values()))


def pad_signal(signal, width, start_point):
    """
    Pads signal on the timeline of length `width` from the left side 
    of `start_point` and from the right side of end of a signal
    ---
    signal :: np.ndarray
    width :: int 
    start_point :: int 
    ---
    -> np.ndarray
    """
    assert isinstance(signal, np.ndarray)
    assert isinstance(width, int)
    assert isinstance(start_point, int)

    padded_signal = np.zeros(width, dtype=np.float32)

    if start_point >= 0:
        signal = signal[: width - start_point]
    else:
        signal = signal[-start_point:]

    padded_signal[max(0, start_point): len(signal)+max(0, start_point)] = signal

    return padded_signal


def align_phases(voltages, currents, period):
    """
    In fact, all the current signals from the same grid are carrying 
    under the same voltage phase. This method takes first voltage signal
    as a reference, then aligns other voltages by phase with it and applies
    corresponding shifts to current signals. The result is a sum of currents.
    ---
    voltages :: list of np.ndarray
    currents :: list of np.ndarray
    period :: int 
    ---
    -> np.ndarray -- merged signal 
    """
    assert isinstance(voltages, list)
    assert isinstance(currents, list)
    assert isinstance(period, int)

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
            scores.append(
                np.sum(np.abs(np.roll(voltage, roll) - reference_voltage)))
        shifts.append(rolls[np.argmin(np.array(scores, dtype=np.float32))])

    # If all the waveforms are with same phase
    if max(np.abs(shifts)) == 0:
        return reference_current + sum(currents), reference_voltage

    for shift, current, voltage in zip(shifts, currents, voltages):
        reference_current += np.roll(current, shift)

    return reference_current, reference_voltage


def save_example(idx, representation, example, path):
    """
    Writes down the aggregated signal into .npy format file 
    ---
    idx :: int
    representation :: int
    example :: tuple -- merged signal and labels 
    path :: str
    ---
    -> None
    """
    assert isinstance(idx, int)
    assert isinstance(representation, int)
    assert isinstance(example, tuple)
    assert isinstance(path, str)

    merged_signal, labels = example
    labels = list(labels)
    w = len(labels)

    os.makedirs(path, exist_ok=True)
    data = {"waveform": merged_signal.astype(np.float32), "labels": labels}

    np.save("./%s/%d-%d-%d" % (path, w, idx, representation), data)

    return None


def save_specgram(idx, representation, example, path):
    """
    Writes down the spectrogram with labels to separate file 
    ---
    idx :: int 
    representation :: int
    example :: tuple -- the spectrogram of merged signal and the labels 
    path :: str
    ---
    -> None
    """
    assert isinstance(idx, int)
    assert isinstance(representation, int)
    assert isinstance(example, tuple)
    assert isinstance(path, str)

    specgram, labels = example
    labels = list(labels)
    w = len(labels)

    os.makedirs(path, exist_ok=True)
    data = {"specgram": specgram.astype(np.float32), "labels": labels}

    np.save("./%s/%d-%d-%d" % (path, w, idx, representation), data)

    return None


def save_feature(idx, representation, example, path):
    """
    Writes down the features with labels to separate file 
    ---
    idx :: int 
    representation :: int
    example :: tuple -- the features of merged signal and the labels 
    path :: str
    ---
    -> None
    """
    assert isinstance(idx, int)
    assert isinstance(representation, int)
    assert isinstance(example, tuple)
    assert isinstance(path, str)

    features, labels = example
    labels = list(labels)
    w = len(labels)

    os.makedirs(path, exist_ok=True)
    data = {"features": features, "labels": labels}

    np.save("./%s/%d-%d-%d" % (path, w, idx, representation), data)

    return None


def get_random_indices(labels, representations, patterns):
    """
    Shuffles indices of patterns for specific label
    ---
    labels :: tuple
    representations :: int
    patterns :: dict -- str: list of waveforms
    ---
    -> np.ndarray
    """
    assert isinstance(labels, tuple)
    assert isinstance(representations, int)
    assert isinstance(patterns, dict)

    indices = {}

    for label in labels:

        # To prevent duplicated distributions of patterns the random state was
        # introduced as a combination of the hash of labels and the hash of a particular label
        hash_base = {"labels": labels, "label": label}
        random_state = hash(frozenset(hash_base.items())) % 10**9
        np.random.seed(random_state)

        channels = patterns[label]
        num_patterns = len(channels["current"])
        uniques = np.arange(num_patterns, dtype=np.int32)

        if representations <= num_patterns:
            random_vector = np.random.choice(
                uniques, size=representations, replace=False)
        else:
            # If label requested more times than corresponding
            # unique patterns exist, then allow repetitions of these patterns
            random_base_vector = np.random.permutation(uniques)
            redundancy = np.random.choice(
                uniques, size=representations - num_patterns, replace=True)

            np.random.seed(random_state)
            random_vector = np.random.permutation(
                np.concatenate([random_base_vector, redundancy]))

        indices[label] = random_vector

    matrix = np.stack([indices[label] for label in labels])
    unique = np.unique(matrix, axis=1)
    indices_without_collisions = {
        label: vec for label, vec in zip(labels, unique)}

    collisions = matrix.shape[-1]-unique.shape[-1]
    if collisions > 0:
        print("[SKIP] # Collisions = %d @ %s" % (collisions, labels))

    return indices_without_collisions, unique.shape[-1]


def select_patterns(labels, representation, indices, patterns, num_forbidden_points,
                    num_signal_points):
    """
    Selects waveforms for given representation from distribution obtained earlier
    ---
    labels :: tuple
    representation :: int
    indices :: dict 
    patterns :: dict
    num_forbidden_points :: int 
    num_signal_points :: int 
    ---
    -> (list, list, list)
    """
    assert isinstance(labels, tuple)
    assert isinstance(representation, int)
    assert isinstance(indices, dict)
    assert isinstance(patterns, dict)
    assert isinstance(num_forbidden_points, int)
    assert isinstance(num_signal_points, int)

    currents = []
    voltages = []
    start_points = []
    patterns_by_id = np.array([indices[k][representation]
                               for k in labels]).astype(np.int32)

    for label, pattern_id in zip(labels, patterns_by_id):
        channels = patterns[label]
        current = channels["current"][pattern_id]
        voltage = channels["voltage"][pattern_id]

        # Random state calculation for reproducability and preventing same
        # start points. Here hash of current label and pattern_id are accounted
        # with support of hash of all patterns appeared for given representation.
        # The last term allows avoid repetitions of start time for given label-pattern_id
        hash_base = {"label": label, "pattern_id": pattern_id, "labels": labels,
                     "patterns_by_id": patterns_by_id.tobytes()}
        random_state = hash(frozenset(hash_base.items())) % 10**9
        np.random.seed(random_state)

        if len(current) <= num_forbidden_points:
            points_to_start = np.arange(
                0, num_signal_points - num_forbidden_points, dtype=np.int32)
        else:
            points_to_start = np.arange(
                num_forbidden_points - len(current),
                num_signal_points - num_forbidden_points, dtype=np.int32)

        # Randomly choose the point to place signal
        start_point = int(np.random.choice(points_to_start))

        start_points.append(start_point)
        currents.append(current)
        voltages.append(voltage)

    return voltages, currents, start_points


def extract_features(current, voltage, time=True, spectral=True,
                     max_harmonics=20, sampling_rate=4000, fundamental=50,
                     thresh=0.01):
    """
    Extracts some time- and frequency-domain features of an aggregated signal
    ---
    current :: np.array
    voltage :: np.array
    time :: bool -- time-domain features
    spectral  :: bool -- frequency-domain features
    max_harmonics :: int
    sampling_rate :: int
    fundamental :: int
    thresh :: int -- for signal extraction
    ---
    -> dict
    """
    assert isinstance(current, np.ndarray)
    assert isinstance(voltage, np.ndarray)
    assert isinstance(time, bool)
    assert isinstance(spectral, bool)
    assert isinstance(max_harmonics, int)
    assert isinstance(sampling_rate, int)
    assert isinstance(fundamental, int)
    assert isinstance(thresh, float)

    if time:
        phase_shift = np.corrcoef(current, voltage)[0, 1]
        mag_std = np.std(current)
        mag_mean = np.mean(current)
        mag_min = np.min(current)
        mag_max = np.max(current)
        form_factor = np.std(current)/np.mean(np.abs(current))

        time_features = {"phase_shift": phase_shift, "std": mag_std,
                         "min": mag_min, "max": mag_max,
                         "form_factor": form_factor}
    else:
        time_features = {}

    if spectral:
        centroid = librosa.feature.spectral_centroid(
            current, sr=sampling_rate, n_fft=len(current),
            hop_length=len(current))[0, 0]
        mask = extract_signal_mask(current, sampling_rate//fundamental, thresh)

        assert len(mask) > 0

        coefs = np.abs(np.fft.rfft(current)*2/len(mask))
        freqs = np.fft.rfftfreq(len(current), 1/sampling_rate)

        spectrum_bins = parse_spectrum_bins(
            coefs, freqs, fundamental=fundamental,
            sampling_rate=sampling_rate, limit=max_harmonics)

        thd = (np.sum(spectrum_bins**2) -
               spectrum_bins[1]**2)**0.5/spectrum_bins[1]
        spectrum_bins_as_features = {"f%d" %
                                     h: v for h, v in enumerate(spectrum_bins)}

        spectral_features = {**{"spectral_centroid": centroid,
                                "thd": thd}, **spectrum_bins_as_features}
    else:
        spectral_features = {}

    features = {**time_features, **spectral_features}

    return features


def generate_example(labels, voltages, currents, start_points,
                     num_signal_points, voltage_magnitude=311.0,
                     fundamental=50, sampling_rate=4000, period=80,
                     transform_spec=None, transform_features=None):
    """
    Takes signals corresponding to each label, places randomly on common
    timeline, pads currents, extrapolates voltage, aligns phases and summarizes 
    ---
    labels :: tuple
    voltages :: list
    currents :: list
    start_points :: list
    num_signal_points :: int 
    voltage_magnitude :: float
    fundamental :: int 
    sampling_rate :: int 
    period :: int 
    transform_spec :: callable 
    transform_features :: callable 
    ---
    -> tuple 
    """
    assert isinstance(labels, tuple)
    assert isinstance(voltages, list)
    assert isinstance(currents, list)
    assert isinstance(start_points, list)
    assert isinstance(num_signal_points, int)
    assert isinstance(voltage_magnitude, float)
    assert isinstance(fundamental, int)
    assert isinstance(sampling_rate, int)
    assert isinstance(period, int)

    if transform_spec is not None:
        assert callable(transform_spec)
    elif transform_features is not None:
        assert callable(transform_features)

    time_as_points = np.arange(
        0, num_signal_points+2*period, dtype=np.int32)

    for i in range(len(labels)):

        currents[i] = pad_signal(
            currents[i], len(time_as_points), start_points[i])
        voltages[i] = extrapolate_voltage(
            time_as_points, voltages[i], start_points[i],
            magnitude=voltage_magnitude,  fundamental=fundamental,
            sampling_rate=sampling_rate)

    merged_signal, voltage = align_phases(
        deepcopy(voltages), deepcopy(currents), period)

    time_as_points = time_as_points[period:-period]
    merged_signal = merged_signal[period:-period]
    voltage = voltage[period:-period]

    assert len(merged_signal) == len(voltage) == num_signal_points

    if transform_spec is not None:
        spectrogram = transform_spec(merged_signal)
        example = (spectrogram, labels)
    elif transform_features is not None:
        features = transform_features(merged_signal, voltage)
        example = (features, labels)
    else:
        example = (merged_signal, labels)

    return example


def create_multiple_examples(idx, config, patterns, signal_duration=5,
                             forbidden_interval=0.5, voltage_magnitude=311.0,
                             fundamental=50, sampling_rate=4000, period=80,
                             transform_spec=None, transform_features=None,
                             saver=save_example):
    """
    Creates examples one by one. Validates all examples on uniqueness 
    ---
    idx :: int
    config :: tuple 
    patterns :: dict
    signal_duration :: int  
    forbidden_interval :: float 
    voltage_magnitude :: float
    fundamental :: int 
    sampling_rate :: int 
    period :: int 
    transform_spec :: callable 
    transform_features :: callable 
    saver :: callable 
    ---
    -> list -- list of tuples from signal and its labels
    """
    assert isinstance(idx, int)
    assert isinstance(config, tuple)
    assert isinstance(patterns, dict)
    assert isinstance(signal_duration, int)
    assert isinstance(forbidden_interval, float)
    assert isinstance(voltage_magnitude, float)
    assert isinstance(fundamental, int)
    assert isinstance(sampling_rate, int)
    assert isinstance(period, int)
    assert callable(saver)

    if transform_spec is not None:
        assert callable(transform_spec)
    elif transform_features is not None:
        assert callable(transform_features)

    labels, representations = config
    indices, unique_representations = get_random_indices(
        labels, representations, patterns)

    # Imagine that generation space is represented as a matrix, where
    # rows are labels, and columns are indices of corresponding patterns,
    # the number of columns equals to number of representations
    for representation in range(unique_representations):
        num_forbidden_points = int(
            sampling_rate * forbidden_interval)+period
        num_signal_points = int(signal_duration * sampling_rate)

        voltages, currents, start_points = select_patterns(
            labels, representation, indices, patterns, num_forbidden_points,
            num_signal_points)
        example = generate_example(
            labels, voltages, currents, start_points, num_signal_points,
            voltage_magnitude=voltage_magnitude, fundamental=fundamental,
            sampling_rate=sampling_rate, period=period,
            transform_spec=transform_spec,
            transform_features=transform_features)

        saver(idx, representation, example)

    return None


def build_dataset(
    name,
    patterns,
    limits,
    collection_path="",
    signal_duration=5,
    forbidden_interval=0.5,
    voltage_magnitude=311.0,
    fundamental=50,
    sampling_rate=4000,
    to_specgrams=True,
    to_features=False,
    p_fft=5,
    p_hop=2,
    verbose=True,
    n_jobs=1,
    random_state=0,
):
    """
    name :: str
    patterns :: tuple
    limits :: dict
    collection_path :: str -- where to store generated examples
    signal_duration :: int
    forbidden_interval :: float
    voltage_magnitude :: float
    fundamental :: int
    sampling_rate :: int
    to_specgrams :: bool -- save only specgrams 
    to_features :: bool -- save only features 
    p_fft :: int -- fft window in terms of periods
    p_hop :: int -- hop length in terms of periods
    verbose :: bool
    n_jobs :: int
    random_state :: int
    ---
    -> None
    """
    assert isinstance(name, str)
    assert isinstance(patterns, dict)
    assert isinstance(limits, dict)
    assert isinstance(collection_path, str)
    assert isinstance(signal_duration, int)
    assert isinstance(forbidden_interval, float)
    assert isinstance(voltage_magnitude, float)
    assert isinstance(fundamental, int)
    assert isinstance(sampling_rate, int)
    assert isinstance(to_specgrams, bool)
    assert isinstance(to_features, bool)
    assert isinstance(p_fft, int)
    assert isinstance(p_hop, int)
    assert isinstance(verbose, bool)
    assert isinstance(n_jobs, int)
    assert isinstance(random_state, int)

    period = sampling_rate//fundamental

    saver = None
    transform_spec = None
    transform_features = None

    if to_specgrams and to_features:
        raise NotImplementedError
    elif to_specgrams:
        to_spectrogram = partial(librosa.core.spectrum._spectrogram,
                                 n_fft=p_fft*period, hop_length=p_hop*period,
                                 center=True, pad_mode="reflect", power=2.0)
        to_decibels = partial(librosa.core.amplitude_to_db,
                              amin=1e-10, top_db=None)

        def transform_spec(x): return to_decibels(to_spectrogram(x)[0])
        saver = partial(save_specgram, path=os.path.join(
            collection_path, "specgrams"))
    elif to_features:
        # TODO add assignment of the rest of keyword arguments 
        transform_features = partial(
            extract_features, time=True, spectral=True,
            sampling_rate=sampling_rate, fundamental=fundamental)
        saver = partial(save_feature, path=os.path.join(
            collection_path, "features"))
    else:
        saver = partial(save_example, path=os.path.join(
            collection_path, "raw"))

    counts = {category: len(channels["current"])
              for category, channels in patterns.items()}

    for cid, w in enumerate(limits.keys()):

        if verbose:
            print("Collection %d is in progress..." % (cid+1))

        distribution = distribute(name, list(patterns.keys()), w,
                                  combination_limit=limits[w][0],
                                  representation_limit=limits[w][1],
                                  counts=counts, random_state=random_state,
                                  verbose=verbose)

        generator = partial(create_multiple_examples, patterns=patterns,
                            signal_duration=signal_duration,
                            forbidden_interval=forbidden_interval,
                            voltage_magnitude=voltage_magnitude,
                            fundamental=fundamental, sampling_rate=sampling_rate,
                            period=period, transform_spec=transform_spec,
                            transform_features=transform_features,
                            saver=saver)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        with multiprocessing.Pool(n_jobs) as parallel:
            parallel.starmap(generator, enumerate(distribution.items()))

        del distribution

        if verbose:
            print("Collection %d is ready." % (cid+1))

    return None


def get_stats(train_patterns, validation_patterns, test_patterns):
    """
    Median values of class-cardinalities and total sizes of subsets
    ---
    train_patterns :: dict
    validation_patterns :: dict
    test_patterns :: dict
    ---
    -> dict
    """
    assert isinstance(train_patterns, dict)
    assert isinstance(validation_patterns, dict)
    assert isinstance(test_patterns, dict)

    train_median = int(np.median([len(v["voltage"])
                                  for _, v in train_patterns.items()]))
    validation_median = int(
        np.median([len(v["voltage"]) for _, v in validation_patterns.items()]))
    test_median = int(np.median([len(v["voltage"])
                                 for _, v in test_patterns.items()]))

    train_size = sum([len(v["voltage"]) for _, v in train_patterns.items()])
    validation_size = sum([len(v["voltage"])
                           for _, v in validation_patterns.items()])
    test_size = sum([len(v["voltage"]) for _, v in test_patterns.items()])

    return {"medians": (train_median, validation_median, test_median),
            "sizes": (train_size, validation_size, test_size)}
