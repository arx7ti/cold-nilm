# Concurrent Loads Disaggregator

_for Non-Intrusive Load Monitoring_

**Summary:** This is a code repository for the paper **Concurrent Loads Disaggregator for Non-Intrusive Load Monitoring** (arXiv:2106.02352). We developed the _Synthesizer of Normalized Signatures_ (SNS) algorithm to simulate the aggregated consumption of a various number of appliances. This algorithm takes signatures of individual appliances (i.e. current and voltage) and converts them into European system (311 V, 50 Hz etc.). Then, the schedule (i.e. which appliances will work together) is randomly sampled from uniform distribution. Finally, corresponding signatures are being aligned by the reference voltage and summed up. This repo also includes implementation of the neural network architecture proposed. We wrap the training&testing pipeline into the Pytorch Lightning framework.

**P.S.** this repo is under the active development.

**Keywords**: _signal decomposition, energy disaggregation, nilm, simultaneous appliances, concurrent loads, synthetic data, neural network_

## Installation

Clone this repository:

```
git clone https://github.com/arx7ti/cold-nilm.git
```

**[Optional]** We used the `nix-shell` only for CUDA support and Jupyter environment. Some of the nix packages were incompatible and we decided to use virtual environment within nix-shell. You can activate it via:

```
nix-shell default.nix
```

The required packages can be installed via standard Python package manager:

```
pip install --upgrade pip
pip install -r requirements.txt
```

The full examples of the code with the experiments of the paper are in corresponding notebooks:

1.  The UK-DALE and REDD datasets statistics: `UK-DALE and REDD statistics.ipynb`
2.  The use of SNS algorithm: `SNS algorithm.ipynb`
3.  The neural network training and testing: `Concurrent Loads Disaggregator (COLD).ipynb`

## Cite our paper

```
@misc{kamyshev2021cold,
      title={COLD: Concurrent Loads Disaggregator for Non-Intrusive Load Monitoring},
      author={Ilia Kamyshev and Dmitrii Kriukov and Elena Gryazina},
      year={2021},
      eprint={2106.02352},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```
