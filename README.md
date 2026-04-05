# Meriti: Remote Sensing Image Segmentation 🛰️

---

![Make](https://img.shields.io/badge/Make-064F8C?logo=make&logoColor=fff&style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=for-the-badge)
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff&&style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-7A76FF?logo=plotly&logoColor=fff&style=for-the-badge)
![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=000&style=for-the-badge)
![Hydra](https://img.shields.io/badge/Hydra-89B4FA?style=for-the-badge&logo=hydra&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-261230?logo=ruff&logoColor=000&style=for-the-badge)
![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=fff&style=for-the-badge)
![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?logo=nvidia&logoColor=fff&style=for-the-badge)

## 📝 Description

Meriti is a deep learning project focused on remote sensing image segmentation, specifically tailored for flood detection tasks. The implementation leverages custom U-Net architectures and advanced loss functions (Composite Loss) to perform pixel-level classification on satellite imagery. It features robust logging and experiment tracking through Weights & Biases, configuration management via Hydra, and performance profiling using NVTX and Nvidia Nsight Systems.

## 📑 Table of Contents

- [Meriti: Remote Sensing Image Segmentation 🛰️](#meriti-remote-sensing-image-segmentation-️)
  - [📝 Description](#-description)
  - [📑 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [📦 Installation](#-installation)
  - [🚀 Usage](#-usage)
    - [Run Experiments](#run-experiments)
    - [Profiling](#profiling)
    - [Makefile](#makefile)
  - [📂 Project Structure](#-project-structure)
  - [📄 Logs and Results](#-logs-and-results)
  - [⚖️ License](#️-license)

## ✨ Features

- 🏗️ **Custom U-Net Architecture:** Modular U-Net implementation with support for various backbones (e.g., ResNet18/34/50).
- 📉 **Advanced Loss Functions:** Includes `CompositeLoss` combining Dice Loss and Focal Loss (Partial Cross Entropy).
- 🧠 **Reproducibility:** Deterministic configuration for experiments.
- 📊 **Robust Monitoring:** Integration with Weights & Biases (W&B) for real-time metric tracking and sweep experiments.
- 🚀 **Performance Profiling and Optimization:** Features NVTX instrumentation for profiling.
- 🔍 **Data Augmentation:** Support for dynamic augmentations via `FloodSegmentationDataset`.

## 🛠️ Tech Stack

- **Core:** Python 3.13+, PyTorch
- **Configuration Management:** Hydra
- **Experiment Tracking:** Weights & Biases
- **Formatting and Linting**: Ruff
- **Logging:** Loguru
- **Data Processing:** OpenCV, Pillow, Scikit-learn
- **Visualization:** Matplotlib, Plotly, Rich

## 📦 Installation

Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13+ installed. This project uses `pyproject.toml` for dependency management.

```bash
# Clone the repository
git clone https://github.com/Harkhymadhe/meriti
cd meriti

# Install dependencies
uv pip install -e .
```

## 🚀 Usage

The primary entry point for running the experiments is `src/main.py`. However, these experiments are designed using sweeps from Weights & Biases, and running them can be quite involved. This is automated and abstracted away via a Makefile.

### Run Experiments

To start and run the experiments, simply execute the Makefile via one of the commands below:

```bash
# Ensure your config.yaml is updated in src/conf/
make
```

```bash
# Ensure your config.yaml is updated in src/conf/
make all
```

### Profiling

To profile the performance of the training process, another script `main_nvtx.py` is provided. It contains the exact same contents as the regular `main.py` script, except for the wandb logging-related calls. To execute the profiling script, run one of the below commands:

```bash
python3 src/main_nvtx.py
```

```bash
make profile
```

### Makefile

In general, the Makefile serves as a perfect entry into exploring the project. It provides the following targets:

| Command         | Usage                                            |
|-----------------|--------------------------------------------------|
|make profile     | Execute profiling script.                        |
|make create      | Create a sweep.                                  |
|make run         | Execute sweep.                                   |
|make pause       | Pause sweep.                                     |
|make resume      | Resume sweep.                                    |
|make stop        | Stop sweep.                                      |
|make cancel      | Cancel all experiments.                          |
|make sync        | Synchronize local logged wandb data to the cloud.|
|make (all)       | Create, run, and sync sweep to the cloud.        |

This will generate an Nvidia Nsight Systems report file (.nsys-rep), which can be visualized in the Nvidia Nsight Systems application.

## 📂 Project Structure

```text
├── dataset/                  # Dataset directory
│   ├── data_mount/           # Dataset mount point
│   └── FloodNet/             # Base dataset directory
├── outputs/                  # Hydra configuration files with corresponding artifacts
├── src/
│   ├── conf/                 # Hydra configuration files
│   ├── remote_segmentation/  # Core logic
│   │   ├── architectures.py  # U-Net & ResNet modules
│   │   ├── dataset.py        # Data loaders & augmentation
│   │   ├── metrics.py        # Loss functions & accuracy tracking
│   │   └── models.py         # Model loading utilities
│   ├── main.py               # Main training entry point
│   └── main_nvtx.py          # Profiling entry point
├── tests/
│   └── architecture.py       # Tests for architecture.py script
├── wandb/                    # Offline storage directory for weights and biases logs
├── Makefile                  # Project entrypoint
├── README.md                 # Project README
├── data.sh                   # Bash script for data mounting
├── profile.sh                # Bash script for setting up profiling via Nvidia Nsight systems
├── pyproject.toml            # Project dependencies
├── sweep.sh                  # Bash script for setting up W&B sweeps
└── sweep.yaml                # Experimental sweep configurations for hyperparametric optimization
```

## 📄 Logs and Results

The generated logs from experimentation can be found on W\&B: [https://wandb.ai/harkhymadhe/meriti](https://wandb.ai/harkhymadhe/meriti)

## ⚖️ License

This project is currently unlicensed.
