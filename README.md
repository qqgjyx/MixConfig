# MixConfig: Mixing Configurations for Downstream Prediction

Official implementation for the paper "Mixing Configurations for Downstream Prediction".

## Abstract

We propose MixConfig, a framework for learning sample-specific mixtures of graph-based configurations to improve downstream prediction tasks. Given a dataset, MixConfig extracts multiple configurations via k-NN graph construction and Parallel-DT decomposition, then learns to adaptively weight these configurations for each sample using an Energy-Aware Selector. The selector leverages sample context, cluster assignment embeddings, and energy statistics to produce optimal configuration weights, yielding a mixed representation that captures multi-scale structural information. We demonstrate state-of-the-art performance across tabular, vision, molecular, and text benchmarks.

## Method Overview

MixConfig consists of two main components:

1. **Configuration Extraction**: Given input data X, we construct a k-NN graph and apply Parallel-DT to obtain multiple configurations (cluster assignments) at different resolutions. The Parallel-DT extraction code is not included in this release; a Python implementation will be provided later.

2. **Energy-Aware Selector**: For each sample x, the selector computes:
   - Sample context: h = MLP_enc(x)
   - Cluster embeddings: c_i = Embed_i(omega_i(x))
   - Energy statistics: e_i = [H_i, h_a^(i), h_r^(i), delta_gamma_i]
   - Compatibility scores: s_i = MLP_score([h; c_i; e_i])
   - Configuration weights: w_i(x) = softmax(s_i)
   - Mixed representation: z(x) = sum_i w_i(x) * c_i

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/mixconfig.git
cd mixconfig

# Create conda environment
conda env create -f environment.yml
conda activate mixconfig
```

## Quick Start

```python
from src.mixconfig import EnergyAwareSelector
import numpy as np
import torch

# Initialize the Energy-Aware Selector
selector = EnergyAwareSelector(
    input_dim=X_train.shape[1],
    n_configs=8,
    context_dim=64,
    cluster_embed_dim=32
)

# Load external configuration assignments and energy statistics
# configs: [n_samples, n_configs] integer assignments
# energy_stats: [n_configs, 4] with [H, h_a, h_r, delta_gamma]
configs = np.load("configs.npy")
energy_stats = np.load("energy_stats.npy")
energy_stats_t = torch.tensor(energy_stats, dtype=torch.float32)

# Use mixed representations for downstream prediction
z = selector.get_mixed_representation(X_train, configs, energy_stats_t)
```

## Reproducing Paper Results

Note: configuration extraction is not included in this release. To reproduce MixConfig variants, you must supply extracted configurations and energy statistics. Base and +Config baselines can be run without the extractor.

### Tabular (OpenML-CC18)

```bash
python experiments/run_tabular.py --dataset openml-cc18 --config configs/datasets/tabular.yaml --mode base
```

### Vision (CIFAR-100, ImageNet-1K)

```bash
python experiments/run_vision.py --dataset cifar100 --config configs/datasets/vision.yaml --mode base
python experiments/run_vision.py --dataset imagenet1k --config configs/datasets/vision.yaml --mode base
```

### Molecular (MolHIV, BBBP, BACE)

```bash
python experiments/run_molecular.py --dataset molhiv --config configs/datasets/molecular.yaml --mode base
python experiments/run_molecular.py --dataset bbbp --config configs/datasets/molecular.yaml --mode base
python experiments/run_molecular.py --dataset bace --config configs/datasets/molecular.yaml --mode base
```

### Text (SST-2, AG News)

```bash
python experiments/run_text.py --dataset sst2 --config configs/datasets/text.yaml --mode base
python experiments/run_text.py --dataset ag_news --config configs/datasets/text.yaml --mode base
```

### Low-data and ablation scripts

```bash
python experiments/run_lowdata.py --dataset bbbp --mode base
python experiments/run_ablation.py --dataset bbbp --ablation full
```

## Project Structure

```
mixconfig/
├── src/
│   ├── mixconfig/           # Core MixConfig implementation
│   │   ├── selector.py      # Energy-Aware Selector
│   │   ├── encoder.py       # Sample context encoder
│   │   ├── embedder.py      # Cluster assignment embedder
│   │   ├── energy.py        # Energy statistics computation
│   │   └── config_extractor.py  # Configuration extraction
│   ├── datasets/            # Dataset loaders
│   │   ├── tabular.py       # OpenML-CC18
│   │   ├── vision.py        # CIFAR-100, ImageNet-1K
│   │   ├── molecular.py     # MolHIV, BBBP, BACE
│   │   └── text.py          # SST-2, AG News
│   ├── predictors/          # Downstream predictors
│   │   ├── neural.py        # MLP, TabPFN, FT-Transformer
│   │   └── classical.py     # XGBoost, RF, Linear
│   ├── models.py            # Model definitions
│   ├── utils.py             # Utility functions
│   └── visuals.py           # Visualization utilities
├── experiments/             # Experiment scripts
│   ├── configs/             # Configuration files
│   ├── run_ablation.py
│   ├── run_lowdata.py
│   ├── run_tabular.py
│   ├── run_vision.py
│   ├── run_molecular.py
│   └── run_text.py
├── environment.yml          # Conda environment
└── README.md
```

## Notes on availability

- Configuration extraction (Parallel-DT) is not included in this release; a Python implementation will be provided later.
- The paper includes QM9 results; QM9 pipelines will be released alongside the extractor.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
