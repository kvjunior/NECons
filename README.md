# NECons: Network-aware Edge-based Consensus for Distributed Blockchain Anomaly Detection

<p align="center">
  <img src="assets/necons_logo.png" alt="NECons Logo" width="200"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.1109/TPDS.2025.xxxxxxx"><img src="https://img.shields.io/badge/DOI-10.1109%2FTPDS.2025.xxxxxxx-blue.svg" alt="DOI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+"></a>
</p>

<p align="center">
  <b>IEEE Transactions on Parallel and Distributed Systems (TPDS) 2025</b>
</p>

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Key Contributions](#-key-contributions)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Experiments](#-experiments)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 📄 Abstract

Blockchain networks processed over 500 million daily transactions in 2024, with security incidents causing losses exceeding $3.8 billion annually. Multi-chain ecosystems have enabled sophisticated anomaly patterns that exploit differences in confirmation times across heterogeneous blockchains. Effective anomaly detection requires distributed monitoring systems capable of operating across geographically dispersed edge nodes while maintaining resilience against adversarial participants.

**NECons** (Network-aware Edge-based Consensus) is a distributed framework that addresses three fundamental challenges in blockchain anomaly detection:

1. **Network Heterogeneity**: Existing GNN methods exhibit performance degradation exceeding 15% when latency surpasses 100ms
2. **Byzantine Adversaries**: Standard federated averaging fails catastrophically under Byzantine attacks
3. **Cross-Chain Correlation**: Single-chain methods cannot detect patterns spanning multiple blockchain networks

Our framework introduces NetworkAwareMGD for adaptive graph learning, Byzantine-resilient consensus with formal guarantees, and cross-chain synchronization for multi-chain anomaly correlation.

---

## 🎯 Key Contributions

| Contribution | Description | Key Metric |
|:-------------|:------------|:-----------|
| **NetworkAwareMGD** | Network-state-adaptive graph attention with multigraph discrepancy analysis | **89.47% F1-score** (+5.35% over SOTA) |
| **Byzantine-Resilient Consensus** | Provably secure aggregation with formal safety, liveness, and accuracy guarantees | **83.45% accuracy** under 33% Byzantine nodes |
| **Cross-Chain Synchronization** | Multi-chain correlation detection across heterogeneous blockchains | **318 additional anomalies** detected |
| **Comprehensive Evaluation** | Rigorous experiments across 4 datasets and 6 baselines | **p < 0.001** statistical significance |

---

## 🏗 System Architecture

<p align="center">
  <img src="figures/necons_system_overview.pdf" alt="NECons System Architecture" width="100%"/>
</p>

**Figure 1**: NECons three-tier architecture comprising multi-chain data ingestion, distributed edge processing with NetworkAwareMGD modules, and Byzantine-resilient consensus aggregation.

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-CHAIN DATA LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │ Ethereum │  │ Bitcoin  │  │ Polygon  │  │   BSC    │                    │
│  │PoS, 12s  │  │PoW, 600s │  │ PoS, 2s  │  │DPoS, 3s  │                    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                    │
│       └─────────────┴──────┬──────┴─────────────┘                          │
│                            │ Cross-Chain Sync Protocol                      │
│                            ▼ O(K·log n + Σₖ Fₖ·Tₖ)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                       DISTRIBUTED EDGE LAYER                                │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐                    │
│  │   Node n₁   │  │   Node n₂   │  ...  │   Node nₖ   │                    │
│  │ ┌─────────┐ │  │ ┌─────────┐ │       │ ┌─────────┐ │                    │
│  │ │NetworkA-│ │  │ │NetworkA-│ │       │ │NetworkA-│ │                    │
│  │ │wareMGD  │ │  │ │wareMGD  │ │       │ │wareMGD  │ │                    │
│  │ └─────────┘ │  │ └─────────┘ │       │ └─────────┘ │                    │
│  │   Δθ₁      │  │   Δθ₂      │       │   Δθₖ      │                    │
│  └──────┬──────┘  └──────┬──────┘       └──────┬──────┘                    │
│         └────────────────┴───────────┬─────────┘                           │
├──────────────────────────────────────┼──────────────────────────────────────┤
│                    BYZANTINE-RESILIENT CONSENSUS                            │
│         ┌────────────────────────────┼────────────────────────────┐        │
│         │                            ▼                            │        │
│         │  ┌─────────┐  ┌─────────────────┐  ┌──────────────┐    │        │
│         │  │ Trust   │→ │   Byzantine     │→ │  Aggregation │    │        │
│         │  │ Scoring │  │   Filtering     │  │  (Weighted)  │    │        │
│         │  │  τᵢ⁽ᵗ⁾   │  │ |z|>3, τᵢ<0.2  │  │ Krum/Bulyan  │    │        │
│         │  └─────────┘  └─────────────────┘  └──────┬───────┘    │        │
│         │                                           │            │        │
│         │               θ' = θ - η·Δθ_final ←───────┘            │        │
│         └────────────────────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────────────────────────────┤
│                         DETECTION OUTPUT                                    │
│    Single-Chain: F1 89.47%  │  Cross-Chain: +318 anomalies                 │
│    Byzantine@33%: 83.45%    │  Scalability: 5,000 nodes, <1.7s             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### NetworkAwareMGD Layer

The core graph neural network module with network-adaptive attention:

```
h_v^(l+1) = σ(W_self · h_v^(l) + Σ_{u∈N(v)} α̃_vu · W_n · [h_u^(l) ‖ Δ_vu])

where:
  α̃_vu = α_vu · φ(N(t)) · ψ(d_vu)     # Network-aware attention
  Δ_vu = h_v^(l) - h_u^(l)             # Multigraph discrepancy
  N(t) = (λ, β, ρ, γ, ξ) ∈ ℝ⁵         # Network state vector
```

### Trust-Weighted Aggregation

Byzantine-resilient model aggregation with temporal trust scoring:

```
τᵢ^(t) = β · τᵢ^(t-1) + (1-β) · [w₁ρᵢ + w₂μᵢ + w₃νᵢ + w₄ςᵢ]

where:
  ρᵢ = gradient consistency score
  μᵢ = magnitude reasonableness score  
  νᵢ = historical accuracy score
  ςᵢ = cross-validation score
```

---

## 💻 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/BlockchainLab/NECons.git
cd NECons

# Create virtual environment
conda create -n necons python=3.10
conda activate necons

# Install PyTorch with CUDA support
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric==2.3.1
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install remaining dependencies
pip install -r requirements.txt
```

### Requirements

```text
# requirements.txt
torch>=2.0.1
torch-geometric>=2.3.1
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
networkx>=3.1
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
wandb>=0.15.0
pyyaml>=6.0
web3>=6.0.0
requests>=2.28.0
scipy>=1.10.0
statsmodels>=0.14.0
```

---

## 🚀 Quick Start

### Single-Chain Anomaly Detection

```python
from necons import NEConsFramework
from necons.data import EthereumDataLoader

# Initialize framework
framework = NEConsFramework(
    num_nodes=10,
    byzantine_ratio=0.3,
    network_aware=True
)

# Load Ethereum transaction data
loader = EthereumDataLoader(
    dataset_path='data/ethereum/',
    batch_size=256
)

# Train model
framework.train(
    train_loader=loader.train,
    val_loader=loader.val,
    epochs=100,
    lr=0.001
)

# Evaluate
results = framework.evaluate(loader.test)
print(f"F1-Score: {results['f1']:.4f}")
print(f"AUC-ROC: {results['auc_roc']:.4f}")
```

### Distributed Multi-Node Training

```bash
# Launch distributed training across 10 nodes
python -m torch.distributed.launch \
    --nproc_per_node=10 \
    --master_addr="localhost" \
    --master_port=12355 \
    train_distributed.py \
    --config configs/distributed_training.yaml \
    --byzantine_ratio 0.3 \
    --aggregation trust_weighted
```

### Cross-Chain Detection

```python
from necons import CrossChainDetector

# Initialize cross-chain detector
detector = CrossChainDetector(
    chains=['ethereum', 'bitcoin', 'polygon', 'bsc'],
    sync_protocol='adaptive'
)

# Run cross-chain analysis
correlations = detector.detect_correlations(
    time_window='1h',
    min_confidence=0.8
)

print(f"Cross-chain anomalies detected: {len(correlations)}")
```

---

## 📊 Datasets

### Supported Datasets

| Dataset | Transactions | Nodes | Edges | Anomaly Ratio | Source |
|:--------|-------------:|------:|------:|--------------:|:-------|
| **Elliptic** | 203,769 | 203,769 | 234,355 | 2.0% | [Elliptic](https://www.kaggle.com/ellipticco/elliptic-data-set) |
| **Ethereum Phishing** | 2,973,489 | 1,165,767 | 2,973,489 | 1.2% | [XBlock](http://xblock.pro/) |
| **Bitcoin OTC** | 35,592 | 5,881 | 35,592 | 3.5% | [SNAP](https://snap.stanford.edu/data/) |
| **Multi-Chain Synthetic** | 5,000,000 | 500,000 | 5,000,000 | 2.5% | Generated |

### Data Preparation

```bash
# Download and preprocess datasets
python scripts/download_datasets.py --all

# Generate synthetic multi-chain data
python scripts/generate_multichain.py \
    --num_transactions 5000000 \
    --chains ethereum bitcoin polygon bsc \
    --anomaly_ratio 0.025
```

### Data Format

```
data/
├── ethereum/
│   ├── transactions.csv      # Transaction records
│   ├── node_features.npy     # Node feature matrix X ∈ ℝ^(|V|×d)
│   ├── edge_index.npy        # Edge connectivity
│   ├── edge_attr.npy         # Edge attributes A ∈ ℝ^(|E|×d')
│   └── labels.npy            # Ground truth labels
├── bitcoin/
│   └── ...
├── polygon/
│   └── ...
└── bsc/
    └── ...
```

---

## 🔬 Experiments

### Reproduce Paper Results

```bash
# Table II: Single-chain detection comparison
python experiments/single_chain_comparison.py \
    --datasets elliptic ethereum_phishing \
    --baselines gcn gat graphsage mgd twodynethnet grabphisher \
    --seeds 42 123 456 789 1000

# Table III: Byzantine resilience evaluation  
python experiments/byzantine_resilience.py \
    --byzantine_ratios 0.0 0.1 0.2 0.3 \
    --attack_types poisoning label_flip gradient_scale collusion \
    --aggregations fedavg krum bulyan trimmed_mean trust_weighted

# Table IV: Cross-chain detection
python experiments/cross_chain_detection.py \
    --chain_pairs eth-btc eth-poly poly-bsc \
    --sync_protocols naive adaptive consensus

# Table V: Scalability analysis
python experiments/scalability.py \
    --num_nodes 100 500 1000 2000 5000 \
    --measure latency throughput accuracy
```

### Configuration Files

```yaml
# configs/default.yaml
model:
  name: NetworkAwareMGD
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3
  
network_aware:
  enabled: true
  state_dim: 5  # (λ, β, ρ, γ, ξ)
  adaptation_rate: 0.1

consensus:
  protocol: trust_weighted
  byzantine_threshold: 0.33
  trust_decay: 0.9
  
training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
```

---

## 📈 Results

### Single-Chain Detection Performance

| Method | Elliptic F1 | Elliptic AUC | Ethereum F1 | Ethereum AUC |
|:-------|------------:|-------------:|------------:|-------------:|
| GCN | 81.23 | 89.45 | 79.89 | 88.12 |
| GAT | 82.56 | 90.23 | 81.34 | 89.56 |
| GraphSAGE | 80.78 | 88.90 | 78.45 | 87.23 |
| MGD | 84.12 | 91.78 | 83.67 | 91.23 |
| **NECons (Ours)** | **89.47** | **95.62** | **87.23** | **94.15** |

### Byzantine Resilience

| Byzantine Ratio | FedAvg | Krum | Bulyan | Trimmed Mean | **NECons** |
|----------------:|-------:|-----:|-------:|-------------:|-----------:|
| 0% | 89.12 | 87.45 | 88.23 | 88.67 | **89.47** |
| 10% | 72.34 | 83.12 | 84.56 | 83.89 | **87.23** |
| 20% | 45.67 | 78.90 | 80.12 | 79.45 | **85.67** |
| 33% | 23.45 | 71.23 | 74.56 | 72.89 | **83.45** |

### Scalability

| Nodes | Consensus Latency | Throughput (tx/s) | Accuracy |
|------:|------------------:|------------------:|---------:|
| 100 | 0.12s | 12,450 | 89.47% |
| 500 | 0.34s | 11,890 | 88.92% |
| 1,000 | 0.67s | 11,234 | 88.45% |
| 2,000 | 1.12s | 10,567 | 87.23% |
| 5,000 | 1.67s | 9,890 | 86.78% |

---

## 📁 Project Structure

```
NECons/
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation
│
├── 📁 necons/                      # Main package
│   ├── 📄 __init__.py
│   ├── 📁 models/                  # Neural network models
│   │   ├── 📄 network_aware_mgd.py # NetworkAwareMGD implementation
│   │   ├── 📄 attention.py         # Network-adaptive attention
│   │   └── 📄 aggregators.py       # Message aggregation
│   ├── 📁 consensus/               # Byzantine-resilient consensus
│   │   ├── 📄 trust_scoring.py     # Trust score computation
│   │   ├── 📄 byzantine_filter.py  # Malicious update filtering
│   │   ├── 📄 aggregation.py       # Weighted aggregation
│   │   └── 📄 pbft.py              # PBFT protocol
│   ├── 📁 crosschain/              # Cross-chain components
│   │   ├── 📄 sync_protocol.py     # Synchronization protocol
│   │   ├── 📄 correlation.py       # Correlation detection
│   │   └── 📄 adapters/            # Chain-specific adapters
│   ├── 📁 data/                    # Data loading utilities
│   │   ├── 📄 loaders.py           # Dataset loaders
│   │   ├── 📄 transforms.py        # Graph transforms
│   │   └── 📄 samplers.py          # Batch sampling
│   └── 📁 utils/                   # Utility functions
│       ├── 📄 metrics.py           # Evaluation metrics
│       ├── 📄 visualization.py     # Plotting utilities
│       └── 📄 logging.py           # Logging configuration
│
├── 📁 configs/                     # Configuration files
│   ├── 📄 default.yaml
│   ├── 📄 distributed_training.yaml
│   └── 📄 experiments/
│
├── 📁 experiments/                 # Experiment scripts
│   ├── 📄 single_chain_comparison.py
│   ├── 📄 byzantine_resilience.py
│   ├── 📄 cross_chain_detection.py
│   └── 📄 scalability.py
│
├── 📁 scripts/                     # Utility scripts
│   ├── 📄 download_datasets.py
│   ├── 📄 generate_multichain.py
│   └── 📄 preprocess.py
│
├── 📁 figures/                     # Paper figures
│   ├── 📄 necons_system_overview.pdf
│   ├── 📄 problem_formalization.pdf
│   └── 📄 results/
│
├── 📁 paper/                       # LaTeX source
│   ├── 📄 main.tex
│   ├── 📄 sections/
│   └── 📄 references.bib
│
└── 📁 tests/                       # Unit tests
    ├── 📄 test_models.py
    ├── 📄 test_consensus.py
    └── 📄 test_crosschain.py
```

---

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{necons2025,
  author    = {Author, First and Author, Second and Author, Third},
  title     = {{NECons}: Network-aware Edge-based Consensus for Distributed 
               Blockchain Anomaly Detection},
  journal   = {IEEE Transactions on Parallel and Distributed Systems},
  year      = {2025},
  volume    = {XX},
  number    = {XX},
  pages     = {1--15},
  doi       = {10.1109/TPDS.2025.XXXXXXX},
  publisher = {IEEE}
}
```

### Related Works

```bibtex
@inproceedings{ding2024effective,
  title     = {Effective Illicit Account Detection on Large Transaction Graphs},
  author    = {Ding, Zhihao and others},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024}
}

@inproceedings{blanchard2017machine,
  title     = {Machine Learning with Adversaries: Byzantine Tolerant 
               Gradient Descent},
  author    = {Blanchard, Peva and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}

@inproceedings{castro1999practical,
  title     = {Practical Byzantine Fault Tolerance},
  author    = {Castro, Miguel and Liskov, Barbara},
  booktitle = {OSDI},
  year      = {1999}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 BlockchainLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

This work was supported by:

- [Funding Agency 1] under Grant No. XXXX
- [Funding Agency 2] under Grant No. XXXX
- [University/Institution] Research Fund

We thank the anonymous reviewers for their valuable feedback that significantly improved this paper.

### Computing Resources

Experiments were conducted on:
- NVIDIA A100 80GB GPUs
- AMD EPYC 7763 64-Core Processors
- 512GB DDR4 RAM per node

---

## 📧 Contact

For questions or collaborations, please contact:

- **First Author**: [email@university.edu](mailto:email@university.edu)
- **Project Issues**: [GitHub Issues](https://github.com/BlockchainLab/NECons/issues)
- **Project Website**: [https://necons-project.github.io](https://necons-project.github.io)

---

<p align="center">
  <i>Star ⭐ this repository if you find it useful!</i>
</p>

<p align="center">
  Made with ❤️ by BlockchainLab
</p>
