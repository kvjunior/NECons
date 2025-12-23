# NECons: Network-aware Edge-based Consensus for Distributed Blockchain Anomaly Detection

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+"></a>
</p>

<p align="center">
  <b>IEEE Transactions on Parallel and Distributed Systems (TPDS) 2025</b>
</p>

---

## 馃搵 Table of Contents

- [Abstract](#-abstract)
- [Key Contributions](#-key-contributions)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Experiments](#-experiments)
- [Results](#-results)
- [Figures](#-figures)
- [Citation](#-citation)
- [License](#-license)

---

## 馃搫 Abstract

Blockchain networks processed over 500 million daily transactions in 2024, with security incidents causing losses exceeding $3.8 billion annually. Multi-chain ecosystems have enabled sophisticated anomaly patterns that exploit differences in confirmation times across heterogeneous blockchains. Effective anomaly detection requires distributed monitoring systems capable of operating across geographically dispersed edge nodes while maintaining resilience against adversarial participants.

**NECons** (Network-aware Edge-based Consensus) is a distributed framework that addresses three fundamental challenges in blockchain anomaly detection:

1. **Network Heterogeneity**: Existing GNN methods exhibit performance degradation exceeding 15% when latency surpasses 100ms
2. **Byzantine Adversaries**: Standard federated averaging fails catastrophically under Byzantine attacks
3. **Cross-Chain Correlation**: Single-chain methods cannot detect patterns spanning multiple blockchain networks

---

## 馃幆 Key Contributions

| Contribution | Description | Key Metric |
|:-------------|:------------|:-----------|
| **NetworkAwareMGD** | Network-state-adaptive graph attention with multigraph discrepancy analysis | **89.47% F1-score** (+5.35% over SOTA) |
| **Byzantine-Resilient Consensus** | Provably secure aggregation with formal safety, liveness, and accuracy guarantees | **83.45% accuracy** under 33% Byzantine nodes |
| **Cross-Chain Synchronization** | Multi-chain correlation detection across heterogeneous blockchains | **318 additional anomalies** detected |
| **Comprehensive Evaluation** | Rigorous experiments across 4 datasets and 6 baselines | **p < 0.001** statistical significance |

---

## 馃彈 System Architecture

<p align="center">
  <img src="figures/necons_system_overview.pdf" alt="NECons System Architecture" width="100%"/>
</p>

**Figure 1**: NECons three-tier architecture comprising multi-chain data ingestion, distributed edge processing with NetworkAwareMGD modules, and Byzantine-resilient consensus aggregation.

### Core Components

| Component | Description | Key Formula |
|:----------|:------------|:------------|
| **NetworkAwareMGD** | Adaptive graph attention with discrepancy | `h_v^(l+1) = 蟽(W路h_v + 危 伪虄_vu路[h_u鈥栁擾vu])` |
| **Trust Scoring** | Temporal Byzantine detection | `蟿岬(t) = 尾路蟿岬(t-1) + (1-尾)路[w鈧佅佱耽+w鈧偽坚耽+w鈧兾结耽]` |
| **Cross-Chain Sync** | Multi-chain correlation | `O(K路log n + 危鈧?F鈧柭稵鈧?` |

---

## 馃捇 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/BlockchainLab/NECons.git
cd NECons

# Create virtual environment
conda create -n necons python=3.10
conda activate necons

# Install dependencies
pip install -r requirements
```

### Requirements

See `requirements` file for complete dependency list including:
- PyTorch 2.0+
- PyTorch Geometric
- NumPy, Pandas, Scikit-learn
- NetworkX, Matplotlib, Seaborn

---

## 馃殌 Quick Start

### Training

```bash
# Train NECons model
python necons_main.py --config config.yaml --mode train

# With custom parameters
python necons_main.py \
    --dataset ethereum \
    --byzantine_ratio 0.3 \
    --epochs 100
```

### Evaluation

```bash
# Evaluate trained model
python necons_main.py --config config.yaml --mode eval

# Run comprehensive evaluation
python necons_eval.py --model_path checkpoints/best_model.pt
```

### Generate Experimental Results

```bash
# Generate all experimental figures
python experimental_results.py --output figures/
```

---

## 馃搧 Project Structure

```
NECons/
鈹?鈹溾攢鈹€ 馃搫 __init__.py                      # Package initialization
鈹溾攢鈹€ 馃搫 config.yaml                      # Configuration file
鈹溾攢鈹€ 馃搫 requirements                     # Python dependencies
鈹溾攢鈹€ 馃搫 README.md                        # Documentation
鈹?鈹溾攢鈹€ 馃搫 necons_main.py                   # Main entry point
鈹溾攢鈹€ 馃搫 necons_core.py                   # Core NECons framework
鈹?                                      #   - NetworkAwareMGD model
鈹?                                      #   - Byzantine consensus protocol
鈹?                                      #   - Cross-chain synchronization
鈹溾攢鈹€ 馃搫 necons_data.py                   # Data loading & preprocessing
鈹?                                      #   - Dataset loaders
鈹?                                      #   - Graph construction
鈹?                                      #   - Feature extraction
鈹溾攢鈹€ 馃搫 necons_train.py                  # Training pipeline
鈹?                                      #   - Distributed training
鈹?                                      #   - Byzantine simulation
鈹?                                      #   - Model optimization
鈹溾攢鈹€ 馃搫 necons_eval.py                   # Evaluation & metrics
鈹?                                      #   - Performance metrics
鈹?                                      #   - Statistical analysis
鈹?                                      #   - Result visualization
鈹溾攢鈹€ 馃搫 experimental_results.py          # Experiment reproduction
鈹?鈹斺攢鈹€ 馃搧 figures/                         # Paper figures
    鈹?    鈹溾攢鈹€ 馃搳 Architecture Diagrams
    鈹?  鈹溾攢鈹€ necons_system_overview.drawio
    鈹?  鈹溾攢鈹€ necons_system_overview.pdf
    鈹?  鈹溾攢鈹€ mgd_architecture.drawio
    鈹?  鈹溾攢鈹€ mgd_architecture.pdf
    鈹?  鈹溾攢鈹€ consensus_protocol.drawio
    鈹?  鈹溾攢鈹€ consensus_protocol.pdf
    鈹?  鈹溾攢鈹€ problem_formalization.drawio
    鈹?  鈹斺攢鈹€ problem_formalization.pdf
    鈹?    鈹斺攢鈹€ 馃搱 Experimental Results
        鈹溾攢鈹€ ablation_study_quad.py
        鈹溾攢鈹€ ablation_study_quad.pdf
        鈹溾攢鈹€ ablation_study_quad.png
        鈹溾攢鈹€ byzantine_resilience_quad.py
        鈹溾攢鈹€ byzantine_resilience_quad.pdf
        鈹溾攢鈹€ byzantine_resilience_quad.png
        鈹溾攢鈹€ scalability_analysis_quad.py
        鈹溾攢鈹€ scalability_analysis_quad.pdf
        鈹斺攢鈹€ scalability_analysis_quad.png
```

---

## 馃敩 Experiments

### Reproduce Paper Results

```bash
# Generate ablation study figures (Figure 6)
python figures/ablation_study_quad.py

# Generate Byzantine resilience figures (Figure 7)
python figures/byzantine_resilience_quad.py

# Generate scalability analysis figures (Figure 8)
python figures/scalability_analysis_quad.py
```

### Configuration

Edit `config.yaml` to customize experiments:

```yaml
model:
  name: NetworkAwareMGD
  hidden_dim: 128
  num_layers: 3
  dropout: 0.3

consensus:
  protocol: trust_weighted
  byzantine_threshold: 0.33
  trust_decay: 0.9

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
```

---

## 馃搱 Results

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

## 馃柤 Figures

### Architecture Diagrams

| Figure | File | Description |
|:-------|:-----|:------------|
| Fig. 1 | `necons_system_overview.pdf` | Complete system architecture |
| Fig. 2 | `problem_formalization.pdf` | Problem formalization diagram |
| Fig. 3 | `mgd_architecture.pdf` | NetworkAwareMGD architecture |
| Fig. 4 | `consensus_protocol.pdf` | Byzantine consensus protocol |

### Experimental Results

| Figure | File | Description |
|:-------|:-----|:------------|
| Fig. 5 | `ablation_study_quad.pdf` | Ablation study (4-panel) |
| Fig. 6 | `byzantine_resilience_quad.pdf` | Byzantine resilience (4-panel) |
| Fig. 7 | `scalability_analysis_quad.pdf` | Scalability analysis (4-panel) |

### Regenerate Figures

```bash
# All figures can be regenerated from source
cd figures/

# Regenerate experimental result figures
python ablation_study_quad.py
python byzantine_resilience_quad.py
python scalability_analysis_quad.py

# Architecture diagrams: Edit .drawio files in Draw.io
# Export as PDF for LaTeX inclusion
```

---

## 馃摑 Citation

If you find this work useful, please cite our paper:

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

---

## 馃摐 License

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```


<p align="center">
  <i>猸?Star this repository if you find it useful!</i>
</p>
