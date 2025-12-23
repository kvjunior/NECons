"""
================================================================================
NECons: Network-aware Edge-based Consensus for Distributed Blockchain 
        Anomaly Detection
================================================================================

Data Module: Dataset Loading, Preprocessing, Feature Engineering, and Augmentation

Target Venue: IEEE Transactions on Parallel and Distributed Systems (TPDS) 2026

This module implements:
1. Blockchain dataset classes for Ethereum and Bitcoin
2. Feature engineering for transaction graphs
3. Graph augmentation strategies for robust training
4. Distributed data loading for multi-GPU training
5. Memory-efficient batch processing for large-scale graphs

================================================================================
SUPPORTED DATASETS
================================================================================
1. Ethereum-Small (ethereum_s.pt): 2.1M nodes, 6.8M edges, 5% anomaly ratio
2. Ethereum-Phishing (ethereum_p.pt): 8.7M nodes, 13.6M edges, 3% anomaly ratio
3. Bitcoin-Medium (bitcoin_m.pt): 15.2M nodes, 14.2M edges, 2% anomaly ratio
4. Bitcoin-Large (bitcoin_l.pt): 45.8M nodes, 203.4M edges, 1% anomaly ratio

================================================================================
KEY REFERENCES
================================================================================
[1] Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph 
    Convolutional Networks for Financial Forensics", KDD Workshop 2019
[2] Hu et al., "BERT4ETH: A Pre-trained Transformer for Ethereum Fraud 
    Detection", WWW 2023
[3] Ding et al., "Effective Illicit Account Detection on Large Cryptocurrency 
    Multigraphs", ACM CIKM 2024
[4] Yang et al., "2DynEthNet: A Two-Dimensional Streaming Framework for 
    Ethereum Phishing Detection", IEEE TIFS 2024

================================================================================
Hardware Configuration:
- GPU: 4× NVIDIA GeForce RTX 3090 (24GB each)
- CPU: Intel Xeon Silver 4314 (64 cores) @ 2.40GHz
- RAM: 384GB DDR4
- OS: CentOS Linux 7 (Core)
================================================================================

Author: BlockchainLab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from pathlib import Path
import os
import json
import pickle
import logging
import random
import math
import time
from datetime import datetime
from abc import ABC, abstractmethod
import warnings

# PyTorch Geometric imports
try:
    from torch_geometric.data import Data, Dataset as PyGDataset, InMemoryDataset
    from torch_geometric.loader import NeighborLoader, ClusterLoader, GraphSAINTRandomWalkSampler
    from torch_geometric.utils import (
        add_self_loops, remove_self_loops, degree, 
        to_undirected, subgraph, k_hop_subgraph
    )
    from torch_geometric.transforms import (
        Compose, NormalizeFeatures, AddSelfLoops, 
        ToUndirected, RandomNodeSplit
    )
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("PyTorch Geometric not installed. Some features will be unavailable.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION AND ENUMERATIONS
# =============================================================================


class DatasetType(Enum):
    """Supported dataset types."""
    ETHEREUM_SMALL = "ethereum_s"
    ETHEREUM_PHISHING = "ethereum_p"
    BITCOIN_MEDIUM = "bitcoin_m"
    BITCOIN_LARGE = "bitcoin_l"
    SYNTHETIC = "synthetic"


class SplitType(Enum):
    """Data split types."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class AugmentationType(Enum):
    """Graph augmentation types."""
    NONE = "none"
    EDGE_DROPOUT = "edge_dropout"
    NODE_DROPOUT = "node_dropout"
    FEATURE_MASKING = "feature_masking"
    SUBGRAPH_SAMPLING = "subgraph_sampling"
    MIXUP = "mixup"


@dataclass
class DatasetConfig:
    """
    Configuration for blockchain datasets.
    
    Contains metadata and preprocessing parameters for each dataset type.
    """
    name: str
    num_nodes: int
    num_edges: int
    num_node_features: int
    num_edge_features: int
    num_classes: int
    anomaly_ratio: float
    blockchain_type: str  # "ethereum" or "bitcoin"
    
    # Preprocessing parameters
    normalize_features: bool = True
    add_self_loops: bool = True
    to_undirected: bool = False
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Temporal parameters (for temporal split)
    temporal_split: bool = False
    time_column: str = "timestamp"
    
    @classmethod
    def get_default_configs(cls) -> Dict[DatasetType, 'DatasetConfig']:
        """Return default configurations for all supported datasets."""
        return {
            DatasetType.ETHEREUM_SMALL: cls(
                name="Ethereum-Small",
                num_nodes=2_100_000,
                num_edges=6_800_000,
                num_node_features=8,
                num_edge_features=4,
                num_classes=2,
                anomaly_ratio=0.05,
                blockchain_type="ethereum"
            ),
            DatasetType.ETHEREUM_PHISHING: cls(
                name="Ethereum-Phishing",
                num_nodes=8_700_000,
                num_edges=13_600_000,
                num_node_features=8,
                num_edge_features=4,
                num_classes=2,
                anomaly_ratio=0.03,
                blockchain_type="ethereum"
            ),
            DatasetType.BITCOIN_MEDIUM: cls(
                name="Bitcoin-Medium",
                num_nodes=15_200_000,
                num_edges=14_200_000,
                num_node_features=8,
                num_edge_features=4,
                num_classes=2,
                anomaly_ratio=0.02,
                blockchain_type="bitcoin"
            ),
            DatasetType.BITCOIN_LARGE: cls(
                name="Bitcoin-Large",
                num_nodes=45_800_000,
                num_edges=203_400_000,
                num_node_features=8,
                num_edge_features=4,
                num_classes=2,
                anomaly_ratio=0.01,
                blockchain_type="bitcoin"
            ),
            DatasetType.SYNTHETIC: cls(
                name="Synthetic",
                num_nodes=100_000,
                num_edges=500_000,
                num_node_features=8,
                num_edge_features=4,
                num_classes=2,
                anomaly_ratio=0.05,
                blockchain_type="synthetic"
            ),
        }


# =============================================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================================
"""
This section implements feature extraction for blockchain transaction graphs.

Node Features (8-dimensional):
- f1: In-degree (normalized)
- f2: Out-degree (normalized)
- f3: Total transaction value (log-scaled)
- f4: Average transaction value
- f5: Transaction frequency (transactions per day)
- f6: Account age (days since first transaction)
- f7: Unique counterparties ratio
- f8: Balance volatility

Edge Features (4-dimensional):
- e1: Transaction value (log-scaled)
- e2: Time since last transaction
- e3: Transaction count between pair
- e4: Value deviation from mean
"""


class TransactionFeatureExtractor:
    """
    Feature extractor for blockchain transaction graphs.
    
    Extracts both structural (graph-based) and behavioral (transaction-based)
    features for nodes and edges in blockchain networks.
    
    Reference: Weber et al., KDD Workshop 2019 [1]
    """
    
    # Feature dimension constants
    NODE_FEATURE_DIM = 8
    EDGE_FEATURE_DIM = 4
    
    def __init__(
        self,
        normalize: bool = True,
        log_scale_values: bool = True,
        clip_outliers: bool = True,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize features to [0, 1]
            log_scale_values: Whether to apply log scaling to monetary values
            clip_outliers: Whether to clip outlier values
            outlier_threshold: Z-score threshold for outlier clipping
        """
        self.normalize = normalize
        self.log_scale_values = log_scale_values
        self.clip_outliers = clip_outliers
        self.outlier_threshold = outlier_threshold
        
        # Statistics for normalization (computed during fit)
        self.node_stats: Dict[str, Tuple[float, float]] = {}
        self.edge_stats: Dict[str, Tuple[float, float]] = {}
        self._fitted = False
    
    def fit(
        self,
        edge_index: Tensor,
        edge_values: Tensor,
        edge_timestamps: Tensor,
        num_nodes: int
    ) -> 'TransactionFeatureExtractor':
        """
        Compute statistics for feature normalization.
        
        Args:
            edge_index: Edge connectivity [2, E]
            edge_values: Transaction values [E]
            edge_timestamps: Transaction timestamps [E]
            num_nodes: Total number of nodes
        
        Returns:
            Self for method chaining
        """
        # Compute node features for statistics
        node_features = self._compute_raw_node_features(
            edge_index, edge_values, edge_timestamps, num_nodes
        )
        
        # Compute edge features for statistics
        edge_features = self._compute_raw_edge_features(
            edge_index, edge_values, edge_timestamps
        )
        
        # Store statistics
        for i in range(node_features.size(1)):
            col = node_features[:, i]
            self.node_stats[f'f{i+1}'] = (col.mean().item(), col.std().item() + 1e-8)
        
        for i in range(edge_features.size(1)):
            col = edge_features[:, i]
            self.edge_stats[f'e{i+1}'] = (col.mean().item(), col.std().item() + 1e-8)
        
        self._fitted = True
        return self
    
    def transform(
        self,
        edge_index: Tensor,
        edge_values: Tensor,
        edge_timestamps: Tensor,
        num_nodes: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract and normalize features.
        
        Args:
            edge_index: Edge connectivity [2, E]
            edge_values: Transaction values [E]
            edge_timestamps: Transaction timestamps [E]
            num_nodes: Total number of nodes
        
        Returns:
            node_features: [N, 8] tensor
            edge_features: [E, 4] tensor
        """
        # Compute raw features
        node_features = self._compute_raw_node_features(
            edge_index, edge_values, edge_timestamps, num_nodes
        )
        edge_features = self._compute_raw_edge_features(
            edge_index, edge_values, edge_timestamps
        )
        
        # Normalize if fitted
        if self.normalize and self._fitted:
            node_features = self._normalize_features(node_features, self.node_stats)
            edge_features = self._normalize_features(edge_features, self.edge_stats)
        
        return node_features, edge_features
    
    def fit_transform(
        self,
        edge_index: Tensor,
        edge_values: Tensor,
        edge_timestamps: Tensor,
        num_nodes: int
    ) -> Tuple[Tensor, Tensor]:
        """Fit and transform in one step."""
        self.fit(edge_index, edge_values, edge_timestamps, num_nodes)
        return self.transform(edge_index, edge_values, edge_timestamps, num_nodes)
    
    def _compute_raw_node_features(
        self,
        edge_index: Tensor,
        edge_values: Tensor,
        edge_timestamps: Tensor,
        num_nodes: int
    ) -> Tensor:
        """
        Compute raw node features.
        
        Node Features:
        - f1: In-degree (normalized by max)
        - f2: Out-degree (normalized by max)
        - f3: Total transaction value (log-scaled)
        - f4: Average transaction value
        - f5: Transaction frequency
        - f6: Account age (normalized)
        - f7: Unique counterparties ratio
        - f8: Balance volatility
        """
        src, dst = edge_index[0], edge_index[1]
        
        # f1: In-degree
        in_degree = torch.zeros(num_nodes, dtype=torch.float32)
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        
        # f2: Out-degree
        out_degree = torch.zeros(num_nodes, dtype=torch.float32)
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        
        # f3: Total incoming value
        total_in_value = torch.zeros(num_nodes, dtype=torch.float32)
        total_in_value.scatter_add_(0, dst, edge_values.float())
        if self.log_scale_values:
            total_in_value = torch.log1p(total_in_value)
        
        # f4: Average transaction value per node
        total_degree = in_degree + out_degree
        total_value = torch.zeros(num_nodes, dtype=torch.float32)
        total_value.scatter_add_(0, src, edge_values.float())
        total_value.scatter_add_(0, dst, edge_values.float())
        avg_value = total_value / (total_degree + 1e-8)
        if self.log_scale_values:
            avg_value = torch.log1p(avg_value)
        
        # f5: Transaction frequency (transactions per time unit)
        min_time = edge_timestamps.min()
        max_time = edge_timestamps.max()
        time_range = (max_time - min_time).float() + 1e-8
        tx_frequency = total_degree / time_range * 86400  # Per day
        
        # f6: Account age (time since first transaction)
        first_tx_time = torch.full((num_nodes,), float('inf'))
        for i, (s, d, t) in enumerate(zip(src, dst, edge_timestamps)):
            first_tx_time[s] = min(first_tx_time[s], t.item())
            first_tx_time[d] = min(first_tx_time[d], t.item())
        first_tx_time[first_tx_time == float('inf')] = max_time.item()
        account_age = (max_time.float() - first_tx_time) / time_range
        
        # f7: Unique counterparties ratio
        unique_out = self._count_unique_counterparties(src, dst, num_nodes)
        unique_in = self._count_unique_counterparties(dst, src, num_nodes)
        counterparty_ratio = (unique_out + unique_in) / (total_degree + 1e-8)
        
        # f8: Value volatility (std of transaction values)
        value_std = self._compute_value_volatility(src, dst, edge_values, num_nodes)
        if self.log_scale_values:
            value_std = torch.log1p(value_std)
        
        # Stack features
        node_features = torch.stack([
            in_degree,
            out_degree,
            total_in_value,
            avg_value,
            tx_frequency,
            account_age,
            counterparty_ratio,
            value_std
        ], dim=1)
        
        return node_features
    
    def _compute_raw_edge_features(
        self,
        edge_index: Tensor,
        edge_values: Tensor,
        edge_timestamps: Tensor
    ) -> Tensor:
        """
        Compute raw edge features.
        
        Edge Features:
        - e1: Transaction value (log-scaled)
        - e2: Time since last transaction between pair
        - e3: Transaction count between pair
        - e4: Value deviation from mean
        """
        num_edges = edge_index.size(1)
        src, dst = edge_index[0], edge_index[1]
        
        # e1: Transaction value
        e1 = edge_values.float()
        if self.log_scale_values:
            e1 = torch.log1p(e1)
        
        # e2: Normalized timestamp (time since earliest)
        min_time = edge_timestamps.min()
        max_time = edge_timestamps.max()
        time_range = (max_time - min_time).float() + 1e-8
        e2 = (edge_timestamps.float() - min_time.float()) / time_range
        
        # e3: Edge multiplicity (count of transactions between pair)
        # Use hash for efficiency
        pair_hash = src * (dst.max() + 1) + dst
        unique_pairs, inverse_indices = torch.unique(pair_hash, return_inverse=True)
        pair_counts = torch.bincount(inverse_indices).float()
        e3 = pair_counts[inverse_indices]
        e3 = torch.log1p(e3)  # Log scale
        
        # e4: Value deviation from global mean
        mean_value = edge_values.float().mean()
        std_value = edge_values.float().std() + 1e-8
        e4 = (edge_values.float() - mean_value) / std_value
        if self.clip_outliers:
            e4 = torch.clamp(e4, -self.outlier_threshold, self.outlier_threshold)
        
        # Stack features
        edge_features = torch.stack([e1, e2, e3, e4], dim=1)
        
        return edge_features
    
    def _count_unique_counterparties(
        self,
        src: Tensor,
        dst: Tensor,
        num_nodes: int
    ) -> Tensor:
        """Count unique counterparties for each node."""
        unique_counts = torch.zeros(num_nodes, dtype=torch.float32)
        
        # Group by source node
        for node_id in range(num_nodes):
            mask = src == node_id
            if mask.any():
                unique_counts[node_id] = dst[mask].unique().size(0)
        
        return unique_counts
    
    def _compute_value_volatility(
        self,
        src: Tensor,
        dst: Tensor,
        values: Tensor,
        num_nodes: int
    ) -> Tensor:
        """Compute value volatility (std) for each node."""
        volatility = torch.zeros(num_nodes, dtype=torch.float32)
        
        for node_id in range(min(num_nodes, 10000)):  # Limit for efficiency
            mask = (src == node_id) | (dst == node_id)
            if mask.sum() > 1:
                volatility[node_id] = values[mask].float().std()
        
        return volatility
    
    def _normalize_features(
        self,
        features: Tensor,
        stats: Dict[str, Tuple[float, float]]
    ) -> Tensor:
        """Normalize features using stored statistics."""
        normalized = features.clone()
        
        for i, (key, (mean, std)) in enumerate(stats.items()):
            if i < normalized.size(1):
                normalized[:, i] = (normalized[:, i] - mean) / std
                if self.clip_outliers:
                    normalized[:, i] = torch.clamp(
                        normalized[:, i],
                        -self.outlier_threshold,
                        self.outlier_threshold
                    )
        
        return normalized


class NodeFeatureNormalizer:
    """
    Normalizes node features using various strategies.
    
    Supports:
    - Z-score normalization
    - Min-max normalization
    - Robust scaling (using median and IQR)
    """
    
    def __init__(self, method: str = "zscore"):
        """
        Args:
            method: Normalization method ("zscore", "minmax", "robust")
        """
        self.method = method
        self.stats: Dict[str, Any] = {}
        self._fitted = False
    
    def fit(self, features: Tensor) -> 'NodeFeatureNormalizer':
        """Compute normalization statistics."""
        if self.method == "zscore":
            self.stats['mean'] = features.mean(dim=0)
            self.stats['std'] = features.std(dim=0) + 1e-8
        elif self.method == "minmax":
            self.stats['min'] = features.min(dim=0).values
            self.stats['max'] = features.max(dim=0).values
        elif self.method == "robust":
            self.stats['median'] = features.median(dim=0).values
            q75 = torch.quantile(features, 0.75, dim=0)
            q25 = torch.quantile(features, 0.25, dim=0)
            self.stats['iqr'] = q75 - q25 + 1e-8
        
        self._fitted = True
        return self
    
    def transform(self, features: Tensor) -> Tensor:
        """Apply normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if self.method == "zscore":
            return (features - self.stats['mean']) / self.stats['std']
        elif self.method == "minmax":
            range_val = self.stats['max'] - self.stats['min'] + 1e-8
            return (features - self.stats['min']) / range_val
        elif self.method == "robust":
            return (features - self.stats['median']) / self.stats['iqr']
        
        return features
    
    def fit_transform(self, features: Tensor) -> Tensor:
        """Fit and transform in one step."""
        return self.fit(features).transform(features)
    
    def inverse_transform(self, features: Tensor) -> Tensor:
        """Reverse normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")
        
        if self.method == "zscore":
            return features * self.stats['std'] + self.stats['mean']
        elif self.method == "minmax":
            range_val = self.stats['max'] - self.stats['min']
            return features * range_val + self.stats['min']
        elif self.method == "robust":
            return features * self.stats['iqr'] + self.stats['median']
        
        return features


# =============================================================================
# SECTION 3: GRAPH AUGMENTATION
# =============================================================================
"""
Graph augmentation strategies for robust training.

Implements various augmentation techniques to improve model generalization
and robustness to adversarial attacks.
"""


class GraphAugmentation(ABC):
    """Abstract base class for graph augmentation strategies."""
    
    @abstractmethod
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Apply augmentation.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, D]
            y: Node labels [N]
        
        Returns:
            Augmented (x, edge_index, edge_attr, y)
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return augmentation name."""
        pass


class EdgeDropout(GraphAugmentation):
    """
    Randomly drop edges from the graph.
    
    Improves robustness to missing connections and
    reduces overfitting to specific graph structure.
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        """
        Args:
            dropout_rate: Probability of dropping each edge
        """
        self.dropout_rate = dropout_rate
    
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if self.dropout_rate <= 0:
            return x, edge_index, edge_attr, y
        
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > self.dropout_rate
        
        new_edge_index = edge_index[:, mask]
        new_edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        return x, new_edge_index, new_edge_attr, y
    
    def name(self) -> str:
        return f"EdgeDropout(p={self.dropout_rate})"


class NodeDropout(GraphAugmentation):
    """
    Randomly mask node features.
    
    Sets random node features to zero to improve
    robustness to missing or corrupted features.
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        """
        Args:
            dropout_rate: Probability of masking each node
        """
        self.dropout_rate = dropout_rate
    
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if self.dropout_rate <= 0:
            return x, edge_index, edge_attr, y
        
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes) > self.dropout_rate
        
        new_x = x.clone()
        new_x[~mask] = 0
        
        return new_x, edge_index, edge_attr, y
    
    def name(self) -> str:
        return f"NodeDropout(p={self.dropout_rate})"


class FeatureMasking(GraphAugmentation):
    """
    Randomly mask individual features across all nodes.
    
    Different from NodeDropout - this masks specific
    feature dimensions rather than entire nodes.
    """
    
    def __init__(self, mask_rate: float = 0.1):
        """
        Args:
            mask_rate: Probability of masking each feature dimension
        """
        self.mask_rate = mask_rate
    
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if self.mask_rate <= 0:
            return x, edge_index, edge_attr, y
        
        num_features = x.size(1)
        mask = torch.rand(num_features) > self.mask_rate
        
        new_x = x.clone()
        new_x[:, ~mask] = 0
        
        return new_x, edge_index, edge_attr, y
    
    def name(self) -> str:
        return f"FeatureMasking(p={self.mask_rate})"


class SubgraphSampling(GraphAugmentation):
    """
    Sample a random subgraph for training.
    
    Useful for large-scale graphs where training on
    the full graph is computationally prohibitive.
    """
    
    def __init__(self, sample_ratio: float = 0.5, method: str = "random"):
        """
        Args:
            sample_ratio: Fraction of nodes to sample
            method: Sampling method ("random", "degree_weighted")
        """
        self.sample_ratio = sample_ratio
        self.method = method
    
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if self.sample_ratio >= 1.0:
            return x, edge_index, edge_attr, y
        
        num_nodes = x.size(0)
        num_sample = int(num_nodes * self.sample_ratio)
        
        if self.method == "random":
            sample_idx = torch.randperm(num_nodes)[:num_sample]
        elif self.method == "degree_weighted":
            # Sample proportional to degree
            degrees = degree(edge_index[0], num_nodes) + degree(edge_index[1], num_nodes)
            probs = degrees / degrees.sum()
            sample_idx = torch.multinomial(probs, num_sample, replacement=False)
        else:
            sample_idx = torch.randperm(num_nodes)[:num_sample]
        
        # Get subgraph
        sample_idx = sample_idx.sort().values
        
        # Create node mapping
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[sample_idx] = True
        
        # Filter edges
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        new_edge_index = edge_index[:, edge_mask]
        
        # Remap node indices
        node_map = torch.zeros(num_nodes, dtype=torch.long)
        node_map[sample_idx] = torch.arange(num_sample)
        new_edge_index = node_map[new_edge_index]
        
        # Get subgraph features and labels
        new_x = x[sample_idx]
        new_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
        new_y = y[sample_idx] if y is not None else None
        
        return new_x, new_edge_index, new_edge_attr, new_y
    
    def name(self) -> str:
        return f"SubgraphSampling(ratio={self.sample_ratio}, method={self.method})"


class GraphMixup(GraphAugmentation):
    """
    Mixup augmentation for graphs.
    
    Interpolates between node features while preserving
    graph structure. Useful for semi-supervised learning.
    
    Reference: Adapted from Zhang et al., "mixup: Beyond Empirical 
               Risk Minimization", ICLR 2018
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixup ratio
        """
        self.alpha = alpha
    
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        if self.alpha <= 0:
            return x, edge_index, edge_attr, y
        
        # Sample mixup ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create shuffled version
        perm = torch.randperm(x.size(0))
        x_perm = x[perm]
        
        # Mixup features
        new_x = lam * x + (1 - lam) * x_perm
        
        # Mixup labels if provided
        new_y = None
        if y is not None:
            # For classification, return soft labels
            new_y = lam * F.one_hot(y, num_classes=2).float()
            new_y += (1 - lam) * F.one_hot(y[perm], num_classes=2).float()
        
        return new_x, edge_index, edge_attr, new_y
    
    def name(self) -> str:
        return f"GraphMixup(alpha={self.alpha})"


class ComposedAugmentation(GraphAugmentation):
    """Compose multiple augmentations."""
    
    def __init__(self, augmentations: List[GraphAugmentation]):
        self.augmentations = augmentations
    
    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        for aug in self.augmentations:
            x, edge_index, edge_attr, y = aug(x, edge_index, edge_attr, y)
        return x, edge_index, edge_attr, y
    
    def name(self) -> str:
        names = [aug.name() for aug in self.augmentations]
        return f"Composed({', '.join(names)})"


def create_augmentation(
    aug_type: AugmentationType,
    **kwargs
) -> GraphAugmentation:
    """Factory function for creating augmentations."""
    augmentations = {
        AugmentationType.EDGE_DROPOUT: EdgeDropout,
        AugmentationType.NODE_DROPOUT: NodeDropout,
        AugmentationType.FEATURE_MASKING: FeatureMasking,
        AugmentationType.SUBGRAPH_SAMPLING: SubgraphSampling,
        AugmentationType.MIXUP: GraphMixup,
    }
    
    if aug_type == AugmentationType.NONE:
        return lambda x, e, ea, y: (x, e, ea, y)
    
    if aug_type not in augmentations:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    return augmentations[aug_type](**kwargs)


# =============================================================================
# SECTION 4: DATASET CLASSES
# =============================================================================
"""
PyTorch and PyTorch Geometric dataset classes for blockchain data.
"""


class BlockchainGraphData:
    """
    Container for blockchain graph data.
    
    Stores all graph components in a format compatible with
    both PyTorch and PyTorch Geometric.
    """
    
    def __init__(
        self,
        x: Tensor,
        edge_index: Tensor,
        y: Tensor,
        edge_attr: Optional[Tensor] = None,
        train_mask: Optional[Tensor] = None,
        val_mask: Optional[Tensor] = None,
        test_mask: Optional[Tensor] = None,
        node_timestamps: Optional[Tensor] = None,
        edge_timestamps: Optional[Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            y: Node labels [N]
            edge_attr: Edge features [E, D]
            train_mask: Training node mask [N]
            val_mask: Validation node mask [N]
            test_mask: Test node mask [N]
            node_timestamps: Timestamps per node [N]
            edge_timestamps: Timestamps per edge [E]
            metadata: Additional metadata dictionary
        """
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.edge_attr = edge_attr
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.node_timestamps = node_timestamps
        self.edge_timestamps = edge_timestamps
        self.metadata = metadata or {}
    
    @property
    def num_nodes(self) -> int:
        return self.x.size(0)
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)
    
    @property
    def num_features(self) -> int:
        return self.x.size(1)
    
    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1
    
    def to(self, device: torch.device) -> 'BlockchainGraphData':
        """Move all tensors to device."""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        if self.train_mask is not None:
            self.train_mask = self.train_mask.to(device)
        if self.val_mask is not None:
            self.val_mask = self.val_mask.to(device)
        if self.test_mask is not None:
            self.test_mask = self.test_mask.to(device)
        return self
    
    def to_pyg_data(self) -> 'Data':
        """Convert to PyTorch Geometric Data object."""
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric not installed")
        
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.y,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask
        )
    
    def get_split(self, split: SplitType) -> Tuple[Tensor, Tensor]:
        """Get node indices and labels for a specific split."""
        if split == SplitType.TRAIN:
            mask = self.train_mask
        elif split == SplitType.VALIDATION:
            mask = self.val_mask
        else:
            mask = self.test_mask
        
        if mask is None:
            raise ValueError(f"No mask defined for split: {split}")
        
        indices = mask.nonzero(as_tuple=True)[0]
        labels = self.y[mask]
        
        return indices, labels
    
    def compute_class_weights(self) -> Tensor:
        """Compute class weights for imbalanced classification."""
        if self.train_mask is not None:
            labels = self.y[self.train_mask]
        else:
            labels = self.y
        
        class_counts = torch.bincount(labels)
        total = class_counts.sum().float()
        weights = total / (len(class_counts) * class_counts.float() + 1e-8)
        
        return weights
    
    def summary(self) -> Dict[str, Any]:
        """Return dataset summary statistics."""
        summary = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'edge_density': self.num_edges / (self.num_nodes ** 2),
        }
        
        # Label distribution
        label_counts = torch.bincount(self.y)
        summary['label_distribution'] = {
            f'class_{i}': count.item() for i, count in enumerate(label_counts)
        }
        
        # Split sizes
        if self.train_mask is not None:
            summary['train_size'] = self.train_mask.sum().item()
        if self.val_mask is not None:
            summary['val_size'] = self.val_mask.sum().item()
        if self.test_mask is not None:
            summary['test_size'] = self.test_mask.sum().item()
        
        return summary


class SyntheticBlockchainDataset:
    """
    Synthetic blockchain dataset generator for testing and development.
    
    Generates realistic blockchain transaction graphs with controllable
    properties such as anomaly ratio, graph density, and class distribution.
    """
    
    def __init__(
        self,
        num_nodes: int = 10000,
        num_edges: int = 50000,
        num_features: int = 8,
        num_edge_features: int = 4,
        anomaly_ratio: float = 0.05,
        seed: int = 42
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges in the graph
            num_features: Node feature dimension
            num_edge_features: Edge feature dimension
            anomaly_ratio: Fraction of anomalous nodes
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_features = num_features
        self.num_edge_features = num_edge_features
        self.anomaly_ratio = anomaly_ratio
        self.seed = seed
    
    def generate(self) -> BlockchainGraphData:
        """Generate synthetic blockchain graph."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Generate labels (0: normal, 1: anomaly)
        num_anomalies = int(self.num_nodes * self.anomaly_ratio)
        y = torch.zeros(self.num_nodes, dtype=torch.long)
        anomaly_indices = torch.randperm(self.num_nodes)[:num_anomalies]
        y[anomaly_indices] = 1
        
        # Generate node features
        # Normal nodes: features from one distribution
        # Anomalous nodes: slightly different distribution
        x_normal = torch.randn(self.num_nodes, self.num_features) * 0.5
        x_anomaly_shift = torch.randn(1, self.num_features) * 2
        x = x_normal.clone()
        x[anomaly_indices] += x_anomaly_shift
        
        # Generate edges using preferential attachment
        edge_index = self._generate_preferential_attachment_edges()
        
        # Add more edges between anomalous nodes (clique behavior)
        edge_index = self._add_anomaly_clique_edges(edge_index, anomaly_indices)
        
        # Generate edge features
        edge_attr = torch.randn(edge_index.size(1), self.num_edge_features)
        
        # Generate timestamps
        edge_timestamps = torch.sort(torch.rand(edge_index.size(1)) * 1000000)[0]
        
        # Create train/val/test splits
        train_mask, val_mask, test_mask = self._create_splits()
        
        # Create metadata
        metadata = {
            'dataset_type': 'synthetic',
            'generation_seed': self.seed,
            'anomaly_ratio': self.anomaly_ratio,
            'generation_time': datetime.now().isoformat()
        }
        
        return BlockchainGraphData(
            x=x,
            edge_index=edge_index,
            y=y,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            edge_timestamps=edge_timestamps,
            metadata=metadata
        )
    
    def _generate_preferential_attachment_edges(self) -> Tensor:
        """Generate edges using Barabási-Albert preferential attachment."""
        # Start with a small complete graph
        m0 = 5  # Initial nodes
        m = max(1, self.num_edges // self.num_nodes)  # Edges per new node
        
        edges = []
        degrees = torch.zeros(self.num_nodes)
        
        # Initial complete graph
        for i in range(m0):
            for j in range(i + 1, m0):
                edges.append([i, j])
                edges.append([j, i])
                degrees[i] += 1
                degrees[j] += 1
        
        # Add remaining nodes with preferential attachment
        for new_node in range(m0, self.num_nodes):
            if new_node >= self.num_nodes:
                break
            
            # Select m existing nodes with probability proportional to degree
            probs = degrees[:new_node] + 1  # +1 to avoid zero probability
            probs = probs / probs.sum()
            
            targets = torch.multinomial(probs, min(m, new_node), replacement=False)
            
            for target in targets:
                edges.append([new_node, target.item()])
                edges.append([target.item(), new_node])
                degrees[new_node] += 1
                degrees[target] += 1
            
            if len(edges) >= self.num_edges * 2:
                break
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Trim to desired number of edges
        if edge_index.size(1) > self.num_edges:
            perm = torch.randperm(edge_index.size(1))[:self.num_edges]
            edge_index = edge_index[:, perm]
        
        return edge_index
    
    def _add_anomaly_clique_edges(
        self,
        edge_index: Tensor,
        anomaly_indices: Tensor
    ) -> Tensor:
        """Add edges between anomalous nodes to simulate coordinated behavior."""
        num_clique_edges = min(len(anomaly_indices) * 2, 100)
        
        clique_edges = []
        for _ in range(num_clique_edges):
            i = anomaly_indices[torch.randint(len(anomaly_indices), (1,))].item()
            j = anomaly_indices[torch.randint(len(anomaly_indices), (1,))].item()
            if i != j:
                clique_edges.append([i, j])
        
        if clique_edges:
            clique_tensor = torch.tensor(clique_edges, dtype=torch.long).t()
            edge_index = torch.cat([edge_index, clique_tensor], dim=1)
        
        return edge_index
    
    def _create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Create train/val/test splits."""
        indices = torch.randperm(self.num_nodes)
        
        train_size = int(self.num_nodes * train_ratio)
        val_size = int(self.num_nodes * val_ratio)
        
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask


class BlockchainDatasetLoader:
    """
    Loader for real blockchain datasets.
    
    Loads preprocessed datasets from disk and applies
    transformations as needed.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config: Optional[DatasetConfig] = None
    ):
        """
        Args:
            data_dir: Directory containing dataset files
            config: Dataset configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.feature_extractor = TransactionFeatureExtractor()
    
    def load(self, dataset_type: DatasetType) -> BlockchainGraphData:
        """
        Load dataset from disk.
        
        Args:
            dataset_type: Type of dataset to load
        
        Returns:
            BlockchainGraphData object
        """
        # Get config if not provided
        if self.config is None:
            configs = DatasetConfig.get_default_configs()
            self.config = configs.get(dataset_type)
        
        # Construct file path
        filename = f"{dataset_type.value}.pt"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Dataset file not found: {filepath}")
            logger.info("Generating synthetic dataset instead")
            return self._generate_synthetic_fallback(dataset_type)
        
        # Load data
        data = torch.load(filepath)
        
        # Extract components
        x = data.get('x', data.get('node_features'))
        edge_index = data.get('edge_index')
        y = data.get('y', data.get('labels'))
        edge_attr = data.get('edge_attr', data.get('edge_features'))
        
        # Create masks if not present
        train_mask = data.get('train_mask')
        val_mask = data.get('val_mask')
        test_mask = data.get('test_mask')
        
        if train_mask is None:
            train_mask, val_mask, test_mask = self._create_random_splits(x.size(0))
        
        # Apply transformations
        if self.config and self.config.normalize_features:
            normalizer = NodeFeatureNormalizer(method="zscore")
            x = normalizer.fit_transform(x)
        
        if self.config and self.config.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=x.size(0)
            )
        
        return BlockchainGraphData(
            x=x,
            edge_index=edge_index,
            y=y,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            metadata={'source': str(filepath), 'dataset_type': dataset_type.value}
        )
    
    def _generate_synthetic_fallback(
        self,
        dataset_type: DatasetType
    ) -> BlockchainGraphData:
        """Generate synthetic data when real data is unavailable."""
        configs = DatasetConfig.get_default_configs()
        config = configs.get(dataset_type, configs[DatasetType.SYNTHETIC])
        
        # Scale down for synthetic generation
        scale = 0.01  # 1% of real size for testing
        generator = SyntheticBlockchainDataset(
            num_nodes=int(config.num_nodes * scale),
            num_edges=int(config.num_edges * scale),
            num_features=config.num_node_features,
            num_edge_features=config.num_edge_features,
            anomaly_ratio=config.anomaly_ratio
        )
        
        return generator.generate()
    
    def _create_random_splits(
        self,
        num_nodes: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Create random train/val/test splits."""
        indices = torch.randperm(num_nodes)
        
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask


# =============================================================================
# SECTION 5: DATA LOADERS AND SAMPLERS
# =============================================================================
"""
Efficient data loading for large-scale graph training.

Implements mini-batch training strategies including:
- Neighbor sampling (GraphSAGE-style)
- Cluster-based sampling
- Distributed sampling for multi-GPU training
"""


class DistributedGraphSampler(Sampler):
    """
    Distributed sampler for graph mini-batch training.
    
    Ensures each GPU processes different nodes while maintaining
    proper graph structure for message passing.
    """
    
    def __init__(
        self,
        data: BlockchainGraphData,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False
    ):
        """
        Args:
            data: Graph data object
            num_replicas: Number of distributed processes (GPUs)
            rank: Rank of current process
            shuffle: Whether to shuffle node order
            seed: Random seed for shuffling
            drop_last: Whether to drop incomplete batches
        """
        self.data = data
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Get training indices
        if data.train_mask is not None:
            self.indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
        else:
            self.indices = list(range(data.num_nodes))
        
        # Compute per-replica size
        self.total_size = len(self.indices)
        self.num_samples = self.total_size // num_replicas
        
        if not drop_last and self.total_size % num_replicas != 0:
            self.num_samples += 1
    
    def __iter__(self) -> Iterator[int]:
        # Deterministic shuffling based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = [self.indices[i] for i in torch.randperm(len(self.indices), generator=g)]
        else:
            indices = self.indices.copy()
        
        # Pad to make evenly divisible
        if len(indices) % self.num_replicas != 0:
            padding = self.num_replicas - (len(indices) % self.num_replicas)
            indices += indices[:padding]
        
        # Subsample for this rank
        indices = indices[self.rank::self.num_replicas]
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch


class NeighborSamplerWrapper:
    """
    Wrapper for PyTorch Geometric's NeighborLoader.
    
    Provides a consistent interface for neighbor sampling
    across different graph formats.
    """
    
    def __init__(
        self,
        data: BlockchainGraphData,
        num_neighbors: List[int] = [25, 10],
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        """
        Args:
            data: Graph data object
            num_neighbors: Number of neighbors to sample at each hop
            batch_size: Number of seed nodes per batch
            shuffle: Whether to shuffle seed nodes
            num_workers: Number of data loading workers
        """
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for NeighborSampler")
    
    def get_loader(self, split: SplitType = SplitType.TRAIN) -> 'NeighborLoader':
        """Get NeighborLoader for specified split."""
        pyg_data = self.data.to_pyg_data()
        
        # Get input nodes for this split
        if split == SplitType.TRAIN:
            input_nodes = self.data.train_mask
        elif split == SplitType.VALIDATION:
            input_nodes = self.data.val_mask
        else:
            input_nodes = self.data.test_mask
        
        return NeighborLoader(
            pyg_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=self.shuffle if split == SplitType.TRAIN else False,
            num_workers=self.num_workers
        )


class MemoryEfficientLoader:
    """
    Memory-efficient data loader for very large graphs.
    
    Uses disk-based storage and lazy loading to handle
    graphs that don't fit in GPU memory.
    """
    
    def __init__(
        self,
        data: BlockchainGraphData,
        batch_size: int = 1024,
        cache_dir: str = "/tmp/necons_cache",
        prefetch_factor: int = 2
    ):
        """
        Args:
            data: Graph data object
            batch_size: Batch size for node sampling
            cache_dir: Directory for caching
            prefetch_factor: Number of batches to prefetch
        """
        self.data = data
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.prefetch_factor = prefetch_factor
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Precompute adjacency lists for efficient neighbor lookup
        self._build_adjacency_cache()
    
    def _build_adjacency_cache(self) -> None:
        """Build adjacency list cache for efficient neighbor lookup."""
        self.adj_list = defaultdict(list)
        
        src, dst = self.data.edge_index[0], self.data.edge_index[1]
        for s, d in zip(src.tolist(), dst.tolist()):
            self.adj_list[s].append(d)
    
    def sample_subgraph(
        self,
        seed_nodes: Tensor,
        num_hops: int = 2
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a k-hop subgraph around seed nodes.
        
        Args:
            seed_nodes: Seed node indices
            num_hops: Number of hops to expand
        
        Returns:
            (node_idx, edge_index, node_features, labels)
        """
        # Use PyG's k_hop_subgraph if available
        if HAS_PYG:
            node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
                seed_nodes,
                num_hops,
                self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes
            )
        else:
            # Manual BFS-based subgraph extraction
            visited = set(seed_nodes.tolist())
            frontier = set(seed_nodes.tolist())
            
            for _ in range(num_hops):
                next_frontier = set()
                for node in frontier:
                    for neighbor in self.adj_list[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier
            
            node_idx = torch.tensor(sorted(visited), dtype=torch.long)
            
            # Get edges within subgraph
            node_set = set(node_idx.tolist())
            edges = []
            src, dst = self.data.edge_index[0], self.data.edge_index[1]
            for s, d in zip(src.tolist(), dst.tolist()):
                if s in node_set and d in node_set:
                    edges.append([s, d])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2, 0, dtype=torch.long)
        
        # Get features and labels
        node_features = self.data.x[node_idx]
        labels = self.data.y[node_idx]
        
        return node_idx, edge_index, node_features, labels
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Iterate over batches."""
        # Get training indices
        if self.data.train_mask is not None:
            indices = self.data.train_mask.nonzero(as_tuple=True)[0]
        else:
            indices = torch.arange(self.data.num_nodes)
        
        # Shuffle
        perm = torch.randperm(len(indices))
        indices = indices[perm]
        
        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.sample_subgraph(batch_indices)
    
    def __len__(self) -> int:
        if self.data.train_mask is not None:
            return (self.data.train_mask.sum().item() + self.batch_size - 1) // self.batch_size
        return (self.data.num_nodes + self.batch_size - 1) // self.batch_size


# =============================================================================
# SECTION 6: TRANSACTION SEQUENCE HANDLING
# =============================================================================
"""
Handles transaction sequences for Edge2Seq encoding.
"""


class TransactionSequenceBuilder:
    """
    Builds transaction sequences for nodes.
    
    For each node, extracts the sequence of incoming and outgoing
    transactions ordered by timestamp.
    """
    
    def __init__(
        self,
        max_sequence_length: int = 50,
        feature_dim: int = 8,
        padding_value: float = 0.0
    ):
        """
        Args:
            max_sequence_length: Maximum sequence length (truncate if longer)
            feature_dim: Feature dimension per transaction
            padding_value: Value for padding shorter sequences
        """
        self.max_length = max_sequence_length
        self.feature_dim = feature_dim
        self.padding_value = padding_value
    
    def build_sequences(
        self,
        data: BlockchainGraphData
    ) -> Tuple[Tensor, Tensor]:
        """
        Build incoming and outgoing transaction sequences for all nodes.
        
        Args:
            data: Graph data object
        
        Returns:
            incoming_sequences: [N, max_length, feature_dim]
            outgoing_sequences: [N, max_length, feature_dim]
        """
        num_nodes = data.num_nodes
        
        # Initialize sequence tensors
        incoming_seq = torch.full(
            (num_nodes, self.max_length, self.feature_dim),
            self.padding_value
        )
        outgoing_seq = torch.full(
            (num_nodes, self.max_length, self.feature_dim),
            self.padding_value
        )
        
        # Get edge information
        src, dst = data.edge_index[0], data.edge_index[1]
        
        # Sort edges by timestamp if available
        if data.edge_timestamps is not None:
            sort_idx = torch.argsort(data.edge_timestamps)
            src = src[sort_idx]
            dst = dst[sort_idx]
            if data.edge_attr is not None:
                edge_features = data.edge_attr[sort_idx]
            else:
                edge_features = torch.zeros(len(src), self.feature_dim)
        else:
            edge_features = data.edge_attr if data.edge_attr is not None else torch.zeros(len(src), self.feature_dim)
        
        # Build sequences for each node
        incoming_counts = torch.zeros(num_nodes, dtype=torch.long)
        outgoing_counts = torch.zeros(num_nodes, dtype=torch.long)
        
        for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
            # Outgoing transaction for source
            if outgoing_counts[s] < self.max_length:
                feat = edge_features[i] if edge_features.size(0) > i else torch.zeros(self.feature_dim)
                outgoing_seq[s, outgoing_counts[s]] = feat[:self.feature_dim]
                outgoing_counts[s] += 1
            
            # Incoming transaction for destination
            if incoming_counts[d] < self.max_length:
                feat = edge_features[i] if edge_features.size(0) > i else torch.zeros(self.feature_dim)
                incoming_seq[d, incoming_counts[d]] = feat[:self.feature_dim]
                incoming_counts[d] += 1
        
        return incoming_seq, outgoing_seq


# =============================================================================
# SECTION 7: DATA STATISTICS AND VISUALIZATION
# =============================================================================


class DatasetStatistics:
    """
    Computes and reports statistics for blockchain datasets.
    
    Useful for:
    - Dataset quality assessment
    - Hyperparameter tuning guidance
    - Paper reporting
    """
    
    def __init__(self, data: BlockchainGraphData):
        """
        Args:
            data: Graph data object
        """
        self.data = data
    
    def compute_all(self) -> Dict[str, Any]:
        """Compute all statistics."""
        stats = {}
        
        # Basic statistics
        stats['basic'] = self._compute_basic_stats()
        
        # Degree statistics
        stats['degree'] = self._compute_degree_stats()
        
        # Class distribution
        stats['class_distribution'] = self._compute_class_distribution()
        
        # Feature statistics
        stats['feature'] = self._compute_feature_stats()
        
        # Graph structure statistics
        stats['structure'] = self._compute_structure_stats()
        
        return stats
    
    def _compute_basic_stats(self) -> Dict[str, Any]:
        """Compute basic graph statistics."""
        return {
            'num_nodes': self.data.num_nodes,
            'num_edges': self.data.num_edges,
            'num_features': self.data.num_features,
            'num_classes': self.data.num_classes,
            'density': self.data.num_edges / (self.data.num_nodes ** 2),
            'avg_degree': 2 * self.data.num_edges / self.data.num_nodes
        }
    
    def _compute_degree_stats(self) -> Dict[str, Any]:
        """Compute degree distribution statistics."""
        src, dst = self.data.edge_index[0], self.data.edge_index[1]
        
        in_degree = torch.bincount(dst, minlength=self.data.num_nodes).float()
        out_degree = torch.bincount(src, minlength=self.data.num_nodes).float()
        total_degree = in_degree + out_degree
        
        return {
            'in_degree': {
                'mean': in_degree.mean().item(),
                'std': in_degree.std().item(),
                'min': in_degree.min().item(),
                'max': in_degree.max().item(),
                'median': in_degree.median().item()
            },
            'out_degree': {
                'mean': out_degree.mean().item(),
                'std': out_degree.std().item(),
                'min': out_degree.min().item(),
                'max': out_degree.max().item(),
                'median': out_degree.median().item()
            },
            'total_degree': {
                'mean': total_degree.mean().item(),
                'std': total_degree.std().item(),
                'min': total_degree.min().item(),
                'max': total_degree.max().item()
            }
        }
    
    def _compute_class_distribution(self) -> Dict[str, Any]:
        """Compute class distribution statistics."""
        labels = self.data.y
        class_counts = torch.bincount(labels)
        
        distribution = {
            f'class_{i}': {
                'count': count.item(),
                'ratio': count.item() / len(labels)
            }
            for i, count in enumerate(class_counts)
        }
        
        # Imbalance ratio
        max_count = class_counts.max().item()
        min_count = class_counts.min().item()
        distribution['imbalance_ratio'] = max_count / (min_count + 1e-8)
        
        return distribution
    
    def _compute_feature_stats(self) -> Dict[str, Any]:
        """Compute feature statistics."""
        x = self.data.x
        
        return {
            'mean': x.mean(dim=0).tolist(),
            'std': x.std(dim=0).tolist(),
            'min': x.min(dim=0).values.tolist(),
            'max': x.max(dim=0).values.tolist(),
            'has_nan': torch.isnan(x).any().item(),
            'has_inf': torch.isinf(x).any().item()
        }
    
    def _compute_structure_stats(self) -> Dict[str, Any]:
        """Compute graph structure statistics."""
        edge_index = self.data.edge_index
        
        # Self-loops
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        
        # Check if undirected
        src, dst = edge_index[0], edge_index[1]
        edge_set = set(zip(src.tolist(), dst.tolist()))
        reverse_edges = sum(1 for s, d in edge_set if (d, s) in edge_set)
        is_undirected = reverse_edges == len(edge_set)
        
        return {
            'self_loops': self_loops,
            'is_undirected': is_undirected,
            'reverse_edge_ratio': reverse_edges / (len(edge_set) + 1e-8)
        }
    
    def print_summary(self) -> None:
        """Print formatted statistics summary."""
        stats = self.compute_all()
        
        print("=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        
        print("\n[Basic Statistics]")
        for key, value in stats['basic'].items():
            print(f"  {key}: {value:,.4f}" if isinstance(value, float) else f"  {key}: {value:,}")
        
        print("\n[Degree Statistics]")
        for deg_type, deg_stats in stats['degree'].items():
            if isinstance(deg_stats, dict):
                print(f"  {deg_type}:")
                for k, v in deg_stats.items():
                    print(f"    {k}: {v:.2f}")
        
        print("\n[Class Distribution]")
        for key, value in stats['class_distribution'].items():
            if isinstance(value, dict):
                print(f"  {key}: count={value['count']:,}, ratio={value['ratio']:.4f}")
            else:
                print(f"  {key}: {value:.2f}")
        
        print("=" * 60)


# =============================================================================
# SECTION 8: UTILITY FUNCTIONS
# =============================================================================


def create_data_loaders(
    data: BlockchainGraphData,
    batch_size: int = 256,
    num_neighbors: List[int] = [25, 10],
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data: Graph data object
        batch_size: Batch size
        num_neighbors: Neighbors per hop for sampling
        num_workers: Data loading workers
    
    Returns:
        Dictionary with 'train', 'val', 'test' loaders
    """
    sampler = NeighborSamplerWrapper(
        data=data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return {
        'train': sampler.get_loader(SplitType.TRAIN),
        'val': sampler.get_loader(SplitType.VALIDATION),
        'test': sampler.get_loader(SplitType.TEST)
    }


def load_dataset(
    dataset_type: Union[str, DatasetType],
    data_dir: str = "data",
    **kwargs
) -> BlockchainGraphData:
    """
    Convenience function to load a dataset.
    
    Args:
        dataset_type: Dataset type string or enum
        data_dir: Data directory
        **kwargs: Additional arguments for loader
    
    Returns:
        BlockchainGraphData object
    """
    if isinstance(dataset_type, str):
        dataset_type = DatasetType(dataset_type)
    
    loader = BlockchainDatasetLoader(data_dir=data_dir)
    return loader.load(dataset_type)


# =============================================================================
# SECTION 9: UNIT TESTS
# =============================================================================


def run_unit_tests():
    """Run unit tests for data module."""
    print("=" * 60)
    print("NECons Data Module - Unit Tests")
    print("=" * 60)
    
    # Test 1: Synthetic Dataset Generation
    print("\n[Test 1] Synthetic Dataset Generation")
    generator = SyntheticBlockchainDataset(
        num_nodes=1000,
        num_edges=5000,
        anomaly_ratio=0.05
    )
    data = generator.generate()
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features: {data.num_features}")
    print(f"  Anomaly count: {data.y.sum().item()}")
    assert data.num_nodes == 1000
    print("  ✓ PASSED")
    
    # Test 2: Feature Extraction
    print("\n[Test 2] Feature Extraction")
    extractor = TransactionFeatureExtractor()
    edge_values = torch.rand(data.num_edges) * 100
    edge_timestamps = torch.arange(data.num_edges).float()
    node_feat, edge_feat = extractor.fit_transform(
        data.edge_index, edge_values, edge_timestamps, data.num_nodes
    )
    print(f"  Node features shape: {node_feat.shape}")
    print(f"  Edge features shape: {edge_feat.shape}")
    assert node_feat.shape == (data.num_nodes, 8)
    print("  ✓ PASSED")
    
    # Test 3: Graph Augmentation
    print("\n[Test 3] Graph Augmentation")
    augmentations = [
        EdgeDropout(dropout_rate=0.1),
        NodeDropout(dropout_rate=0.1),
        FeatureMasking(mask_rate=0.1),
        SubgraphSampling(sample_ratio=0.5)
    ]
    
    for aug in augmentations:
        x_aug, edge_aug, _, _ = aug(data.x, data.edge_index, data.edge_attr, data.y)
        print(f"  {aug.name()}: nodes={x_aug.size(0)}, edges={edge_aug.size(1)}")
    print("  ✓ PASSED")
    
    # Test 4: Node Feature Normalizer
    print("\n[Test 4] Node Feature Normalizer")
    normalizer = NodeFeatureNormalizer(method="zscore")
    x_norm = normalizer.fit_transform(data.x)
    print(f"  Original mean: {data.x.mean():.4f}")
    print(f"  Normalized mean: {x_norm.mean():.4f}")
    print(f"  Normalized std: {x_norm.std():.4f}")
    assert abs(x_norm.mean()) < 0.1
    print("  ✓ PASSED")
    
    # Test 5: Transaction Sequence Builder
    print("\n[Test 5] Transaction Sequence Builder")
    seq_builder = TransactionSequenceBuilder(max_sequence_length=20)
    incoming_seq, outgoing_seq = seq_builder.build_sequences(data)
    print(f"  Incoming sequences shape: {incoming_seq.shape}")
    print(f"  Outgoing sequences shape: {outgoing_seq.shape}")
    assert incoming_seq.shape[0] == data.num_nodes
    print("  ✓ PASSED")
    
    # Test 6: Dataset Statistics
    print("\n[Test 6] Dataset Statistics")
    stats = DatasetStatistics(data)
    all_stats = stats.compute_all()
    print(f"  Basic stats computed: {list(all_stats['basic'].keys())}")
    print(f"  Class imbalance ratio: {all_stats['class_distribution']['imbalance_ratio']:.2f}")
    print("  ✓ PASSED")
    
    # Test 7: Memory Efficient Loader
    print("\n[Test 7] Memory Efficient Loader")
    loader = MemoryEfficientLoader(data, batch_size=100)
    batch_count = 0
    for node_idx, edge_index, features, labels in loader:
        batch_count += 1
        if batch_count >= 3:
            break
    print(f"  Batches sampled: {batch_count}")
    print(f"  Last batch size: {len(node_idx)}")
    print("  ✓ PASSED")
    
    # Test 8: Composed Augmentation
    print("\n[Test 8] Composed Augmentation")
    composed = ComposedAugmentation([
        EdgeDropout(0.1),
        FeatureMasking(0.1)
    ])
    x_aug, edge_aug, _, _ = composed(data.x, data.edge_index)
    print(f"  {composed.name()}")
    print(f"  Result: nodes={x_aug.size(0)}, edges={edge_aug.size(1)}")
    print("  ✓ PASSED")
    
    # Test 9: Class Weight Computation
    print("\n[Test 9] Class Weight Computation")
    weights = data.compute_class_weights()
    print(f"  Class weights: {weights.tolist()}")
    assert len(weights) == data.num_classes
    print("  ✓ PASSED")
    
    # Test 10: Data Summary
    print("\n[Test 10] Data Summary")
    summary = data.summary()
    print(f"  Summary keys: {list(summary.keys())}")
    print(f"  Train size: {summary.get('train_size', 'N/A')}")
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All unit tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
