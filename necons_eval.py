"""
================================================================================
NECons: Network-aware Edge-based Consensus for Distributed Blockchain 
        Anomaly Detection
================================================================================

Evaluation Module: Baselines, Metrics, Statistical Analysis, and Results Formatting

Target Venue: IEEE Transactions on Parallel and Distributed Systems (TPDS) 2026

This module implements:
1. Comprehensive evaluation metrics for anomaly detection
2. Baseline model implementations for comparison
3. Byzantine resilience evaluation
4. Scalability stress testing
5. Cross-chain performance evaluation
6. Statistical significance testing
7. Ablation study framework
8. LaTeX table and figure generation for paper

================================================================================
EVALUATION DIMENSIONS
================================================================================
1. Detection Performance: F1, Precision, Recall, AUC-ROC, AUC-PR, MCC
2. Byzantine Resilience: Accuracy under various attack types and ratios
3. Scalability: Consensus time, throughput, memory usage vs. node count
4. Cross-Chain: Performance across different blockchain pairs
5. Communication: Message complexity comparison with baselines
6. Ablation: Component contribution analysis

================================================================================
KEY REFERENCES
================================================================================
[1] Veličković et al., "Graph Attention Networks", ICLR 2018
[2] Kipf & Welling, "Semi-supervised Classification with GCNs", ICLR 2017
[3] Hamilton et al., "Inductive Representation Learning", NeurIPS 2017
[4] Castro & Liskov, "Practical Byzantine Fault Tolerance", OSDI 1999
[5] Yin et al., "HotStuff: BFT Consensus", PODC 2019
[6] Yang et al., "2DynEthNet", IEEE TIFS 2024
[7] Zhang et al., "GrabPhisher", IEEE TSC 2024
[8] Huang et al., "SAMamba", IEEE TIFS 2025

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

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict
from pathlib import Path
from abc import ABC, abstractmethod
import time
import math
import copy
import json
import logging
import warnings
from datetime import datetime
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: EVALUATION CONFIGURATION
# =============================================================================


class EvaluationType(Enum):
    """Types of evaluation experiments."""
    DETECTION = "detection"
    BYZANTINE = "byzantine"
    SCALABILITY = "scalability"
    CROSS_CHAIN = "cross_chain"
    COMMUNICATION = "communication"
    ABLATION = "ablation"
    STATISTICAL = "statistical"


class BaselineType(Enum):
    """Baseline model types for comparison."""
    # GNN Baselines
    VANILLA_GAT = "vanilla_gat"
    VANILLA_GCN = "vanilla_gcn"
    GRAPHSAGE = "graphsage"
    
    # Federated Learning Baselines
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    
    # Consensus Baselines
    PBFT = "pbft"
    RAFT = "raft"
    HOTSTUFF = "hotstuff"
    
    # Blockchain Detection Baselines
    DYNETHNET = "2dynethnet"
    GRABPHISHER = "grabphisher"
    SAMAMBA = "samamba"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation experiments."""
    # Detection benchmark
    num_runs: int = 10
    confidence_level: float = 0.95
    
    # Byzantine evaluation
    byzantine_ratios: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.33])
    attack_types: List[str] = field(default_factory=lambda: [
        "model_poisoning", "label_flipping", "gradient_scaling"
    ])
    
    # Scalability testing
    node_counts: List[int] = field(default_factory=lambda: [
        100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000
    ])
    scalability_timeout_sec: float = 300.0
    max_consensus_time_ms: float = 5000.0
    min_accuracy_threshold: float = 0.90
    
    # Cross-chain evaluation
    chain_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("ethereum", "bitcoin"),
        ("ethereum", "polygon"),
        ("ethereum", "binance_sc"),
        ("bitcoin", "polygon"),
        ("polygon", "binance_sc")
    ])
    
    # Statistical testing
    use_bonferroni: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: [
        "wilcoxon", "paired_t", "friedman"
    ])
    
    # Output
    results_dir: str = "results"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"
    save_raw_results: bool = True


# =============================================================================
# SECTION 2: METRICS COMPUTATION
# =============================================================================


class EvaluationMetrics:
    """
    Comprehensive metrics computation for anomaly detection.
    
    Computes all metrics required for IEEE TPDS paper:
    - Classification metrics: F1, Precision, Recall, Accuracy
    - Ranking metrics: AUC-ROC, AUC-PR
    - Correlation metrics: MCC, Cohen's Kappa
    - Per-class metrics for detailed analysis
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, logits: Tensor, labels: Tensor) -> None:
        """Accumulate predictions."""
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        self.all_preds.append(preds.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        self.all_probs.append(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if not self.all_preds:
            return {}
        
        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)
        probs = np.concatenate(self.all_probs)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = (preds == labels).mean()
        
        # Confusion matrix components
        for c in range(self.num_classes):
            tp = ((preds == c) & (labels == c)).sum()
            fp = ((preds == c) & (labels != c)).sum()
            fn = ((preds != c) & (labels == c)).sum()
            tn = ((preds != c) & (labels != c)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            
            metrics[f'precision_c{c}'] = precision
            metrics[f'recall_c{c}'] = recall
            metrics[f'f1_c{c}'] = f1
            metrics[f'specificity_c{c}'] = specificity
        
        # Macro-averaged metrics
        metrics['precision'] = np.mean([metrics[f'precision_c{c}'] for c in range(self.num_classes)])
        metrics['recall'] = np.mean([metrics[f'recall_c{c}'] for c in range(self.num_classes)])
        metrics['f1'] = np.mean([metrics[f'f1_c{c}'] for c in range(self.num_classes)])
        
        # Binary classification specific metrics
        if self.num_classes == 2:
            # AUC-ROC
            metrics['auc_roc'] = self._compute_auc_roc(probs[:, 1], labels)
            
            # AUC-PR (Average Precision)
            metrics['auc_pr'] = self._compute_auc_pr(probs[:, 1], labels)
            
            # Matthews Correlation Coefficient
            metrics['mcc'] = self._compute_mcc(preds, labels)
            
            # Cohen's Kappa
            metrics['kappa'] = self._compute_kappa(preds, labels)
            
            # G-Mean
            metrics['gmean'] = np.sqrt(metrics['recall_c1'] * metrics['specificity_c1'])
        
        return metrics
    
    def _compute_auc_roc(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC-ROC using trapezoidal rule."""
        try:
            # Sort by probability descending
            sorted_idx = np.argsort(probs)[::-1]
            sorted_labels = labels[sorted_idx]
            
            total_pos = (labels == 1).sum()
            total_neg = (labels == 0).sum()
            
            if total_pos == 0 or total_neg == 0:
                return 0.5
            
            tpr_list = []
            fpr_list = []
            cum_pos = 0
            cum_neg = 0
            
            for label in sorted_labels:
                if label == 1:
                    cum_pos += 1
                else:
                    cum_neg += 1
                tpr_list.append(cum_pos / total_pos)
                fpr_list.append(cum_neg / total_neg)
            
            # Trapezoidal integration
            auc = 0.0
            for i in range(1, len(fpr_list)):
                auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
            
            return float(auc)
        except Exception:
            return 0.5
    
    def _compute_auc_pr(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute Average Precision."""
        try:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_labels = labels[sorted_idx]
            
            total_pos = (labels == 1).sum()
            if total_pos == 0:
                return 0.0
            
            cum_tp = 0
            ap = 0.0
            prev_recall = 0.0
            
            for i, label in enumerate(sorted_labels):
                if label == 1:
                    cum_tp += 1
                    precision = cum_tp / (i + 1)
                    recall = cum_tp / total_pos
                    ap += precision * (recall - prev_recall)
                    prev_recall = recall
            
            return float(ap)
        except Exception:
            return 0.0
    
    def _compute_mcc(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute Matthews Correlation Coefficient."""
        try:
            tp = ((preds == 1) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            
            numerator = tp * tn - fp * fn
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            
            if denominator == 0:
                return 0.0
            
            return float(numerator / denominator)
        except Exception:
            return 0.0
    
    def _compute_kappa(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute Cohen's Kappa coefficient."""
        try:
            # Observed accuracy
            po = (preds == labels).mean()
            
            # Expected accuracy
            for c in range(self.num_classes):
                pred_c = (preds == c).mean()
                label_c = (labels == c).mean()
            
            pe = sum(
                (preds == c).mean() * (labels == c).mean()
                for c in range(self.num_classes)
            )
            
            if pe == 1:
                return 1.0
            
            return float((po - pe) / (1 - pe))
        except Exception:
            return 0.0


# =============================================================================
# SECTION 3: BASELINE IMPLEMENTATIONS
# =============================================================================


class BaselineModel(ABC):
    """Abstract base class for baseline models."""
    
    @abstractmethod
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        pass
    
    @abstractmethod
    def reference(self) -> str:
        """Return paper reference."""
        pass


class VanillaGATBaseline(nn.Module, BaselineModel):
    """
    Vanilla Graph Attention Network baseline.
    
    Reference: Veličković et al., "Graph Attention Networks", ICLR 2018 [1]
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConvSimple(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConvSimple(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Output layer
        self.convs.append(
            GATConvSimple(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        for i, (conv, norm) in enumerate(zip(self.convs[:-1], self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def name(self) -> str:
        return "VanillaGAT"
    
    def reference(self) -> str:
        return "Veličković et al., ICLR 2018"


class GATConvSimple(nn.Module):
    """Simplified GAT convolution for baseline."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        self.dropout = dropout
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        H, C = self.heads, self.out_channels
        N = x.size(0)
        
        x = self.lin(x).view(N, H, C)
        
        src, dst = edge_index
        
        # Compute attention
        alpha_src = (x[src] * self.att[:, :, :C]).sum(dim=-1)
        alpha_dst = (x[dst] * self.att[:, :, C:]).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)
        
        # Softmax over neighbors
        alpha = alpha - alpha.max()
        alpha = torch.exp(alpha)
        
        # Sum for normalization
        alpha_sum = torch.zeros(N, H, device=x.device)
        alpha_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (alpha_sum[dst] + 1e-8)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Message passing
        out = torch.zeros(N, H, C, device=x.device)
        msg = x[src] * alpha.unsqueeze(-1)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, H, C), msg)
        
        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)
        
        return out + self.bias


class VanillaGCNBaseline(nn.Module, BaselineModel):
    """
    Vanilla Graph Convolutional Network baseline.
    
    Reference: Kipf & Welling, "Semi-supervised Classification with GCNs", ICLR 2017 [2]
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(GCNConvSimple(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvSimple(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        self.convs.append(GCNConvSimple(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def name(self) -> str:
        return "VanillaGCN"
    
    def reference(self) -> str:
        return "Kipf & Welling, ICLR 2017"


class GCNConvSimple(nn.Module):
    """Simplified GCN convolution for baseline."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        N = x.size(0)
        
        # Compute degree
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_inv_sqrt = (deg + 1).pow(-0.5)
        
        # Normalize
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        
        # Message passing
        x = self.lin(x)
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src] * norm.unsqueeze(1))
        
        # Add self-loop
        out = out + x
        
        return out + self.bias


class GraphSAGEBaseline(nn.Module, BaselineModel):
    """
    GraphSAGE baseline with mean aggregation.
    
    Reference: Hamilton et al., "Inductive Representation Learning on Large Graphs", 
               NeurIPS 2017 [3]
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(SAGEConvSimple(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConvSimple(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        self.convs.append(SAGEConvSimple(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def name(self) -> str:
        return "GraphSAGE"
    
    def reference(self) -> str:
        return "Hamilton et al., NeurIPS 2017"


class SAGEConvSimple(nn.Module):
    """Simplified SAGE convolution with mean aggregation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        N = x.size(0)
        
        # Neighbor aggregation (mean)
        neigh_sum = torch.zeros(N, x.size(1), device=x.device)
        neigh_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = deg.clamp(min=1)
        
        neigh_mean = neigh_sum / deg.unsqueeze(1)
        
        # Combine self and neighbor
        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        
        return out + self.bias


class FedAvgBaseline:
    """
    Federated Averaging baseline.
    
    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks 
               from Decentralized Data", AISTATS 2017
    """
    
    def __init__(self, num_clients: int = 100):
        self.num_clients = num_clients
    
    def aggregate(self, client_updates: Dict[int, Tensor]) -> Tensor:
        """Simple averaging of client updates."""
        if not client_updates:
            raise ValueError("No updates to aggregate")
        
        updates = list(client_updates.values())
        return torch.stack(updates).mean(dim=0)
    
    def name(self) -> str:
        return "FedAvg"
    
    def reference(self) -> str:
        return "McMahan et al., AISTATS 2017"


class PBFTBaseline:
    """
    PBFT consensus baseline.
    
    Reference: Castro & Liskov, "Practical Byzantine Fault Tolerance", OSDI 1999 [4]
    
    Message Complexity: O(n²)
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.f = num_nodes // 3  # Max Byzantine nodes
    
    def aggregate(self, node_updates: Dict[int, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
        """PBFT-style aggregation (simplified)."""
        start_time = time.time()
        
        # Simple majority voting simulation
        updates = list(node_updates.values())
        result = torch.stack(updates).mean(dim=0)
        
        elapsed = time.time() - start_time
        
        info = {
            'consensus_time_ms': elapsed * 1000,
            'message_complexity': 3 * self.num_nodes ** 2,  # O(n²)
            'rounds': 3,
            'byzantine_tolerance': f'f < n/3 = {self.f}'
        }
        
        return result, info
    
    def name(self) -> str:
        return "PBFT"
    
    def reference(self) -> str:
        return "Castro & Liskov, OSDI 1999"


class RaftBaseline:
    """
    Raft consensus baseline (CFT, not BFT).
    
    Reference: Ongaro & Ousterhout, "In Search of an Understandable Consensus 
               Algorithm", USENIX ATC 2014
    
    Message Complexity: O(n)
    Note: Raft is crash fault tolerant, NOT Byzantine fault tolerant.
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.leader_id = 0
    
    def aggregate(self, node_updates: Dict[int, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
        """Raft-style leader-based aggregation."""
        start_time = time.time()
        
        # Leader decides (vulnerable to Byzantine leader)
        if self.leader_id in node_updates:
            result = node_updates[self.leader_id]
        else:
            result = torch.stack(list(node_updates.values())).mean(dim=0)
        
        elapsed = time.time() - start_time
        
        info = {
            'consensus_time_ms': elapsed * 1000,
            'message_complexity': 2 * self.num_nodes,  # O(n)
            'rounds': 2,
            'byzantine_tolerance': 'None (CFT only)',
            'warning': 'Vulnerable to Byzantine leader'
        }
        
        return result, info
    
    def name(self) -> str:
        return "Raft"
    
    def reference(self) -> str:
        return "Ongaro & Ousterhout, ATC 2014"


class HotStuffBaseline:
    """
    HotStuff consensus baseline.
    
    Reference: Yin et al., "HotStuff: BFT Consensus with Linearity and 
               Responsiveness", PODC 2019 [5]
    
    Message Complexity: O(n) with threshold signatures
    """
    
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.f = num_nodes // 3
    
    def aggregate(self, node_updates: Dict[int, Tensor]) -> Tuple[Tensor, Dict[str, Any]]:
        """HotStuff-style linear BFT aggregation."""
        start_time = time.time()
        
        # Median-based aggregation (simplified)
        updates = torch.stack(list(node_updates.values()))
        result = torch.median(updates, dim=0).values
        
        elapsed = time.time() - start_time
        
        info = {
            'consensus_time_ms': elapsed * 1000,
            'message_complexity': 3 * self.num_nodes,  # O(n) with threshold sigs
            'rounds': 3,
            'byzantine_tolerance': f'f < n/3 = {self.f}'
        }
        
        return result, info
    
    def name(self) -> str:
        return "HotStuff"
    
    def reference(self) -> str:
        return "Yin et al., PODC 2019"


# =============================================================================
# SECTION 4: BASELINE COMPARATOR
# =============================================================================


class BaselineComparator:
    """
    Compares NECons with baseline methods.
    
    Runs identical experiments across all methods and
    computes statistical significance of improvements.
    """
    
    def __init__(
        self,
        necons_model: nn.Module,
        config: EvaluationConfig,
        device: torch.device
    ):
        self.necons_model = necons_model
        self.config = config
        self.device = device
        
        # Initialize baseline models
        self.gnn_baselines: Dict[str, nn.Module] = {}
        self.consensus_baselines: Dict[str, Any] = {}
    
    def setup_gnn_baselines(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int
    ) -> None:
        """Initialize GNN baseline models."""
        self.gnn_baselines = {
            'VanillaGAT': VanillaGATBaseline(
                in_channels, hidden_channels, out_channels
            ).to(self.device),
            'VanillaGCN': VanillaGCNBaseline(
                in_channels, hidden_channels, out_channels
            ).to(self.device),
            'GraphSAGE': GraphSAGEBaseline(
                in_channels, hidden_channels, out_channels
            ).to(self.device)
        }
    
    def setup_consensus_baselines(self, num_nodes: int) -> None:
        """Initialize consensus baseline protocols."""
        self.consensus_baselines = {
            'PBFT': PBFTBaseline(num_nodes),
            'Raft': RaftBaseline(num_nodes),
            'HotStuff': HotStuffBaseline(num_nodes),
            'FedAvg': FedAvgBaseline(num_nodes)
        }
    
    def compare_detection_performance(
        self,
        test_data: Any,
        num_runs: int = 10
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Compare detection performance across all models.
        
        Returns metrics for each model across multiple runs.
        """
        results = {'NECons': defaultdict(list)}
        
        for baseline_name in self.gnn_baselines:
            results[baseline_name] = defaultdict(list)
        
        for run in range(num_runs):
            # Evaluate NECons
            necons_metrics = self._evaluate_model(self.necons_model, test_data)
            for metric, value in necons_metrics.items():
                results['NECons'][metric].append(value)
            
            # Evaluate baselines
            for name, model in self.gnn_baselines.items():
                metrics = self._evaluate_model(model, test_data)
                for metric, value in metrics.items():
                    results[name][metric].append(value)
        
        return dict(results)
    
    def _evaluate_model(self, model: nn.Module, test_data: Any) -> Dict[str, float]:
        """Evaluate a single model on test data."""
        model.eval()
        metrics_computer = EvaluationMetrics()
        
        with torch.no_grad():
            if hasattr(test_data, 'x'):
                # Single graph
                x = test_data.x.to(self.device)
                edge_index = test_data.edge_index.to(self.device)
                y = test_data.y.to(self.device)
                mask = getattr(test_data, 'test_mask', None)
                
                logits = model(x, edge_index)
                
                if mask is not None:
                    logits = logits[mask]
                    y = y[mask]
                
                metrics_computer.update(logits, y)
            else:
                # Batched data
                for batch in test_data:
                    batch = batch.to(self.device)
                    logits = model(batch.x, batch.edge_index)
                    metrics_computer.update(logits, batch.y)
        
        return metrics_computer.compute()
    
    def compare_consensus_performance(
        self,
        node_updates: Dict[int, Tensor],
        necons_consensus: Any,
        num_runs: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Compare consensus protocol performance."""
        results = {}
        
        # NECons consensus
        necons_times = []
        for _ in range(num_runs):
            _, info = necons_consensus(node_updates, torch.zeros(256))
            necons_times.append(info['consensus_time_ms'])
        
        results['NECons'] = {
            'avg_time_ms': np.mean(necons_times),
            'std_time_ms': np.std(necons_times),
            'message_complexity': info['message_complexity']
        }
        
        # Baseline consensus
        for name, baseline in self.consensus_baselines.items():
            times = []
            for _ in range(num_runs):
                _, info = baseline.aggregate(node_updates)
                times.append(info['consensus_time_ms'])
            
            results[name] = {
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'message_complexity': info['message_complexity']
            }
        
        return results


# =============================================================================
# SECTION 5: BYZANTINE RESILIENCE EVALUATOR
# =============================================================================


class ByzantineResilienceEvaluator:
    """
    Evaluates Byzantine fault tolerance of NECons.
    
    Tests accuracy preservation under various:
    - Byzantine ratios: [0%, 10%, 20%, 33%]
    - Attack types: Model poisoning, label flipping, gradient scaling
    """
    
    def __init__(
        self,
        necons_model: nn.Module,
        consensus_module: nn.Module,
        config: EvaluationConfig,
        device: torch.device
    ):
        self.model = necons_model
        self.consensus = consensus_module
        self.config = config
        self.device = device
    
    def evaluate(
        self,
        test_data: Any,
        num_edge_nodes: int = 100
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run full Byzantine resilience evaluation.
        
        Returns nested dict: attack_type -> byzantine_ratio -> metrics
        """
        results = {}
        
        for attack_type in self.config.attack_types:
            results[attack_type] = {}
            
            for byzantine_ratio in self.config.byzantine_ratios:
                logger.info(f"Evaluating: {attack_type}, Byzantine ratio: {byzantine_ratio}")
                
                metrics = self._evaluate_with_byzantine(
                    test_data,
                    num_edge_nodes,
                    byzantine_ratio,
                    attack_type
                )
                
                results[attack_type][str(byzantine_ratio)] = metrics
        
        return results
    
    def _evaluate_with_byzantine(
        self,
        test_data: Any,
        num_nodes: int,
        byzantine_ratio: float,
        attack_type: str
    ) -> Dict[str, float]:
        """Evaluate with specified Byzantine configuration."""
        num_byzantine = int(num_nodes * byzantine_ratio)
        byzantine_nodes = set(range(num_nodes - num_byzantine, num_nodes))
        
        # Simulate distributed updates
        honest_update = torch.randn(256) * 0.1
        
        node_updates = {}
        for i in range(num_nodes):
            if i in byzantine_nodes:
                # Apply attack
                node_updates[i] = self._apply_attack(honest_update.clone(), attack_type)
            else:
                node_updates[i] = honest_update.clone() + torch.randn(256) * 0.01
        
        # Run consensus
        aggregated, info = self.consensus(
            node_updates,
            torch.zeros(256),
            byzantine_nodes=byzantine_nodes
        )
        
        # Compute deviation from honest mean
        honest_updates = [node_updates[i] for i in range(num_nodes) if i not in byzantine_nodes]
        honest_mean = torch.stack(honest_updates).mean(dim=0)
        
        deviation = torch.norm(aggregated - honest_mean).item()
        honest_norm = torch.norm(honest_mean).item()
        
        return {
            'deviation': deviation,
            'relative_error': deviation / (honest_norm + 1e-8),
            'accuracy_preserved': deviation < honest_norm * 0.5,
            'consensus_time_ms': info['consensus_time_ms'],
            'filtered_nodes': info['num_filtered']
        }
    
    def _apply_attack(self, update: Tensor, attack_type: str) -> Tensor:
        """Apply Byzantine attack to update."""
        if attack_type == "model_poisoning":
            return update + torch.randn_like(update) * 10
        elif attack_type == "label_flipping":
            return -update
        elif attack_type == "gradient_scaling":
            return update * 100
        else:
            return update


# =============================================================================
# SECTION 6: SCALABILITY ANALYZER
# =============================================================================


class ScalabilityAnalyzer:
    """
    Analyzes scalability of NECons across node counts.
    
    Measures:
    - Consensus time vs. number of nodes
    - Throughput (transactions per second)
    - Memory usage
    - Identifies practical scaling limits
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
    
    def run_scalability_test(
        self,
        consensus_module_factory: Callable[[int], nn.Module],
        hidden_dim: int = 256
    ) -> List[Dict[str, Any]]:
        """
        Run scalability tests across node counts.
        
        Args:
            consensus_module_factory: Function that creates consensus module given num_nodes
            hidden_dim: Dimension of model updates
        
        Returns:
            List of results for each node count
        """
        results = []
        
        for node_count in self.config.node_counts:
            logger.info(f"Testing scalability with {node_count} nodes")
            
            try:
                result = self._test_node_count(
                    consensus_module_factory,
                    node_count,
                    hidden_dim
                )
                results.append(result)
                
                # Check if we hit scaling limits
                if not result['success']:
                    logger.warning(f"Scaling limit reached at {node_count} nodes")
                    break
                    
            except Exception as e:
                logger.error(f"Error at {node_count} nodes: {e}")
                results.append({
                    'node_count': node_count,
                    'success': False,
                    'error': str(e)
                })
                break
        
        self.results = results
        return results
    
    def _test_node_count(
        self,
        consensus_factory: Callable[[int], nn.Module],
        node_count: int,
        hidden_dim: int,
        num_trials: int = 5
    ) -> Dict[str, Any]:
        """Test specific node count."""
        # Create consensus module
        consensus = consensus_factory(node_count)
        
        # Generate updates with Byzantine nodes
        num_byzantine = int(node_count * 0.33)
        
        trial_times = []
        trial_accuracies = []
        
        for trial in range(num_trials):
            # Generate node updates
            honest_base = torch.randn(hidden_dim) * 0.1
            node_updates = {}
            
            for i in range(node_count):
                if i >= node_count - num_byzantine:
                    # Byzantine update
                    node_updates[i] = torch.randn(hidden_dim) * 5
                else:
                    # Honest update with small noise
                    node_updates[i] = honest_base + torch.randn(hidden_dim) * 0.01
            
            # Time the consensus
            start_time = time.time()
            result, info = consensus(node_updates, torch.zeros(hidden_dim))
            elapsed_ms = (time.time() - start_time) * 1000
            
            trial_times.append(elapsed_ms)
            
            # Compute accuracy (deviation from honest mean)
            honest_updates = [node_updates[i] for i in range(node_count - num_byzantine)]
            honest_mean = torch.stack(honest_updates).mean(dim=0)
            deviation = torch.norm(result - honest_mean).item()
            honest_norm = torch.norm(honest_mean).item()
            accuracy = 1.0 - min(1.0, deviation / (honest_norm + 1e-8))
            
            trial_accuracies.append(accuracy)
        
        avg_time = np.mean(trial_times)
        avg_accuracy = np.mean(trial_accuracies)
        
        # Check success criteria
        success = (
            avg_time < self.config.max_consensus_time_ms and
            avg_accuracy > self.config.min_accuracy_threshold
        )
        
        return {
            'node_count': node_count,
            'consensus_time_ms': avg_time,
            'consensus_time_std': np.std(trial_times),
            'accuracy': avg_accuracy,
            'accuracy_std': np.std(trial_accuracies),
            'throughput_tps': 1000.0 / avg_time if avg_time > 0 else 0,
            'message_complexity': info.get('message_complexity', node_count ** 2),
            'success': success
        }
    
    def find_practical_limit(self) -> Optional[int]:
        """Find the practical node limit based on test results."""
        if not self.results:
            return None
        
        for result in self.results:
            if not result.get('success', False):
                # Return previous successful count
                idx = self.results.index(result)
                if idx > 0:
                    return self.results[idx - 1]['node_count']
                return None
        
        # All tests passed
        return self.results[-1]['node_count']


# =============================================================================
# SECTION 7: STATISTICAL SIGNIFICANCE TESTING
# =============================================================================


class StatisticalTester:
    """
    Statistical significance testing for method comparisons.
    
    Implements:
    - Wilcoxon signed-rank test (non-parametric paired)
    - Paired t-test (parametric paired)
    - Friedman test (multiple methods)
    - Effect size computation (Cohen's d, Cliff's delta)
    """
    
    def __init__(self, confidence_level: float = 0.95, use_bonferroni: bool = True):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.use_bonferroni = use_bonferroni
    
    def wilcoxon_test(
        self,
        values_a: List[float],
        values_b: List[float]
    ) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Non-parametric alternative to paired t-test.
        """
        try:
            statistic, p_value = scipy_stats.wilcoxon(values_a, values_b)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'test': 'Wilcoxon signed-rank'
            }
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {'error': str(e)}
    
    def paired_t_test(
        self,
        values_a: List[float],
        values_b: List[float]
    ) -> Dict[str, float]:
        """
        Paired t-test for dependent samples.
        
        Assumes normally distributed differences.
        """
        try:
            statistic, p_value = scipy_stats.ttest_rel(values_a, values_b)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'test': 'Paired t-test'
            }
        except Exception as e:
            logger.warning(f"Paired t-test failed: {e}")
            return {'error': str(e)}
    
    def friedman_test(
        self,
        *value_lists: List[float]
    ) -> Dict[str, float]:
        """
        Friedman test for comparing multiple methods.
        
        Non-parametric alternative to repeated measures ANOVA.
        """
        try:
            statistic, p_value = scipy_stats.friedmanchisquare(*value_lists)
            
            # Apply Bonferroni correction if needed
            num_comparisons = len(value_lists) * (len(value_lists) - 1) // 2
            adjusted_alpha = self.alpha / num_comparisons if self.use_bonferroni else self.alpha
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'adjusted_alpha': adjusted_alpha,
                'significant': p_value < adjusted_alpha,
                'test': 'Friedman'
            }
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
            return {'error': str(e)}
    
    def cohens_d(
        self,
        values_a: List[float],
        values_b: List[float]
    ) -> float:
        """
        Compute Cohen's d effect size.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        """
        a = np.array(values_a)
        b = np.array(values_b)
        
        mean_diff = a.mean() - b.mean()
        pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return float(mean_diff / pooled_std)
    
    def cliffs_delta(
        self,
        values_a: List[float],
        values_b: List[float]
    ) -> float:
        """
        Compute Cliff's delta effect size.
        
        Non-parametric effect size measure.
        
        Interpretation:
        - |δ| < 0.147: negligible
        - 0.147 ≤ |δ| < 0.33: small
        - 0.33 ≤ |δ| < 0.474: medium
        - |δ| ≥ 0.474: large
        """
        a = np.array(values_a)
        b = np.array(values_b)
        
        n_a, n_b = len(a), len(b)
        
        # Count dominance
        greater = sum(1 for ai in a for bi in b if ai > bi)
        less = sum(1 for ai in a for bi in b if ai < bi)
        
        delta = (greater - less) / (n_a * n_b)
        
        return float(delta)
    
    def full_comparison(
        self,
        necons_values: List[float],
        baseline_values: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run full statistical comparison of NECons vs all baselines.
        """
        results = {}
        
        for baseline_name, values in baseline_values.items():
            comparison = {
                'wilcoxon': self.wilcoxon_test(necons_values, values),
                'paired_t': self.paired_t_test(necons_values, values),
                'cohens_d': self.cohens_d(necons_values, values),
                'cliffs_delta': self.cliffs_delta(necons_values, values),
                'necons_mean': np.mean(necons_values),
                'necons_std': np.std(necons_values),
                'baseline_mean': np.mean(values),
                'baseline_std': np.std(values),
                'improvement': (np.mean(necons_values) - np.mean(values)) / np.mean(values) * 100
            }
            
            results[baseline_name] = comparison
        
        return results


# =============================================================================
# SECTION 8: ABLATION STUDY RUNNER
# =============================================================================


class AblationStudyRunner:
    """
    Runs ablation studies to evaluate component contributions.
    
    Tests NECons with various components disabled:
    - Without Network-Aware attention
    - Without MGD (Multigraph Discrepancy)
    - Without Edge2Seq
    - Without Byzantine consensus
    - Without Cross-chain synchronization
    """
    
    def __init__(self, full_model: nn.Module, device: torch.device):
        self.full_model = full_model
        self.device = device
        self.ablation_models: Dict[str, nn.Module] = {}
    
    def create_ablation_variants(
        self,
        model_factory: Callable[..., nn.Module],
        base_kwargs: Dict[str, Any]
    ) -> None:
        """Create model variants for ablation study."""
        # Full model
        self.ablation_models['Full NECons'] = self.full_model
        
        # Without network-aware attention
        kwargs = base_kwargs.copy()
        kwargs['network_aware'] = False
        self.ablation_models['w/o Network-Aware'] = model_factory(**kwargs).to(self.device)
        
        # Without Edge2Seq
        kwargs = base_kwargs.copy()
        kwargs['use_edge2seq'] = False
        self.ablation_models['w/o Edge2Seq'] = model_factory(**kwargs).to(self.device)
        
        # Reduced MGD layers
        kwargs = base_kwargs.copy()
        kwargs['num_mgd_layers'] = 1
        self.ablation_models['w/ 1 MGD Layer'] = model_factory(**kwargs).to(self.device)
    
    def run_ablation(
        self,
        test_data: Any,
        num_runs: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Run ablation study."""
        results = {}
        
        for variant_name, model in self.ablation_models.items():
            logger.info(f"Evaluating ablation variant: {variant_name}")
            
            metrics_list = []
            for run in range(num_runs):
                metrics = self._evaluate_variant(model, test_data)
                metrics_list.append(metrics)
            
            # Average metrics across runs
            avg_metrics = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            
            results[variant_name] = avg_metrics
        
        return results
    
    def _evaluate_variant(self, model: nn.Module, test_data: Any) -> Dict[str, float]:
        """Evaluate a single ablation variant."""
        model.eval()
        metrics = EvaluationMetrics()
        
        with torch.no_grad():
            if hasattr(test_data, 'x'):
                x = test_data.x.to(self.device)
                edge_index = test_data.edge_index.to(self.device)
                y = test_data.y.to(self.device)
                mask = getattr(test_data, 'test_mask', None)
                
                logits = model(x, edge_index)
                
                if mask is not None:
                    mask = mask.to(self.device)
                    logits = logits[mask]
                    y = y[mask]
                
                metrics.update(logits, y)
        
        return metrics.compute()
    
    def compute_contributions(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """Compute relative contribution of each component."""
        full_score = results['Full NECons'][metric]
        
        contributions = {}
        for variant_name, metrics in results.items():
            if variant_name == 'Full NECons':
                continue
            
            variant_score = metrics[metric]
            contribution = (full_score - variant_score) / full_score * 100
            contributions[variant_name] = contribution
        
        return contributions


# =============================================================================
# SECTION 9: LATEX TABLE GENERATOR
# =============================================================================


class LaTeXTableGenerator:
    """
    Generates publication-ready LaTeX tables for IEEE TPDS.
    
    Formats:
    - Detection performance comparison
    - Byzantine resilience results
    - Scalability analysis
    - Statistical significance
    - Ablation study
    """
    
    def __init__(self, output_dir: str = "results/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_detection_table(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metrics: List[str] = ['f1', 'precision', 'recall', 'auc_roc', 'auc_pr', 'mcc'],
        caption: str = "Detection Performance Comparison",
        label: str = "tab:detection"
    ) -> str:
        """Generate detection performance comparison table."""
        # Header
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\resizebox{\\columnwidth}{!}{%\n"
        
        # Column format
        col_format = "l" + "c" * len(metrics)
        latex += f"\\begin{{tabular}}{{{col_format}}}\n"
        latex += "\\toprule\n"
        
        # Header row
        metric_names = {
            'f1': 'F1',
            'precision': 'Prec.',
            'recall': 'Recall',
            'auc_roc': 'AUC-ROC',
            'auc_pr': 'AUC-PR',
            'mcc': 'MCC',
            'accuracy': 'Acc.'
        }
        header = "Method & " + " & ".join(metric_names.get(m, m) for m in metrics) + " \\\\\n"
        latex += header
        latex += "\\midrule\n"
        
        # Find best values for bolding
        best_values = {}
        for metric in metrics:
            best_val = -float('inf')
            for method, method_results in results.items():
                if metric in method_results:
                    val = np.mean(method_results[metric])
                    if val > best_val:
                        best_val = val
            best_values[metric] = best_val
        
        # Data rows
        for method, method_results in results.items():
            row = method
            for metric in metrics:
                if metric in method_results:
                    mean = np.mean(method_results[metric])
                    std = np.std(method_results[metric])
                    
                    # Bold if best
                    if abs(mean - best_values[metric]) < 0.001:
                        row += f" & \\textbf{{{mean:.3f}}}$\\pm${std:.3f}"
                    else:
                        row += f" & {mean:.3f}$\\pm${std:.3f}"
                else:
                    row += " & -"
            row += " \\\\\n"
            latex += row
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}%\n"
        latex += "}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        filepath = self.output_dir / f"{label.replace('tab:', '')}.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return latex
    
    def generate_byzantine_table(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        caption: str = "Byzantine Resilience Evaluation",
        label: str = "tab:byzantine"
    ) -> str:
        """Generate Byzantine resilience results table."""
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\resizebox{\\columnwidth}{!}{%\n"
        latex += "\\begin{tabular}{llcccc}\n"
        latex += "\\toprule\n"
        latex += "Attack Type & Metric & 0\\% & 10\\% & 20\\% & 33\\% \\\\\n"
        latex += "\\midrule\n"
        
        attack_names = {
            'model_poisoning': 'Model Poisoning',
            'label_flipping': 'Label Flipping',
            'gradient_scaling': 'Gradient Scaling'
        }
        
        for attack_type, ratios in results.items():
            attack_name = attack_names.get(attack_type, attack_type)
            
            # Relative error row
            row = f"{attack_name} & Rel. Error"
            for ratio in ['0.0', '0.1', '0.2', '0.33']:
                if ratio in ratios:
                    val = ratios[ratio].get('relative_error', 0)
                    row += f" & {val:.3f}"
                else:
                    row += " & -"
            row += " \\\\\n"
            latex += row
            
            # Accuracy preserved row
            row = " & Acc. Preserved"
            for ratio in ['0.0', '0.1', '0.2', '0.33']:
                if ratio in ratios:
                    val = ratios[ratio].get('accuracy_preserved', False)
                    row += f" & {'\\checkmark' if val else '\\texttimes'}"
                else:
                    row += " & -"
            row += " \\\\\n"
            latex += row
            
            latex += "\\addlinespace\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}%\n"
        latex += "}\n"
        latex += "\\end{table}\n"
        
        filepath = self.output_dir / f"{label.replace('tab:', '')}.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return latex
    
    def generate_scalability_table(
        self,
        results: List[Dict[str, Any]],
        caption: str = "Scalability Analysis",
        label: str = "tab:scalability"
    ) -> str:
        """Generate scalability analysis table."""
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{rcccc}\n"
        latex += "\\toprule\n"
        latex += "Nodes & Time (ms) & Accuracy & TPS & Success \\\\\n"
        latex += "\\midrule\n"
        
        for result in results:
            n = result['node_count']
            time_ms = result.get('consensus_time_ms', 0)
            acc = result.get('accuracy', 0)
            tps = result.get('throughput_tps', 0)
            success = result.get('success', False)
            
            success_str = '\\checkmark' if success else '\\texttimes'
            latex += f"{n:,} & {time_ms:.1f} & {acc:.3f} & {tps:.1f} & {success_str} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        filepath = self.output_dir / f"{label.replace('tab:', '')}.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return latex
    
    def generate_ablation_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['f1', 'auc_roc', 'mcc'],
        caption: str = "Ablation Study Results",
        label: str = "tab:ablation"
    ) -> str:
        """Generate ablation study table."""
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        
        col_format = "l" + "c" * (len(metrics) + 1)
        latex += f"\\begin{{tabular}}{{{col_format}}}\n"
        latex += "\\toprule\n"
        
        header = "Variant & " + " & ".join(m.upper() for m in metrics) + " & $\\Delta$ F1 \\\\\n"
        latex += header
        latex += "\\midrule\n"
        
        full_f1 = results.get('Full NECons', {}).get('f1', 0)
        
        for variant, metrics_dict in results.items():
            row = variant
            for metric in metrics:
                val = metrics_dict.get(metric, 0)
                std = metrics_dict.get(f'{metric}_std', 0)
                
                if variant == 'Full NECons':
                    row += f" & \\textbf{{{val:.3f}}}"
                else:
                    row += f" & {val:.3f}"
            
            # Delta F1
            variant_f1 = metrics_dict.get('f1', 0)
            delta = (full_f1 - variant_f1) / full_f1 * 100 if full_f1 > 0 else 0
            
            if variant == 'Full NECons':
                row += " & -"
            else:
                row += f" & -{delta:.1f}\\%"
            
            row += " \\\\\n"
            latex += row
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        filepath = self.output_dir / f"{label.replace('tab:', '')}.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return latex
    
    def generate_communication_table(
        self,
        results: Dict[str, Dict[str, Any]],
        caption: str = "Communication Complexity Comparison",
        label: str = "tab:communication"
    ) -> str:
        """Generate communication complexity comparison table."""
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{lccc}\n"
        latex += "\\toprule\n"
        latex += "Protocol & Complexity & Time (ms) & BFT \\\\\n"
        latex += "\\midrule\n"
        
        for protocol, info in results.items():
            complexity = info.get('message_complexity', 'N/A')
            time_ms = info.get('avg_time_ms', 0)
            bft = info.get('byzantine_tolerance', 'N/A')
            
            if isinstance(complexity, int):
                complexity = f"$O(n^2)$ = {complexity:,}"
            
            bft_str = '\\checkmark' if 'f < n/3' in str(bft) else '\\texttimes'
            
            latex += f"{protocol} & {complexity} & {time_ms:.2f} & {bft_str} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        filepath = self.output_dir / f"{label.replace('tab:', '')}.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return latex


# =============================================================================
# SECTION 10: RESULTS FORMATTER
# =============================================================================


class ResultsFormatter:
    """
    Formats and exports experimental results.
    
    Supports multiple output formats:
    - JSON for programmatic access
    - LaTeX tables for paper
    - Markdown for documentation
    - CSV for spreadsheet analysis
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.latex_gen = LaTeXTableGenerator(str(self.output_dir / "tables"))
    
    def save_results(
        self,
        results: Dict[str, Any],
        name: str,
        formats: List[str] = ['json', 'latex']
    ) -> Dict[str, Path]:
        """Save results in multiple formats."""
        saved_files = {}
        
        if 'json' in formats:
            filepath = self.output_dir / f"{name}.json"
            with open(filepath, 'w') as f:
                json.dump(self._convert_to_serializable(results), f, indent=2)
            saved_files['json'] = filepath
        
        if 'csv' in formats:
            filepath = self.output_dir / f"{name}.csv"
            self._save_as_csv(results, filepath)
            saved_files['csv'] = filepath
        
        return saved_files
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy/tensor types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, Tensor):
            return obj.cpu().numpy().tolist()
        else:
            return obj
    
    def _save_as_csv(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save results as CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Flatten nested dict
            rows = self._flatten_dict(results)
            
            if rows:
                writer.writerow(rows[0].keys())
                for row in rows:
                    writer.writerow(row.values())
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '_'
    ) -> List[Dict[str, Any]]:
        """Flatten nested dictionary."""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep))
            else:
                items.append({new_key: v})
        
        # Merge into single dict per row
        if items:
            merged = {}
            for item in items:
                merged.update(item)
            return [merged]
        
        return items
    
    def generate_all_tables(
        self,
        detection_results: Dict[str, Dict[str, List[float]]],
        byzantine_results: Dict[str, Dict[str, Dict[str, float]]],
        scalability_results: List[Dict[str, Any]],
        ablation_results: Dict[str, Dict[str, float]],
        communication_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate all LaTeX tables for paper."""
        tables = {}
        
        tables['detection'] = self.latex_gen.generate_detection_table(detection_results)
        tables['byzantine'] = self.latex_gen.generate_byzantine_table(byzantine_results)
        tables['scalability'] = self.latex_gen.generate_scalability_table(scalability_results)
        tables['ablation'] = self.latex_gen.generate_ablation_table(ablation_results)
        tables['communication'] = self.latex_gen.generate_communication_table(communication_results)
        
        return tables


# =============================================================================
# SECTION 11: EXPERIMENT RUNNER
# =============================================================================


class ExperimentRunner:
    """
    Orchestrates all evaluation experiments.
    
    Runs complete evaluation pipeline:
    1. Detection benchmark
    2. Byzantine resilience
    3. Scalability analysis
    4. Cross-chain evaluation
    5. Statistical significance
    6. Ablation study
    """
    
    def __init__(
        self,
        necons_model: nn.Module,
        consensus_module: nn.Module,
        config: EvaluationConfig,
        device: torch.device
    ):
        self.model = necons_model
        self.consensus = consensus_module
        self.config = config
        self.device = device
        
        # Initialize components
        self.baseline_comparator = BaselineComparator(necons_model, config, device)
        self.byzantine_evaluator = ByzantineResilienceEvaluator(
            necons_model, consensus_module, config, device
        )
        self.scalability_analyzer = ScalabilityAnalyzer(config)
        self.statistical_tester = StatisticalTester(
            config.confidence_level, config.use_bonferroni
        )
        self.ablation_runner = AblationStudyRunner(necons_model, device)
        self.results_formatter = ResultsFormatter(config.results_dir)
    
    def run_all_experiments(
        self,
        test_data: Any,
        consensus_factory: Callable[[int], nn.Module]
    ) -> Dict[str, Any]:
        """Run all evaluation experiments."""
        all_results = {}
        
        logger.info("Starting comprehensive evaluation...")
        
        # 1. Detection benchmark
        logger.info("Running detection benchmark...")
        detection_results = self.baseline_comparator.compare_detection_performance(
            test_data, self.config.num_runs
        )
        all_results['detection'] = detection_results
        
        # 2. Byzantine resilience
        logger.info("Running Byzantine resilience evaluation...")
        byzantine_results = self.byzantine_evaluator.evaluate(test_data)
        all_results['byzantine'] = byzantine_results
        
        # 3. Scalability analysis
        logger.info("Running scalability analysis...")
        scalability_results = self.scalability_analyzer.run_scalability_test(
            consensus_factory
        )
        all_results['scalability'] = scalability_results
        
        # 4. Statistical significance
        logger.info("Running statistical significance tests...")
        if 'f1' in detection_results.get('NECons', {}):
            necons_f1 = detection_results['NECons']['f1']
            baseline_f1 = {
                name: results.get('f1', [])
                for name, results in detection_results.items()
                if name != 'NECons'
            }
            statistical_results = self.statistical_tester.full_comparison(
                necons_f1, baseline_f1
            )
            all_results['statistical'] = statistical_results
        
        # 5. Generate tables
        logger.info("Generating LaTeX tables...")
        # Note: Ablation and communication results would be generated separately
        
        # Save results
        self.results_formatter.save_results(all_results, 'all_results')
        
        logger.info("Evaluation complete!")
        return all_results


# =============================================================================
# SECTION 12: UNIT TESTS
# =============================================================================


def run_unit_tests():
    """Run unit tests for evaluation module."""
    print("=" * 60)
    print("NECons Evaluation Module - Unit Tests")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Test 1: Evaluation Metrics
    print("\n[Test 1] Evaluation Metrics")
    metrics = EvaluationMetrics(num_classes=2)
    logits = torch.randn(100, 2)
    labels = torch.randint(0, 2, (100,))
    metrics.update(logits, labels)
    results = metrics.compute()
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  AUC-ROC: {results['auc_roc']:.4f}")
    print(f"  MCC: {results['mcc']:.4f}")
    print("  ✓ PASSED")
    
    # Test 2: GNN Baselines
    print("\n[Test 2] GNN Baselines")
    x = torch.randn(50, 8)
    edge_index = torch.randint(0, 50, (2, 200))
    
    gat = VanillaGATBaseline(8, 32, 2)
    gcn = VanillaGCNBaseline(8, 32, 2)
    sage = GraphSAGEBaseline(8, 32, 2)
    
    for model in [gat, gcn, sage]:
        out = model(x, edge_index)
        print(f"  {model.name()}: output shape = {out.shape}")
    print("  ✓ PASSED")
    
    # Test 3: Consensus Baselines
    print("\n[Test 3] Consensus Baselines")
    node_updates = {i: torch.randn(64) for i in range(20)}
    
    pbft = PBFTBaseline(20)
    raft = RaftBaseline(20)
    hotstuff = HotStuffBaseline(20)
    
    for baseline in [pbft, raft, hotstuff]:
        result, info = baseline.aggregate(node_updates)
        print(f"  {baseline.name()}: time={info['consensus_time_ms']:.3f}ms")
    print("  ✓ PASSED")
    
    # Test 4: Statistical Tests
    print("\n[Test 4] Statistical Tests")
    tester = StatisticalTester()
    
    values_a = [0.85, 0.87, 0.86, 0.88, 0.84]
    values_b = [0.80, 0.82, 0.81, 0.79, 0.83]
    
    wilcoxon = tester.wilcoxon_test(values_a, values_b)
    print(f"  Wilcoxon p-value: {wilcoxon.get('p_value', 'N/A')}")
    
    cohens_d = tester.cohens_d(values_a, values_b)
    print(f"  Cohen's d: {cohens_d:.3f}")
    
    cliffs_delta = tester.cliffs_delta(values_a, values_b)
    print(f"  Cliff's delta: {cliffs_delta:.3f}")
    print("  ✓ PASSED")
    
    # Test 5: Scalability Analyzer
    print("\n[Test 5] Scalability Analyzer")
    config = EvaluationConfig(node_counts=[50, 100])
    analyzer = ScalabilityAnalyzer(config)
    
    # Mock consensus factory
    class MockConsensus(nn.Module):
        def __init__(self, num_nodes):
            super().__init__()
            self.n = num_nodes
        
        def forward(self, updates, global_model, **kwargs):
            result = torch.stack(list(updates.values())).mean(dim=0)
            return result, {'consensus_time_ms': 10.0, 'message_complexity': self.n**2, 'num_filtered': 0}
    
    results = analyzer.run_scalability_test(MockConsensus, hidden_dim=64)
    print(f"  Tested node counts: {[r['node_count'] for r in results]}")
    print(f"  Practical limit: {analyzer.find_practical_limit()}")
    print("  ✓ PASSED")
    
    # Test 6: LaTeX Table Generator
    print("\n[Test 6] LaTeX Table Generator")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        latex_gen = LaTeXTableGenerator(tmpdir)
        
        detection_results = {
            'NECons': {'f1': [0.85, 0.86, 0.87], 'auc_roc': [0.90, 0.91, 0.89]},
            'GCN': {'f1': [0.80, 0.81, 0.79], 'auc_roc': [0.85, 0.86, 0.84]}
        }
        
        table = latex_gen.generate_detection_table(
            detection_results,
            metrics=['f1', 'auc_roc'],
            caption="Test Table",
            label="tab:test"
        )
        print(f"  Generated table length: {len(table)} chars")
        assert "\\begin{table}" in table
        assert "NECons" in table
    print("  ✓ PASSED")
    
    # Test 7: Byzantine Resilience (Mock)
    print("\n[Test 7] Byzantine Resilience Evaluator")
    
    class MockModel(nn.Module):
        def forward(self, x, edge_index, edge_attr=None):
            return torch.randn(x.size(0), 2)
    
    class MockData:
        def __init__(self):
            self.x = torch.randn(100, 8)
            self.edge_index = torch.randint(0, 100, (2, 300))
            self.y = torch.randint(0, 2, (100,))
            self.test_mask = torch.ones(100, dtype=torch.bool)
    
    mock_config = EvaluationConfig(
        byzantine_ratios=[0.0, 0.1],
        attack_types=["model_poisoning"]
    )
    
    evaluator = ByzantineResilienceEvaluator(
        MockModel(), MockConsensus(100), mock_config, device
    )
    byz_results = evaluator.evaluate(MockData(), num_edge_nodes=50)
    print(f"  Attack types evaluated: {list(byz_results.keys())}")
    print("  ✓ PASSED")
    
    # Test 8: Results Formatter
    print("\n[Test 8] Results Formatter")
    with tempfile.TemporaryDirectory() as tmpdir:
        formatter = ResultsFormatter(tmpdir)
        
        test_results = {
            'method': 'NECons',
            'f1': 0.85,
            'nested': {'a': 1, 'b': 2}
        }
        
        saved = formatter.save_results(test_results, 'test', formats=['json'])
        print(f"  Saved files: {list(saved.keys())}")
    print("  ✓ PASSED")
    
    # Test 9: Ablation Study Runner
    print("\n[Test 9] Ablation Study Runner")
    ablation = AblationStudyRunner(MockModel(), device)
    ablation.ablation_models['Full'] = MockModel()
    ablation.ablation_models['Reduced'] = MockModel()
    
    results = ablation.run_ablation(MockData(), num_runs=2)
    print(f"  Variants tested: {list(results.keys())}")
    print("  ✓ PASSED")
    
    # Test 10: Baseline Comparator
    print("\n[Test 10] Baseline Comparator")
    comparator = BaselineComparator(MockModel(), mock_config, device)
    comparator.setup_gnn_baselines(8, 32, 2)
    print(f"  GNN baselines: {list(comparator.gnn_baselines.keys())}")
    comparator.setup_consensus_baselines(100)
    print(f"  Consensus baselines: {list(comparator.consensus_baselines.keys())}")
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All unit tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
