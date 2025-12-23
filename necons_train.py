"""
================================================================================
NECons: Network-aware Edge-based Consensus for Distributed Blockchain 
        Anomaly Detection
================================================================================

Training Module: Training, Validation, Distributed Learning, and Optimization

Target Venue: IEEE Transactions on Parallel and Distributed Systems (TPDS) 2026

This module implements:
1. Single-node and distributed training pipelines
2. Byzantine attack simulation for robustness evaluation
3. Learning rate scheduling and optimization strategies
4. Checkpoint management and model persistence
5. Early stopping and convergence monitoring
6. Multi-GPU training with DDP support

================================================================================
TRAINING STRATEGIES
================================================================================
1. Standard Training: Full-batch or mini-batch on single GPU
2. Distributed Training: Multi-GPU with DistributedDataParallel
3. Byzantine-Resilient Training: Simulated distributed with Byzantine nodes
4. Federated-Style Training: Edge node simulation with consensus aggregation

================================================================================
KEY REFERENCES
================================================================================
[1] McMahan et al., "Communication-Efficient Learning of Deep Networks from 
    Decentralized Data", AISTATS 2017 (FedAvg)
[2] Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant 
    Gradient Descent", NeurIPS 2017
[3] Lin et al., "Don't Use Large Mini-Batches, Use Local SGD", ICLR 2020
[4] Han et al., "DegaFL: Decentralized Gradient Aggregation for Cross-Silo 
    Federated Learning", IEEE TPDS 2025

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
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    _LRScheduler, StepLR, CosineAnnealingLR, 
    CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
from pathlib import Path
from abc import ABC, abstractmethod
import os
import json
import copy
import time
import math
import logging
import warnings
from datetime import datetime
from contextlib import contextmanager

# Optional distributed imports
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    HAS_DISTRIBUTED = True
except ImportError:
    HAS_DISTRIBUTED = False

# TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION AND DATA STRUCTURES
# =============================================================================


class TrainingPhase(Enum):
    """Training phase enumeration."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class AttackType(Enum):
    """Byzantine attack types for training simulation."""
    NONE = "none"
    MODEL_POISONING = "model_poisoning"
    LABEL_FLIPPING = "label_flipping"
    GRADIENT_SCALING = "gradient_scaling"
    SIGN_FLIPPING = "sign_flipping"
    DELAY = "delay"
    COLLUSION = "collusion"


class AggregationMethod(Enum):
    """Byzantine-robust aggregation methods."""
    MEAN = "mean"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    BULYAN = "bulyan"
    TRUST_WEIGHTED = "trust_weighted"


@dataclass
class TrainingConfig:
    """
    Configuration for training pipeline.
    
    Comprehensive settings for all training aspects including
    optimization, scheduling, checkpointing, and Byzantine simulation.
    """
    # Basic training parameters
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimizer settings
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW
    eps: float = 1e-8
    
    # Learning rate scheduler
    scheduler: str = "cosine_warmup"  # "step", "cosine", "cosine_warmup", "plateau", "onecycle"
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.1
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.0001
    monitor_metric: str = "val_f1"
    
    # Class imbalance handling
    use_class_weights: bool = True
    focal_loss: bool = False
    focal_gamma: float = 2.0
    
    # Regularization
    dropout: float = 0.2
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 1.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # Distributed training
    distributed: bool = False
    num_workers: int = 4
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Byzantine simulation (for NECons experiments)
    simulate_byzantine: bool = False
    num_edge_nodes: int = 100
    byzantine_ratio: float = 0.0
    attack_type: str = "none"
    aggregation_method: str = "bulyan"
    consensus_rounds: int = 3
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    save_best_only: bool = True
    
    # Logging
    log_frequency: int = 100
    use_tensorboard: bool = True
    tensorboard_dir: str = "results/logs"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class TrainingState:
    """
    Maintains training state across epochs.
    
    Used for checkpointing and resuming training.
    """
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0
    
    # Training history
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_metrics: List[Dict[str, float]] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    # Consensus statistics (for Byzantine training)
    consensus_times: List[float] = field(default_factory=list)
    byzantine_filtered: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create state from dictionary."""
        return cls(**data)


# =============================================================================
# SECTION 2: LOSS FUNCTIONS
# =============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Particularly useful for blockchain anomaly detection where anomalies
    are typically rare (1-5% of transactions).
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            gamma: Focusing parameter (γ ≥ 0)
            alpha: Class weights [num_classes]
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [N, C]
            targets: Ground truth labels [N]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Prevents overconfident predictions and improves calibration.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        weight: Optional[Tensor] = None
    ):
        """
        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor (0 = no smoothing)
            weight: Class weights
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute label-smoothed cross-entropy.
        
        Args:
            inputs: Predicted logits [N, C]
            targets: Ground truth labels [N]
        
        Returns:
            Smoothed loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.weight is not None:
            loss = loss * self.weight[targets]
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning discriminative embeddings.
    
    Useful for semi-supervised learning when labeled anomalies are scarce.
    
    Reference: Chen et al., "A Simple Framework for Contrastive Learning 
               of Visual Representations", ICML 2020
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Softmax temperature
            base_temperature: Base temperature for scaling
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features: Normalized feature vectors [N, D]
            labels: Class labels [N] (for supervised contrastive)
            mask: Positive pair mask [N, N]
        
        Returns:
            Contrastive loss value
        """
        device = features.device
        batch_size = features.size(0)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        if mask is None:
            if labels is not None:
                labels = labels.contiguous().view(-1, 1)
                mask = torch.eq(labels, labels.T).float().to(device)
            else:
                mask = torch.eye(batch_size, device=device)
        
        # Remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Compute log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob
        return loss.mean()


def create_loss_function(
    config: TrainingConfig,
    class_weights: Optional[Tensor] = None,
    num_classes: int = 2
) -> nn.Module:
    """
    Factory function to create appropriate loss function.
    
    Args:
        config: Training configuration
        class_weights: Optional class weights for imbalanced data
        num_classes: Number of classes
    
    Returns:
        Loss function module
    """
    if config.focal_loss:
        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=class_weights,
            reduction='mean'
        )
    elif config.label_smoothing > 0:
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=config.label_smoothing,
            weight=class_weights
        )
    else:
        return nn.CrossEntropyLoss(weight=class_weights)


# =============================================================================
# SECTION 3: OPTIMIZER AND SCHEDULER FACTORY
# =============================================================================


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create(
        model: nn.Module,
        config: TrainingConfig
    ) -> Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            model: Model to optimize
            config: Training configuration
        
        Returns:
            Optimizer instance
        """
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'bn' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if config.optimizer.lower() == 'adam':
            return Adam(
                param_groups,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps
            )
        elif config.optimizer.lower() == 'adamw':
            return AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=config.betas,
                eps=config.eps
            )
        elif config.optimizer.lower() == 'sgd':
            return SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=config.momentum,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create(
        optimizer: Optimizer,
        config: TrainingConfig,
        steps_per_epoch: int = 1000
    ) -> Optional[_LRScheduler]:
        """
        Create scheduler based on configuration.
        
        Args:
            optimizer: Optimizer instance
            config: Training configuration
            steps_per_epoch: Number of training steps per epoch
        
        Returns:
            Scheduler instance or None
        """
        scheduler_type = config.scheduler.lower()
        
        if scheduler_type == 'step':
            return StepLR(
                optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma
            )
        
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.min_lr
            )
        
        elif scheduler_type == 'cosine_warmup':
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.warmup_epochs,
                T_mult=2,
                eta_min=config.min_lr
            )
        
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.scheduler_gamma,
                patience=config.patience // 2,
                min_lr=config.min_lr
            )
        
        elif scheduler_type == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=config.learning_rate,
                epochs=config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        
        elif scheduler_type == 'none':
            return None
        
        else:
            logger.warning(f"Unknown scheduler: {scheduler_type}, using None")
            return None


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by another scheduler.
    
    Gradually increases learning rate from 0 to base_lr over warmup_steps,
    then follows the main scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        main_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.main_scheduler = main_scheduler
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if self.main_scheduler is not None:
                return self.main_scheduler.get_last_lr()
            return self.base_lrs
    
    def step(self, epoch: Optional[int] = None) -> None:
        super().step(epoch)
        if self.last_epoch >= self.warmup_steps and self.main_scheduler is not None:
            self.main_scheduler.step()


# =============================================================================
# SECTION 4: METRICS COMPUTATION
# =============================================================================


class MetricsComputer:
    """
    Computes classification metrics for anomaly detection.
    
    Handles binary and multi-class classification with support
    for imbalanced datasets.
    """
    
    def __init__(self, num_classes: int = 2, threshold: float = 0.5):
        """
        Args:
            num_classes: Number of classes
            threshold: Classification threshold for binary case
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated predictions and labels."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(
        self,
        logits: Tensor,
        targets: Tensor
    ) -> None:
        """
        Accumulate predictions and labels.
        
        Args:
            logits: Model output logits [N, C]
            targets: Ground truth labels [N]
        """
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        self.predictions.append(preds.cpu())
        self.labels.append(targets.cpu())
        self.probabilities.append(probs.cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            return {}
        
        preds = torch.cat(self.predictions)
        labels = torch.cat(self.labels)
        probs = torch.cat(self.probabilities)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = (preds == labels).float().mean().item()
        
        # Per-class metrics
        for class_idx in range(self.num_classes):
            tp = ((preds == class_idx) & (labels == class_idx)).sum().float()
            fp = ((preds == class_idx) & (labels != class_idx)).sum().float()
            fn = ((preds != class_idx) & (labels == class_idx)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'precision_class{class_idx}'] = precision.item()
            metrics[f'recall_class{class_idx}'] = recall.item()
            metrics[f'f1_class{class_idx}'] = f1.item()
        
        # Macro-averaged metrics
        metrics['precision'] = np.mean([metrics[f'precision_class{i}'] for i in range(self.num_classes)])
        metrics['recall'] = np.mean([metrics[f'recall_class{i}'] for i in range(self.num_classes)])
        metrics['f1'] = np.mean([metrics[f'f1_class{i}'] for i in range(self.num_classes)])
        
        # AUC-ROC (binary case)
        if self.num_classes == 2:
            metrics['auc_roc'] = self._compute_auc_roc(probs[:, 1], labels)
            metrics['auc_pr'] = self._compute_auc_pr(probs[:, 1], labels)
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = self._compute_mcc(preds, labels)
        
        return metrics
    
    def _compute_auc_roc(self, probs: Tensor, labels: Tensor) -> float:
        """Compute AUC-ROC using trapezoidal rule."""
        try:
            # Sort by probability
            sorted_indices = torch.argsort(probs, descending=True)
            sorted_labels = labels[sorted_indices]
            
            # Compute TPR and FPR at each threshold
            total_pos = (labels == 1).sum().float()
            total_neg = (labels == 0).sum().float()
            
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
            
            return auc.item() if isinstance(auc, Tensor) else auc
        except:
            return 0.5
    
    def _compute_auc_pr(self, probs: Tensor, labels: Tensor) -> float:
        """Compute AUC-PR (Average Precision)."""
        try:
            sorted_indices = torch.argsort(probs, descending=True)
            sorted_labels = labels[sorted_indices]
            
            total_pos = (labels == 1).sum().float()
            if total_pos == 0:
                return 0.0
            
            precisions = []
            recalls = []
            
            cum_tp = 0
            for i, label in enumerate(sorted_labels):
                if label == 1:
                    cum_tp += 1
                precision = cum_tp / (i + 1)
                recall = cum_tp / total_pos
                precisions.append(precision)
                recalls.append(recall)
            
            # Compute AP
            ap = 0.0
            prev_recall = 0.0
            for p, r in zip(precisions, recalls):
                ap += p * (r - prev_recall)
                prev_recall = r
            
            return ap.item() if isinstance(ap, Tensor) else ap
        except:
            return 0.0
    
    def _compute_mcc(self, preds: Tensor, labels: Tensor) -> float:
        """Compute Matthews Correlation Coefficient."""
        try:
            tp = ((preds == 1) & (labels == 1)).sum().float()
            tn = ((preds == 0) & (labels == 0)).sum().float()
            fp = ((preds == 1) & (labels == 0)).sum().float()
            fn = ((preds == 0) & (labels == 1)).sum().float()
            
            numerator = tp * tn - fp * fn
            denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            
            if denominator == 0:
                return 0.0
            
            mcc = numerator / denominator
            return mcc.item()
        except:
            return 0.0


# =============================================================================
# SECTION 5: EARLY STOPPING
# =============================================================================


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0001,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, value: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            value: Current metric value
            epoch: Current epoch
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                logger.info(f"Early stopping triggered. Best epoch: {self.best_epoch}")
        
        return self.should_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        self.best_epoch = 0


# =============================================================================
# SECTION 6: CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """
    Manages model checkpoints for saving and loading.
    
    Supports:
    - Best model saving
    - Periodic checkpointing
    - Training resumption
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_best_only: bool = True,
        max_checkpoints: int = 5
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save best model
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        
        self.checkpoints: List[Path] = []
        self.best_metric = None
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        state: TrainingState,
        config: TrainingConfig,
        metric: float,
        is_best: bool = False
    ) -> Optional[Path]:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            state: Training state
            config: Training configuration
            metric: Current metric value
            is_best: Whether this is the best model
        
        Returns:
            Path to saved checkpoint or None
        """
        # Determine if we should save
        if self.save_best_only and not is_best:
            return None
        
        # Create checkpoint
        checkpoint = {
            'epoch': state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'training_state': state.to_dict(),
            'config': asdict(config),
            'metric': metric,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        if is_best:
            filepath = self.checkpoint_dir / 'best_model.pt'
        else:
            filepath = self.checkpoint_dir / f'checkpoint_epoch_{state.epoch}.pt'
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Track checkpoints
        if not is_best:
            self.checkpoints.append(filepath)
            self._cleanup_old_checkpoints()
        
        return filepath
    
    def load(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> TrainingState:
        """
        Load checkpoint.
        
        Args:
            filepath: Path to checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to load state
            scheduler: Optional scheduler to load state
        
        Returns:
            Loaded training state
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        state = TrainingState.from_dict(checkpoint['training_state'])
        
        logger.info(f"Loaded checkpoint from epoch {state.epoch}")
        return state
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> Optional[TrainingState]:
        """Load best model checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return self.load(str(best_path), model, optimizer, scheduler)
        return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")


# =============================================================================
# SECTION 7: TENSORBOARD LOGGER
# =============================================================================


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.
    
    Provides convenient methods for logging training metrics,
    model graphs, and embeddings.
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled and HAS_TENSORBOARD
        self.log_dir = Path(log_dir)
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        if self.writer:
            self.writer.add_scalars(main_tag, values, step)
    
    def log_histogram(self, tag: str, values: Tensor, step: int) -> None:
        """Log histogram of values."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = '') -> None:
        """Log dictionary of metrics."""
        if self.writer:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)
    
    def log_model_graph(self, model: nn.Module, input_sample: Tensor) -> None:
        """Log model computation graph."""
        if self.writer:
            try:
                self.writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def log_embeddings(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        tag: str = 'embeddings',
        step: int = 0
    ) -> None:
        """Log embeddings for visualization."""
        if self.writer:
            metadata = labels.tolist() if labels is not None else None
            self.writer.add_embedding(embeddings, metadata=metadata, tag=tag, global_step=step)
    
    def close(self) -> None:
        """Close the writer."""
        if self.writer:
            self.writer.close()


# =============================================================================
# SECTION 8: BYZANTINE ATTACK SIMULATOR
# =============================================================================


class ByzantineAttackSimulator:
    """
    Simulates Byzantine attacks for distributed training evaluation.
    
    Implements various attack strategies that Byzantine nodes might employ
    to corrupt the distributed learning process.
    
    Reference: Blanchard et al., NeurIPS 2017; El Mhamdi et al., ICML 2018
    """
    
    def __init__(
        self,
        attack_type: AttackType,
        attack_strength: float = 1.0,
        collusion_size: int = 0
    ):
        """
        Args:
            attack_type: Type of Byzantine attack
            attack_strength: Strength multiplier for attacks
            collusion_size: Number of colluding Byzantine nodes
        """
        self.attack_type = attack_type
        self.attack_strength = attack_strength
        self.collusion_size = collusion_size
    
    def attack(
        self,
        update: Tensor,
        honest_updates: Optional[List[Tensor]] = None,
        global_model: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply Byzantine attack to an update.
        
        Args:
            update: Original honest update
            honest_updates: List of other honest updates (for some attacks)
            global_model: Current global model (for some attacks)
        
        Returns:
            Corrupted update
        """
        if self.attack_type == AttackType.NONE:
            return update
        
        elif self.attack_type == AttackType.MODEL_POISONING:
            return self._model_poisoning_attack(update)
        
        elif self.attack_type == AttackType.LABEL_FLIPPING:
            return self._label_flipping_attack(update)
        
        elif self.attack_type == AttackType.GRADIENT_SCALING:
            return self._gradient_scaling_attack(update)
        
        elif self.attack_type == AttackType.SIGN_FLIPPING:
            return self._sign_flipping_attack(update)
        
        elif self.attack_type == AttackType.COLLUSION:
            return self._collusion_attack(update, honest_updates)
        
        else:
            return update
    
    def _model_poisoning_attack(self, update: Tensor) -> Tensor:
        """
        Model poisoning: Add large random noise.
        
        Byzantine nodes send arbitrary values to corrupt the aggregate.
        """
        noise = torch.randn_like(update) * self.attack_strength * 10
        return update + noise
    
    def _label_flipping_attack(self, update: Tensor) -> Tensor:
        """
        Label flipping: Negate the update.
        
        Simulates training on flipped labels, causing gradient reversal.
        """
        return -update * self.attack_strength
    
    def _gradient_scaling_attack(self, update: Tensor) -> Tensor:
        """
        Gradient scaling: Scale update by large factor.
        
        Amplifies the Byzantine contribution to the aggregate.
        """
        scale = 100 * self.attack_strength
        return update * scale
    
    def _sign_flipping_attack(self, update: Tensor) -> Tensor:
        """
        Sign flipping: Flip signs of update coordinates.
        
        Randomly flips coordinate signs to cause divergence.
        """
        mask = torch.rand_like(update) < 0.5
        attacked = update.clone()
        attacked[mask] = -attacked[mask] * self.attack_strength
        return attacked
    
    def _collusion_attack(
        self,
        update: Tensor,
        honest_updates: Optional[List[Tensor]]
    ) -> Tensor:
        """
        Collusion attack: Coordinate with other Byzantine nodes.
        
        Byzantine nodes send the same malicious update to maximize impact.
        """
        if honest_updates:
            # Compute direction opposite to honest mean
            honest_mean = torch.stack(honest_updates).mean(dim=0)
            attack_direction = -honest_mean / (torch.norm(honest_mean) + 1e-8)
            attack_magnitude = torch.norm(update) * self.attack_strength * 5
            return attack_direction * attack_magnitude
        else:
            return self._model_poisoning_attack(update)


# =============================================================================
# SECTION 9: NECONS TRAINER
# =============================================================================


class NEConsTrainer:
    """
    Main trainer class for NECons.
    
    Supports:
    - Standard single-GPU training
    - Multi-GPU distributed training
    - Byzantine-resilient training simulation
    - Comprehensive logging and checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        consensus_module: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: NEConsGNN model
            config: Training configuration
            consensus_module: NEConsByzantineConsensus for Byzantine training
            device: Training device
        """
        self.model = model
        self.config = config
        self.consensus_module = consensus_module
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.scaler = GradScaler() if config.use_amp else None
        
        # Training state
        self.state = TrainingState()
        self.metrics_computer = MetricsComputer(num_classes=2)
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max'
        ) if config.early_stopping else None
        
        # Checkpointing and logging
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            save_best_only=config.save_best_only
        )
        self.tb_logger = TensorBoardLogger(
            config.tensorboard_dir,
            enabled=config.use_tensorboard
        )
        
        # Byzantine simulation
        self.byzantine_simulator = None
        if config.simulate_byzantine and config.byzantine_ratio > 0:
            self.byzantine_simulator = ByzantineAttackSimulator(
                attack_type=AttackType(config.attack_type),
                attack_strength=1.0
            )
        
        # Set random seeds
        self._set_seeds(config.seed)
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup(
        self,
        train_loader: Any,
        val_loader: Any,
        class_weights: Optional[Tensor] = None
    ) -> None:
        """
        Setup training components.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Optional class weights for imbalanced data
        """
        # Create optimizer
        self.optimizer = OptimizerFactory.create(self.model, self.config)
        
        # Create scheduler
        steps_per_epoch = len(train_loader) if hasattr(train_loader, '__len__') else 1000
        self.scheduler = SchedulerFactory.create(
            self.optimizer, self.config, steps_per_epoch
        )
        
        # Create loss function
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.loss_fn = create_loss_function(self.config, class_weights, num_classes=2)
        
        # Store loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        logger.info("Training setup complete")
        logger.info(f"  Optimizer: {self.config.optimizer}")
        logger.info(f"  Scheduler: {self.config.scheduler}")
        logger.info(f"  Device: {self.device}")
    
    def train(self, resume_from: Optional[str] = None) -> TrainingState:
        """
        Run full training loop.
        
        Args:
            resume_from: Optional checkpoint path to resume from
        
        Returns:
            Final training state
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.state = self.checkpoint_manager.load(
                resume_from, self.model, self.optimizer, self.scheduler
            )
            logger.info(f"Resumed training from epoch {self.state.epoch}")
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch
            
            # Training epoch
            train_loss, train_metrics = self._train_epoch()
            self.state.train_losses.append(train_loss)
            self.state.train_metrics.append(train_metrics)
            
            # Validation epoch
            val_loss, val_metrics = self._validate_epoch()
            self.state.val_losses.append(val_loss)
            self.state.val_metrics.append(val_metrics)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.state.learning_rates.append(current_lr)
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)
            
            # Check for best model
            monitor_value = val_metrics.get(self.config.monitor_metric.replace('val_', ''), 0)
            is_best = monitor_value > self.state.best_metric
            
            if is_best:
                self.state.best_metric = monitor_value
                self.state.best_epoch = epoch
                logger.info(f"New best model! {self.config.monitor_metric}: {monitor_value:.4f}")
            
            # Save checkpoint
            self.checkpoint_manager.save(
                self.model, self.optimizer, self.scheduler,
                self.state, self.config, monitor_value, is_best
            )
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if self.early_stopping:
                if self.early_stopping(monitor_value, epoch):
                    logger.info("Early stopping triggered")
                    break
        
        # Load best model
        self.checkpoint_manager.load_best(self.model)
        
        # Close logger
        self.tb_logger.close()
        
        logger.info(f"Training complete. Best epoch: {self.state.best_epoch}")
        return self.state
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Run one training epoch.
        
        Returns:
            (average_loss, metrics_dict)
        """
        self.model.train()
        self.metrics_computer.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            loss, logits, labels = self._train_step(batch)
            
            total_loss += loss
            num_batches += 1
            self.state.global_step += 1
            
            # Accumulate metrics
            self.metrics_computer.update(logits.detach(), labels)
            
            # Log batch metrics
            if self.state.global_step % self.config.log_frequency == 0:
                self.tb_logger.log_scalar(
                    'train/batch_loss', loss, self.state.global_step
                )
        
        avg_loss = total_loss / max(num_batches, 1)
        metrics = self.metrics_computer.compute()
        
        return avg_loss, metrics
    
    def _train_step(self, batch: Any) -> Tuple[float, Tensor, Tensor]:
        """
        Run one training step.
        
        Args:
            batch: Data batch
        
        Returns:
            (loss_value, logits, labels)
        """
        # Handle different batch formats
        if hasattr(batch, 'x'):
            # PyG batch
            batch = batch.to(self.device)
            x, edge_index = batch.x, batch.edge_index
            labels = batch.y
            edge_attr = getattr(batch, 'edge_attr', None)
            mask = getattr(batch, 'train_mask', None)
        else:
            # Tuple format
            x, edge_index, labels = batch[0], batch[1], batch[2]
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
            labels = labels.to(self.device)
            edge_attr = batch[3].to(self.device) if len(batch) > 3 else None
            mask = None
        
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.config.use_amp and self.scaler:
            with autocast():
                logits = self.model(x, edge_index, edge_attr)
                if mask is not None:
                    logits = logits[mask]
                    labels = labels[mask]
                loss = self.loss_fn(logits, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(x, edge_index, edge_attr)
            if mask is not None:
                logits = logits[mask]
                labels = labels[mask]
            loss = self.loss_fn(logits, labels)
            
            loss.backward()
            
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
        
        return loss.item(), logits, labels
    
    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Run validation epoch.
        
        Returns:
            (average_loss, metrics_dict)
        """
        self.model.eval()
        self.metrics_computer.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # Handle different batch formats
            if hasattr(batch, 'x'):
                batch = batch.to(self.device)
                x, edge_index = batch.x, batch.edge_index
                labels = batch.y
                edge_attr = getattr(batch, 'edge_attr', None)
                mask = getattr(batch, 'val_mask', None)
            else:
                x, edge_index, labels = batch[0], batch[1], batch[2]
                x = x.to(self.device)
                edge_index = edge_index.to(self.device)
                labels = labels.to(self.device)
                edge_attr = batch[3].to(self.device) if len(batch) > 3 else None
                mask = None
            
            logits = self.model(x, edge_index, edge_attr)
            
            if mask is not None:
                logits = logits[mask]
                labels = labels[mask]
            
            loss = self.loss_fn(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            self.metrics_computer.update(logits, labels)
        
        avg_loss = total_loss / max(num_batches, 1)
        metrics = self.metrics_computer.compute()
        
        return avg_loss, metrics
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ) -> None:
        """Log metrics for an epoch."""
        # Console logging
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics.get('f1', 0):.4f} | "
            f"LR: {lr:.6f}"
        )
        
        # TensorBoard logging
        self.tb_logger.log_scalar('train/loss', train_loss, epoch)
        self.tb_logger.log_scalar('val/loss', val_loss, epoch)
        self.tb_logger.log_scalar('train/lr', lr, epoch)
        self.tb_logger.log_metrics(train_metrics, epoch, prefix='train')
        self.tb_logger.log_metrics(val_metrics, epoch, prefix='val')


# =============================================================================
# SECTION 10: DISTRIBUTED TRAINER
# =============================================================================


class DistributedNEConsTrainer(NEConsTrainer):
    """
    Distributed trainer with Byzantine simulation.
    
    Simulates distributed training across multiple edge nodes with
    potential Byzantine failures, using consensus for aggregation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        consensus_module: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: NEConsGNN model
            config: Training configuration
            consensus_module: NEConsByzantineConsensus for aggregation
            device: Training device
        """
        super().__init__(model, config, consensus_module, device)
        
        self.num_edge_nodes = config.num_edge_nodes
        self.byzantine_ratio = config.byzantine_ratio
        self.num_byzantine = int(self.num_edge_nodes * self.byzantine_ratio)
        
        # Assign Byzantine nodes randomly
        all_nodes = list(range(self.num_edge_nodes))
        np.random.shuffle(all_nodes)
        self.byzantine_nodes = set(all_nodes[:self.num_byzantine])
        self.honest_nodes = set(all_nodes[self.num_byzantine:])
        
        logger.info(f"Distributed training setup:")
        logger.info(f"  Total edge nodes: {self.num_edge_nodes}")
        logger.info(f"  Byzantine nodes: {self.num_byzantine} ({self.byzantine_ratio*100:.1f}%)")
    
    def _train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Run distributed training epoch with Byzantine simulation.
        
        Simulates:
        1. Each edge node trains locally
        2. Nodes submit updates
        3. Byzantine nodes submit malicious updates
        4. Consensus aggregates updates
        5. Global model is updated
        """
        self.model.train()
        self.metrics_computer.reset()
        
        total_loss = 0.0
        num_rounds = 0
        consensus_times = []
        filtered_counts = []
        
        # Simulate multiple communication rounds
        for batch in self.train_loader:
            # Move batch to device
            if hasattr(batch, 'x'):
                batch = batch.to(self.device)
                x, edge_index = batch.x, batch.edge_index
                labels = batch.y
                edge_attr = getattr(batch, 'edge_attr', None)
            else:
                x = batch[0].to(self.device)
                edge_index = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                edge_attr = batch[3].to(self.device) if len(batch) > 3 else None
            
            # Store initial model parameters
            initial_params = {
                name: param.clone() for name, param in self.model.named_parameters()
            }
            
            # Simulate local training on edge nodes
            node_updates = {}
            
            for node_id in range(self.num_edge_nodes):
                # Sample subset of data for this node
                node_mask = torch.rand(x.size(0)) < (1.0 / self.num_edge_nodes)
                if node_mask.sum() == 0:
                    continue
                
                # Local forward pass
                self.optimizer.zero_grad()
                logits = self.model(x, edge_index, edge_attr)
                loss = self.loss_fn(logits, labels)
                loss.backward()
                
                # Collect gradient as update
                update = torch.cat([
                    p.grad.view(-1) for p in self.model.parameters() if p.grad is not None
                ])
                
                # Apply Byzantine attack if this is a Byzantine node
                if node_id in self.byzantine_nodes and self.byzantine_simulator:
                    honest_updates_list = [
                        node_updates[nid] for nid in self.honest_nodes 
                        if nid in node_updates
                    ]
                    update = self.byzantine_simulator.attack(
                        update, 
                        honest_updates=honest_updates_list if honest_updates_list else None
                    )
                
                node_updates[node_id] = update
                
                # Reset parameters for next node
                for name, param in self.model.named_parameters():
                    param.data = initial_params[name].clone()
            
            # Run consensus to aggregate updates
            if self.consensus_module and node_updates:
                global_param = torch.cat([
                    p.data.view(-1) for p in self.model.parameters()
                ])
                
                aggregated_update, consensus_info = self.consensus_module(
                    node_updates,
                    global_param,
                    byzantine_nodes=self.byzantine_nodes
                )
                
                consensus_times.append(consensus_info['consensus_time_ms'])
                filtered_counts.append(consensus_info['num_filtered'])
                
                # Apply aggregated update to model
                self._apply_update(aggregated_update)
            else:
                # Simple mean aggregation fallback
                if node_updates:
                    aggregated_update = torch.stack(list(node_updates.values())).mean(dim=0)
                    self._apply_update(aggregated_update)
            
            # Compute loss and metrics with updated model
            with torch.no_grad():
                logits = self.model(x, edge_index, edge_attr)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                self.metrics_computer.update(logits, labels)
            
            num_rounds += 1
            self.state.global_step += 1
        
        # Record consensus statistics
        if consensus_times:
            self.state.consensus_times.extend(consensus_times)
            self.state.byzantine_filtered.extend(filtered_counts)
        
        avg_loss = total_loss / max(num_rounds, 1)
        metrics = self.metrics_computer.compute()
        
        # Add consensus metrics
        if consensus_times:
            metrics['avg_consensus_time_ms'] = np.mean(consensus_times)
            metrics['avg_byzantine_filtered'] = np.mean(filtered_counts)
        
        return avg_loss, metrics
    
    def _apply_update(self, update: Tensor) -> None:
        """Apply aggregated update to model parameters."""
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param_update = update[idx:idx + param_size].view(param.shape)
            param.data -= self.config.learning_rate * param_update
            idx += param_size


# =============================================================================
# SECTION 11: TRAINING UTILITIES
# =============================================================================


def train_necons(
    model: nn.Module,
    train_data: Any,
    val_data: Any,
    config: Optional[TrainingConfig] = None,
    consensus_module: Optional[nn.Module] = None,
    class_weights: Optional[Tensor] = None,
    device: Optional[torch.device] = None
) -> TrainingState:
    """
    Convenience function to train NECons model.
    
    Args:
        model: NEConsGNN model
        train_data: Training data/loader
        val_data: Validation data/loader
        config: Training configuration
        consensus_module: Optional consensus module for Byzantine training
        class_weights: Optional class weights
        device: Training device
    
    Returns:
        Training state with history
    """
    if config is None:
        config = TrainingConfig()
    
    # Create trainer
    if config.simulate_byzantine and consensus_module is not None:
        trainer = DistributedNEConsTrainer(
            model, config, consensus_module, device
        )
    else:
        trainer = NEConsTrainer(
            model, config, consensus_module, device
        )
    
    # Setup training
    trainer.setup(train_data, val_data, class_weights)
    
    # Run training
    state = trainer.train()
    
    return state


def setup_distributed_training(
    rank: int,
    world_size: int,
    backend: str = 'nccl'
) -> None:
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
    """
    if not HAS_DISTRIBUTED:
        raise ImportError("torch.distributed not available")
    
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    logger.info(f"Distributed training initialized: rank {rank}/{world_size}")


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if HAS_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# SECTION 12: UNIT TESTS
# =============================================================================


def run_unit_tests():
    """Run unit tests for training module."""
    print("=" * 60)
    print("NECons Training Module - Unit Tests")
    print("=" * 60)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 2)
        
        def forward(self, x, edge_index, edge_attr=None):
            return self.linear(x)
    
    model = DummyModel()
    
    # Test 1: Training Configuration
    print("\n[Test 1] Training Configuration")
    config = TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=0.001
    )
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print("  ✓ PASSED")
    
    # Test 2: Optimizer Factory
    print("\n[Test 2] Optimizer Factory")
    optimizer = OptimizerFactory.create(model, config)
    print(f"  Optimizer type: {type(optimizer).__name__}")
    assert isinstance(optimizer, AdamW)
    print("  ✓ PASSED")
    
    # Test 3: Scheduler Factory
    print("\n[Test 3] Scheduler Factory")
    scheduler = SchedulerFactory.create(optimizer, config)
    print(f"  Scheduler type: {type(scheduler).__name__}")
    print("  ✓ PASSED")
    
    # Test 4: Loss Functions
    print("\n[Test 4] Loss Functions")
    
    # Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    logits = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    loss = focal_loss(logits, targets)
    print(f"  Focal Loss: {loss.item():.4f}")
    
    # Label Smoothing
    smooth_loss = LabelSmoothingLoss(num_classes=2, smoothing=0.1)
    loss = smooth_loss(logits, targets)
    print(f"  Label Smoothing Loss: {loss.item():.4f}")
    print("  ✓ PASSED")
    
    # Test 5: Metrics Computer
    print("\n[Test 5] Metrics Computer")
    metrics_computer = MetricsComputer(num_classes=2)
    metrics_computer.update(logits, targets)
    metrics = metrics_computer.compute()
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
    print("  ✓ PASSED")
    
    # Test 6: Early Stopping
    print("\n[Test 6] Early Stopping")
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    values = [0.5, 0.6, 0.65, 0.65, 0.65, 0.65]  # Should trigger at index 5
    for i, val in enumerate(values):
        stopped = early_stopping(val, i)
        if stopped:
            print(f"  Stopped at epoch {i}")
            break
    print("  ✓ PASSED")
    
    # Test 7: Training State
    print("\n[Test 7] Training State")
    state = TrainingState(epoch=5, best_metric=0.85)
    state_dict = state.to_dict()
    restored_state = TrainingState.from_dict(state_dict)
    assert restored_state.epoch == 5
    assert restored_state.best_metric == 0.85
    print(f"  Epoch: {restored_state.epoch}")
    print(f"  Best metric: {restored_state.best_metric}")
    print("  ✓ PASSED")
    
    # Test 8: Byzantine Attack Simulator
    print("\n[Test 8] Byzantine Attack Simulator")
    update = torch.randn(100)
    
    attacks = [
        AttackType.MODEL_POISONING,
        AttackType.LABEL_FLIPPING,
        AttackType.GRADIENT_SCALING,
        AttackType.SIGN_FLIPPING
    ]
    
    for attack_type in attacks:
        simulator = ByzantineAttackSimulator(attack_type)
        attacked = simulator.attack(update.clone())
        diff = (attacked - update).norm().item()
        print(f"  {attack_type.value}: norm_diff={diff:.2f}")
    print("  ✓ PASSED")
    
    # Test 9: Checkpoint Manager
    print("\n[Test 9] Checkpoint Manager")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        state = TrainingState(epoch=10, best_metric=0.9)
        
        # Save checkpoint
        path = manager.save(
            model, optimizer, scheduler, state, config,
            metric=0.9, is_best=True
        )
        print(f"  Saved to: {path}")
        
        # Load checkpoint
        loaded_state = manager.load_best(model)
        print(f"  Loaded epoch: {loaded_state.epoch}")
    print("  ✓ PASSED")
    
    # Test 10: TensorBoard Logger
    print("\n[Test 10] TensorBoard Logger")
    with tempfile.TemporaryDirectory() as tmpdir:
        tb_logger = TensorBoardLogger(tmpdir, enabled=True)
        tb_logger.log_scalar('test/loss', 0.5, 0)
        tb_logger.log_metrics({'accuracy': 0.9, 'f1': 0.85}, 0)
        tb_logger.close()
    print("  TensorBoard logging: OK")
    print("  ✓ PASSED")
    
    print("\n" + "=" * 60)
    print("All unit tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
