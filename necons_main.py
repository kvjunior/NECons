"""
================================================================================
NECons: Network-aware Edge-based Consensus for Distributed Blockchain 
        Anomaly Detection
================================================================================

Main Module: CLI, Experiment Orchestration, and Entry Points

Target Venue: IEEE Transactions on Parallel and Distributed Systems (TPDS) 2026

This module provides:
1. Command-line interface for all experiments
2. Configuration management via YAML files
3. Experiment orchestration and reproducibility
4. Logging and monitoring setup
5. Multi-GPU coordination
6. Results aggregation and reporting

================================================================================
USAGE EXAMPLES
================================================================================
# Train NECons on Ethereum dataset
python necons_main.py train --dataset ethereum_s --epochs 200

# Evaluate with Byzantine simulation
python necons_main.py evaluate --checkpoint best_model.pt --byzantine-ratio 0.33

# Run full benchmark suite
python necons_main.py benchmark --full --output results/

# Run scalability analysis
python necons_main.py scalability --max-nodes 5000

# Generate paper tables
python necons_main.py tables --results-dir results/ --output tables/

# Run ablation study
python necons_main.py ablation --config configs/ablation.yaml

================================================================================
Hardware Configuration:
- GPU: 4× NVIDIA GeForce RTX 3090 (24GB each)
- CPU: Intel Xeon Silver 4314 (64 cores) @ 2.40GHz
- RAM: 384GB DDR4
- OS: CentOS Linux 7 (Core)
================================================================================

Author: BlockchainLab
"""

import os
import sys
import argparse
import logging
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Import NECons modules
try:
    from necons_core import (
        NEConsGNN, NetworkAwareMGD, NEConsByzantineConsensus,
        CrossChainSyncProtocol, NetworkState, ChainType,
        AttackType, ConsensusVerifier, NodeScalabilityAnalyzer,
        CommunicationAnalyzer
    )
    from necons_data import (
        BlockchainGraphData, SyntheticBlockchainDataset,
        BlockchainDatasetLoader, DatasetType, TransactionSequenceBuilder,
        DatasetStatistics, create_augmentation, AugmentationType
    )
    from necons_train import (
        NEConsTrainer, DistributedNEConsTrainer, TrainingConfig,
        TrainingState, train_necons
    )
    from necons_eval import (
        EvaluationConfig, EvaluationMetrics, BaselineComparator,
        ByzantineResilienceEvaluator, ScalabilityAnalyzer,
        StatisticalTester, AblationStudyRunner, LaTeXTableGenerator,
        ResultsFormatter, ExperimentRunner
    )
except ImportError as e:
    # Handle import when running as standalone
    print(f"Warning: Could not import NECons modules: {e}")
    print("Running in standalone mode with mock implementations.")

# YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Using JSON for configuration.")

warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: LOGGING SETUP
# =============================================================================


def setup_logging(
    log_dir: str = "results/logs",
    log_level: str = "INFO",
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging for experiments.
    
    Creates both file and console handlers with proper formatting.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        experiment_name: Optional experiment name for log file
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_file = log_path / f"{experiment_name}_{timestamp}.log"
    else:
        log_file = log_path / f"necons_{timestamp}.log"
    
    # Configure logging
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Get numeric level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("NECons")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# =============================================================================
# SECTION 2: CONFIGURATION MANAGEMENT
# =============================================================================


@dataclass
class NEConsConfig:
    """
    Master configuration for NECons experiments.
    
    Combines model, training, data, and evaluation configurations
    into a single manageable structure.
    """
    # Experiment identification
    experiment_name: str = "necons_experiment"
    seed: int = 42
    
    # Hardware
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    num_gpus: int = 1
    num_workers: int = 4
    
    # Model architecture
    node_input_dim: int = 8
    edge_input_dim: int = 4
    hidden_dim: int = 256
    output_dim: int = 2
    num_mgd_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    network_aware: bool = True
    use_edge2seq: bool = True
    max_sequence_length: int = 50
    
    # Training
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 10
    use_amp: bool = True
    gradient_clip_norm: float = 1.0
    
    # Loss function
    use_class_weights: bool = True
    focal_loss: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 0.0001
    monitor_metric: str = "val_f1"
    
    # Data
    dataset: str = "ethereum_s"
    data_dir: str = "data"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Augmentation
    use_augmentation: bool = True
    edge_dropout: float = 0.1
    node_dropout: float = 0.1
    feature_masking: float = 0.1
    
    # Byzantine simulation
    simulate_byzantine: bool = False
    num_edge_nodes: int = 100
    byzantine_ratio: float = 0.0
    attack_type: str = "none"
    aggregation_method: str = "bulyan"
    consensus_rounds: int = 3
    
    # Cross-chain
    enable_cross_chain: bool = True
    supported_chains: List[str] = field(default_factory=lambda: [
        "ethereum", "bitcoin", "polygon"
    ])
    
    # Evaluation
    num_eval_runs: int = 10
    confidence_level: float = 0.95
    
    # Output
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 10
    save_best_only: bool = True
    
    # Logging
    log_dir: str = "results/logs"
    log_level: str = "INFO"
    use_tensorboard: bool = True
    tensorboard_dir: str = "results/tensorboard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NEConsConfig':
        """Create config from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'NEConsConfig':
        """Load config from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML config files")
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'NEConsConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def save_yaml(self, filepath: str) -> None:
        """Save config to YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML config files")
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def save_json(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: Optional[str] = None) -> NEConsConfig:
    """
    Load configuration from file or return defaults.
    
    Args:
        config_path: Path to config file (YAML or JSON)
    
    Returns:
        NEConsConfig instance
    """
    if config_path is None:
        return NEConsConfig()
    
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix in ['.yaml', '.yml']:
        return NEConsConfig.from_yaml(config_path)
    elif path.suffix == '.json':
        return NEConsConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def create_default_config(output_path: str = "config.yaml") -> None:
    """Create default configuration file."""
    config = NEConsConfig()
    
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        config.save_yaml(output_path)
    else:
        config.save_json(output_path)
    
    print(f"Created default config: {output_path}")


# =============================================================================
# SECTION 3: DEVICE MANAGEMENT
# =============================================================================


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get appropriate device based on configuration and availability.
    
    Args:
        device_str: Device specification ("auto", "cuda", "cuda:0", "cpu")
    
    Returns:
        torch.device instance
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    elif device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(device_str)
    else:
        device = torch.device("cpu")
    
    return device


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}")


def print_system_info() -> None:
    """Print system information for reproducibility."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    
    print(f"NumPy: {np.__version__}")
    print("=" * 60 + "\n")


# =============================================================================
# SECTION 4: MODEL FACTORY
# =============================================================================


def create_model(config: NEConsConfig, device: torch.device) -> nn.Module:
    """
    Create NEConsGNN model from configuration.
    
    Args:
        config: Model configuration
        device: Target device
    
    Returns:
        Initialized model on device
    """
    try:
        model = NEConsGNN(
            node_input_dim=config.node_input_dim,
            edge_input_dim=config.edge_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_mgd_layers=config.num_mgd_layers,
            num_heads=config.num_heads,
            max_sequence_length=config.max_sequence_length,
            dropout=config.dropout,
            network_aware=config.network_aware,
            use_edge2seq=config.use_edge2seq
        )
    except NameError:
        # Fallback for standalone testing
        print("Warning: Using mock model for testing")
        model = nn.Sequential(
            nn.Linear(config.node_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
    
    model = model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB")
    
    return model


def create_consensus_module(
    config: NEConsConfig,
    device: torch.device
) -> Optional[nn.Module]:
    """
    Create Byzantine consensus module from configuration.
    
    Args:
        config: Configuration
        device: Target device
    
    Returns:
        Consensus module or None if not simulating Byzantine
    """
    if not config.simulate_byzantine:
        return None
    
    try:
        consensus = NEConsByzantineConsensus(
            num_nodes=config.num_edge_nodes,
            hidden_dim=config.hidden_dim,
            byzantine_threshold=config.byzantine_ratio,
            aggregation_strategy=config.aggregation_method,
            consensus_rounds=config.consensus_rounds,
            network_aware=config.network_aware
        )
    except NameError:
        print("Warning: Using mock consensus for testing")
        return None
    
    return consensus


# =============================================================================
# SECTION 5: DATA LOADING
# =============================================================================


def load_data(config: NEConsConfig) -> Tuple[Any, Any, Any]:
    """
    Load training, validation, and test data.
    
    Args:
        config: Data configuration
    
    Returns:
        (train_data, val_data, test_data) tuple
    """
    print(f"\nLoading dataset: {config.dataset}")
    
    try:
        # Try to load real dataset
        dataset_type = DatasetType(config.dataset)
        loader = BlockchainDatasetLoader(data_dir=config.data_dir)
        data = loader.load(dataset_type)
    except (NameError, ValueError, FileNotFoundError):
        # Fall back to synthetic data
        print("Using synthetic dataset for testing")
        try:
            generator = SyntheticBlockchainDataset(
                num_nodes=10000,
                num_edges=50000,
                num_features=config.node_input_dim,
                anomaly_ratio=0.05,
                seed=config.seed
            )
            data = generator.generate()
        except NameError:
            # Create minimal mock data
            print("Creating minimal mock data")
            data = create_mock_data(config)
    
    # Print data statistics
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {data.num_classes}")
    
    # Compute class distribution
    if hasattr(data, 'y'):
        class_counts = torch.bincount(data.y)
        for i, count in enumerate(class_counts):
            print(f"  Class {i}: {count:,} ({count/len(data.y)*100:.1f}%)")
    
    return data, data, data  # Using same data object with masks


def create_mock_data(config: NEConsConfig) -> Any:
    """Create mock data for testing."""
    
    class MockData:
        def __init__(self, config):
            self.x = torch.randn(1000, config.node_input_dim)
            self.edge_index = torch.randint(0, 1000, (2, 5000))
            self.y = torch.randint(0, 2, (1000,))
            self.edge_attr = torch.randn(5000, config.edge_input_dim)
            
            # Create masks
            indices = torch.randperm(1000)
            train_size = int(1000 * config.train_ratio)
            val_size = int(1000 * config.val_ratio)
            
            self.train_mask = torch.zeros(1000, dtype=torch.bool)
            self.val_mask = torch.zeros(1000, dtype=torch.bool)
            self.test_mask = torch.zeros(1000, dtype=torch.bool)
            
            self.train_mask[indices[:train_size]] = True
            self.val_mask[indices[train_size:train_size+val_size]] = True
            self.test_mask[indices[train_size+val_size:]] = True
        
        @property
        def num_nodes(self):
            return self.x.size(0)
        
        @property
        def num_edges(self):
            return self.edge_index.size(1)
        
        @property
        def num_features(self):
            return self.x.size(1)
        
        @property
        def num_classes(self):
            return int(self.y.max().item()) + 1
        
        def to(self, device):
            self.x = self.x.to(device)
            self.edge_index = self.edge_index.to(device)
            self.y = self.y.to(device)
            self.edge_attr = self.edge_attr.to(device)
            self.train_mask = self.train_mask.to(device)
            self.val_mask = self.val_mask.to(device)
            self.test_mask = self.test_mask.to(device)
            return self
    
    return MockData(config)


# =============================================================================
# SECTION 6: TRAINING PIPELINE
# =============================================================================


def run_training(
    config: NEConsConfig,
    model: nn.Module,
    train_data: Any,
    val_data: Any,
    consensus: Optional[nn.Module],
    device: torch.device,
    logger: logging.Logger
) -> TrainingState:
    """
    Run complete training pipeline.
    
    Args:
        config: Training configuration
        model: Model to train
        train_data: Training data
        val_data: Validation data
        consensus: Optional consensus module
        device: Training device
        logger: Logger instance
    
    Returns:
        Final training state
    """
    logger.info("Starting training pipeline")
    
    # Create training config
    try:
        train_config = TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            warmup_epochs=config.warmup_epochs,
            use_amp=config.use_amp,
            gradient_clip_norm=config.gradient_clip_norm,
            use_class_weights=config.use_class_weights,
            focal_loss=config.focal_loss,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            early_stopping=config.early_stopping,
            patience=config.patience,
            min_delta=config.min_delta,
            monitor_metric=config.monitor_metric,
            simulate_byzantine=config.simulate_byzantine,
            num_edge_nodes=config.num_edge_nodes,
            byzantine_ratio=config.byzantine_ratio,
            attack_type=config.attack_type,
            aggregation_method=config.aggregation_method,
            checkpoint_dir=config.checkpoint_dir,
            save_frequency=config.save_frequency,
            save_best_only=config.save_best_only,
            use_tensorboard=config.use_tensorboard,
            tensorboard_dir=config.tensorboard_dir,
            seed=config.seed
        )
    except NameError:
        logger.warning("Using mock training config")
        train_config = None
    
    # Compute class weights if needed
    class_weights = None
    if config.use_class_weights and hasattr(train_data, 'y'):
        if hasattr(train_data, 'train_mask'):
            labels = train_data.y[train_data.train_mask]
        else:
            labels = train_data.y
        
        class_counts = torch.bincount(labels)
        total = class_counts.sum().float()
        class_weights = total / (len(class_counts) * class_counts.float() + 1e-8)
        class_weights = class_weights.to(device)
        logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Create trainer
    try:
        if config.simulate_byzantine and consensus is not None:
            trainer = DistributedNEConsTrainer(
                model, train_config, consensus, device
            )
        else:
            trainer = NEConsTrainer(
                model, train_config, consensus, device
            )
        
        # Setup and run training
        trainer.setup(train_data, val_data, class_weights)
        state = trainer.train()
        
    except NameError:
        logger.warning("Running simplified training loop")
        state = run_simple_training(
            model, train_data, val_data, config, device, logger
        )
    
    logger.info(f"Training complete. Best epoch: {state.best_epoch}")
    logger.info(f"Best {config.monitor_metric}: {state.best_metric:.4f}")
    
    return state


def run_simple_training(
    model: nn.Module,
    train_data: Any,
    val_data: Any,
    config: NEConsConfig,
    device: torch.device,
    logger: logging.Logger
) -> Any:
    """Simplified training loop for testing."""
    
    @dataclass
    class SimpleState:
        epoch: int = 0
        best_epoch: int = 0
        best_metric: float = 0.0
        train_losses: List[float] = field(default_factory=list)
        val_losses: List[float] = field(default_factory=list)
    
    state = SimpleState()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Move data to device
    train_data = train_data.to(device)
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(train_data.x, train_data.edge_index)
        
        if hasattr(train_data, 'train_mask'):
            loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
        else:
            loss = criterion(out, train_data.y)
        
        loss.backward()
        optimizer.step()
        
        state.train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(train_data.x, train_data.edge_index)
            
            if hasattr(train_data, 'val_mask'):
                val_loss = criterion(out[train_data.val_mask], train_data.y[train_data.val_mask])
                preds = out[train_data.val_mask].argmax(dim=1)
                labels = train_data.y[train_data.val_mask]
            else:
                val_loss = criterion(out, train_data.y)
                preds = out.argmax(dim=1)
                labels = train_data.y
            
            acc = (preds == labels).float().mean().item()
        
        state.val_losses.append(val_loss.item())
        
        if acc > state.best_metric:
            state.best_metric = acc
            state.best_epoch = epoch
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Acc={acc:.4f}")
        
        state.epoch = epoch
        
        # Early stopping
        if config.early_stopping and epoch - state.best_epoch > config.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    return state


# =============================================================================
# SECTION 7: EVALUATION PIPELINE
# =============================================================================


def run_evaluation(
    config: NEConsConfig,
    model: nn.Module,
    test_data: Any,
    consensus: Optional[nn.Module],
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    Args:
        config: Evaluation configuration
        model: Trained model
        test_data: Test data
        consensus: Optional consensus module
        device: Evaluation device
        logger: Logger instance
    
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Starting evaluation pipeline")
    
    results = {}
    
    # Detection evaluation
    logger.info("Running detection evaluation...")
    model.eval()
    
    try:
        metrics = EvaluationMetrics(num_classes=config.output_dim)
    except NameError:
        metrics = None
    
    with torch.no_grad():
        test_data = test_data.to(device)
        logits = model(test_data.x, test_data.edge_index)
        
        if hasattr(test_data, 'test_mask'):
            logits = logits[test_data.test_mask]
            labels = test_data.y[test_data.test_mask]
        else:
            labels = test_data.y
        
        if metrics:
            metrics.update(logits, labels)
            detection_results = metrics.compute()
        else:
            # Simple metrics
            preds = logits.argmax(dim=1)
            detection_results = {
                'accuracy': (preds == labels).float().mean().item(),
                'num_samples': len(labels)
            }
    
    results['detection'] = detection_results
    logger.info(f"Detection results: {detection_results}")
    
    # Byzantine evaluation (if enabled)
    if config.simulate_byzantine and consensus is not None:
        logger.info("Running Byzantine evaluation...")
        
        try:
            eval_config = EvaluationConfig(
                byzantine_ratios=[0.0, 0.1, 0.2, 0.33],
                attack_types=["model_poisoning", "label_flipping", "gradient_scaling"]
            )
            
            evaluator = ByzantineResilienceEvaluator(
                model, consensus, eval_config, device
            )
            
            byzantine_results = evaluator.evaluate(test_data, config.num_edge_nodes)
            results['byzantine'] = byzantine_results
            
        except NameError:
            logger.warning("Byzantine evaluation not available")
    
    return results


# =============================================================================
# SECTION 8: BENCHMARK SUITE
# =============================================================================


def run_benchmark(
    config: NEConsConfig,
    device: torch.device,
    logger: logging.Logger,
    run_all: bool = True
) -> Dict[str, Any]:
    """
    Run complete benchmark suite.
    
    Args:
        config: Configuration
        device: Device
        logger: Logger
        run_all: Whether to run all benchmarks
    
    Returns:
        Complete benchmark results
    """
    logger.info("Starting benchmark suite")
    
    results = {}
    
    # Create model and data
    model = create_model(config, device)
    train_data, val_data, test_data = load_data(config)
    consensus = create_consensus_module(config, device)
    
    # 1. Training benchmark
    logger.info("\n[1/5] Training benchmark")
    start_time = time.time()
    train_state = run_training(
        config, model, train_data, val_data, consensus, device, logger
    )
    training_time = time.time() - start_time
    
    results['training'] = {
        'time_seconds': training_time,
        'epochs': train_state.epoch + 1,
        'best_epoch': train_state.best_epoch,
        'best_metric': train_state.best_metric
    }
    
    # 2. Detection benchmark
    logger.info("\n[2/5] Detection benchmark")
    eval_results = run_evaluation(
        config, model, test_data, consensus, device, logger
    )
    results['detection'] = eval_results.get('detection', {})
    
    # 3. Scalability benchmark
    if run_all:
        logger.info("\n[3/5] Scalability benchmark")
        try:
            scalability_config = EvaluationConfig(
                node_counts=[100, 250, 500, 1000]
            )
            analyzer = ScalabilityAnalyzer(scalability_config)
            
            def consensus_factory(n):
                return NEConsByzantineConsensus(
                    num_nodes=n,
                    hidden_dim=config.hidden_dim,
                    byzantine_threshold=0.33,
                    aggregation_strategy="bulyan"
                )
            
            scalability_results = analyzer.run_scalability_test(
                consensus_factory, config.hidden_dim
            )
            results['scalability'] = scalability_results
            
        except NameError:
            logger.warning("Scalability benchmark not available")
    
    # 4. Communication complexity
    if run_all:
        logger.info("\n[4/5] Communication complexity")
        try:
            comm_analyzer = CommunicationAnalyzer(
                num_nodes=config.num_edge_nodes
            )
            comm_results = comm_analyzer.compare_with_baselines()
            results['communication'] = comm_results
            
        except NameError:
            logger.warning("Communication analysis not available")
    
    # 5. Save results
    logger.info("\n[5/5] Saving results")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")
    
    return results


# =============================================================================
# SECTION 9: TABLE GENERATION
# =============================================================================


def generate_tables(
    results_dir: str,
    output_dir: str,
    logger: logging.Logger
) -> Dict[str, str]:
    """
    Generate LaTeX tables from results.
    
    Args:
        results_dir: Directory containing result files
        output_dir: Output directory for tables
        logger: Logger instance
    
    Returns:
        Dictionary of table names to file paths
    """
    logger.info("Generating LaTeX tables")
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find latest results file
    result_files = list(results_path.glob("*.json"))
    
    if not result_files:
        logger.error("No result files found")
        return {}
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Generate tables
    try:
        latex_gen = LaTeXTableGenerator(str(output_path))
        
        tables = {}
        
        # Detection table
        if 'detection' in results:
            # Format for table generator
            detection_data = {
                'NECons': {k: [v] for k, v in results['detection'].items() if isinstance(v, (int, float))}
            }
            table = latex_gen.generate_detection_table(detection_data)
            tables['detection'] = str(output_path / 'detection.tex')
        
        # Scalability table
        if 'scalability' in results:
            table = latex_gen.generate_scalability_table(results['scalability'])
            tables['scalability'] = str(output_path / 'scalability.tex')
        
        # Communication table
        if 'communication' in results:
            table = latex_gen.generate_communication_table(results['communication'])
            tables['communication'] = str(output_path / 'communication.tex')
        
        logger.info(f"Generated {len(tables)} tables")
        return tables
        
    except NameError:
        logger.warning("LaTeX table generation not available")
        return {}


# =============================================================================
# SECTION 10: CLI ARGUMENT PARSER
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    
    parser = argparse.ArgumentParser(
        description="NECons: Network-aware Edge-based Consensus for Blockchain Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python necons_main.py train --dataset ethereum_s

  # Train with custom config
  python necons_main.py train --config configs/custom.yaml

  # Evaluate pretrained model
  python necons_main.py evaluate --checkpoint best_model.pt

  # Run full benchmark
  python necons_main.py benchmark --full

  # Generate paper tables
  python necons_main.py tables --results-dir results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # =========================================================================
    # Train command
    # =========================================================================
    train_parser = subparsers.add_parser('train', help='Train NECons model')
    
    train_parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    train_parser.add_argument(
        '--dataset', type=str, default='ethereum_s',
        choices=['ethereum_s', 'ethereum_p', 'bitcoin_m', 'bitcoin_l', 'synthetic'],
        help='Dataset to use'
    )
    train_parser.add_argument(
        '--epochs', type=int, default=200,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size'
    )
    train_parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--hidden-dim', type=int, default=256,
        help='Hidden dimension'
    )
    train_parser.add_argument(
        '--num-layers', type=int, default=3,
        help='Number of MGD layers'
    )
    train_parser.add_argument(
        '--byzantine', action='store_true',
        help='Enable Byzantine simulation'
    )
    train_parser.add_argument(
        '--byzantine-ratio', type=float, default=0.33,
        help='Ratio of Byzantine nodes'
    )
    train_parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for checkpoints'
    )
    train_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    # =========================================================================
    # Evaluate command
    # =========================================================================
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    
    eval_parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    eval_parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    eval_parser.add_argument(
        '--dataset', type=str, default='ethereum_s',
        help='Dataset to evaluate on'
    )
    eval_parser.add_argument(
        '--byzantine-ratio', type=float, default=0.0,
        help='Byzantine ratio for evaluation'
    )
    eval_parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for results'
    )
    
    # =========================================================================
    # Benchmark command
    # =========================================================================
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmark suite')
    
    bench_parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    bench_parser.add_argument(
        '--full', action='store_true',
        help='Run full benchmark suite'
    )
    bench_parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory'
    )
    
    # =========================================================================
    # Scalability command
    # =========================================================================
    scale_parser = subparsers.add_parser('scalability', help='Run scalability analysis')
    
    scale_parser.add_argument(
        '--max-nodes', type=int, default=5000,
        help='Maximum number of nodes to test'
    )
    scale_parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory'
    )
    
    # =========================================================================
    # Tables command
    # =========================================================================
    table_parser = subparsers.add_parser('tables', help='Generate LaTeX tables')
    
    table_parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory containing results'
    )
    table_parser.add_argument(
        '--output', type=str, default='results/tables',
        help='Output directory for tables'
    )
    
    # =========================================================================
    # Ablation command
    # =========================================================================
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    
    ablation_parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    ablation_parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory'
    )
    
    # =========================================================================
    # Config command
    # =========================================================================
    config_parser = subparsers.add_parser('config', help='Configuration utilities')
    
    config_parser.add_argument(
        '--create-default', type=str, default=None,
        help='Create default config file at specified path'
    )
    config_parser.add_argument(
        '--show', type=str, default=None,
        help='Show configuration from file'
    )
    
    # =========================================================================
    # Global arguments
    # =========================================================================
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device to use (auto, cuda, cuda:0, cpu)'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--log-dir', type=str, default='results/logs',
        help='Directory for log files'
    )
    
    return parser


# =============================================================================
# SECTION 11: MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for NECons."""
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=args.log_level,
        experiment_name=args.command
    )
    
    # Print header
    print("\n" + "=" * 70)
    print("NECons: Network-aware Edge-based Consensus")
    print("Distributed Blockchain Anomaly Detection")
    print("Target Venue: IEEE TPDS 2026")
    print("=" * 70)
    
    # Print system info
    print_system_info()
    
    # Get device
    device = get_device(args.device)
    
    # Handle commands
    if args.command == 'config':
        handle_config_command(args, logger)
    
    elif args.command == 'train':
        handle_train_command(args, device, logger)
    
    elif args.command == 'evaluate':
        handle_evaluate_command(args, device, logger)
    
    elif args.command == 'benchmark':
        handle_benchmark_command(args, device, logger)
    
    elif args.command == 'scalability':
        handle_scalability_command(args, device, logger)
    
    elif args.command == 'tables':
        handle_tables_command(args, logger)
    
    elif args.command == 'ablation':
        handle_ablation_command(args, device, logger)
    
    else:
        parser.print_help()


def handle_config_command(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Handle config command."""
    if args.create_default:
        create_default_config(args.create_default)
        logger.info(f"Created default config: {args.create_default}")
    
    elif args.show:
        config = load_config(args.show)
        print("\nConfiguration:")
        print(json.dumps(config.to_dict(), indent=2))


def handle_train_command(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger
) -> None:
    """Handle train command."""
    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = NEConsConfig()
    
    # Override with CLI arguments
    config.dataset = args.dataset
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.hidden_dim = args.hidden_dim
    config.num_mgd_layers = args.num_layers
    config.simulate_byzantine = args.byzantine
    config.byzantine_ratio = args.byzantine_ratio
    config.checkpoint_dir = args.checkpoint_dir
    config.seed = args.seed
    
    # Set seeds
    set_seeds(config.seed)
    
    # Create model and data
    model = create_model(config, device)
    train_data, val_data, test_data = load_data(config)
    consensus = create_consensus_module(config, device)
    
    # Run training
    state = run_training(
        config, model, train_data, val_data, consensus, device, logger
    )
    
    # Save final config
    config_path = Path(config.checkpoint_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save_json(str(config_path))
    
    logger.info("Training complete!")


def handle_evaluate_command(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger
) -> None:
    """Handle evaluate command."""
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = NEConsConfig()
    
    config.dataset = args.dataset
    config.byzantine_ratio = args.byzantine_ratio
    config.output_dir = args.output
    
    # Load model
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from: {args.checkpoint}")
    
    # Load data
    _, _, test_data = load_data(config)
    
    # Create consensus if needed
    consensus = create_consensus_module(config, device) if args.byzantine_ratio > 0 else None
    
    # Run evaluation
    results = run_evaluation(
        config, model, test_data, consensus, device, logger
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_file}")


def handle_benchmark_command(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger
) -> None:
    """Handle benchmark command."""
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = NEConsConfig()
    
    config.output_dir = args.output
    
    # Set seeds
    set_seeds(config.seed)
    
    # Run benchmark
    results = run_benchmark(config, device, logger, run_all=args.full)
    
    logger.info("Benchmark complete!")


def handle_scalability_command(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger
) -> None:
    """Handle scalability command."""
    logger.info(f"Running scalability analysis up to {args.max_nodes} nodes")
    
    try:
        config = EvaluationConfig(
            node_counts=[100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]
        )
        config.node_counts = [n for n in config.node_counts if n <= args.max_nodes]
        
        analyzer = ScalabilityAnalyzer(config)
        
        def consensus_factory(n):
            return NEConsByzantineConsensus(
                num_nodes=n,
                hidden_dim=256,
                byzantine_threshold=0.33,
                aggregation_strategy="bulyan"
            )
        
        results = analyzer.run_scalability_test(consensus_factory, hidden_dim=256)
        
        # Print results
        print("\nScalability Results:")
        print("-" * 60)
        print(f"{'Nodes':>8} {'Time (ms)':>12} {'Accuracy':>10} {'TPS':>10} {'Success':>8}")
        print("-" * 60)
        
        for r in results:
            success_str = '✓' if r.get('success', False) else '✗'
            print(f"{r['node_count']:>8} {r['consensus_time_ms']:>12.1f} {r['accuracy']:>10.3f} {r['throughput_tps']:>10.1f} {success_str:>8}")
        
        # Save results
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"scalability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
    except NameError:
        logger.error("Scalability analysis requires NECons modules")


def handle_tables_command(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Handle tables command."""
    tables = generate_tables(args.results_dir, args.output, logger)
    
    if tables:
        print("\nGenerated tables:")
        for name, path in tables.items():
            print(f"  {name}: {path}")


def handle_ablation_command(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger
) -> None:
    """Handle ablation command."""
    logger.info("Running ablation study")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = NEConsConfig()
    
    # Create full model
    model = create_model(config, device)
    
    # Load data
    _, _, test_data = load_data(config)
    
    try:
        # Create ablation runner
        ablation = AblationStudyRunner(model, device)
        
        # Create variants
        def model_factory(**kwargs):
            merged_kwargs = {
                'node_input_dim': config.node_input_dim,
                'hidden_dim': config.hidden_dim,
                'output_dim': config.output_dim,
                'num_mgd_layers': config.num_mgd_layers,
                'num_heads': config.num_heads,
                'dropout': config.dropout,
                'network_aware': config.network_aware,
                'use_edge2seq': config.use_edge2seq
            }
            merged_kwargs.update(kwargs)
            return NEConsGNN(**merged_kwargs)
        
        ablation.create_ablation_variants(model_factory, {})
        
        # Run ablation
        results = ablation.run_ablation(test_data, num_runs=config.num_eval_runs)
        
        # Print results
        print("\nAblation Study Results:")
        print("-" * 60)
        print(f"{'Variant':<25} {'F1':>10} {'AUC-ROC':>10} {'ΔF1':>10}")
        print("-" * 60)
        
        full_f1 = results.get('Full NECons', {}).get('f1', 0)
        
        for variant, metrics in results.items():
            f1 = metrics.get('f1', 0)
            auc = metrics.get('auc_roc', 0)
            delta = (full_f1 - f1) / full_f1 * 100 if full_f1 > 0 and variant != 'Full NECons' else 0
            delta_str = f"-{delta:.1f}%" if delta > 0 else "-"
            print(f"{variant:<25} {f1:>10.3f} {auc:>10.3f} {delta_str:>10}")
        
        # Save results
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
    except NameError:
        logger.error("Ablation study requires NECons modules")


# =============================================================================
# SECTION 12: ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    main()
