"""
NECons: Network-aware Edge-based Consensus for Distributed Blockchain Anomaly Detection

This package implements the NECons framework for Byzantine-resilient distributed
blockchain anomaly detection using network-aware graph neural networks and
formal consensus protocols.

Modules:
    necons_core: Core architectures, protocols, and algorithms
    necons_data: Data loading, preprocessing, and augmentation
    necons_train: Training, optimization, and Byzantine simulation
    necons_eval: Evaluation, baselines, and statistical analysis
    necons_main: CLI and experiment orchestration

Target Venue: IEEE Transactions on Parallel and Distributed Systems (TPDS) 2026

Author: BlockchainLab
"""

from .necons_core import (
    NEConsGNN,
    NetworkAwareMGD,
    MGDLayer,
    DistributedEdge2Seq,
    NEConsByzantineConsensus,
    CrossChainSyncProtocol,
    NetworkState,
    ConsensusPhase,
    AttackType,
    ChainType,
    KrumAggregation,
    TrimmedMeanAggregation,
    BulyanAggregation,
    TrustWeightedAggregation,
    ConsensusVerifier,
    NodeScalabilityAnalyzer,
    CommunicationAnalyzer,
)

from .necons_data import (
    BlockchainGraphData,
    SyntheticBlockchainDataset,
    BlockchainDatasetLoader,
    DatasetType,
    DatasetConfig,
    TransactionFeatureExtractor,
    TransactionSequenceBuilder,
    DatasetStatistics,
    EdgeDropout,
    NodeDropout,
    FeatureMasking,
    SubgraphSampling,
    GraphMixup,
    create_augmentation,
    AugmentationType,
)

from .necons_train import (
    NEConsTrainer,
    DistributedNEConsTrainer,
    TrainingConfig,
    TrainingState,
    FocalLoss,
    LabelSmoothingLoss,
    MetricsComputer,
    EarlyStopping,
    CheckpointManager,
    ByzantineAttackSimulator,
    train_necons,
)

from .necons_eval import (
    EvaluationConfig,
    EvaluationMetrics,
    BaselineComparator,
    ByzantineResilienceEvaluator,
    ScalabilityAnalyzer,
    StatisticalTester,
    AblationStudyRunner,
    LaTeXTableGenerator,
    ResultsFormatter,
    ExperimentRunner,
    VanillaGATBaseline,
    VanillaGCNBaseline,
    GraphSAGEBaseline,
    PBFTBaseline,
    RaftBaseline,
    HotStuffBaseline,
)

__version__ = "1.0.0"
__author__ = "BlockchainLab"
__target_venue__ = "IEEE TPDS 2026"

__all__ = [
    # Core
    "NEConsGNN",
    "NetworkAwareMGD",
    "MGDLayer",
    "DistributedEdge2Seq",
    "NEConsByzantineConsensus",
    "CrossChainSyncProtocol",
    "NetworkState",
    "ConsensusPhase",
    "AttackType",
    "ChainType",
    "KrumAggregation",
    "TrimmedMeanAggregation",
    "BulyanAggregation",
    "TrustWeightedAggregation",
    "ConsensusVerifier",
    "NodeScalabilityAnalyzer",
    "CommunicationAnalyzer",
    # Data
    "BlockchainGraphData",
    "SyntheticBlockchainDataset",
    "BlockchainDatasetLoader",
    "DatasetType",
    "DatasetConfig",
    "TransactionFeatureExtractor",
    "TransactionSequenceBuilder",
    "DatasetStatistics",
    "EdgeDropout",
    "NodeDropout",
    "FeatureMasking",
    "SubgraphSampling",
    "GraphMixup",
    "create_augmentation",
    "AugmentationType",
    # Training
    "NEConsTrainer",
    "DistributedNEConsTrainer",
    "TrainingConfig",
    "TrainingState",
    "FocalLoss",
    "LabelSmoothingLoss",
    "MetricsComputer",
    "EarlyStopping",
    "CheckpointManager",
    "ByzantineAttackSimulator",
    "train_necons",
    # Evaluation
    "EvaluationConfig",
    "EvaluationMetrics",
    "BaselineComparator",
    "ByzantineResilienceEvaluator",
    "ScalabilityAnalyzer",
    "StatisticalTester",
    "AblationStudyRunner",
    "LaTeXTableGenerator",
    "ResultsFormatter",
    "ExperimentRunner",
    "VanillaGATBaseline",
    "VanillaGCNBaseline",
    "GraphSAGEBaseline",
    "PBFTBaseline",
    "RaftBaseline",
    "HotStuffBaseline",
]
