"""Configuration dataclasses for AlignIt using draccus."""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and loading."""

    path: str = field(
        default="./data/duck", metadata={"help": "Path to the dataset directory"}
    )


@dataclass
class ModelConfig:
    """Configuration for AlignNet model."""

    backbone: str = field(
        default="efficientnet_b0",
        metadata={"help": "Backbone architecture: 'efficientnet_b0' or 'resnet18'"},
    )
    backbone_weights: str = field(
        default="DEFAULT", metadata={"help": "Backbone weights: 'DEFAULT' or None"}
    )
    use_vector_input: bool = field(
        default=False, metadata={"help": "Whether to use vector input"}
    )
    fc_layers: List[int] = field(
        default_factory=lambda: [256, 128],
        metadata={"help": "Hidden layer sizes for FC head"},
    )
    vector_hidden_dim: int = field(
        default=64, metadata={"help": "Output dimension of vector MLP"}
    )
    output_dim: int = field(
        default=9,
        metadata={"help": "Final output dimension (3 translation + 6 rotation)"},
    )
    feature_agg: str = field(
        default="mean", metadata={"help": "Feature aggregation method: 'mean' or 'max'"}
    )
    path: str = field(
        default="alignnet_model.pth",
        metadata={"help": "Path to save/load trained model"},
    )


@dataclass
class TrajectoryConfig:
    """Configuration for spiral trajectory generation."""

    z_step: float = field(
        default=0.002, metadata={"help": "Z step size for spiral trajectory"}
    )
    radius_step: float = field(
        default=0.001, metadata={"help": "Radius step size for spiral trajectory"}
    )
    num_steps: int = field(
        default=50, metadata={"help": "Number of steps in spiral trajectory"}
    )
    cone_angle: float = field(default=30.0, metadata={"help": "Cone angle in degrees"})
    visible_sweep: float = field(
        default=60.0, metadata={"help": "Visible sweep angle in degrees"}
    )
    viewing_angle_offset: float = field(
        default=-120.0, metadata={"help": "Viewing angle offset in degrees"}
    )
    angular_resolution: float = field(
        default=10.0, metadata={"help": "Angular resolution in degrees"}
    )
    include_cone_poses: bool = field(
        default=False, metadata={"help": "Include cone poses in trajectory"}
    )
    lift_height_before_spiral: float = field(
        default=0.01, metadata={"help": "Lift height before spiral in meters"}
    )


@dataclass
class RecordConfig:
    """Configuration for data recording."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    episodes: int = field(default=10, metadata={"help": "Number of episodes to record"})
    lin_tol_alignment: float = field(
        default=0.015, metadata={"help": "Linear tolerance for alignment servo"}
    )
    ang_tol_alignment: float = field(
        default=0.015, metadata={"help": "Angular tolerance for alignment servo"}
    )
    lin_tol_trajectory: float = field(
        default=0.05, metadata={"help": "Linear tolerance for trajectory servo"}
    )
    ang_tol_trajectory: float = field(
        default=0.05, metadata={"help": "Angular tolerance for trajectory servo"}
    )


@dataclass
class TrainConfig:
    """Configuration for model training."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = field(default=8, metadata={"help": "Training batch size"})
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Learning rate for optimizer"}
    )
    epochs: int = field(default=100, metadata={"help": "Number of training epochs"})
    test_size: float = field(
        default=0.2, metadata={"help": "Fraction of data for testing"}
    )
    random_seed: int = field(
        default=42, metadata={"help": "Random seed for train/test split"}
    )


@dataclass
class InferConfig:
    """Configuration for inference/alignment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    start_pose_xyz: List[float] = field(
        default_factory=lambda: [0.33, 0.0, 0.35],
        metadata={"help": "Starting pose XYZ coordinates"},
    )
    start_pose_rpy: List[float] = field(
        default_factory=lambda: [np.pi, 0.0, 0.0],
        metadata={"help": "Starting pose RPY angles"},
    )
    lin_tolerance: float = field(
        default=2e-3, metadata={"help": "Linear tolerance for convergence (meters)"}
    )
    ang_tolerance: float = field(
        default=2, metadata={"help": "Angular tolerance for convergence (degrees)"}
    )
    max_iterations: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum iterations before stopping (None = infinite)"},
    )
    debug_output: bool = field(
        default=True, metadata={"help": "Print debug information during inference"}
    )
    debouncing_count: int = field(
        default=5,
        metadata={"help": "Number of iterations within tolerance before stopping"},
    )


@dataclass
class VisualizeConfig:
    """Configuration for dataset visualization."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    share: bool = field(default=False, metadata={"help": "Create a public Gradio link"})
    server_name: Optional[str] = field(
        default=None, metadata={"help": "Server name for Gradio interface"}
    )
    server_port: Optional[int] = field(
        default=None, metadata={"help": "Server port for Gradio interface"}
    )
