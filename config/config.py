import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    
    datasets: dict = field(default_factory=lambda: {
        "asl_alphabet": "asl_alphabet/asl_alphabet_train",
        "asl_hg_raw": "asl_hg_raw/asl_dataset",
        "asl_hg_processed": "asl_hg_processed/asl_processed/train",
        "mendeley_210k_raw": "mendeley_210k/Root/Type_01_(Raw_Gesture)",
        "mendeley_210k_keypoint": "mendeley_210k/Root/Type_02_(Keypoint Based)",
        "sign_mnist": "sign_mnist",
        "signalphaset_static": "signalphaset_static/SignAlphaSet",
    })
    
    image_size: int = 224
    num_channels: int = 3
    
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    
    batch_size: int = 256
    num_workers: int = 16  
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    model_name: str = "vit_large_patch16_224"
    
    patch_size: int = 16
    hidden_dim: int = 1024
    
    num_layers: int = 24
    num_heads: int = 16
    mlp_dim: int = 4096  # 4 * hidden_dim
    
    num_classes: int = 26  # A-Z
    
    dropout: float = 0.0 
    drop_path_rate: float = 0.2 
    attention_dropout: float = 0.0
    
    pretrained: bool = True
    pretrained_weights: str = "imagenet21k" 
    
    use_fp16: bool = False 
    use_bf16: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 100
    seed: int = 42
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    scheduler: str = "cosine" 
    warmup_epochs: int = 10
    warmup_lr_init: float = 1e-7
    
    gradient_clip_max_norm: float = 1.0
    gradient_accumulation_steps: int = 1  
    
    label_smoothing: float = 0.1
    
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    save_best_only: bool = True
    save_every_n_epochs: int = 10

    log_interval: int = 50  # Log every N batches
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    resume_from_checkpoint: Optional[str] = None


@dataclass
class AugmentationConfig:

    random_crop: bool = True
    crop_scale: Tuple[float, float] = (0.8, 1.0)
    crop_ratio: Tuple[float, float] = (0.9, 1.1)
    
    random_horizontal_flip: float = 0.5
    random_rotation_degrees: int = 15
    
    random_affine: bool = True
    affine_translate: Tuple[float, float] = (0.1, 0.1)
    affine_scale: Tuple[float, float] = (0.9, 1.1)
    affine_shear: int = 10
    
    color_jitter: bool = True
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.1
    
    gaussian_blur: bool = True
    blur_kernel_size: int = 3
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    
    random_erasing: bool = True
    erasing_probability: float = 0.25
    erasing_scale: Tuple[float, float] = (0.02, 0.2)
    
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_cutmix_prob: float = 0.5


@dataclass
class OutputConfig:
    """Output paths and settings."""

    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    logs_dir: str = "./outputs/logs"
    tensorboard_dir: str = "./outputs/tensorboard"

    export_onnx: bool = True
    export_torchscript: bool = True
    
    experiment_name: str = "signlex_vit_large"


@dataclass
class Config:
    """Master configuration combining all configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        """Create output directories."""
        for dir_path in [
            self.output.output_dir,
            self.output.checkpoint_dir,
            self.output.logs_dir,
            self.output.tensorboard_dir,
            self.data.processed_data_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "augmentation": self.augmentation.__dict__,
            "output": self.output.__dict__,
        }

ASL_LABELS = {
    'A': 0,  'B': 1,  'C': 2,  'D': 3,  'E': 4,
    'F': 5,  'G': 6,  'H': 7,  'I': 8,  'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
    'Z': 25
}

ASL_LABELS_INV = {v: k for k, v in ASL_LABELS.items()}

NUM_CLASSES = len(ASL_LABELS)


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


def get_b200_optimized_config() -> Config:
    config = Config()
    
    config.data.batch_size = 512
    config.data.num_workers = 24
    
    config.model.use_bf16 = True
    config.model.use_fp16 = False
    
    config.training.learning_rate = 2e-4
    
    return config


if __name__ == "__main__":
    # Print config for verification
    config = get_config()
    print("=== SignLex ViT-Large Configuration ===")
    print(f"\nModel: {config.model.model_name}")
    print(f"  - Hidden dim: {config.model.hidden_dim}")
    print(f"  - Layers: {config.model.num_layers}")
    print(f"  - Heads: {config.model.num_heads}")
    print(f"  - MLP dim: {config.model.mlp_dim}")
    print(f"  - Patch size: {config.model.patch_size}")
    print(f"  - Classes: {config.model.num_classes}")
    print(f"\nTraining:")
    print(f"  - Epochs: {config.training.epochs}")
    print(f"  - Batch size: {config.data.batch_size}")
    print(f"  - Learning rate: {config.training.learning_rate}")
    print(f"  - Weight decay: {config.training.weight_decay}")
    print(f"\nData:")
    print(f"  - Image size: {config.data.image_size}")
    print(f"  - Train/Val/Test: {config.data.train_ratio}/{config.data.val_ratio}/{config.data.test_ratio}")
