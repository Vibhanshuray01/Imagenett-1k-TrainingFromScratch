from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data paths
    data_dir: str = "/media/data/ILSVRC/Data/CLS-LOC"
    output_dir: str = "/media/data/saved_models"
    
    # Training parameters
    batch_size: int = 64  # Adjust based on GPU memory
    num_workers: int = 8
    max_epochs: int = 90
    
    # Model parameters
    num_classes: int = 1000
    image_size: int = 224
    
    # Optimizer parameters
    base_lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Training techniques
    precision: str = "16-mixed"
    label_smoothing: float = 0.1
