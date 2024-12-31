from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Training params
    batch_size: int = 32  # Consider increasing if memory allows
    num_workers: int = 8
    max_epochs: int = 90  # Recommend increasing for ImageNet training
    precision: str = "16-mixed"
    
    # Optimizer params
    base_lr: float = 0.1 * (batch_size / 256)  # This is good
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Data params
    image_size: int = 224
    num_classes: int = 1000
    
    # Training techniques
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Scheduler params
    warmup_epochs: int = 5  # Increase warmup epochs for better stability
    min_lr: float = 1e-6  # Slightly lower minimum learning rate
    
    # Save model path (you might also want to keep this configurable in your main script)
    checkpoint_dir: str = "/media/data/saved_models"
