import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.config import TrainingConfig
from src.dataset import ImageNetLocalDataset, get_transforms  # Update import
from src.model import ResNet50Module
import torch

def main():
    # Initialize config
    config = TrainingConfig()

    # Path to the root directory of ImageNet dataset
    imagenet_root = "/mnt/imagenet"  # Change this to the mount path of your volume

    # Create datasets
    train_dataset = ImageNetLocalDataset(
        root_dir=imagenet_root,
        split="train",
        transform=get_transforms(config, is_train=True)
    )

    val_dataset = ImageNetLocalDataset(
        root_dir=imagenet_root,
        split="val",  # Validation folder
        transform=get_transforms(config, is_train=False)
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True  # Shuffle for training
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False  # No shuffle for validation
    )

    # Create model
    model = ResNet50Module(config)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc1',
            mode='max',
            save_top_k=1,
            filename='resnet50-{epoch:02d}-{val_acc1:.2f}'
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        deterministic=False,
        devices=1,  # Use 1 GPU
        accelerator="gpu"  # Specify GPU
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
