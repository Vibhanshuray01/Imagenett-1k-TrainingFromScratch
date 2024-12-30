import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from src.config import TrainingConfig
from src.dataset import ImageNetLocalDataset, get_transforms
from src.model import ResNet50Module
import torch

def main():
    # Initialize config
    config = TrainingConfig()

    # Path to the root directory of ImageNet dataset
    imagenet_root = "/mnt/imagenet"  # Change this to the mount path of your volume

    # Create datasets
    train_dataset = ImageNetLocalDataset(
        root_dir="/media/data/ILSVRC/Data/CLS-LOC/train",
        split="train",
        transform=get_transforms(config, is_train=True)
    )

    val_dataset = ImageNetLocalDataset(
        root_dir="/media/data/ILSVRC/Data/CLS-LOC/val",
        split="val",
        transform=get_transforms(config, is_train=False)
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False
    )

    # Create model
    model = ResNet50Module(config)

    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor="val_acc1", patience=5, mode="max"),  # Early stopping
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
        accumulate_grad_batches=2,  # Simulate larger batch size
        check_val_every_n_epoch=2,  # Validate every 2 epochs
        devices=1,  # Use 1 GPU
        accelerator="gpu"
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
