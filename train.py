import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from src.config import TrainingConfig
from src.dataset import ImageNetLocalDataset, get_transforms
from src.model import ResNet50Module
import torch
import os


class MetricsLoggerCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_acc1 = metrics.get("val_acc1", "N/A")
        val_acc5 = metrics.get("val_acc5", "N/A")
        val_loss = metrics.get("val_loss", "N/A")
        print(f"Epoch {epoch} - val_acc1: {val_acc1}, val_acc5: {val_acc5}, val_loss: {val_loss}")


def main():
    # Initialize config
    config = TrainingConfig()

    # Path to the root directory of ImageNet dataset
    imagenet_root = "/mnt/imagenet"  # Change this to the mount path of your volume

    # Create datasets
    train_dataset = ImageNetLocalDataset(
        root_dir="/media/data/ILSVRC/Data/CLS-LOC",
        split="train",
        transform=get_transforms(config, is_train=True)
    )

    val_dataset = ImageNetLocalDataset(
        root_dir="/media/data/ILSVRC/Data/CLS-LOC",
        split="val",
        transform=get_transforms(config, is_train=False)
    )

    # Print dataset sizes for debugging
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True  # Ensures no incomplete batches
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False  # Validation can handle incomplete batches
    )

    # Debugging: Print actual batch size
    for images, targets in train_loader:
        print(f"Batch size during training: {images.size(0)}")
        break  # Check only the first batch

    # Create model
    model = ResNet50Module(config)

    # Specify the directory for saving checkpoints
    checkpoint_dir = config.checkpoint_dir  # This uses the value from config

    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor="val_acc1", patience=10, mode="max"),  # Increased patience
        ModelCheckpoint(
            monitor='val_acc1',
            mode='max',
            save_top_k=1,
            filename='resnet50-{epoch:02d}-{val_acc1:.2f}',
            dirpath=checkpoint_dir  # Specify the directory to save the checkpoints
        ),
        LearningRateMonitor(logging_interval='step'),
        MetricsLoggerCallback()  # Custom callback to log and print metrics
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Simulate larger batch size
        check_val_every_n_epoch=1,  # Validate every epoch
        devices=1,  # Use 1 GPU
        accelerator="gpu"
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Optionally, log final validation accuracy after training
    metrics = trainer.callback_metrics
    print(f"Final validation accuracy: {metrics.get('val_acc1', 'N/A')}")


if __name__ == "__main__":
    main()
