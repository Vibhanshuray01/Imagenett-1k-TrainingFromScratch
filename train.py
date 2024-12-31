import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from src.config import TrainingConfig
from src.dataset import ImageNetDataset, get_transforms
from src.model import ResNet50
import torch

def main():
    # Enable tensor cores
    torch.set_float32_matmul_precision('high')
    
    # Initialize config
    config = TrainingConfig()
    
    # Create datasets
    train_dataset = ImageNetDataset(
        root_dir=config.data_dir,
        split="train",
        transform=get_transforms(config.image_size, is_training=True)
    )
    
    val_dataset = ImageNetDataset(
        root_dir=config.data_dir,
        split="val",
        transform=get_transforms(config.image_size, is_training=False)
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
    model = ResNet50(config)
    
    # Setup logging
    logger = TensorBoardLogger("logs", name="resnet50")
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.output_dir,
            filename='resnet50-{epoch:02d}-{val_acc1:.2f}',
            monitor='val_acc1',
            mode='max',
            save_top_k=3
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar()
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        deterministic=True,
        log_every_n_steps=50
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
