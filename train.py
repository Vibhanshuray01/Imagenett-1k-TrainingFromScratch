import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from src.config import TrainingConfig
from src.dataset import ImageNetLocalDataset, get_transforms
from src.model import ResNet50Module
import torch
import os

def main():
    # Set tensor core precision
    torch.set_float32_matmul_precision('high')
    
    # Initialize config
    config = TrainingConfig()
    
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

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
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

    # Setup logging
    logger = TensorBoardLogger("logs", name="resnet50")

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc1',
            mode='max',
            save_top_k=3,
            filename='resnet50-epoch{epoch:02d}-val_acc{val_acc1:.2f}',
            dirpath=config.checkpoint_dir
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar()
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        devices=1,
        accelerator="gpu",
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
        strategy='ddp_find_unused_parameters_false'
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
