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
    
    # Create only training dataset for now
    train_dataset = ImageNetLocalDataset(
        root_dir="/media/data/ILSVRC/Data/CLS-LOC",
        split="train",
        transform=get_transforms(config, is_train=True)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    # Create model
    model = ResNet50Module(config)

    # Modify callbacks to remove validation-dependent ones
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            monitor='train_loss',  # Changed to monitor training loss instead
            mode='min',
            save_top_k=1,
            filename='resnet50-{epoch:02d}-{train_loss:.2f}',
            dirpath=config.checkpoint_dir
        )
    ]

    # Initialize trainer without validation
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        devices=1,
        accelerator="gpu",
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
        strategy='ddp_find_unused_parameters_false'
    )

    # Train model without validation
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
