import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from src.config import TrainingConfig
from src.dataset import ImageNetLocalDataset, get_transforms
from src.model import ResNet50Module
import torch
import os
import torch.distributed as dist
from pytorch_lightning.callbacks import RichProgressBar


class MetricsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch}: train_loss = {trainer.callback_metrics['train_loss']:.3f}")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(f"Validation: loss = {metrics['val_loss']:.3f}, acc@1 = {metrics['val_acc1']:.2f}%, acc@5 = {metrics['val_acc5']:.2f}%")


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
        ModelCheckpoint(
            monitor='val_acc1',
            mode='max',
            save_top_k=3,
            filename='resnet50-epoch{epoch:02d}-val_acc{val_acc1:.2f}',
            dirpath=config.checkpoint_dir
        ),
        LearningRateMonitor(logging_interval='step'),
        MetricsCallback(),
        RichProgressBar()
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
