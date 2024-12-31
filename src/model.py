import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

class ResNet50(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = resnet50(weights=None)  # Training from scratch
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        acc1, acc5 = self._accuracy(outputs, targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc1', acc1, prog_bar=True, on_epoch=True)
        self.log('train_acc5', acc5, prog_bar=True, on_epoch=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy
        acc1, acc5 = self._accuracy(outputs, targets)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc1', acc1, prog_bar=True, sync_dist=True)
        self.log('val_acc5', acc5, prog_bar=True, sync_dist=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.config.base_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.base_lr,
            epochs=self.config.max_epochs,
            steps_per_epoch=self.trainer.estimated_stepping_batches // self.config.max_epochs,
            pct_start=self.config.warmup_epochs / self.config.max_epochs,
            div_factor=10,
            final_div_factor=100,
        )
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def _accuracy(self, output, target, topk=(1, 5)):
        """Computes the accuracy over the k top predictions"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
