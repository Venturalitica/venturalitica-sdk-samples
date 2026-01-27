
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as L
import torchmetrics
from typing import Optional, List

class MultiTaskResNet(L.LightningModule):
    def __init__(self, lds_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['lds_weights'])
        
        # Dual Backbone (MiVOLO inspired)
        self.face_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.body_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        num_features = self.face_backbone.fc.in_features
        self.face_backbone.fc = nn.Identity() 
        self.body_backbone.fc = nn.Identity()
        
        # Simple Concat Fusion (to be upgraded to Cross-Attention if needed)
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Heads
        self.fc_gender = nn.Linear(num_features, 2)
        self.fc_race = nn.Linear(num_features, 7)
        self.fc_age = nn.Linear(num_features, 9)
        
        # Loss Functions
        from venturalitica.fairness.multiclass import GroupFairnessLoss
        
        self.register_buffer('lds_weights', lds_weights if lds_weights is not None else torch.ones(9))
        self.criterion_age = nn.CrossEntropyLoss(weight=self.lds_weights)
        
        base_criterion = nn.CrossEntropyLoss(reduction='none') 
        self.criterion_gender_fair = GroupFairnessLoss(
            base_criterion=base_criterion, 
            protected_attr_key='gender_str',
            alpha=0.1
        )
        self.criterion_std = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc_gender = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.train_acc_race = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.train_acc_age = torchmetrics.Accuracy(task="multiclass", num_classes=9)
        
        self.val_acc_gender = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc_race = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.val_acc_age = torchmetrics.Accuracy(task="multiclass", num_classes=9)
        
    def forward(self, face_img, body_img=None):
        if body_img is None:
            body_img = face_img # Fallback if body not available
            
        f_feat = self.face_backbone(face_img)
        b_feat = self.body_backbone(body_img)
        
        combined = torch.cat([f_feat, b_feat], dim=1)
        features = self.fusion(combined)
        
        return {
            'gender': self.fc_gender(features),
            'race': self.fc_race(features),
            'age': self.fc_age(features)
        }

    def training_step(self, batch, batch_idx):
        x_face, labels, meta = batch
        x_body = meta.get('body_img', x_face) 
        
        logits = self(x_face, x_body)
        
        # Gender use In-processing Fairness Loss
        loss_gender = self.criterion_gender_fair(logits['gender'], labels['gender'], meta)
        loss_race = self.criterion_std(logits['race'], labels['race'])
        loss_age = self.criterion_age(logits['age'], labels['age'])
        
        total_loss = loss_gender + loss_race + loss_age
        
        # Diagnostics
        acc_g = self.train_acc_gender(logits['gender'], labels['gender'])
        acc_r = self.train_acc_race(logits['race'], labels['race'])
        acc_a = self.train_acc_age(logits['age'], labels['age'])
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("acc_g", acc_g, on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc_r", acc_r, on_step=True, on_epoch=False, prog_bar=True)
        self.log("acc_a", acc_a, on_step=True, on_epoch=False, prog_bar=True)

        # Gradient Norm Tracking (Andrew's Tip #11: Monitor health)
        if batch_idx % 10 == 0:
            total_grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            self.log("grad_norm", total_grad_norm ** 0.5, on_step=True, on_epoch=False)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x_face, labels, meta = batch
        x_body = meta.get('body_img', x_face)
        
        logits = self(x_face, x_body)
        
        # Update metrics
        self.val_acc_gender(logits['gender'], labels['gender'])
        self.val_acc_race(logits['race'], labels['race'])
        self.val_acc_age(logits['age'], labels['age'])
        
        self.log("val_acc_g", self.val_acc_gender, on_epoch=True, prog_bar=True)
        self.log("val_acc_r", self.val_acc_race, on_epoch=True, prog_bar=True)
        self.log("val_acc_a", self.val_acc_age, on_epoch=True, prog_bar=True)
        
        return {
            "preds_gender": torch.argmax(logits['gender'], dim=1), 
            "targets_gender": labels['gender'],
            "preds_race": torch.argmax(logits['race'], dim=1),
            "targets_race": labels['race'],
            "preds_age": torch.argmax(logits['age'], dim=1),
            "targets_age": labels['age'],
            "race": meta['race_str'], 
            "gender": meta['gender_str'],
            "age_str": meta['age_str']
        }

    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(), lr=0.0001)
