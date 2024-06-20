import sys
import numpy as np
import torchvision
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from happypose.pose_estimators.megapose.config import LOCAL_DATA_DIR, EXP_DIR
from happypose.toolbox.utils.distributed import reduce_dict


class MaskRCNN(pl.LightningModule):
    @staticmethod
    def _get_detector_model(num_classes: int):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)#weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        return model

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = MaskRCNN._get_detector_model(num_classes)
        #from src.mask_rcnn import DetectorMaskRCNN
        #self.model = DetectorMaskRCNN(input_resize=(480,640), n_classes=num_classes)
        self.automatic_optimization = True
        self.save_hyperparameters()

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        self.model.train()
        images, targets = batch
        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = self.model(images, targets)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if np.isinf(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        self.log(
            "train_loss",
            loss_value,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            batch_size=len(images),
            sync_dist=True,
        )

        return losses_reduced

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        self.model.eval()

        if np.isinf(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        self.log(
            "val_loss",
            loss_value,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
            batch_size=len(images),
            sync_dist=True,
        )
        return losses_reduced

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    