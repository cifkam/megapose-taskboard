import sys
import numpy as np
import torchvision
import pandas as pd

import lightning.pytorch as pl

import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import happypose.toolbox.utils.tensor_collection as tc
from happypose.pose_estimators.megapose.config import EXP_DIR
from happypose.toolbox.utils.distributed import reduce_dict
from happypose.toolbox.inference.utils import filter_detections, add_instance_id

def load_detector(exp_id: str):
    model = MaskRCNN.load_from_checkpoint(str(EXP_DIR / exp_id / "model.ckpt"))
    model.eval()
    return model

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
    
    @torch.no_grad()
    def get_detections(self, observation,
        detection_th = None,
        output_masks = False,
        mask_th = 0.8,
        one_instance_per_class = False
    ):
        images = observation.images
        outputs_ = self(images)

        def category_id_to_label(category_id):
            return "obj_"+str(category_id).zfill(6)

        infos = []
        bboxes = []
        masks = []
        for n, outputs_n in enumerate(outputs_):
            outputs_n["labels"] = [
                category_id_to_label(category_id.item())
                for category_id in outputs_n["labels"]
            ]
            for obj_id in range(len(outputs_n["boxes"])):
                bbox = outputs_n["boxes"][obj_id]
                info = {
                    "batch_im_id": n,
                    "label": outputs_n["labels"][obj_id],
                    "score": outputs_n["scores"][obj_id].item(),
                }
                mask = outputs_n["masks"][obj_id, 0] > mask_th
                bboxes.append(torch.as_tensor(bbox))
                masks.append(torch.as_tensor(mask))
                infos.append(info)

        if len(bboxes) > 0:
            bboxes = torch.stack(bboxes).to(self.device).float()
            masks = torch.stack(masks).to(self.device)
        else:
            infos = {"score": [], "label": [], "batch_im_id": []}
            bboxes = torch.empty(0, 4).to(self.device).float()
            masks = torch.empty(
                0,
                images.shape[1],
                images.shape[2],
                dtype=torch.bool,
            ).to(self.device)

        detections = tc.PandasTensorCollection(
            infos=pd.DataFrame(infos),
            bboxes=bboxes,
        )
        if output_masks:
            detections.register_tensor("masks", masks)
        if detection_th is not None:
            keep = np.where(detections.infos["score"] > detection_th)[0]
            detections = detections[keep]

        # Keep only the top-detection for each class label
        if one_instance_per_class:
            detections = filter_detections(
                detections,
                one_instance_per_class=True,
            )

        # Add instance_id column to dataframe
        # Each detection is now associated with an `instance_id` that
        # identifies multiple instances of the same object
        detections = add_instance_id(detections)
        return detections
    