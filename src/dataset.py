import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader
from src.transforms import GaussianNoise

from happypose.pose_estimators.megapose.config import LOCAL_DATA_DIR
from happypose.toolbox.datasets.bop_scene_dataset import BOPDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def make_train_val_datasets(ds_name, split, label_format, train_test_ratio=0.85):
    ds = make_dataset(ds_name, split, label_format)

    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    i = int(train_test_ratio*len(ds))

    train_ds = Subset(ds, indices[:i])
    val_ds = Subset(ds, indices[i:])
    
    return train_ds, val_ds
    
def make_dataset(ds_name, split, label_format):
    ds = BOPDataset(LOCAL_DATA_DIR / ds_name, label_format=label_format, split=split)
    return DetectionDataset(ds)

    
class DetectionDataset(torch.utils.data.Dataset):
    class Iterator:
        def __init__(self, ds):
            self.ds = ds
            self.idx = -1
        def __iter__(self):
            return self
        def __next__(self):
            self.idx += 1
            if self.idx < len(self.ds):
                return self.ds[self.idx]
            raise StopIteration
    ####################################

    def __init__(self, ds):
        self.current = 0
        self.ds = ds
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.GaussianBlur(3, sigma=(0.1, 2.0)),
                #GaussianNoise(mean=0.0, var=0.01, p=0.3),
            ])

        """
        self.transforms = A.Compose(
            [
                A.Blur(blur_limit=(3,5), p=0.5),
                A.Downscale(scale_min=0.5, scale_max=0.99, interpolation=1, p=0.125),
                A.ISONoise(color_shift=(0,0.2), intensity=(0.1,1), p=0.25),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1,0.25), p=0.5),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
            p=0.8
        )
        """

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        return self.Iterator(self)
    
    def __getitem__(self, idx):
        for _ in range(3):

            x = self.ds[idx]
            object_ids = set(np.unique(x.segmentation)) - {0}
            if len(object_ids) == 0: # if no objects in the image, replace it with random image
                idx = np.random.randint(0, len(self.ds))
                continue

            labels = [y.unique_id for y in x.object_datas]
            bboxes = [y.bbox_modal for y in x.object_datas]
            masks = []
            for object_id in object_ids:
                masks.append((x.segmentation == object_id)[None])

            rgb = self.transforms(x.rgb)
            
            target = {
                "boxes" : torch.as_tensor(np.array(bboxes), dtype=torch.float32),
                "labels" : torch.as_tensor(np.array(labels), dtype=torch.int64),
                "masks" : torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id" : x.infos.view_id,
                "area" : torch.as_tensor(np.array([np.sum(y) for y in masks]), dtype=torch.float32),
                "iscrowd" : torch.zeros((len(labels),), dtype=torch.int64),
            }

            return rgb, target
        raise ValueError("Couldn't find valid sample!")
