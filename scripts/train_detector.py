import os
import numpy as np
import datetime
from argparse import ArgumentParser
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from happypose.pose_estimators.megapose.config import LOCAL_DATA_DIR, EXP_DIR

from src.dataset import make_train_val_datasets
from src.detector import MaskRCNN


def collate_fn(batch):
    return tuple(zip(*batch))


def main(args):
    np.random.seed(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
                            

    #ds = make_dataset("custom_dataset", "train_pbr", "custom-{label}")
    ds_train, ds_val = make_train_val_datasets(args.custom_ds_name, "train_pbr", "custom-{label}", train_test_ratio=0.85)
    for x in ds_train: break

    """
    count = 0
    count_all = 0
    for image,target in ds_train:
        count_all += 1
        if target['masks'].shape == torch.Size([0]):
            count += 1
        print(f"{count}/{count_all}")
    breakpoint()
    """

    dataloader_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=False)
    dataloader_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=False)

    num_classes  = 1 + len([x for x in (LOCAL_DATA_DIR / args.custom_ds_name / 'models').iterdir() if str(x)[-4:] == '.ply'])

    output_dir = EXP_DIR / args.exp_id
    output_dir.mkdir(exist_ok=True, parents=True)
    

    logger = CSVLogger(output_dir, name="", version="")

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir, save_top_k=1, monitor="val_loss"
    )

    model = MaskRCNN(num_classes)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.num_epochs,
        default_root_dir=output_dir,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, dataloader_train, dataloader_val)
    Path(checkpoint_callback.best_model_path).rename(output_dir / "model.ckpt")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)

    args = parser.parse_args()

    args.num_workers = 8
    if args.exp_id is None:
        args.exp_id = f'detector-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")}'
    

    args.custom_ds_name = "custom_dataset"

    main(args)


