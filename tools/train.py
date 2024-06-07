import os
import sys
import logging
import argparse
import yaml
sys.path.insert(0, os.getcwd())

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import VehicleDataset, TrainTransform, TestTransform
from src.model import build_model
from src.optim import build_optimizer
from src.logger_helper import setup_logger
import src.trainer as trainer


logger = setup_logger(level=logging.INFO)


def main():
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    df_train = pd.read_csv(args.train_csv)
    df_query = pd.read_csv(args.guery_csv)
    df_gallery = pd.read_csv(args.gallery_csv)

    df_val = pd.concat([df_query, df_gallery], axis=0, ignore_index=True)
    num_query = df_query.index.size
    num_cam = df_train["cam_id"].unique().size
    num_train_vehicle_ids = df_train["vehicle_id"].unique().size

    logger.info(f"train size: {df_train.index.size}")
    logger.info(f"query size: {num_query}")
    logger.info(f"gallery size: {df_gallery.index.size}")
    logger.info(f"number of train vids: {num_train_vehicle_ids}")
    logger.info(f"number of camera: {num_cam}")
    
    train_transform = TrainTransform(
        mean=config["TRANSFORM"]["TRAIN"]["MEAN"],
        std=config["TRANSFORM"]["TRAIN"]["STD"],
        size=config["TRANSFORM"]["TRAIN"]["SIZE"],
        aug_prob=config["TRANSFORM"]["TRAIN"]["AUG_PROB"],
        re_prob=config["TRANSFORM"]["TRAIN"]["RE_PROB"]
    )
    val_transform = TestTransform(
        mean=config["TRANSFORM"]["TEST"]["MEAN"],
        std=config["TRANSFORM"]["TEST"]["STD"],
        size=config["TRANSFORM"]["TEST"]["SIZE"],
    )

    train_dataset = VehicleDataset(df_train, args.image_dir, train_transform, relabel=True)
    val_dataset = VehicleDataset(df_val, args.image_dir, val_transform, relabel=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=2 * args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(
        sie_cam=True, cam_num=config["CAM_NUM"], 
        s=config["ARCFACE_RADIUS"], m=config["ARCFACE_MARGIN"], 
        num_vids=num_train_vehicle_ids, is_train=True
    ).to(args.device)
    logger.info("Build model successfully")
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model, base_lr=config["TRAIN"]["BASE_LR"], 
        weight_decay=config["TRAIN"]["WEIGHT_DECAY"], 
        bias_lr_factor=config["TRAIN"]["BASE_LR_FACTOR"],
        weight_decay_bias=config["TRAIN"]["WEIGHT_DECAY_BIAS"], 
        fc_lr_factor=config["TRAIN"]["FC_LR_FACTOR"]
    )
    logger.info("Start training ...")
    trainer.train(
        model, optimizer, loss_fn, train_loader, val_loader, num_query, 
        num_epoch=args.num_epochs, device=args.device, save_dir=f"./checkpoints/{args.tag}",
        metric=config["TEST"]["METRIC"], max_rank=config["TEST"]["MAX_RANK"], 
        eval_freq=config["TEST"]["EVAL_FREQ"], save_freq=config["TEST"]["SAVE_FREQ"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--query_csv", type=str, required=True)
    parser.add_argument("--gallery_csv", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main()