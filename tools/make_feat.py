import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.logger_helper import setup_logger
from src.data import InferenceDataset, TestTransform
from src.model import build_model, load_checkpoint
from src.eval import eval_model


logger = setup_logger(level=logging.INFO)


def main():
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    val_transform = TestTransform(
        mean=config["TRANSFORM"]["TEST"]["MEAN"],
        std=config["TRANSFORM"]["TEST"]["STD"],
        size=config["TRANSFORM"]["TEST"]["SIZE"],
    )

    df_info = pd.read_csv(args.csv_path)
    logger.info(f"Read csv file successfully. Found {len(df_info)} of records")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(sie_cam=args.sie_cam, cam_num=config["CAM_NUM"]).to(device)
    msg = load_checkpoint(model, args.ckpt)
    logger.info(f"Load day model successfully. {msg}")

    for timestamp, same_timestap_df in df_info.groupby("timestamp", observed=True):
        val_dataset = InferenceDataset(
            args.image_dir, 
            same_timestap_df["filename"].values, 
            same_timestap_df["cam_id"].values, 
            transform=val_transform
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        feats = eval_model(model, val_loader, device)

        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(args.save_dir, f"{timestamp}.npy")
        np.save(save_file, feats)
        logger.info(f"Save feature at {save_file}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--sie_cam", action="store_true")
    args = parser.parse_args()
    main()