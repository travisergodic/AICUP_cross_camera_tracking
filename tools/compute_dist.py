import os
import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())

import torch
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pairwise

from src.logger_helper import setup_logger
from src.model import build_inference_model, load_checkpoint
from src.eval import make_image_features, compute_topk
from src.dataset import VehicleDataset, val_transform
from torch.utils.data import DataLoader

logger = setup_logger(level=logging.INFO)


def main():
    # build model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    day_model = build_inference_model().to(device)
    night_model = build_inference_model().to(device)

    if args.day_ckpt: 
        msg = load_checkpoint(day_model, args.day_ckpt)
        logger.info(f"Load day model successfully. {msg}")

    if args.night_ckpt: 
        msg = load_checkpoint(day_model, args.night_ckpt)
        logger.info(f"Load night model successfully. {msg}")

    all_query = pd.read_csv(args.query_csv)
    all_gallery = pd.read_csv(args.gallery_csv)

    Path(f"./checkpoints/{args.tag}").mkdir(exist_ok=True, parents=True)

    for timestamp, model in [("150000_151900", day_model), ("190000_191900", night_model)]:
        # build loader
        df_query = all_query[all_query["timestamp"].apply(lambda s: timestamp in s)]
        df_gallery = all_gallery[all_gallery["timestamp"].apply(lambda s: timestamp in s)]
        df_val = pd.concat([df_query, df_gallery], axis=0, ignore_index=True)
        val_dataset = VehicleDataset(df_val, args.image_dir, val_transform, relabel=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # make features
        logger.info("Making image features ...")
        data_dict = make_image_features(model, val_loader, device=device)
        feats = data_dict.pop("feat")
        
        # distance matrix
        logger.info("Compute pairwise distances ...")
        distmat = getattr(pairwise, args.metric)(feats[:len(df_query)], feats[len(df_query):])

        # cross cam 
        cc_query, cc_topk_gallery, cc_topk_distmat = compute_topk(distmat, topk=args.topk, mode="cc", exclude_mismatch=args.exclude_mis, **data_dict)
        dist_arr = cc_topk_distmat.reshape(-1)
        gt = (cc_query.reshape(-1, 1) == cc_topk_gallery).flatten().astype(bool)

        logger.info(f"positive cc samples:{gt.sum()}")
        logger.info(f"negative cc samples:{(~gt).sum()}")

        plt.title(f"{timestamp}: cross camera, topk: {args.topk}")
        plt.hist(dist_arr[gt], bins=100, color='red', alpha=0.5, label='Class 1')
        plt.hist(dist_arr[~gt], bins=100, color='blue', alpha=0.5, label='Class 2')
        plt.savefig(f"./checkpoints/{args.tag}/{timestamp}_cc.png")
        
        # single cam
        sc_query, sc_topk_gallery, sc_topk_distmat = compute_topk(distmat, topk=args.topk, mode="sc", exclude_mismatch=args.exclude_mis, **data_dict)
        dist_arr = sc_topk_distmat.reshape(-1)
        gt = (sc_query.reshape(-1, 1) == sc_topk_gallery).flatten().astype(bool)

        logger.info(f"positive sc samples:{gt.sum()}")
        logger.info(f"negative sc samples:{(~gt).sum()}")

        plt.title(f"{timestamp} single camera, topk: {args.topk}")
        plt.hist(dist_arr[gt], bins=100, color='red', alpha=0.5, label='Class 1')
        plt.hist(dist_arr[~gt], bins=100, color='blue', alpha=0.5, label='Class 2')
        plt.savefig(f"./checkpoints/{args.tag}/{timestamp}_sc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--tag", type=str)
    parser.add_argument("--query_csv", type=str)
    parser.add_argument("--gallery_csv", type=str)
    parser.add_argument("--day_model", type=str)
    parser.add_argument("--night_model", type=str)
    parser.add_argument("--metric", type=str, choices=["cosine_similarity", "euclidean_distances"])
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--exclude_mis", action="store_true")
    args = parser.parse_args()
    main()