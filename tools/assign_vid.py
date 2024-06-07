import os
import sys
import json
import logging
import argparse
from collections import defaultdict
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise

from src.logger_helper import setup_logger
from src.assign import assign_single_camera, make_topk_avg_features, cross_camera_match


logger = setup_logger(level=logging.INFO)


def main():
    with open(args.margin_cfg, "r") as f:
        MARGIN = json.load(f)

    df_info = pd.read_csv(args.csv_path)
    logger.info(f"Read csv file successfully. Found {len(df_info)} of records")

    pairwise_dist = getattr(pairwise, args.metric)
    res = dict()
    for timestamp, same_timestamp_df in df_info.groupby("timestamp", observed=True):
        logger.info(f"Processing timestamp {timestamp} ...")
        # load feats
        feats = np.load(os.path.join(args.feat_dir, f"{timestamp}.npy"))

        idx_to_vid = assign_single_camera(
            same_timestamp_df, feats, 
            pairwise_dist=pairwise_dist, 
            pre=args.pre, margin=MARGIN
        )
        vid_to_idx = defaultdict(list)
        for idx, vid in idx_to_vid.items():
            vid_to_idx[vid].append(idx)

        avg_feats, vids, camids = make_topk_avg_features(
            feats, same_timestamp_df, vid_to_idx, topk=args.topk
        )
        update_vid = cross_camera_match(avg_feats, camids, vids, pairwise_dist, MARGIN)

        updated_idx_to_vid = {}
        for idx, vid in idx_to_vid.items():
            if vid in update_vid:
                updated_idx_to_vid[int(idx)] = int(update_vid[vid])
            else:
                updated_idx_to_vid[int(idx)] = int(vid)
        res[timestamp] = updated_idx_to_vid

    with open(args.save_path, 'w') as f:
        json.dump(res, f, indent=4)
    logger.info(f"Save matching result at {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--feat_dir", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--metric", type=str, choices=["euclidean_distances", "cosine_distances"])
    parser.add_argument("--pre", type=int)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--margin_cfg", type=str, required=True)
    args = parser.parse_args()
    main()