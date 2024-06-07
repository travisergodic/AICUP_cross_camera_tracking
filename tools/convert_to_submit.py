import os
import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())

import json
import pandas as pd

from src.logger_helper import setup_logger


logger = setup_logger(level=logging.INFO)


def main():
    df_info = pd.read_csv(args.csv_path)
    logger.info(f"Read csv file successfully. Found {len(df_info)} of records")

    df_info = df_info.sort_values(["timestamp", "cam_id", "frame"])
    for timestamp, same_timestamp_df in df_info.groupby("timestamp", observed=True):
        images = sorted(os.listdir(os.path.join(args.image_dir, timestamp)))
        image_to_frame_id = dict(zip(images, range(1, len(images) + 1)))
        df_info.loc[df_info["timestamp"]==timestamp, "frame_id"] = df_info.loc[df_info["timestamp"]==timestamp, "path"].apply(os.path.basename).map(image_to_frame_id)

    with open(args.json_path, "r") as f:
        idx_to_vid = json.load(f)

    Path(args.save_dir).mkdir(exist_ok=True, parents=True)

    for timestamp, same_timestamp_df in df_info.groupby("timestamp", observed=True):
        logger.info(f"Processing timestamp {timestamp} ...") 
        same_timestamp_df["frame_id"] = same_timestamp_df["frame_id"].astype(int)
        # vid start from 1
        same_timestamp_df["vid"] = same_timestamp_df["idx"].astype(int).astype(str).map(idx_to_vid[timestamp]) + 1 
        same_timestamp_df["width"] = (same_timestamp_df["x2"] - same_timestamp_df["x1"]).apply(lambda x: f"{x:.2f}")
        same_timestamp_df["height"] = (same_timestamp_df["y2"] - same_timestamp_df["y1"]).apply(lambda x: f"{x:.2f}")
        same_timestamp_df["x1"] = same_timestamp_df["x1"].apply(lambda x: f"{x:.2f}")
        same_timestamp_df["y1"] = same_timestamp_df["y1"].apply(lambda x: f"{x:.2f}") 
        same_timestamp_df["3d_x"] = -1
        same_timestamp_df["3d_y"] = -1
        same_timestamp_df["3d_z"] = -1
        same_timestamp_df["conf"] = same_timestamp_df["conf"].apply(lambda x: f"{x:.2f}")
        same_timestamp_df[["frame_id", "vid", "x1", "y1", "width", "height", "conf", "3d_x", "3d_y", "3d_z"]].to_csv(
            os.path.join(args.save_dir, f"{timestamp}.txt"), index=False, header=None, sep=","
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    main()