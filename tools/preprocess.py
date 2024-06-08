import os
import sys
import logging
import argparse
from pathlib import Path
sys.path.insert(0, os.getcwd())

import cv2
from tqdm import tqdm

from src.logger_helper import setup_logger
from src.utils import make_df_info, train_val_split, query_gallery_split


logger = setup_logger(level=logging.INFO)


def main():
    txt_pattern = os.path.join(args.data_root, "*/*.txt") 
    df_info = make_df_info(txt_pattern)
    df_info.to_csv(os.path.join(args.save_dir, "crop_vehicle.csv"), index=False)

    crop_img_dir = os.path.join(args.save_dir, f"images_{str(args.margin)}/")
    Path(crop_img_dir).mkdir(exist_ok=True, parents=True)

    logger.info("Cropping images ...")
    for path, gp_df in tqdm(df_info.groupby("path", observed=True)):
        img = cv2.imread(path)
        for _, row in gp_df.iterrows():
            save_path = os.path.join(crop_img_dir, row["filename"])
            x1, x2, y1, y2 = row["x1"], row["x2"], row["y1"], row["y2"]
            x1, x2, y1, y2 = max(row["x1"]-args.margin, 0), min(row["x2"]+args.margin, 1280), max(row["y1"]-args.margin, 0), min(row["y2"]+args.margin, 720)
            cv2.imwrite(save_path, img[y1:y2, x1:x2, :])
    logger.info(f"Save all cropping images in {args.save_dir}")    

    logger.info("Doing train, query, gallery split ...")
    df_train, df_val = train_val_split(df_info)
    df_query, df_gallery = query_gallery_split(df_val, query_size=args.query_ratio, random_state=42)

    # save
    df_train.to_csv(os.path.join(args.save_dir, "train.csv"), index=False)
    df_query.to_csv(os.path.join(args.save_dir, "query.csv"), index=False)
    df_gallery.to_csv(os.path.join(args.save_dir, "gallery.csv"), index=False)
    logger.info(f"Save train.csv & query.csv & gallery.csv in {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--margin", type=int, required=True)
    parser.add_argument("--query_ratio", type=float, default=0.4)
    args = parser.parse_args()
    main()