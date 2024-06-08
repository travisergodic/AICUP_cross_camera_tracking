import os
import argparse
from pathlib import Path

import cv2
import pandas as pd

H, W = 720, 1280

def main():
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    df_info = pd.read_csv(args.csv_path)

    for path, same_frame_df in df_info.groupby("path", observed=True):
        img = cv2.imread(path)
        for _, row in same_frame_df.iterrows():
            x1, x2, y1, y2 = row["x1"], row["x2"], row["y1"], row["y2"]
            x1, x2, y1, y2 = max(0, x1 - args.margin), min(W, x2 + args.margin), max(0, y1 - args.margin), min(H, y2 + args.margin)
            save_path = os.path.join(args.save_dir, row["filename"])
            cv2.imwrite(save_path, img[y1:y2, x1:x2, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--margin", type=int, required=True)
    args = parser.parse_args()
    main()