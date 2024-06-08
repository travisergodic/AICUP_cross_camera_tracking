import os
import math
from pathlib import Path

import glob
import numpy as np
import pandas as pd


H, W = 720, 1280


def make_df_info(txt_pattern):
    df_list = []
    for path in glob.glob(txt_pattern):
        df = pd.read_csv(
            path, sep=" ", header=None, names=["class", "center_x", "center_y", "width", "height", "vehicle_id"]
        )
        filename = os.path.basename(path)
        df["frame"] = int(filename.replace(".txt", "").split("_")[-1])
        df["cam_id"] = int(filename.split("_")[0])
        df["track_id"] = 0
        df["path"] = os.path.abspath(path.replace("labels", "images").replace(".txt", ".jpg"))
        df["timestamp"] = Path(path).parent.name 
        df_list.append(df)

    df_res = pd.concat(df_list, axis=0, ignore_index=True)
    df_res["x1"] = ((df_res["center_x"] - df_res["width"] / 2) * W).astype(int)
    df_res["x2"] = ((df_res["center_x"] + df_res["width"] / 2) * W).astype(int)
    df_res["y1"] = ((df_res["center_y"] - df_res["height"] / 2) * H).astype(int)
    df_res["y2"] = ((df_res["center_y"] + df_res["height"] / 2) * H).astype(int)
    df_res["area"] = (df_res["x2"]-df_res["x1"]) * (df_res["y2"]-df_res["y1"])

    df_res = df_res.reset_index(drop=True)
    df_res["filename"] = [f"{ele}.jpg" for ele in range(len(df_res))]
    return df_res


def train_val_split(df_info):
    val_mask = df_info["timestamp"].isin(["1016_150000_151900", "1016_190000_191900"])
    return df_info[~val_mask], df_info[val_mask]

# query 圖像不能太小
def query_gallery_split(df_val, query_size=0.2, random_state=42, query_area_margin=1400):
    query_list, gallery_list = [], [] 
    for vehicle_id, gp_vehicle_df in df_val.groupby("vehicle_id", observed=True):
        # one sample cam
        cam_value_count = gp_vehicle_df["cam_id"].value_counts()
        one_sample_cam_ids = cam_value_count[cam_value_count == 1].index
        if len(one_sample_cam_ids) == 1:
            area = gp_vehicle_df.loc[gp_vehicle_df["cam_id"].isin(one_sample_cam_ids), "area"].item()
            filename = gp_vehicle_df.loc[gp_vehicle_df["cam_id"].isin(one_sample_cam_ids), "filename"].item()
            if area >= query_area_margin:
                query_list.append(filename)
            else:
                gallery_list.append(filename)
                
        elif len(one_sample_cam_ids) > 1:
            num_select_query = math.ceil(len(one_sample_cam_ids) * query_size)
            count = 0
            for cam_id in np.random.permutation(one_sample_cam_ids, random_state=random_state):
                area = gp_vehicle_df.loc[gp_vehicle_df["cam_id"] == cam_id, "area"].item()
                filename = gp_vehicle_df.loc[gp_vehicle_df["cam_id"] == cam_id, "filename"].item()
                if (area >= query_area_margin) and count < num_select_query:
                    query_list.append(filename)
                    count += 1
                else:
                    gallery_list.append(filename)
        
        # multi sample cam
        multi_sample_cam_ids = cam_value_count[cam_value_count > 1].index
        
        for cam_id in multi_sample_cam_ids:
            df_curr = gp_vehicle_df[gp_vehicle_df["cam_id"]==cam_id].sample(frac=1, random_state=random_state).reset_index(drop=True)
            num_select_query = math.ceil(len(df_curr) * query_size)
            count = 0 
            for _, row in df_curr.iterrows():
                if (row["area"] >= query_area_margin) and count < num_select_query:
                    query_list.append(row["filename"])
                    count += 1
                else:
                    gallery_list.append(row["filename"])
    return df_val[df_val["filename"].isin(set(query_list))], df_val[df_val["filename"].isin(set(gallery_list))]