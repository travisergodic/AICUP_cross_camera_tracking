import os

from PIL import Image
from torch.utils.data import Dataset


class VehicleDataset(Dataset):
    cols = ["filename", "timestamp", "vehicle_id", "cam_id", "track_id", "frame", "x1", "x2", "y1", "y2"]

    def __init__(self, df_info, image_dir, transform, relabel):
        self.df_info = df_info
        self.image_dir = image_dir
        self.transform = transform
        self.relabel = relabel
        if self.relabel:
            vehicle_ids = sorted(df_info["vehicle_id"].unique())
            self.relabel_map = {vehicle_id: new_id for new_id, vehicle_id in enumerate(vehicle_ids)}
            self.df_info["vehicle_id"] = self.df_info["vehicle_id"].map(self.relabel_map)

    def __getitem__(self, idx):
        record = self.df_info.loc[idx, self.cols].to_dict()
        path = os.path.join(self.image_dir, record["filename"])
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return {
            "data": image,
            "vehicle_id": record["vehicle_id"],
            "cam_id": record["cam_id"],
            "frame": record["frame"],
            "path": path,
            "timestamp": record["timestamp"]
        }

    def __len__(self):
        return self.df_info.index.size    


class InferenceDataset(Dataset):
    def __init__(self, image_dir, filenames, cam_ids, transform=None):
        self.image_dir = image_dir
        self.filenames = filenames 
        self.cam_ids = cam_ids
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_dir, self.filenames[index]))
        if self.transform:
            image = self.transform(image)
        return image, self.cam_ids[index]
    
    def __len__(self):
        return len(self.filenames)
