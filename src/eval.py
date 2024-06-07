import torch
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()

    feats, camids = [], []
    for data, camid, in tqdm(loader):
        data = data.to(device)
        camid = camid.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            feat = model(data, cam_id=camid)
        feats.append(feat.cpu())
        camids.append(camid.cpu())

    # concat
    feats = torch.concat(feats, dim=0).numpy()
    camids = torch.concat(camids, dim=0).numpy()
    return feats


@torch.no_grad()
def make_image_features(model, loader, device):
    model.eval()
    feats, vids, camids, frames, timestamps = [], [], [], [], []
    for batch in tqdm(loader):
        X = batch["data"].to(device)
        cam_id = batch["cam_id"].to(device)
        with torch.autocast(device_type="cuda"):
            feat = model(X, cam_id=cam_id)

        feats.append(feat.cpu())
        vids.append(batch["vehicle_id"])
        camids.append(cam_id.cpu())
        frames.append(batch["frame"])
        timestamps += batch["timestamp"]
    # concat
    feats = torch.concat(feats, dim=0).numpy()
    vids = torch.concat(vids, dim=0).numpy()
    camids = torch.concat(camids, dim=0).numpy()
    frames = torch.concat(frames, dim=0).numpy()
    timestamps = np.array(timestamps)
    return {"feat": feats, "vids": vids, "camids": camids, "frames": frames, "timestamps": timestamps}


def compute_topk(distmat, vids, cam_ids, frames, timestamps, topk, mode, exclude_mismatch=False):
    num_q, num_g = distmat.shape
    q_vids, g_vids = vids[:num_q], vids[num_q:]
    q_cam_ids, g_cam_ids = cam_ids[:num_q], cam_ids[num_q:]
    q_frames, g_frames = frames[:num_q], frames[num_q:]
    q_timestamps, g_timestamps = timestamps[:num_q], timestamps[num_q:]

    indices = np.argsort(distmat, axis=1)
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)

    topk_dist, topk_g, valid_q, num_valid_q = [], [], [], 0
    for q_idx in range(num_q):
        # query
        q_vid = vids[q_idx]
        q_camid = cam_ids[q_idx]
        q_frame = frames[q_idx]
        q_timestamp = timestamps[q_idx]

        # gallery
        order = indices[q_idx]
        curr_g_vids = g_vids[order]
        curr_g_camids = g_cam_ids[order]
        curr_g_frames = g_frames[order]
        curr_g_timestamps = g_timestamps[order]
        curr_dist = distmat[q_idx][order]

        if mode == "cc":
            # (移除同個 timestamp, camera 底下，鄰近的幀且相通 vid 的車子) 或 (timestamp 不同的車子)
            remove = ((curr_g_vids == q_vid) & (curr_g_camids == q_camid) & ((curr_g_frames - q_frame) < 10)) | \
                     (curr_g_timestamps != q_timestamp)
        elif mode == "sc":
            # 移除不同 camera 或 不同 timestamp 的車子
            remove = (curr_g_camids != q_camid) | (curr_g_timestamps != q_timestamp)

        curr_g_vids = curr_g_vids[~remove]
        qg_dist = curr_dist[~remove]

        if qg_dist.size < topk:
            continue

        if exclude_mismatch:
            if not np.any((curr_g_vids == q_vid)[:topk]):
                continue

        valid_q.append(q_vid)
        topk_g.append(curr_g_vids[:topk])
        topk_dist.append(qg_dist[:topk])
    return np.array(valid_q), np.stack(topk_g, axis=0), np.stack(topk_dist , axis=0)