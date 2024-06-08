import os
import logging
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter

import sklearn.metrics.pairwise

logger=logging.getLogger(__name__)


def train_one_epoch(model, optimizer, loss_fn, loader, device, scaler):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader)
    for batch in pbar:
        y = batch["vehicle_id"].to(device)
        cam_ids = batch["cam_id"].to(device)
        with torch.cuda.amp.autocast(enabled=True):
            y_hat = model(batch["data"].to(device), y, cam_ids)
            loss = loss_fn(y_hat, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item()})
        loss_meter.update(loss, y_hat.size(0))
    return loss_meter.avg


@torch.no_grad()
def eval_model(model, loader, normalize, num_query, metric, max_rank, device):
    model.eval()
    feats, vehicle_ids, cam_ids, frames, timestamps = [], [], [], [], []
    for batch in loader:
        with torch.cuda.amp.autocast(enabled=True):
            X = batch["data"].to(device)
            cam_id = batch["cam_id"].to(device)
            feat = model(X, cam_id=cam_id)
        feats.append(feat.cpu())
        vehicle_ids.append(batch["vehicle_id"])
        cam_ids.append(cam_id.cpu())
        frames.append(batch["frame"])
        timestamps += batch["timestamp"]
    # concat
    feats = torch.concat(feats, dim=0).numpy()
    vehicle_ids = torch.concat(vehicle_ids, dim=0).numpy()
    cam_ids = torch.concat(cam_ids, dim=0).numpy()
    frames = torch.concat(frames, dim=0).numpy()
    timestamps = np.array(timestamps)
    if normalize:
        feats = feats / np.linalg.norm(feats, ord=2, axis=1, keepdims=True)

    distmat = getattr(sklearn.metrics.pairwise, metric)(feats[:num_query], feats[num_query:])
    cc_dict = compute_map_cmc(distmat, vehicle_ids, cam_ids, frames, timestamps, max_rank=max_rank, mode="cc")
    sc_dict = compute_map_cmc(distmat, vehicle_ids, cam_ids, frames, timestamps, max_rank=5, mode="sc")
    return {**cc_dict, **sc_dict}


def compute_map_cmc(distmat, vehicle_ids, cam_ids, frames, timestamps, max_rank=50, mode="cc"):
    assert mode in ("sc", "cc")
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    q_vids, g_vids = vehicle_ids[:num_q], vehicle_ids[num_q:]
    q_cam_ids, g_cam_ids = cam_ids[:num_q], cam_ids[num_q:]
    q_frames, g_frames = frames[:num_q], frames[num_q:]
    q_timestamps, g_timestamps = timestamps[:num_q], timestamps[num_q:]

    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)
    all_cmc, all_AP, num_valid_q = [], [], 0
    for q_idx in range(num_q):
        q_vid = vehicle_ids[q_idx]
        q_camid = cam_ids[q_idx]
        q_frame = frames[q_idx]
        q_timestamp = timestamps[q_idx]

        order = indices[q_idx]
        if mode == "cc":
            remove = ((g_vids[order] == q_vid) & (g_cam_ids[order] == q_camid) & ((g_frames[order] - q_frame) < 5)) | \
                    (g_timestamps[order] != q_timestamp)
        elif mode == "sc":
            remove = (g_cam_ids[order] != q_camid) | (g_timestamps[order] != q_timestamp)

        orig_cmc = matches[q_idx][~remove]
        if len(orig_cmc) < max_rank:
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        if num_rel > 0:
            y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
            tmp_cmc = tmp_cmc / y
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        else:
            all_AP.append(0)
    all_cmc = np.stack(all_cmc, axis=0).astype(np.float32)
    # all_cmc = np.asarray(all_cmc).astype(np.float32)
    print(num_valid_q)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return {f"{mode}_CMC": all_cmc, f"{mode}_mAP": mAP}


def train(model, optimizer, loss_fn, train_loader, val_loader, num_query, metric, max_rank, num_epoch, device, eval_freq, save_freq, save_dir):
    for epoch in range(num_epoch):
        scaler = torch.cuda.amp.GradScaler()
        avg_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, device, scaler)
        logger.info(f"Epoch [{epoch+1}/{num_epoch}] loss: {avg_loss}")
        if (epoch + 1) % eval_freq == 0:
            # cross camera
            score_dict = eval_model(model, val_loader, normalize=False, num_query=num_query, metric=metric, max_rank=max_rank, device=device)
            cmc = score_dict["cc_CMC"]
            map = score_dict["cc_mAP"]
            logger.info("--------- cross camera ---------")
            logger.info(f"CMC(1): {cmc[0]}, CMC(5): {cmc[4]}, CMC(10): {cmc[9]}")
            logger.info(f"mAP: {map}")

            # single cam
            cmc = score_dict["sc_CMC"]
            map = score_dict["sc_mAP"]
            logger.info("--------- single camera ---------")
            logger.info(f"CMC(1): {cmc[0]}, CMC(5): {cmc[4]}")
            logger.info(f"mAP: {map}")

        if (epoch + 1) % save_freq == 0:
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"e{epoch}.pt"))