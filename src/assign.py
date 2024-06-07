from collections import defaultdict

import numpy as np


def assign_single_camera(df_info, feats, pairwise_dist, pre, margin):
    init_vid = 0  # 當下 camera 的起始 vid 編號 
    idx_to_vid = dict()

    for camid, same_camid_df in df_info.groupby("cam_id", observed=True, sort=True):
        # 先掃一次，僅看前面幾幀是否有 match 的車輛
        curr_idx_to_vid = dict()
        curr_vid_to_frames = defaultdict(set)
        vid = init_vid

        for frame, same_frame_df in same_camid_df.groupby("frame", observed=True, sort=True):
            if same_frame_df.index.size == 0:
                continue
            
            if vid == init_vid: # 第一個 frame 直接 assign vid
                for _, row in same_frame_df.iterrows():
                    curr_idx_to_vid[row["idx"]] = vid
                    curr_vid_to_frames[vid].add(frame) 
                    vid += 1
            else:
                # gallery: 僅看前面幾個 frame 
                stage1_gallery_idxs = same_camid_df.loc[(same_camid_df["frame"] < frame) & (same_camid_df["frame"] >= frame - pre), "idx"].values
                stage1_gallery_feats = feats[stage1_gallery_idxs]

                ## query: 當下的 frame
                query_idxs = same_frame_df["idx"].values
                query_feats = feats[query_idxs]

                matched_vids, matched_query_idxs = set(), set()
                ## stage1 distmat
                if stage1_gallery_feats.size != 0:
                    distmat1 = pairwise_dist(query_feats, stage1_gallery_feats)
                    flat_indices1 = np.argsort(distmat1, axis=None)
                    two_d_indices1 = np.array(np.unravel_index(flat_indices1, distmat1.shape)).T

                    # stage1 match
                    for q_idx, g_idx in two_d_indices1:
                        # 距離大於 margin 直接退出
                        if distmat1[q_idx, g_idx] > margin[str(int(camid))]:
                            break

                        # 確保不出現一對多關係
                        q, g = query_idxs[q_idx], stage1_gallery_idxs[g_idx]
                        if (curr_idx_to_vid[g] in matched_vids) or (q_idx in matched_query_idxs):
                            continue
                            
                        match_vid = curr_idx_to_vid[g]
                        curr_idx_to_vid[q] = match_vid
                        curr_vid_to_frames[match_vid].add(frame)

                        matched_vids.add(match_vid)
                        matched_query_idxs.add(q_idx)        

                # 沒有匹配的車輛直接分配新的 vid
                for idx in (ele for ele in query_idxs if ele not in curr_idx_to_vid):
                    curr_idx_to_vid[idx] = vid
                    curr_vid_to_frames[vid].add(frame)
                    vid += 1

        init_vid = vid

        # 計算特徵平均
        curr_vid_to_idxs = defaultdict(list)
        for idx, vid in curr_idx_to_vid.items():
            curr_vid_to_idxs[vid].append(idx)

        avg_feats, vids = [], []
        for vid, idxs in curr_vid_to_idxs.items():
            curr_feats = feats[idxs]
            curr_feats = curr_feats / np.linalg.norm(curr_feats, axis=1, keepdims=True, ord=2)
            avg_feats.append(curr_feats.mean(axis=0))
            vids.append(vid)
        avg_feats = np.stack(avg_feats, axis=0)
        vids = np.array(vids)

        ## stage2 matching
        # vid -> partition set
        vid_to_partition = {vid: {vid} for vid in curr_vid_to_idxs.keys()}

        distmat2 = pairwise_dist(avg_feats, avg_feats)
        flat_indices2 = np.argsort(distmat2, axis=None)
        two_d_indices2 = np.array(np.unravel_index(flat_indices2, distmat2.shape)).T

        for q_idx, g_idx in two_d_indices2:
            # 距離大於 margin 直接退出
            if distmat2[q_idx, g_idx] > margin[str(int(camid))]:
                break

            # 出現在同一個 frame 的兩台車一定是不同車子，直接略過
            if curr_vid_to_frames[vids[q_idx]] & curr_vid_to_frames[vids[g_idx]]:
                continue

            # 要求 q_idx 要 大於 g_idx
            if g_idx >= q_idx:
                continue

            q_partition = vid_to_partition[vids[q_idx]]
            g_partition = vid_to_partition[vids[g_idx]]

            q_frames = set.union(*[curr_vid_to_frames[vid] for vid in q_partition])
            g_frames = set.union(*[curr_vid_to_frames[vid] for vid in g_partition])

            # 出現在同一個 frame 的一定是不同車子，直接略過
            if q_frames & g_frames:
                continue

            union_set = vid_to_partition[vids[q_idx]] | vid_to_partition[vids[g_idx]]
            
            # 更新
            for vid in union_set:
                vid_to_partition[vid] = union_set

        update_vid = {vid: min(partition) for vid, partition in vid_to_partition.items()}

        for idx, vid in curr_idx_to_vid.items():
            curr_idx_to_vid[idx] = update_vid[vid]
        idx_to_vid.update(curr_idx_to_vid)
    
    # reindex
    old_vid_to_new_vid = {old_vid: new_vid for new_vid, old_vid in enumerate(sorted(set(idx_to_vid.values())))}

    # Create a new dictionary with reindexed values
    new_idx_to_vid = {idx: old_vid_to_new_vid[old_vid] for idx, old_vid in idx_to_vid.items()}
    return new_idx_to_vid


def make_topk_avg_features(feats, df_info, vid_to_idx, topk=3):
    vids = sorted(vid_to_idx.keys())
    assert (max(vids) - min(vids) + 1) == len(vids)

    average_feats, camids = [], []
    for vid in vids:
        idxs = vid_to_idx[vid]
        topk_area_df = df_info[df_info["idx"].isin(idxs)].sort_values(by='area', ascending=False).head(topk) 
        # average feat
        topk_idxs = topk_area_df["idx"].values
        curr_feats = feats[topk_idxs]
        curr_feats = curr_feats / np.linalg.norm(curr_feats, axis=1, keepdims=True, ord=2)
        average_feats.append(curr_feats.mean(axis=0))
        camids.append(topk_area_df["cam_id"].iloc[0])
    return np.stack(average_feats, axis=0), np.array(vids), np.array(camids)


def cross_camera_match(average_feats, camids, vids, pairwise_dist, margin):
    idx_to_partition = {idx: {idx} for idx in range(len(vids))}

    distmat = pairwise_dist(average_feats, average_feats)
    flat_indices = np.argsort(distmat, axis=None)
    two_d_indices = np.array(np.unravel_index(flat_indices, distmat.shape)).T

    for q_idx, g_idx in two_d_indices:
        # 距離大於 margin 直接退出
        if distmat[q_idx, g_idx] > margin["cc"]:
            break

        # 不允許匹配的車輛出現在同一個 cam
        if camids[q_idx] == camids[g_idx]:
            break

        # 要求 q_idx 要 大於 g_idx
        if g_idx >= q_idx:
            continue

        q_idx_partition = idx_to_partition[q_idx]
        g_idx_partition = idx_to_partition[g_idx]

        q_camids = set(camids[i] for i in q_idx_partition)
        g_camids = set(camids[i] for i in g_idx_partition)

        # 不允許匹配的車輛出現在同一個 cam，直接略過
        if q_camids & g_camids:
            continue

        idx_union_set = idx_to_partition[q_idx] | idx_to_partition[g_idx]
        
        # 更新
        for idx in idx_union_set:
            idx_to_partition[idx] = idx_union_set

    update_idx = {idx: min(partition) for idx, partition in idx_to_partition.items()}
    # update vid
    old_vid_to_new_vid = {}
    for idx, old_vid in enumerate(vids):
        new_vid = vids[update_idx[idx]]
        old_vid_to_new_vid[old_vid] = new_vid

    # Create a new dictionary with reindexed values
    reindex_vid_map = {old_vid: new_vid for new_vid, old_vid in enumerate(sorted(set(old_vid_to_new_vid.values())))}

    final_vid_map = {}
    for raw_vid, old_vid in old_vid_to_new_vid.items():
        final_vid_map[raw_vid] = reindex_vid_map[old_vid]
    return final_vid_map