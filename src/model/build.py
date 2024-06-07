import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_reid import build_reid_model, load_pretrained


logger = logging.getLogger(__name__)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, input, label):
        return input
    

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.training:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros(cosine.size()).to(cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            return output * self.s
        return input


class ViTModel(nn.Module):
    def __init__(self, backbone, head, sie_cam=False, cam_num=None, sie_coe=3.):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.sie_cam = sie_cam
        self.cam_num = cam_num
        self.sie_coe = sie_coe
        if self.sie_cam:
            self.cv_embed = nn.Parameter(torch.zeros(self.cam_num, 768))

    def forward(self, X, y=None, cam_id=None):
        if self.sie_cam:
            cv_embed = self.sie_coe * self.cv_embed[cam_id]
            _, feat, _ = self.backbone(X, cv_embed)
        else:
            _, feat, _ = self.backbone(X)

        feat = feat[:,0]
        return self.head(feat, y)


def build_model(sie_cam=True, cam_num=None, is_train=False, num_vids=None, s=30.0, m=0.05):
    backbone_cfg = dict(
        type ='ViT', sie_coe = 3.0, image_size = [256, 256], stride = [12, 12], 
        prompt = 'vehicle', sie_cam = True, sie_view = False
    )
    head_cfg = dict(type = "linear", neck_feat = "after")
    reid_model = build_reid_model(
        num_classes=1000, cam_num=8, view_num=1, backbone_cfg=backbone_cfg, head_cfg=head_cfg
    )
    load_pretrained(reid_model, pretrained="./weight/VehicleID_clipreid_12x12sie_ViT-B-16_60.pt")
    backbone = reid_model.backbone.image_encoder
    if is_train:
        model = ViTModel(
            backbone=backbone, head=ArcFace(s=s, m=m, in_features=768, out_features=num_vids), 
            sie_cam=sie_cam, cam_num=cam_num
        )         
    else:
        model = ViTModel(
            backbone=backbone, head=Identity(), 
            sie_cam=sie_cam, cam_num=cam_num
        )
    return model