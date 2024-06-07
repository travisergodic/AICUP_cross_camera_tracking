import logging

import torch
import torch.nn as nn

from .backbone import VisionTransformer, RN50

logger = logging.getLogger(__name__)


class ReIDModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        feats = self.backbone(x=x, label=label, get_image=get_image, get_text=get_text, cam_label=cam_label, view_label=view_label)
        if get_image or get_text:
            return feats
        return self.head(*feats, label=label)

def load_pretrained(model, pretrained):
    state_dict = torch.load(pretrained, map_location="cpu")
    backbone_state_dict, head_state_dict = {}, {}
    for k, v in state_dict.items():
        if k.startswith("bottleneck") or k.startswith("bottleneck_proj"):
            head_state_dict[k] = state_dict[k]
        elif not (k.startswith("classifier") or k.startswith("classifier_proj") or k.startswith("prompt_learner")):
            backbone_state_dict[k] = state_dict[k]
    backbone_msg = model.backbone.load_state_dict(backbone_state_dict, strict=False)
    head_msg = model.head.load_state_dict(head_state_dict, strict=False)
    logger.info(f"backbone msg: {backbone_msg}")
    logger.info(f"head msg: {head_msg}")
    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ClipReidLinearHead(nn.Module):
    def __init__(self, num_classes, in_planes, in_planes_proj, neck_feat):
        super().__init__()
        self.in_planes = in_planes
        self.in_planes_proj = in_planes_proj
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        # classifier
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        # bottleneck
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

    def forward(self, img_feature_last, img_feature, img_feature_proj, label=None):
        # feat = self.bottleneck(img_feature) 
        # feat_proj = self.bottleneck_proj(img_feature_proj) 
        feat = img_feature
        feat_proj = img_feature_proj

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


def build_reid_model(num_classes, cam_num, view_num, backbone_cfg, head_cfg):
    # build backbone
    backbone_type = backbone_cfg.pop("type")
    backbone_cfg["num_classes"] = num_classes
    backbone_cfg["camera_num"] = cam_num
    backbone_cfg["view_num"] = view_num
    if backbone_type == "ViT":
        backbone = VisionTransformer(**backbone_cfg)
    elif backbone_type == "RN50":
        backbone = RN50(**backbone_cfg)

    # build head
    head_type = head_cfg.pop("type")
    head_cfg["num_classes"] = num_classes
    head_cfg["in_planes"] = backbone.in_planes
    head_cfg["in_planes_proj"] = backbone.in_planes_proj
    head = ClipReidLinearHead(**head_cfg)
    return ReIDModel(backbone=backbone, head=head)