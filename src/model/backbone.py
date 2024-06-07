import logging

import torch
import torch.nn as nn

from .clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x
    

class PromptLearner(nn.Module):
    def __init__(self, num_class, prompt, dtype, token_embedding):
        super().__init__()
        ctx_init = f"A photo of a X X X X {prompt}."
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.size(0)
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts
    

def build_model(**kwargs):
    name = kwargs.pop("name")
    if name == "ViT-B-16":
        return VisionTransformer(**kwargs)
    elif name == "RN50":
        return RN50(**kwargs)
    else:
        raise ValueError()


class ReIDBackbone(nn.Module):
    def __init__(
            self, num_classes, camera_num, view_num, sie_cam, sie_view,
            sie_coe, image_size, stride, prompt='vehicle', **kwargs
        ):
        super().__init__()
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = sie_coe
        self.prompt = prompt
        self.sie_cam = sie_cam
        self.sie_view = sie_view

        self.h_resolution = int((image_size[0]-16) // stride[0] + 1)
        self.w_resolution = int((image_size[1]-16) // stride[1] + 1)
        self.vision_stride_size = stride[0]

        if sie_cam and sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            logger.info('camera number is : {}'.format(camera_num))
        elif sie_cam:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            logger.info('camera number is : {}'.format(camera_num))
        elif sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            logger.info('camera number is : {}'.format(view_num))

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        logger.info('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        logger.info('Loading pretrained model for finetuning from {}'.format(model_path))


class VisionTransformer(ReIDBackbone):
    in_planes = 768
    in_planes_proj = 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        clip_model = load_clip_to_cpu("ViT-B-16", self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.image_encoder = clip_model.visual
        self.prompt_learner = PromptLearner(self.num_classes, self.prompt, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            return image_features_proj[:,0]
        
        if self.sie_cam and self.sie_view:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif self.sie_cam:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif self.sie_view:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None
        image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
        img_feature_last = image_features_last[:,0]
        img_feature = image_features[:,0]
        img_feature_proj = image_features_proj[:,0]
        return img_feature_last, img_feature, img_feature_proj
            

class RN50(ReIDBackbone):
    in_planes = 2048
    in_planes_proj = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        clip_model = load_clip_to_cpu("RN50", self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.image_encoder = clip_model.visual
        self.prompt_learner = PromptLearner(self.num_classes, self.prompt, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            return image_features_proj[0]
        
        image_features_last, image_features, image_features_proj = self.image_encoder(x) 
        img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
        img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
        img_feature_proj = image_features_proj[0]
        return img_feature_last, img_feature, img_feature_proj