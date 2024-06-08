import torch
import torch.nn as nn


# def load_pretrained_weight(model, ckpt_path, num_classes):
#     state_dict = torch.load(ckpt_path, map_location="cpu")
#     msg = model.load_state_dict(state_dict=state_dict, strict=False)
#     print(msg)

#     # classifier
#     in_features = model.classifier.in_features 
#     model.classifier = nn.Linear(in_features, num_classes)

#     # classifier_proj
#     in_features = model.classifier_proj.in_features 
#     model.classifier_proj = nn.Linear(in_features, num_classes)

#     # cls_ctx
#     _, n_cls_ctx, ctx_dim = model.prompt_learner.cls_ctx.size()
#     cls_vectors = torch.empty(num_classes, n_cls_ctx, ctx_dim, dtype=model.prompt_learner.cls_ctx.dtype) 
#     nn.init.normal_(cls_vectors, std=0.02)
#     model.prompt_learner.cls_ctx = nn.Parameter(cls_vectors)
#     return model


def load_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt)
    remove_keys = []
    for k, _ in state_dict.items():
        if k.startswith("head."):
            remove_keys.append(k)

    for k in remove_keys:
        state_dict.pop(k)
    return model.load_state_dict(state_dict, strict=False)