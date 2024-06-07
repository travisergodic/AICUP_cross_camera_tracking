import logging

import torch


logger = logging.getLogger(__name__)


# optimizer
def build_optimizer(model, base_lr, weight_decay, bias_lr_factor, weight_decay_bias, fc_lr_factor=None):
    params, keys = [], []

    for key, value in model.named_parameters():
        if ".bias" in key:
            curr_lr = base_lr * bias_lr_factor
            curr_weight_decay = weight_decay_bias
        else:
            curr_lr = base_lr
            curr_weight_decay = weight_decay

        if fc_lr_factor:
            if key.startswith("head."):
                curr_lr = base_lr * fc_lr_factor
                logger.info(f'Using {fc_lr_factor} times learning rate for fc ')

        params += [{"params": [value], "lr": curr_lr, "weight_decay": curr_weight_decay}]
        keys += [key]
    return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)