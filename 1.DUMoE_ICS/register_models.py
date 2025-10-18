import sys
import torch
from timm.models.registry import register_model
from moe_duns import *


def load_pretrained_model(ratio, model_name, model):
    path = f"./model/checkpoint-{model_name}-{ratio}-best.pth"
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@register_model
def dumoe(ratio, pretrained=False, **kwargs):
    depth = 5
    dim = 32
    mult = 1.5

    model = DUMoE(
        ratio=ratio,
        dim=dim,
        mult=mult,
        depth=depth,
    )
    if pretrained:
        return load_pretrained_model(ratio, sys._getframe().f_code.co_name, model)
    return model
