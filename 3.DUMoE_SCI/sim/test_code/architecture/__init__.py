import torch
from .moe_duns import DUMoE


def model_generator(method, pretrained_model_path=None):
    if method == "dumoe":
        depth = 5

        model = DUMoE(
            depth=depth,
        )

    if pretrained_model_path is not None:
        print(f"load model from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(
            {k.replace("module.", ""): v for k, v in checkpoint.items()}, strict=True
        )
    return model
