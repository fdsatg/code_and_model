import torch
from .moe_duns import DUMoE

def model_generator(opt, device="cuda"):
    if opt.method == 'dumoe':
        depth = 5

        model = DUMoE(
            depth=depth,
        ).to(device)
    else:
        print(f'opt.Method {opt.method} is not defined !!!!')
    
    return model