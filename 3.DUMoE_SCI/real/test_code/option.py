import argparse
import template
import platform

parser = argparse.ArgumentParser(
    description="HyperSpectral Image Reconstruction Toolbox"
)
parser.add_argument(
    "--template", default="dumoe", help="You can set various templates in option.py"
)

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default="0")

# Data specifications
parser.add_argument(
    "--data_root", type=str, default=r"./data/SCI", help="dataset directory"
)

# Saving specifications
parser.add_argument("--outf", type=str, default="./exp/dumoe/", help="saving_path")

# Model specifications
parser.add_argument("--method", type=str, default="dumoe", help="method name")
parser.add_argument(
    "--pretrained_model_path", type=str, default="./model/dumoe_sci_real.pth", help="pretrained model directory"
)
parser.add_argument(
    "--input_setting",
    type=str,
    default="Y",
    help="the input measurement of the network: H, HM or Y",
)
parser.add_argument(
    "--input_mask",
    type=str,
    default="Phi",
    help="the input mask of the network: Phi, Phi_PhiPhiT, Mask or None",
)  # Phi: shift_mask   Mask: mask


# Training specifications
parser.add_argument("--height", default=660, type=int, help="cropped patch height")
parser.add_argument("--width", default=660, type=int, help="cropped patch width")

opt = parser.parse_known_args()[0]
template.set_template(opt)

opt.data_path = f"{opt.data_root}/TSA_real_data/Measurements/"
opt.mask_path = f"{opt.data_root}/TSA_real_data/mask_3d_shift.mat"


for arg in vars(opt):
    if vars(opt)[arg] == "True":
        vars(opt)[arg] = True
    elif vars(opt)[arg] == "False":
        vars(opt)[arg] = False
