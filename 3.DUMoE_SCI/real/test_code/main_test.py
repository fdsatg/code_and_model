from architecture import *
import numpy as np
from scipy import io as sio
import torch
from torch.autograd import Variable
import os
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(path, file_num, height=660):
    HR_HSI = np.zeros((((height, 714, file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + "scene" + str(idx + 1) + ".mat"
        data = sio.loadmat(path1)
        HR_HSI[:, :, idx] = data["meas_real"][:height, :]
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI


def load_mask(path):
    ## load mask
    data = sio.loadmat(path)
    mask_3d_shift = data["mask_3d_shift"]
    mask_3d_shift_s = np.sum(mask_3d_shift**2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)


HR_HSI = prepare_data(opt.data_path, 5, height=opt.height)
mask_3d_shift, mask_3d_shift_s = load_mask(opt.mask_path)

# model
model = model_generator(opt, device=device)

print(f"===> Loading Checkpoint from {opt.pretrained_model_path}")
checkpoint = torch.load(opt.pretrained_model_path, map_location=device)["model"]
model.load_state_dict(
    {k.replace("module.", ""): v for k, v in checkpoint.items()}, strict=True
)
model.eval()

save_path = "./Results/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
res = []
for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:, :, j]
        # meas = meas / meas.max() * 0.8
        meas = meas / (meas.max() + 1e-7) * 0.9
        meas = torch.FloatTensor(meas)
        input = meas.unsqueeze(0)
        input = Variable(input)
        input = input.to(device)
        mask_3d_shift = mask_3d_shift.to(device)
        mask_3d_shift_s = mask_3d_shift_s.to(device)
        result = model(input, mask_3d_shift)
        result = result.clamp(min=0.0, max=1.0)
    res.append(result.cpu().permute(0, 2, 3, 1).numpy())

    save_file = save_path + f"{j}.mat"
    sio.savemat(save_file, {"res": result.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()})

save_file = save_path + f"Real_result.mat"
res = np.concatenate(res, axis=0)
print(res.shape)
sio.savemat(save_file, {"pred": res})
