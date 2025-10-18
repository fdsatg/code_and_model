from architecture import *
from utils import *
import scipy.io as scio
import torch
from ssim_torch import *
import os
import numpy as np
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    raise Exception("NO GPU!")

# Intialize mask
mask3d_batch = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)


def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, mask3d_batch)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(
        np.float32
    )
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()
    return pred, truth


def torch_psnr(img, ref):  # input [28,256,256]
    img = (img * 256).round()
    ref = (ref * 256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255 * 255) / mse)
    return psnr / nC


def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))


def main():
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, truth = test(model)
    name = opt.outf + f"{opt.method}.mat"
    print(f"Save reconstructed HSIs as {name}.")
    scio.savemat(name, {"truth": truth, "pred": pred})


def results(path):
    mat = scio.loadmat(path)
    method = path.split("/")[-1].split("\\")[-1].split(".")[0]
    with open("results.csv", "a+") as f:
        f.write(f"{method},PSNR,SSIM\n")
    truth, pred = mat["truth"], mat["pred"]
    truth = np.transpose(truth, (0, 3, 1, 2)).astype(np.float32)
    pred = np.transpose(pred, (0, 3, 1, 2)).astype(np.float32)

    psnrs, ssims = [], []
    for k in range(pred.shape[0]):
        psnr_val = torch_psnr(
            torch.from_numpy(truth[k, :, :, :]), torch.from_numpy(pred[k, :, :, :])
        )
        ssim_val = torch_ssim(
            torch.from_numpy(truth[k, :, :, :]), torch.from_numpy(pred[k, :, :, :])
        )
        psnrs.append(psnr_val.detach().cpu().numpy())
        ssims.append(ssim_val.detach().cpu().numpy())
        with open("results.csv", "a+") as f:
            f.write(f"{k+1},{psnr_val:.2f},{ssim_val:.3f}\n")
            print(f"{k+1},{psnr_val:.2f},{ssim_val:.3f}")
    with open("results.csv", "a+") as f:
        f.write(f"Avg.,{np.average(psnrs):.2f},{np.average(ssims):.3f}\n")
        print(f"Avg.,{np.average(psnrs):.2f},{np.average(ssims):.3f}")


if __name__ == "__main__":
    main()
    path = "./exp/dumoe.mat"
    results(path)
