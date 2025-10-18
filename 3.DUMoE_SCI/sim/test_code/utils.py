import scipy.io as sio
import os
import numpy as np
import torch
import logging


def generate_shift_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + "/mask_3d_shift.mat")
    mask_3d_shift = mask["mask_3d_shift"]
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2, 1)
    Phi_s_batch[Phi_s_batch == 0] = 1
    # print(Phi_batch.shape, Phi_s_batch.shape)
    return Phi_batch, Phi_s_batch


def LoadTest(path_test):
    scene_list = os.listdir(path_test)
    scene_list.sort()
    test_data = np.zeros((len(scene_list), 256, 256, 28))
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]
        img = sio.loadmat(scene_path)["img"]
        test_data[i, :, :, :] = img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))
    return test_data


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = (
        year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second
    )
    return time_filename


def gen_meas_torch(data_batch, mask3d_batch):
    [batch_size, nC, H, W] = data_batch.shape
    step = 2
    gt_batch = torch.zeros(batch_size, nC, H, W + step * (nC - 1)).to(data_batch.device)
    gt_batch[:, :, :, 0:W] = data_batch
    gt_shift_batch = shift(gt_batch)
    meas = torch.sum(mask3d_batch * gt_shift_batch, 1)
    meas = meas / nC * 2
    return meas


def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(
            inputs[:, i, :, :], shifts=(-1) * step * i, dims=2
        )
    return inputs


def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + "/log.txt"
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def init_mask(mask_path, mask_type, batch_size):
    if mask_type == "Phi_PhiPhiT":
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
    return Phi_batch


def init_meas(gt, mask, input_setting):
    if input_setting == "Y":
        input_meas = gen_meas_torch(gt, mask)
    return input_meas
