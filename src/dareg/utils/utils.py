import glob
import math
import os
import random
from math import log10

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio


def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))


def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))


def read_images(file_names):
    images = []
    for file_name in file_names:
        image = cv2.imread(file_name, -1)
        image = np.float32(image / 2**16)
        images.append(np.clip(image, 0, 1))
    return np.array(images)


def read_label(file_path, file_name):
    label = imageio.imread(os.path.join(file_path, file_name), format="hdr")
    return label[:, :, [2, 1, 0]]


def ldr_to_hdr(imgs, expo, gamma):
    return (imgs**gamma) / (expo + 1e-8)


def range_compressor(x, mu=5000):
    return np.log1p(mu * x) / np.log1p(mu)


def range_compressor_cuda(hdr_img, mu=5000):
    return torch.log1p(mu * hdr_img) / math.log1p(mu)


def psnr(x, target):
    mse = np.mean((x - target) ** 2)
    return 10 * log10(1 / mse)


def batch_psnr(img, imclean, data_range):
    img_np = img.detach().cpu().numpy().astype(np.float32)
    clean_np = imclean.detach().cpu().numpy().astype(np.float32)
    score = 0
    for i in range(img_np.shape[0]):
        score += peak_signal_noise_ratio(clean_np[i], img_np[i], data_range=data_range)
    return score / img_np.shape[0]


def batch_psnr_mu(img, imclean, data_range):
    return batch_psnr(range_compressor_cuda(img), range_compressor_cuda(imclean), data_range)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def init_parameters(net):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
            init.constant_(module.bias, 0)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def ssim(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[2] == 3:
            return np.array([ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]).mean()
        if img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("Wrong input image dimensions.")


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))

        image = np.maximum(image, 0.0).astype(np.float32)
        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        nonzero = brightest > 1e-32
        mantissa[nonzero], exponent[nonzero] = np.frexp(brightest[nonzero])

        scale = np.zeros_like(brightest)
        scale[nonzero] = mantissa[nonzero] * 255.0 / brightest[nonzero]
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scale[..., None]).clip(0, 255).astype(np.uint8)
        rgbe[..., 3] = np.where(nonzero, np.around(exponent + 128), 0).astype(np.uint8)
        rgbe.flatten().tofile(f)


def save_hdr(path, image):
    return radiance_writer(path, image)
