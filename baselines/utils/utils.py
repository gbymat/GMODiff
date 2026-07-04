#-*- coding:utf-8 -*-  
import numpy as np
import os, glob
import cv2
import math
import imageio
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))


def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)
        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)


def read_label(file_path, file_name):
    label = imageio.imread(os.path.join(file_path, file_name), 'hdr')
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def gamma_correction( img, expo, gamma):
    return (img ** gamma) / 2.0 ** expo

def ldr_to_hdr(imgs, expo, gamma):
    return (imgs ** gamma) / (expo + 1e-8)

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def range_compressor_cuda(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

def set_random_seed(seed):
    """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
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
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def save_hdr(path, image):
    return radiance_writer(path, image)

# def flow_to_image(flow):
#     """
#     将光流（H, W, 2）可视化为RGB图片，返回uint8格式的BGR图像（适合cv2.imwrite）。
#     """
#     if flow.ndim == 3 and flow.shape[2] == 2:
#         u = flow[..., 0]
#         v = flow[..., 1]
#     elif flow.ndim == 2:
#         raise ValueError('flow_to_image: 输入flow应为(H, W, 2)')
#     else:
#         raise ValueError('flow_to_image: 输入flow维度不正确')
#
#     rad = np.sqrt(u ** 2 + v ** 2)
#     ang = np.arctan2(v, u)
#     hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
#     hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # H: 0~180
#     hsv[..., 1] = 255
#     hsv[..., 2] = np.clip(rad / (np.max(rad) + 1e-5) * 255, 0, 255).astype(np.uint8)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return bgr

def resize(x, scale_factor):
    return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False, recompute_scale_factor=True)

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

from torchvision.utils import flow_to_image
import torch
def flow_to_image_torch(flow):
    flow = torch.from_numpy(np.transpose(flow, [2, 0, 1]))
    flow_im = flow_to_image(flow)
    img = np.transpose(flow_im.numpy(), [1, 2, 0])
    # print(img.shape)
    return img


# import math
# import numpy as np
# import torch
# import torch.nn.functional as F
#
#
#
#
# def weight_3expo_low_tog17(img):
#     w = torch.zeros_like(img)
#     mask1 = img < 0.5
#     w[mask1] = 0.0
#     mask2 = img >= 0.50
#     w[mask2] = img[mask2] - 0.5
#     w /= 0.5
#     return w
#
# def weight_3expo_mid_tog17(img):
#     w = torch.zeros_like(img)
#     mask1 = img < 0.5
#     w[mask1] = img[mask1]
#     mask2 = img >= 0.5
#     w[mask2] = 1.0 - img[mask2]
#     w /= 0.5
#     return w
#
# def weight_3expo_high_tog17(img):
#     w = torch.zeros_like(img)
#     mask1 = img < 0.5
#     w[mask1] = 0.5 - img[mask1]
#     mask2 = img >= 0.5
#     w[mask2] = 0.0
#     w /= 0.5
#     return w
#
# def merge_hdr(ldr_imgs, lin_imgs, mask0, mask2):
#     sum_img = torch.zeros_like(ldr_imgs[1])
#     sum_w = torch.zeros_like(ldr_imgs[1])
#     w_low = weight_3expo_low_tog17(ldr_imgs[1]) * mask0
#     w_mid = weight_3expo_mid_tog17(ldr_imgs[1]) + weight_3expo_low_tog17(ldr_imgs[1]) * (1.0 - mask0) + weight_3expo_high_tog17(ldr_imgs[1]) * (1.0 - mask2)
#     w_high = weight_3expo_high_tog17(ldr_imgs[1]) * mask2
#     w_list = [w_low, w_mid, w_high]
#     for i in range(len(ldr_imgs)):
#         sum_w += w_list[i]
#         sum_img += w_list[i] * lin_imgs[i]
#     hdr_img = sum_img / (sum_w + 1e-9)
#     return hdr_img
#
#
#
# def calculate_psnr(img1, img2):
#     psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
#     return psnr
