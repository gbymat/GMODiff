import argparse
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from dataset.dataset_sig17 import SIG17_Validation_Dataset
from models.nafnet import NAFNet
from utils.gainmap import recover_hdr_from_gm
from utils.utils import AverageMeter, calculate_ssim, range_compressor, save_hdr


def get_args():
    parser = argparse.ArgumentParser(
        description="Run pretrained HDR gain-map model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--pretrained_model", type=str, default="../../checkpoints/dareg_stage1/best_checkpoint.pth")
    parser.add_argument("--save_results", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="../../results/dareg_stage1")
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model(checkpoint_path, device):
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model = NAFNet(use_mask_head=False, freeze_backbone=False).to(device)
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    model.eval()
    return model


def save_gain_map(save_dir, idx, gain_map):
    gain_map_img = np.clip(gain_map, 0.0, 1.0)
    gain_map_img = (gain_map_img * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite(osp.join(save_dir, f"{idx}_gain_map.png"), gain_map_img)
    np.save(osp.join(save_dir, f"{idx}_gain_map.npy"), gain_map)


def save_prediction(save_dir, idx, pred_img, gain_map):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    pred_hdr = pred_img.copy().transpose(1, 2, 0)[..., ::-1]
    save_hdr(osp.join(save_dir, f"{idx}_pred.hdr"), pred_hdr)
    save_gain_map(save_dir, idx, gain_map)


def main():
    args = get_args()
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = load_model(args.pretrained_model, device)

    dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False, crop_size=512)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    total_time = 0.0

    for idx, batch_data in enumerate(dataloader):
        batch_ldr0 = batch_data["input0"].to(device)
        batch_ldr1 = batch_data["input1"].to(device)
        batch_ldr2 = batch_data["input2"].to(device)
        label = batch_data["label"].to(device)
        ldr1 = batch_data["ldr1"].to(device)

        start_time = time.time()
        with torch.no_grad():
            gain_map = model(batch_ldr0, batch_ldr1, batch_ldr2)
            pred_img = recover_hdr_from_gm(ldr1, gain_map)
        total_time += time.time() - start_time

        gain_map = torch.squeeze(gain_map.detach().cpu()).numpy().astype(np.float32)
        pred_img = torch.squeeze(pred_img.detach().cpu()).numpy().astype(np.float32)
        label = torch.squeeze(label.detach().cpu()).numpy().astype(np.float32)

        scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)
        scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)

        pred_img_vis = np.clip(pred_img * 255.0, 0.0, 255.0).transpose(1, 2, 0)
        label_vis = np.clip(label * 255.0, 0.0, 255.0).transpose(1, 2, 0)
        scene_ssim_l = calculate_ssim(pred_img_vis, label_vis)

        pred_img_mu_vis = np.clip(pred_img_mu * 255.0, 0.0, 255.0).transpose(1, 2, 0)
        label_mu_vis = np.clip(label_mu * 255.0, 0.0, 255.0).transpose(1, 2, 0)
        scene_ssim_mu = calculate_ssim(pred_img_mu_vis, label_mu_vis)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)

        if args.save_results:
            save_prediction(args.save_dir, idx, pred_img, gain_map)

    num_images = len(dataloader)
    avg_time = total_time / max(num_images, 1)
    print(f"Total inference time: {total_time:.4f} s")
    print(f"Average inference time per image: {avg_time:.4f} s")
    print(f"Average PSNR_mu: {psnr_mu.avg:.4f}  PSNR_l: {psnr_l.avg:.4f}")
    print(f"Average SSIM_mu: {ssim_mu.avg:.4f}  SSIM_l: {ssim_l.avg:.4f}")
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == "__main__":
    main()
