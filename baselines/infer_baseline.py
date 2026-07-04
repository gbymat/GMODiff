import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader

from dataset.dataset_sig17 import SIG17_Test_Dataset, SIG17_Validation_Dataset
from models.AFUNet import AFUNet
from models.SCTNet import SCTNet
from models.hdr_transformer import HDRTransformer
from utils.utils import AverageMeter, calculate_ssim, range_compressor, save_hdr


MODEL_CHOICES = {
    1: "cavit",
    2: "afunet",
    3: "sctnet",
}

CHECKPOINTS = {
    "cavit": "checkpoints/cavit.pth",
    "afunet": "checkpoints/afunet.pth",
    "sctnet": "checkpoints/sctnet.pth",
}


def build_model(name, device):
    window_size = 8
    if name == "cavit":
        return HDRTransformer(
            embed_dim=60,
            depths=[6, 6, 6],
            num_heads=[6, 6, 6],
            mlp_ratio=2,
            in_chans=6,
        ).to(device)
    if name == "afunet":
        height = (128 // 4 // window_size + 1) * window_size
        width = (128 // 4 // window_size + 1) * window_size
        return AFUNet(
            img_size=(height, width),
            in_chans=18,
            window_size=window_size,
            img_range=1.0,
            drop_path_rate=0.1,
            depths=[5, 5, 5, 5],
            embed_dim=72,
            num_heads=[4, 4, 4, 4],
            mlp_ratio=2,
        ).to(device)
    if name == "sctnet":
        height = (256 // 4 // window_size + 1) * window_size
        width = (256 // 4 // window_size + 1) * window_size
        return SCTNet(
            upscale=2,
            img_size=(height, width),
            in_chans=18,
            window_size=window_size,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
        ).to(device)
    raise ValueError(f"Unsupported baseline: {name}")


def default_checkpoint(name):
    root = osp.dirname(osp.abspath(__file__))
    return osp.join(root, CHECKPOINTS[name])


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_model(name, model, sample, device):
    input0 = sample["input0"].to(device)
    input1 = sample["input1"].to(device)
    input2 = sample["input2"].to(device)
    if name == "afunet":
        return model(torch.cat([input0, input1, input2], dim=1))
    return model(input0, input1, input2)


def evaluate_pair(pred_img, label):
    psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
    label_mu = range_compressor(label)
    pred_mu = range_compressor(pred_img)
    psnr_mu = peak_signal_noise_ratio(label_mu, pred_mu, data_range=1.0)

    pred_vis = np.clip(pred_img * 255.0, 0.0, 255.0).transpose(1, 2, 0)
    label_vis = np.clip(label * 255.0, 0.0, 255.0).transpose(1, 2, 0)
    ssim_l = calculate_ssim(pred_vis, label_vis)

    pred_mu_vis = np.clip(pred_mu * 255.0, 0.0, 255.0).transpose(1, 2, 0)
    label_mu_vis = np.clip(label_mu * 255.0, 0.0, 255.0).transpose(1, 2, 0)
    ssim_mu = calculate_ssim(pred_mu_vis, label_mu_vis)
    return psnr_l, psnr_mu, ssim_l, ssim_mu


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline HDR inference.")
    parser.add_argument("--model_id", type=int, choices=sorted(MODEL_CHOICES), default=2)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), default=None)
    parser.add_argument("--mode", choices=["full", "patch"], default="full")
    parser.add_argument("--dataset_dir", type=str, required=False, default='../../data')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def run_full_inference(args, model_name, model, device, save_dir):
    dataset = SIG17_Validation_Dataset(
        root_dir=args.dataset_dir,
        is_training=False,
        crop=args.crop,
        crop_size=args.crop_size,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    psnr_l_meter = AverageMeter()
    psnr_mu_meter = AverageMeter()
    ssim_l_meter = AverageMeter()
    ssim_mu_meter = AverageMeter()
    total_time = 0.0
    for idx, sample in enumerate(dataloader):
        start = time.time()
        with torch.no_grad():
            pred = run_model(model_name, model, sample, device)
        total_time += time.time() - start

        pred_img = torch.squeeze(pred.detach().cpu()).numpy().astype(np.float32)
        label = torch.squeeze(sample["label"]).numpy().astype(np.float32)
        pred_hdr = pred_img.transpose(1, 2, 0)[..., ::-1]
        save_hdr(osp.join(save_dir, f"{idx}.hdr"), pred_hdr)

        psnr_l, psnr_mu, ssim_l, ssim_mu = evaluate_pair(pred_img, label)
        psnr_l_meter.update(psnr_l)
        psnr_mu_meter.update(psnr_mu)
        ssim_l_meter.update(ssim_l)
        ssim_mu_meter.update(ssim_mu)

    return total_time, len(dataloader), psnr_l_meter, psnr_mu_meter, ssim_l_meter, ssim_mu_meter


def run_patch_inference(args, model_name, model, device, save_dir):
    datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size)
    psnr_l_meter = AverageMeter()
    psnr_mu_meter = AverageMeter()
    ssim_l_meter = AverageMeter()
    ssim_mu_meter = AverageMeter()
    total_time = 0.0
    num_images = 0

    for idx, img_dataset in enumerate(datasets):
        dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
        start = time.time()
        with torch.no_grad():
            for sample in dataloader:
                pred = run_model(model_name, model, sample, device)
                img_dataset.update_result(torch.squeeze(pred.detach().cpu()).numpy().astype(np.float32))
        total_time += time.time() - start
        num_images += 1

        pred_img, label = img_dataset.rebuild_result()
        pred_hdr = pred_img.transpose(1, 2, 0)[..., ::-1]
        save_hdr(osp.join(save_dir, f"{idx}.hdr"), pred_hdr)

        psnr_l, psnr_mu, ssim_l, ssim_mu = evaluate_pair(pred_img, label)
        psnr_l_meter.update(psnr_l)
        psnr_mu_meter.update(psnr_mu)
        ssim_l_meter.update(ssim_l)
        ssim_mu_meter.update(ssim_mu)

    return total_time, num_images, psnr_l_meter, psnr_mu_meter, ssim_l_meter, ssim_mu_meter


def main():
    args = parse_args()
    model_name = args.model or MODEL_CHOICES[args.model_id]
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint_path = args.checkpoint or default_checkpoint(model_name)
    save_dir = args.save_dir or osp.join("results", model_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Baseline choices: 1=cavit, 2=afunet, 3=sctnet")
    print(f"Selected baseline: {model_name}")
    print(f"Inference mode: {args.mode}")
    print(f"Load weights from: {checkpoint_path}")
    print(f"Save HDR results to: {save_dir}")

    model = build_model(model_name, device)
    model = load_checkpoint(model, checkpoint_path)

    if args.mode == "patch":
        stats = run_patch_inference(args, model_name, model, device, save_dir)
    else:
        stats = run_full_inference(args, model_name, model, device, save_dir)

    total_time, num_images, psnr_l_meter, psnr_mu_meter, ssim_l_meter, ssim_mu_meter = stats
    avg_time = total_time / max(num_images, 1)
    print(f"Total inference time: {total_time:.4f} s")
    print(f"Average inference time per image: {avg_time:.4f} s")
    print(f"Average PSNR_mu: {psnr_mu_meter.avg:.4f}  PSNR_l: {psnr_l_meter.avg:.4f}")
    print(f"Average SSIM_mu: {ssim_mu_meter.avg:.4f}  SSIM_l: {ssim_l_meter.avg:.4f}")


if __name__ == "__main__":
    main()
