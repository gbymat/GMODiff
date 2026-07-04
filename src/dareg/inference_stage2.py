import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch

from dataset.dataset_sig17 import build_network_input, to_chw_tensor
from models.nafnet import NAFNet
from utils.gainmap import recover_hdr_from_gm
from utils.utils import list_all_files_sorted, read_expo_times, read_images, read_label


def get_args():
    parser = argparse.ArgumentParser(
        description="Run stage-2 mask inference and save gain map, predicted mask, and GT mask images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="../../results/dareg_stage2")
    parser.add_argument("--checkpoint", type=str, default="../../checkpoints/dareg_stage2/best_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask_threshold", type=float, default=0.02)
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model(checkpoint_path, device):
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = NAFNet(use_mask_head=True, freeze_backbone=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    model.eval()
    return model


def find_scene_dirs(input_dir):
    if osp.isfile(osp.join(input_dir, "exposure.txt")):
        return [input_dir]

    scene_dirs = []
    for name in sorted(os.listdir(input_dir)):
        scene_dir = osp.join(input_dir, name)
        if osp.isdir(scene_dir) and osp.isfile(osp.join(scene_dir, "exposure.txt")):
            scene_dirs.append(scene_dir)
    return scene_dirs


def find_label_name(scene_dir):
    for label_name in ("HDRImg.hdr", "label.hdr"):
        if osp.isfile(osp.join(scene_dir, label_name)):
            return label_name
    return None


def load_scene(scene_dir, device):
    ldr_paths = list_all_files_sorted(scene_dir, ".tif")
    if len(ldr_paths) < 3:
        raise ValueError(f"Need at least 3 .tif images in {scene_dir}, found {len(ldr_paths)}")

    expo_times = read_expo_times(osp.join(scene_dir, "exposure.txt"))
    ldr_images = read_images(ldr_paths[:3])
    inputs = [
        to_chw_tensor(build_network_input(ldr_images[i], expo_times[i])).unsqueeze(0).to(device)
        for i in range(3)
    ]
    ldr1 = to_chw_tensor(ldr_images[1].astype(np.float32)).unsqueeze(0).to(device)

    label = None
    label_name = find_label_name(scene_dir)
    if label_name is not None:
        label_np = read_label(scene_dir, label_name)
        label = to_chw_tensor(label_np).unsqueeze(0).to(device)

    return inputs, ldr1, label


def build_mask_target(pred, label, threshold):
    error = (pred.detach() - label.detach()).abs()
    luminance_error = 0.299 * error[:, 0] + 0.587 * error[:, 1] + 0.114 * error[:, 2]
    return (luminance_error >= threshold).float().unsqueeze(1)


def save_gain_map(gain_map, save_path):
    gain_map = np.clip(gain_map, 0.0, 1.0)
    gain_map = (gain_map * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite(save_path, gain_map)


def save_mask(mask, save_path):
    mask = np.squeeze(mask)
    mask = np.clip(mask, 0.0, 1.0)
    mask = (mask * 255.0).round().astype(np.uint8)
    cv2.imwrite(save_path, mask)


def infer_scene(model, scene_dir, output_dir, device, mask_threshold):
    inputs, ldr1, label = load_scene(scene_dir, device)
    with torch.no_grad():
        gain_map, mask_pred = model(*inputs)
        mask_gt = None
        if label is not None:
            pred_hdr = recover_hdr_from_gm(ldr1, gain_map)
            mask_gt = build_mask_target(pred_hdr, label, mask_threshold)

    scene_name = osp.basename(osp.normpath(scene_dir))
    scene_output_dir = osp.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)

    gain_map_np = gain_map.squeeze(0).detach().cpu().numpy()
    mask_pred_np = mask_pred.squeeze(0).detach().cpu().numpy()

    save_gain_map(gain_map_np, osp.join(scene_output_dir, "gain_map.png"))
    save_mask(mask_pred_np, osp.join(scene_output_dir, "mask_pred.png"))
    np.save(osp.join(scene_output_dir, "gain_map.npy"), gain_map_np)
    np.save(osp.join(scene_output_dir, "mask_pred.npy"), mask_pred_np)

    if mask_gt is not None:
        mask_gt_np = mask_gt.squeeze(0).detach().cpu().numpy()
        save_mask(mask_gt_np, osp.join(scene_output_dir, "mask_gt.png"))
        np.save(osp.join(scene_output_dir, "mask_gt.npy"), mask_gt_np)
    else:
        print(f"Warning: no HDRImg.hdr or label.hdr found in {scene_dir}; skipped GT mask.")

    return scene_output_dir


def main():
    args = get_args()
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    scene_dirs = find_scene_dirs(args.input_dir)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders with exposure.txt found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    for scene_dir in scene_dirs:
        scene_output_dir = infer_scene(model, scene_dir, args.output_dir, device, args.mask_threshold)
        print(f"Saved {scene_dir} -> {scene_output_dir}")


if __name__ == "__main__":
    main()
