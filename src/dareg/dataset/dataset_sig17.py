import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.gainmap import calculate_gm
from utils.utils import ldr_to_hdr, list_all_files_sorted, read_expo_times, read_images, read_label


class SIG17_Training_Dataset(Dataset):
    def __init__(self, sub_set, is_training=True):
        self.sub_set = sub_set if isinstance(sub_set, list) else [sub_set]
        self.is_training = is_training
        self.image_list = []

        for scenes_dir in self.sub_set:
            if not osp.exists(scenes_dir):
                print(f"Warning: Path {scenes_dir} does not exist, skipping...")
                continue

            for scene in sorted(os.listdir(scenes_dir)):
                scene_dir = osp.join(scenes_dir, scene)
                self.image_list.append(
                    [
                        osp.join(scene_dir, "exposure.txt"),
                        list_all_files_sorted(scene_dir, ".tif"),
                        scene_dir,
                    ]
                )

    def __getitem__(self, index):
        exposure_path, ldr_paths, label_path = self.image_list[index]
        expo_times = read_expo_times(exposure_path)
        ldr_images = read_images(ldr_paths)
        label_np = read_label(label_path, "label.hdr")

        pre_img0 = build_network_input(ldr_images[0], expo_times[0])
        pre_img1 = build_network_input(ldr_images[1], expo_times[1])
        pre_img2 = build_network_input(ldr_images[2], expo_times[2])

        ldr1_np = ldr_images[1].astype(np.float32)
        gm_np = calculate_gm(label_np.astype(np.float32), ldr1_np)

        return {
            "input0": to_chw_tensor(pre_img0),
            "input1": to_chw_tensor(pre_img1),
            "input2": to_chw_tensor(pre_img2),
            "label": to_chw_tensor(label_np),
            "gm": to_chw_tensor(gm_np),
            "ldr1": to_chw_tensor(ldr1_np),
        }

    def __len__(self):
        return len(self.image_list)


class SIG17_Validation_Dataset(Dataset):
    def __init__(self, root_dir, is_training=False, crop=True, crop_size=512):
        self.root_dir = root_dir
        self.is_training = is_training
        self.crop = crop
        self.crop_size = crop_size
        self.scenes_dir = osp.join(root_dir, "Test")
        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        self.image_list = []

        for scene in self.scenes_list:
            scene_dir = osp.join(self.scenes_dir, scene)
            self.image_list.append(
                [
                    osp.join(scene_dir, "exposure.txt"),
                    list_all_files_sorted(scene_dir, ".tif"),
                    scene_dir,
                ]
            )

    def __getitem__(self, index):
        exposure_path, ldr_paths, label_path = self.image_list[index]
        expo_times = read_expo_times(exposure_path)
        ldr_images = read_images(ldr_paths)
        label_np = read_label(label_path, "HDRImg.hdr")

        pre_img0 = build_network_input(ldr_images[0], expo_times[0])
        pre_img1 = build_network_input(ldr_images[1], expo_times[1])
        pre_img2 = build_network_input(ldr_images[2], expo_times[2])

        if self.crop:
            size = self.crop_size
            pre_img0 = pre_img0[:size, :size]
            pre_img1 = pre_img1[:size, :size]
            pre_img2 = pre_img2[:size, :size]
            label_np = label_np[:size, :size]

        ldr1_np = ldr_images[1].astype(np.float32)
        if self.crop:
            ldr1_np = ldr1_np[: self.crop_size, : self.crop_size]
        gm_np = calculate_gm(label_np.astype(np.float32), ldr1_np)

        return {
            "input0": to_chw_tensor(pre_img0),
            "input1": to_chw_tensor(pre_img1),
            "input2": to_chw_tensor(pre_img2),
            "label": to_chw_tensor(label_np),
            "gm": to_chw_tensor(gm_np),
            "ldr1": to_chw_tensor(ldr1_np),
        }

    def __len__(self):
        return len(self.scenes_list)


def build_network_input(ldr_image, exposure_time):
    hdr_image = ldr_to_hdr(ldr_image, exposure_time, gamma=2.2)
    return np.concatenate((hdr_image, ldr_image), axis=2)


def to_chw_tensor(image):
    return torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))
