import os
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils import ldr_to_hdr, list_all_files_sorted, read_expo_times, read_images, read_label


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
            exposure_file_path = osp.join(scene_dir, "exposure.txt")
            ldr_file_path = list_all_files_sorted(scene_dir, ".tif")
            self.image_list.append([exposure_file_path, ldr_file_path, scene_dir])

    def __getitem__(self, index):
        expo_times = read_expo_times(self.image_list[index][0])
        ldr_images = read_images(self.image_list[index][1])
        label_np = read_label(self.image_list[index][2], "HDRImg.hdr")

        pre_img0 = ldr_to_hdr(ldr_images[0], expo_times[0], 2.2)
        pre_img1 = ldr_to_hdr(ldr_images[1], expo_times[1], 2.2)
        pre_img2 = ldr_to_hdr(ldr_images[2], expo_times[2], 2.2)

        pre_img0 = np.concatenate((pre_img0, ldr_images[0]), 2)
        pre_img1 = np.concatenate((pre_img1, ldr_images[1]), 2)
        pre_img2 = np.concatenate((pre_img2, ldr_images[2]), 2)

        if self.crop:
            x, y = 0, 0
            pre_img0 = pre_img0[x : x + self.crop_size, y : y + self.crop_size]
            pre_img1 = pre_img1[x : x + self.crop_size, y : y + self.crop_size]
            pre_img2 = pre_img2[x : x + self.crop_size, y : y + self.crop_size]
            label_np = label_np[x : x + self.crop_size, y : y + self.crop_size]

        img0 = torch.from_numpy(pre_img0.astype(np.float32).transpose(2, 0, 1))
        img1 = torch.from_numpy(pre_img1.astype(np.float32).transpose(2, 0, 1))
        img2 = torch.from_numpy(pre_img2.astype(np.float32).transpose(2, 0, 1))
        label = torch.from_numpy(label_np.astype(np.float32).transpose(2, 0, 1))

        return {
            "input0": img0,
            "input1": img1,
            "input2": img2,
            "label": label,
        }

    def __len__(self):
        return len(self.scenes_list)


class SIG17PatchDataset(Dataset):
    def __init__(self, ldr_path, label_path, exposure_path, patch_size):
        self.ldr_images = read_images(ldr_path)
        self.label = read_label(label_path, "HDRImg.hdr")
        self.expo_times = read_expo_times(exposure_path)
        self.patch_size = patch_size
        self.ldr_patches = self._get_ordered_patches()
        self.result = []

    def __getitem__(self, index):
        ldr0, ldr1, ldr2 = self.ldr_patches[index]
        pre_img0 = np.concatenate((ldr_to_hdr(ldr0, self.expo_times[0], 2.2), ldr0), 2)
        pre_img1 = np.concatenate((ldr_to_hdr(ldr1, self.expo_times[1], 2.2), ldr1), 2)
        pre_img2 = np.concatenate((ldr_to_hdr(ldr2, self.expo_times[2], 2.2), ldr2), 2)

        return {
            "input0": torch.from_numpy(pre_img0.astype(np.float32).transpose(2, 0, 1)),
            "input1": torch.from_numpy(pre_img1.astype(np.float32).transpose(2, 0, 1)),
            "input2": torch.from_numpy(pre_img2.astype(np.float32).transpose(2, 0, 1)),
        }

    def __len__(self):
        return len(self.ldr_patches)

    def _get_ordered_patches(self):
        h, w, c = self.label.shape
        n_h = h // self.patch_size + 1
        n_w = w // self.patch_size + 1
        tmp_h = n_h * self.patch_size
        tmp_w = n_w * self.patch_size

        padded = []
        for image in self.ldr_images:
            tmp = np.ones((tmp_h, tmp_w, c), dtype=np.float32)
            tmp[:h, :w] = image
            padded.append(tmp)

        patches = []
        for x in range(n_w):
            for y in range(n_h):
                patches.append(
                    [
                        padded[0][y * self.patch_size : (y + 1) * self.patch_size, x * self.patch_size : (x + 1) * self.patch_size],
                        padded[1][y * self.patch_size : (y + 1) * self.patch_size, x * self.patch_size : (x + 1) * self.patch_size],
                        padded[2][y * self.patch_size : (y + 1) * self.patch_size, x * self.patch_size : (x + 1) * self.patch_size],
                    ]
                )
        return patches

    def update_result(self, tensor):
        self.result.append(tensor)

    def rebuild_result(self):
        h, w, c = self.label.shape
        n_h = h // self.patch_size + 1
        n_w = w // self.patch_size + 1
        tmp_h = n_h * self.patch_size
        tmp_w = n_w * self.patch_size
        pred = np.empty((c, tmp_h, tmp_w), dtype=np.float32)

        for x in range(n_w):
            for y in range(n_h):
                pred[
                    :,
                    y * self.patch_size : (y + 1) * self.patch_size,
                    x * self.patch_size : (x + 1) * self.patch_size,
                ] = self.result[x * n_h + y]
        return pred[:, :h, :w], self.label.transpose(2, 0, 1)


def SIG17_Test_Dataset(root_dir, patch_size):
    scenes_dir = osp.join(root_dir, "Test")
    scenes_list = sorted(os.listdir(scenes_dir))
    for scene in scenes_list:
        scene_dir = osp.join(scenes_dir, scene)
        exposure_file_path = osp.join(scene_dir, "exposure.txt")
        ldr_file_path = list_all_files_sorted(scene_dir, ".tif")
        yield SIG17PatchDataset(ldr_file_path, scene_dir, exposure_file_path, patch_size)
