import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

import utils.utils_image as util


def gamma_correction(img, expo, gamma):
    return (img ** gamma) / (expo + 1e-8)


def load_ldr(path, n_channels=3):
    img = cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)
    img = img / 2 ** 16
    img = np.float32(img)
    img.clip(0, 1)
    return np.array(img)


def load_hdr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load HDR image from {path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def range_compressor(x):
    return np.log(1 + 100 * x) / np.log(1 + 100)


class DatasetHDRFusion(data.Dataset):
    def __init__(self, opt, normalize=True):
        super().__init__()
        self.opt = opt
        self.n_channels = opt.get('n_channels', 3)
        self.patch_size = opt.get('H_size', 64)
        self.normalize = normalize
        self.base_dataroots = opt['dataroot_H']
        self.target_phase = opt['phase'].lower()

        if self.target_phase not in {'train', 'test'}:
            raise ValueError(f"Unsupported dataset phase: {self.target_phase}")

        self.scene_dirs = self._collect_scenes()
        if not self.scene_dirs:
            raise ValueError(
                f"No valid scenes found in dataset paths: {self.base_dataroots}\n"
                f"Expected {'Training' if self.target_phase == 'train' else 'Test'} subdirectories."
            )
        self.paths_H = [os.path.join(scene, 'HDRImg.hdr') for scene in self.scene_dirs]
        self._print_dataset_summary()

    def _collect_scenes(self):
        all_scenes = []
        phase_dir_name = 'Training' if self.target_phase == 'train' else 'Test'

        for base_dir in self.base_dataroots:
            full_phase_dir = os.path.join(base_dir, phase_dir_name)
            if not os.path.exists(full_phase_dir):
                print(f"Warning: Phase directory '{full_phase_dir}' does not exist, skipping.")
                continue

            scene_dirs = [
                os.path.join(full_phase_dir, d)
                for d in os.listdir(full_phase_dir)
                if os.path.isdir(os.path.join(full_phase_dir, d))
            ]
            all_scenes.extend(scene_dirs)

        return all_scenes

    def _print_dataset_summary(self):
        phase_desc = 'Training' if self.target_phase == 'train' else 'Test'
        print("Dataset initialization complete:")
        print(f"  - Target phase: {self.target_phase.capitalize()}")
        print(f"  - Dataset split: {phase_desc}")
        print(f"  - Total number of valid scenes: {len(self.scene_dirs)}")

    def _process_ldr_images(self, img_Ls, expo_times):
        processed = []
        for i in range(3):
            orig = img_Ls[i]
            corrected = gamma_correction(orig, expo_times[i], 2.2)
            processed.append(np.concatenate((corrected, orig), axis=2))
        return processed

    def _to_tensor(self, img):
        if np.any(np.array(img.strides) < 0):
            img = img.copy()
        return torch.from_numpy(img.transpose(2, 0, 1))

    def _normalize_tensor(self, tensor):
        num_channels = tensor.shape[0]
        return F.normalize(tensor, mean=[0.5] * num_channels, std=[0.5] * num_channels)

    def _load_scene_data(self, index):
        scene_dir = self.scene_dirs[index]
        ldr_files = sorted(f for f in os.listdir(scene_dir) if f.endswith('.tif'))
        assert len(ldr_files) == 3, f"Scene {scene_dir} needs 3 LDRs, found {len(ldr_files)}"

        ldr_paths = [os.path.join(scene_dir, f) for f in ldr_files]
        hdr_path = self.paths_H[index]
        exposure_path = os.path.join(scene_dir, 'exposure.txt')
        expo_times = np.power(2, np.loadtxt(exposure_path))
        assert len(expo_times) == 3, f"exposure.txt in {scene_dir} needs 3 values, found {len(expo_times)}"

        img_Ls = [load_ldr(path, self.n_channels) for path in ldr_paths]
        img_H = load_hdr(hdr_path)
        return img_Ls, img_H, ldr_paths, hdr_path, expo_times

    def __getitem__(self, index):
        img_Ls, img_H, L_paths, H_path, expo_times = self._load_scene_data(index)
        pre_imgs = self._process_ldr_images(img_Ls, expo_times)

        if self.target_phase == 'train':
            h_full, w_full = img_H.shape[:2]
            patch_size_plus_margin = self.patch_size + 8
            rnd_h = random.randint(0, max(0, h_full - patch_size_plus_margin))
            rnd_w = random.randint(0, max(0, w_full - patch_size_plus_margin))
            patch_Ls = [
                img_Ls[i][rnd_h:rnd_h + patch_size_plus_margin, rnd_w:rnd_w + patch_size_plus_margin]
                for i in range(3)
            ]
            patch_H = img_H[rnd_h:rnd_h + patch_size_plus_margin, rnd_w:rnd_w + patch_size_plus_margin]

            mode = random.randint(0, 7)
            patch_Ls = [util.augment_img(patch, mode=mode) for patch in patch_Ls]
            patch_H = util.augment_img(patch_H, mode=mode)

            h_margin, w_margin = patch_H.shape[:2]
            if random.random() > 0.5:
                crop_h = random.randint(0, h_margin - self.patch_size)
                crop_w = random.randint(0, w_margin - self.patch_size)
            else:
                crop_h, crop_w = 0, 0

            img_Ls = [patch_Ls[i][crop_h:crop_h + self.patch_size, crop_w:crop_w + self.patch_size] for i in range(3)]
            img_H = patch_H[crop_h:crop_h + self.patch_size, crop_w:crop_w + self.patch_size]
            pre_imgs = self._process_ldr_images(img_Ls, expo_times)

        img_gm = range_compressor(img_H / (img_Ls[1] + 1 / 64))
        L0, L1, L2 = [self._to_tensor(img) for img in pre_imgs]
        H = self._to_tensor(img_H)
        gm = self._to_tensor(img_gm)
        orig_L0, orig_L1, orig_L2 = [self._to_tensor(img) for img in img_Ls]

        result = {
            'L0': L0,
            'L1': L1,
            'L2': L2,
            'H': H,
            'L_paths': L_paths,
            'H_path': H_path,
            'orig_L1': orig_L1,
        }

        if self.normalize:
            result.update({
                'L0_norm': self._normalize_tensor(orig_L0),
                'L1_norm': self._normalize_tensor(orig_L1),
                'L2_norm': self._normalize_tensor(orig_L2),
                'H_norm': self._normalize_tensor(H),
                'gm_norm': self._normalize_tensor(gm),
            })

        return result

    def __len__(self):
        return len(self.scene_dirs)