import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import utils.utils_image as utils
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from dareg.models.nafnet import NAFNet
from data.dataset_hdr import DatasetHDRFusion
from diffusion.gmodiff import GMODiff_test
from utils import utils_option as option

warnings.filterwarnings("ignore")


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))
        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)


def inverse_range_compressor_tensor(y):
    return (torch.pow(1 + 100, y) - 1) / 100


def load_dareg(dareg_path):
    dareg = NAFNet()
    checkpoint = torch.load(dareg_path, map_location=torch.device("cpu"))
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
    dareg.load_state_dict(state_dict, strict=True)
    dareg.eval()
    for param in dareg.parameters():
        param.requires_grad = False
    return dareg.to("cuda")


def build_test_dataloader(args):
    datasets = option.parse_dataset(args.datasets)["datasets"]
    for phase, dataset_opt in datasets.items():
        if phase != "train":
            test_set = DatasetHDRFusion(dataset_opt)
            test_set.normalize = True
            return DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
    raise ValueError(f"No test dataset found in {args.datasets}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process_size", type=int, default=128)
    parser.add_argument("--gmodiff_path", type=str, required=True)
    parser.add_argument("--dareg_path", type=str, required=True)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--merge_and_unload_lora", action="store_true")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=512)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--datasets", default="../../configs/gmodiff.json")
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    model = GMODiff_test(args)
    dl_test = build_test_dataloader(args)
    dareg = load_dareg(args.dareg_path)
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, batch in enumerate(dl_test):
        lq_raw = [batch["L0"].to("cuda"), batch["L1"].to("cuda"), batch["L2"].to("cuda")]
        condition = batch["L0_norm"].to("cuda")
        h, w = condition.size()[-2:]
        padding_bottom = int(np.ceil(h / 32) * 32 - h)
        padding_right = int(np.ceil(w / 32) * 32 - w)
        pad = nn.ReplicationPad2d((0, padding_right, 0, padding_bottom))
        lq_raw = [pad(item) for item in lq_raw]
        condition = pad(condition)

        with torch.no_grad():
            visual_embedding, lq, mask_embedding = dareg.get_visual_embedding(lq_raw[0], lq_raw[1], lq_raw[2])
            img_e = model(lq, [visual_embedding, mask_embedding], condition)

        gain_map = (img_e[0] * 0.5 + 0.5)[:, :h, :w]
        hdr = inverse_range_compressor_tensor(gain_map) * (lq_raw[1][0][3:6, :h, :w] + 1 / 64)
        hdr = torch.squeeze(hdr.detach().cpu()).numpy().astype(np.float32).transpose(1, 2, 0)
        radiance_writer(os.path.join(args.output_dir, f"{idx}.hdr"), hdr)

        img_e = transforms.ToPILImage()(gain_map.cpu())
        utils.imsave(np.array(img_e), os.path.join(args.output_dir, f"{idx}.png"))


if __name__ == "__main__":
    main(parse_args())
