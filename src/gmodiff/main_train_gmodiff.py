import argparse
import math
import os
import random
import sys
import warnings
from pathlib import Path

import diffusers
import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from dareg.models.nafnet import NAFNet
from data.dataset_hdr import DatasetHDRFusion
from diffusion.gmodiff import GMODiff_train
from diffusion.models.discriminator import Discriminator
from utils import utils_option as option

warnings.filterwarnings("ignore")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_train_steps", type=int, default=300000)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--output_dir", type=str, default="../../results/train_gmodiff")
    parser.add_argument("--pretrained_model", default=None, type=str)
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--datasets", default="../../configs/gmodiff.json")
    parser.add_argument("--gmodiff_path", type=str, default=None)
    parser.add_argument("--dareg_path", type=str, required=True)
    parser.add_argument("--dareg_paths", type=str, default=None)
    parser.add_argument("--num_daregs", type=int, default=1)
    parser.add_argument("--use_gan_loss", action="store_true")
    parser.add_argument("--hdr_loss_type", type=str, default="l1mu", choices=["l1mu", "joint_perceptual"])
    parser.add_argument("--hdr_perceptual_alpha", type=float, default=0.01)
    parser.add_argument("--hdr_loss_mu", type=float, default=5000)
    parser.add_argument("--gan_gen_weight", type=float, default=5e-3)
    parser.add_argument("--gan_dis_weight", type=float, default=1e-2)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


def get_wrapped_model(model):
    return model.module if hasattr(model, "module") else model


def inverse_range_compressor_tensor(y):
    return (torch.pow(1 + 100, y) - 1) / 100


def recover_hdr_from_gm(ldr, gm_compressed):
    return (ldr + 1 / 64) * inverse_range_compressor_tensor(gm_compressed.float())


def range_compressor(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super().__init__()
        self.mu = mu

    def forward(self, pred, label):
        return nn.L1Loss()(range_compressor(pred, self.mu), range_compressor(label, self.mu))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval(),
            vgg[16:23].eval(),
        ])
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=(0, 1, 2, 3), style_layers=()):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = F.interpolate(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = F.interpolate(target, mode="bilinear", size=(224, 224), align_corners=False)

        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                loss += F.l1_loss(act_x @ act_x.permute(0, 2, 1), act_y @ act_y.permute(0, 2, 1))
        return loss


class JointReconPerceptualLoss(nn.Module):
    def __init__(self, alpha=0.01, mu=5000):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.loss_recon = L1MuLoss(mu)
        self.loss_vgg = VGGPerceptualLoss(resize=False)

    def forward(self, input, target):
        input_mu = range_compressor(input, self.mu)
        target_mu = range_compressor(target, self.mu)
        return self.loss_recon(input, target) + self.alpha * self.loss_vgg(input_mu, target_mu)


def build_hdr_loss(args):
    if args.hdr_loss_type == "l1mu":
        return L1MuLoss(mu=args.hdr_loss_mu)
    if args.hdr_loss_type == "joint_perceptual":
        return JointReconPerceptualLoss(alpha=args.hdr_perceptual_alpha, mu=args.hdr_loss_mu)
    raise ValueError(f"Unsupported HDR loss type: {args.hdr_loss_type}")


def parse_dareg_paths(args):
    if args.num_daregs < 1:
        raise ValueError("--num_daregs must be >= 1")
    if args.dareg_paths is None:
        if args.num_daregs != 1:
            raise ValueError("--num_daregs > 1 requires --dareg_paths with comma-separated checkpoints")
        return [args.dareg_path]

    paths = [item.strip() for item in args.dareg_paths.split(",") if item.strip()]
    if len(paths) < args.num_daregs:
        raise ValueError(f"--num_daregs={args.num_daregs}, but only {len(paths)} paths were provided")
    return paths[:args.num_daregs]


def load_dareg(dareg_path):
    dareg = NAFNet()
    checkpoint = torch.load(dareg_path, map_location=torch.device("cpu"))
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
    dareg.load_state_dict(state_dict, strict=True)
    dareg.eval()
    for param in dareg.parameters():
        param.requires_grad = False
    return dareg.to("cuda")


def load_daregs(args):
    daregs = []
    for dareg_path in parse_dareg_paths(args):
        daregs.append(load_dareg(dareg_path))
    return daregs


def build_train_dataloader(args):
    datasets = option.parse_dataset(args.datasets)["datasets"]
    for phase, dataset_opt in datasets.items():
        if phase == "train":
            train_set = DatasetHDRFusion(dataset_opt)
            train_set.normalize = True
            return DataLoader(
                train_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"],
                drop_last=True,
                pin_memory=True,
            )
    raise ValueError(f"No train dataset found in {args.datasets}")


def build_optimizer_and_scheduler(args, params, accelerator):
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    return optimizer, scheduler


def collect_generator_params(model_gen):
    layers_to_opt = []
    for name, param in model_gen.unet.named_parameters():
        if "lora" in name:
            layers_to_opt.append(param)
    layers_to_opt += list(model_gen.unet.conv_in.parameters())
    if hasattr(model_gen, "proj"):
        layers_to_opt += list(model_gen.proj.parameters())
    for name, param in model_gen.vae.named_parameters():
        if "lora" in name or "skip" in name:
            layers_to_opt.append(param)
    for name, param in model_gen.vae_c.named_parameters():
        if "lora" in name:
            layers_to_opt.append(param)
    return layers_to_opt


def collect_discriminator_params(model_reg):
    model_reg.set_train()
    return [param for param in model_reg.parameters() if param.requires_grad]


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model_gen = GMODiff_train(args)
    model_gen.set_train()
    if args.gradient_checkpointing:
        model_gen.unet.enable_gradient_checkpointing()

    model_reg = None
    optimizer_reg = None
    lr_scheduler_reg = None
    layers_to_opt_reg = None
    if args.use_gan_loss:
        model_reg = Discriminator(args=args, accelerator=accelerator)
        model_reg.set_train()
        if args.gradient_checkpointing:
            model_reg.unet.enable_gradient_checkpointing()
        layers_to_opt_reg = collect_discriminator_params(model_reg)
        optimizer_reg, lr_scheduler_reg = build_optimizer_and_scheduler(args, layers_to_opt_reg, accelerator)

    loss_fn = pyiqa.create_metric("dists", device=accelerator.device, as_loss=True)
    hdr_loss = build_hdr_loss(args).to(accelerator.device)
    layers_to_opt = collect_generator_params(model_gen)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(args, layers_to_opt, accelerator)
    dl_train = build_train_dataloader(args)
    daregs = load_daregs(args)

    if args.use_gan_loss:
        model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg = accelerator.prepare(
            model_gen, model_reg, optimizer, optimizer_reg, dl_train, lr_scheduler, lr_scheduler_reg
        )
    else:
        model_gen, optimizer, dl_train, lr_scheduler = accelerator.prepare(model_gen, optimizer, dl_train, lr_scheduler)

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
        total=args.max_train_steps,
    )

    global_step = 0
    while True:
        for batch in dl_train:
            global_step += 1
            if global_step > args.max_train_steps:
                return

            with accelerator.accumulate(model_gen):
                condition = batch["L0_norm"].to("cuda")
                x_tgt_norm = batch["gm_norm"].to("cuda")
                orig_l1 = batch["orig_L1"].to("cuda")
                hdr_gt = batch["H"].to("cuda")
                x_src_dareg = [batch["L0"], batch["L1"], batch["L2"]]

                with torch.no_grad():
                    dareg = random.choice(daregs)
                    visual_embedding, x_src, mask_embedding = dareg.get_visual_embedding(
                        x_src_dareg[0], x_src_dareg[1], x_src_dareg[2]
                    )

                x_pred, latents_pred = model_gen(x_src, [visual_embedding, mask_embedding], condition)
                x_pred = x_pred * 0.5 + 0.5
                x_tgt = x_tgt_norm * 0.5 + 0.5
                pred_hdr = recover_hdr_from_gm(orig_l1, x_pred)

                loss_l1_hdr = hdr_loss(pred_hdr.float(), hdr_gt.float())
                loss_l1_gm = nn.L1Loss()(x_pred.float(), x_tgt.float())
                loss_dists = loss_fn(range_compressor(pred_hdr).float(), range_compressor(hdr_gt).float())
                loss = loss_l1_gm + loss_dists + loss_l1_hdr
                loss_gan_gen = None

                if args.use_gan_loss:
                    gan_model = get_wrapped_model(model_reg)
                    loss_gan_gen = gan_model.compute_generator_loss(latents_pred)
                    loss = loss + loss_gan_gen * args.gan_gen_weight

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            loss_gan_dis = None
            if args.use_gan_loss:
                with accelerator.accumulate(model_reg):
                    gan_model = get_wrapped_model(model_reg)
                    gt_latents = gan_model.compute_gt_latents(x_tgt_norm)
                    loss_gan_dis = gan_model.compute_discriminator_loss(gt_latents, latents_pred) * args.gan_dis_weight
                    accelerator.backward(loss_gan_dis)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt_reg, args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if accelerator.is_main_process:
                    logs = {
                        "loss_l1_gm": loss_l1_gm.detach().item(),
                        "loss_dists": loss_dists.detach().item(),
                        "loss_l1_hdr": loss_l1_hdr.detach().item(),
                    }
                    if loss_gan_gen is not None:
                        logs["loss_gan_gen"] = loss_gan_gen.detach().item()
                    if loss_gan_dis is not None:
                        logs["loss_gan_dis"] = loss_gan_dis.detach().item()
                    progress_bar.set_postfix(**logs)

            if global_step % args.checkpointing_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                outf = os.path.join(checkpoint_dir, f"gmodiff_{global_step}.pkl")
                accelerator.unwrap_model(model_gen).save_model(outf)
                accelerator.unwrap_model(model_gen).set_train()


if __name__ == "__main__":
    main(parse_args())
