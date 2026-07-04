import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_sig17 import SIG17_Training_Dataset, SIG17_Validation_Dataset
from models.loss import JointReconPerceptualLoss, L1MuLoss
from models.nafnet import NAFNet
from utils.gainmap import recover_hdr_from_gm
from utils.utils import AverageMeter, adjust_learning_rate, batch_psnr, batch_psnr_mu, init_parameters, set_random_seed


def get_args():
    parser = argparse.ArgumentParser(
        description="Pretrain NAFNet for HDR gain-map reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, default="../../data")
    parser.add_argument(
        "--sub_set",
        nargs="+",
        default=[
            "./data/sig17_training_crop256_stride128",
            "./data/tel_training_crop256_stride128"
            "./data/c123_training_crop256_stride128"
        ],
    )
    parser.add_argument("--logdir", type=str, default="../../checkpoints/dareg_stage1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--init_weights", action="store_true", default=False)
    parser.add_argument("--loss_func", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_decay_interval", type=int, default=60)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=500)
    return parser.parse_args()


def strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_checkpoint(model, optimizer, checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]), strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_checkpoint(args, accelerator, model, optimizer, epoch, metrics, name):
    if not accelerator.is_local_main_process:
        return

    save_dict = {
        "epoch": epoch + 1,
        "state_dict": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    accelerator.save(save_dict, os.path.join(args.logdir, name))


def append_val_log(args, epoch, metrics, is_best):
    record = {"epoch": epoch, "is_best": is_best, **metrics}
    with open(os.path.join(args.logdir, "val_metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_loss(gm_pred, pred_hdr, gm_gt, label, criterion):
    loss_gm = nn.L1Loss()(gm_pred, gm_gt)
    loss_hdr = criterion(pred_hdr, label)
    return loss_gm + loss_hdr, loss_gm, loss_hdr


def train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, accelerator):
    model.train()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    progress_bar = tqdm(
        train_loader,
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
    )

    for batch_idx, batch_data in enumerate(progress_bar):
        data_time.update(time.time() - end)

        batch_ldr0 = batch_data["input0"]
        batch_ldr1 = batch_data["input1"]
        batch_ldr2 = batch_data["input2"]
        label = batch_data["label"]
        gm = batch_data["gm"]
        ldr1 = batch_data["ldr1"]

        gm_pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        pred_hdr = recover_hdr_from_gm(ldr1, gm_pred)
        loss, loss_gm, loss_hdr = compute_loss(gm_pred, pred_hdr, gm, label, criterion)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        loss_meter.update(float(loss.item()), batch_ldr0.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()

        if accelerator.is_local_main_process and batch_idx % args.log_interval == 0:
            print(
                f"Train epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"loss={loss.item():.6f} gm={loss_gm.item():.6f} hdr={loss_hdr.item():.6f} "
                f"time={batch_time.val:.3f} data={data_time.val:.3f}"
            )
        progress_bar.set_postfix(loss=float(loss.item()), epoch=epoch)

    return {"train_loss": loss_meter.avg}


def validation(model, val_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    mu_psnr_meter = AverageMeter()

    with torch.no_grad():
        for batch_data in val_loader:
            batch_ldr0 = batch_data["input0"]
            batch_ldr1 = batch_data["input1"]
            batch_ldr2 = batch_data["input2"]
            label = batch_data["label"]
            gm = batch_data["gm"]
            ldr1 = batch_data["ldr1"]

            gm_pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            pred_hdr = recover_hdr_from_gm(ldr1, gm_pred)
            loss, _, _ = compute_loss(gm_pred, pred_hdr, gm, label, criterion)

            batch_size = batch_ldr0.shape[0]
            loss_meter.update(float(loss.item()), batch_size)
            psnr_meter.update(float(batch_psnr(pred_hdr, label, 1.0)), batch_size)
            mu_psnr_meter.update(float(batch_psnr_mu(pred_hdr, label, 1.0)), batch_size)

    model.train()
    return {
        "val_loss": loss_meter.avg,
        "val_psnr": psnr_meter.avg,
        "val_mu_psnr": mu_psnr_meter.avg,
    }


def main():
    args = get_args()
    set_random_seed(args.seed)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )

    if accelerator.is_local_main_process:
        os.makedirs(args.logdir, exist_ok=True)

    model = NAFNet(use_mask_head=False, freeze_backbone=False)
    if args.init_weights:
        init_parameters(model)

    loss_dict = {0: L1MuLoss, 1: JointReconPerceptualLoss}
    criterion = loss_dict[args.loss_func]()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    checkpoint = load_checkpoint(model, optimizer, args.resume)
    if checkpoint is not None:
        args.start_epoch = checkpoint.get("epoch", args.start_epoch)
        if accelerator.is_local_main_process:
            print(f"Loaded checkpoint from {args.resume}, start epoch: {args.start_epoch}")

    train_dataset = SIG17_Training_Dataset(sub_set=args.sub_set, is_training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    if accelerator.is_local_main_process:
        print(
            "===> Start HDR gain-map pretraining\n"
            f"Dataset dir:    {args.dataset_dir}\n"
            f"Subsets:        {args.sub_set}\n"
            f"Epochs:         {args.epochs}\n"
            f"Start epoch:    {args.start_epoch}\n"
            f"Batch size:     {args.batch_size}\n"
            f"Loss function:  {args.loss_func}\n"
            f"Learning rate:  {args.lr}\n"
            f"Training size:  {len(train_loader.dataset)}\n"
            f"Validation size:{len(val_loader.dataset)}\n"
            f"Device:         {accelerator.device}\n"
            f"Distributed:    {accelerator.num_processes > 1}"
        )

    best_mu_psnr = -1.0
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train_metrics = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, accelerator)
        val_metrics = validation(model, val_loader, criterion)
        metrics = {**train_metrics, **val_metrics}
        is_best = val_metrics["val_mu_psnr"] > best_mu_psnr

        if accelerator.is_local_main_process:
            print(
                f"Validation epoch {epoch}: "
                f"loss={val_metrics['val_loss']:.6f}, "
                f"psnr={val_metrics['val_psnr']:.4f}, "
                f"mu_psnr={val_metrics['val_mu_psnr']:.4f}"
            )
            append_val_log(args, epoch, metrics, is_best)

        save_checkpoint(args, accelerator, model, optimizer, epoch, metrics, "latest_checkpoint.pth")
        save_checkpoint(args, accelerator, model, optimizer, epoch, metrics, f"latest_checkpoint_epoch_{epoch}.pth")
        if is_best:
            best_mu_psnr = val_metrics["val_mu_psnr"]
            save_checkpoint(args, accelerator, model, optimizer, epoch, metrics, "best_checkpoint.pth")
            if accelerator.is_local_main_process:
                with open(os.path.join(args.logdir, "best_checkpoint.json"), "w", encoding="utf-8") as f:
                    json.dump({"best_epoch": epoch, **metrics}, f, indent=2)


if __name__ == "__main__":
    main()
