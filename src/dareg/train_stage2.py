import argparse
import json
import os
import time

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset_sig17 import SIG17_Training_Dataset, SIG17_Validation_Dataset
from models.nafnet import NAFNet
from utils.gainmap import recover_hdr_from_gm
from utils.utils import AverageMeter, adjust_learning_rate, init_parameters, set_random_seed


def get_args():
    parser = argparse.ArgumentParser(
        description="Train HDR gain-map mask model with Accelerate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, default="../../data/")
    parser.add_argument(
        "--sub_set",
        nargs="+",
        default=[
            "./data/sig17_training_crop256_stride128",
            "./data/tel_training_crop256_stride128"
            "./data/c123_training_crop256_stride128"
        ],
    )
    parser.add_argument("--logdir", type=str, default="../../checkpoints/dareg_stage2")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        type=str,
        default="../../checkpoints/dareg_stage1/latest_checkpoint.pth",
    )
    parser.add_argument("--seed", type=int, default=443)
    parser.add_argument("--init_weights", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_decay_interval", type=int, default=60)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--val_interval", type=int, default=1000)
    return parser.parse_args()


def build_mask_target(pred, label, threshold=0.02):
    error = (pred.detach() - label.detach()).abs()
    luminance_error = 0.299 * error[:, 0] + 0.587 * error[:, 1] + 0.114 * error[:, 2]
    return (luminance_error >= threshold).float().unsqueeze(1)


def strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        print(f"===> No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]), strict=False)


def save_checkpoint(args, accelerator, model, optimizer, epoch, global_step, val_metrics, name):
    if not accelerator.is_local_main_process:
        return

    save_dict = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    accelerator.save(save_dict, os.path.join(args.logdir, name))


def append_val_log(args, epoch, global_step, val_metrics, is_best):
    log_path = os.path.join(args.logdir, "val_mask_metrics.jsonl")
    record = {
        "epoch": epoch,
        "global_step": global_step,
        "is_best": is_best,
        **val_metrics,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def validation(model, val_loader, accelerator):
    model.eval()
    mask_loss = nn.BCELoss()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    with torch.no_grad():
        for batch_data in val_loader:
            batch_ldr0 = batch_data["input0"]
            batch_ldr1 = batch_data["input1"]
            batch_ldr2 = batch_data["input2"]
            label = batch_data["label"]
            ldr1 = batch_data["ldr1"]

            gm_pred, mask_pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            pred = recover_hdr_from_gm(ldr1, gm_pred)
            mask_gt = build_mask_target(pred, label)

            loss = mask_loss(mask_pred, mask_gt)
            mae = torch.mean(torch.abs(mask_pred - mask_gt))
            batch_size = mask_pred.shape[0]
            loss_meter.update(float(loss.item()), batch_size)
            mae_meter.update(float(mae.item()), batch_size)

    model.train()
    return {
        "val_mask_bce": loss_meter.avg,
        "val_mask_mae": mae_meter.avg,
    }


def maybe_run_validation(args, model, val_loader, optimizer, epoch, global_step, best_val_loss, accelerator):
    if global_step == 0 or global_step % args.val_interval != 0:
        return best_val_loss

    val_metrics = validation(model, val_loader, accelerator)
    val_loss = val_metrics["val_mask_bce"]
    is_best = val_loss < best_val_loss

    if accelerator.is_local_main_process:
        print(
            f"Validation step {global_step}: "
            f"mask_bce={val_metrics['val_mask_bce']:.6f}, "
            f"mask_mae={val_metrics['val_mask_mae']:.6f}"
        )
        append_val_log(args, epoch, global_step, val_metrics, is_best)

    save_checkpoint(
        args,
        accelerator,
        model,
        optimizer,
        epoch,
        global_step,
        val_metrics,
        "latest_checkpoint.pth",
    )

    if is_best:
        save_checkpoint(
            args,
            accelerator,
            model,
            optimizer,
            epoch,
            global_step,
            val_metrics,
            "best_checkpoint.pth",
        )
        best_val_loss = val_loss

    return best_val_loss


def train_one_epoch(args, model, train_loader, val_loader, optimizer, epoch, global_step, best_val_loss, accelerator):
    model.train()
    mask_loss = nn.BCELoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    progress_bar = tqdm(
        train_loader,
        total=len(train_loader),
        disable=not accelerator.is_local_main_process,
    )

    for batch_data in progress_bar:
        data_time.update(time.time() - end)

        batch_ldr0 = batch_data["input0"]
        batch_ldr1 = batch_data["input1"]
        batch_ldr2 = batch_data["input2"]
        label = batch_data["label"]
        ldr1 = batch_data["ldr1"]

        gm_pred, mask_pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
        pred = recover_hdr_from_gm(ldr1, gm_pred)
        mask_gt = build_mask_target(pred, label)
        loss = mask_loss(mask_pred, mask_gt)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        best_val_loss = maybe_run_validation(
            args,
            model,
            val_loader,
            optimizer,
            epoch,
            global_step,
            best_val_loss,
            accelerator,
        )

        batch_time.update(time.time() - end)
        end = time.time()
        progress_bar.set_postfix(loss=float(loss.item()), epoch=epoch, step=global_step)

    return global_step, best_val_loss


def main():
    args = get_args()
    set_random_seed(args.seed)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )

    if accelerator.is_local_main_process:
        os.makedirs(args.logdir, exist_ok=True)

    model = NAFNet(use_mask_head=True, freeze_backbone=True)
    if args.init_weights and accelerator.is_local_main_process:
        init_parameters(model)

    load_checkpoint(model, args.resume)

    train_dataset = SIG17_Training_Dataset(sub_set=args.sub_set, is_training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        val_loader,
    )

    if accelerator.is_local_main_process:
        print(
            "===> Start training HDR mask model\n"
            f"Dataset dir:    {args.dataset_dir}\n"
            f"Subset:         {args.sub_set}\n"
            f"Epochs:         {args.epochs}\n"
            f"Batch size:     {args.batch_size}\n"
            f"Val interval:   {args.val_interval}\n"
            f"Learning rate:  {args.lr}\n"
            f"Training size:  {len(train_loader.dataset)}\n"
            f"Validation size:{len(val_loader.dataset)}\n"
            f"Device:         {accelerator.device}"
        )

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        global_step, best_val_loss = train_one_epoch(
            args,
            model,
            train_loader,
            val_loader,
            optimizer,
            epoch,
            global_step,
            best_val_loss,
            accelerator,
        )


if __name__ == "__main__":
    main()
