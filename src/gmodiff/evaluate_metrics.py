import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch


NO_REFERENCE_METRICS = ("maniqa", "musiq", "clipiqa")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate stage-3 GMODiff HDR outputs with DISTS, MANIQA, MUSIQ, and CLIPIQA."
    )
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted .hdr files.")
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Directory containing ground-truth HDR files. Required for DISTS.",
    )
    parser.add_argument("--output_csv", type=str, default="../../results/gmodiff_metrics.csv")
    parser.add_argument("--summary_json", type=str, default="../../results/gmodiff_metrics_summary.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mu", type=float, default=5000.0)
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Do not clip mu-compressed HDR values to [0, 1]. Use this to match older hdr_mu evaluation scripts.",
    )
    parser.add_argument(
        "--allow_resize",
        action="store_true",
        help="Resize GT to prediction size if shapes differ. Disabled by default to catch evaluation mistakes.",
    )
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def natural_key(path):
    stem = Path(path).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def list_pred_hdrs(pred_dir):
    paths = sorted(Path(pred_dir).glob("*.hdr"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .hdr predictions found in {pred_dir}")
    return paths


def list_gt_hdrs(gt_dir):
    root = Path(gt_dir)
    preferred = sorted(root.rglob("HDRImg.hdr"), key=lambda p: str(p).lower())
    paths = preferred if preferred else sorted(root.rglob("*.hdr"), key=lambda p: str(p).lower())
    if not paths:
        raise FileNotFoundError(f"No .hdr ground-truth files found in {gt_dir}")
    return paths


def read_hdr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read HDR image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.maximum(image.astype(np.float32), 0.0)


def range_compress(image, mu):
    return np.log1p(mu * image) / np.log1p(mu)


def to_tensor(image, device, clip=True):
    if clip:
        image = np.clip(image, 0.0, 1.0)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.to(device)


def image_stats(image, prefix):
    return {
        f"{prefix}_min": float(np.min(image)),
        f"{prefix}_max": float(np.max(image)),
        f"{prefix}_mean": float(np.mean(image)),
    }


def build_metrics(device, has_gt):
    metrics = {name: pyiqa.create_metric(name, device=device) for name in NO_REFERENCE_METRICS}
    if has_gt:
        metrics["dists"] = pyiqa.create_metric("dists", device=device)
    for metric in metrics.values():
        metric.eval()
        if hasattr(metric, "net"):
            metric.net.eval()
    return metrics


def evaluate_pair(metrics, pred_tensor, gt_tensor=None):
    scores = {}
    with torch.no_grad():
        for name in NO_REFERENCE_METRICS:
            scores[name] = float(metrics[name](pred_tensor).detach().cpu().item())
        if gt_tensor is not None:
            scores["dists"] = float(metrics["dists"](pred_tensor, gt_tensor).detach().cpu().item())
    return scores


def summarize(rows, metric_names):
    summary = {}
    for metric in metric_names:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values),
            }
    return summary


def package_version(module):
    return getattr(module, "__version__", "unknown")


def build_metadata(args, metric_names):
    return {
        "metrics": metric_names,
        "input": {
            "pred_dir": args.pred_dir,
            "gt_dir": args.gt_dir,
            "mu": args.mu,
            "seed": args.seed,
            "clip_to_01": not args.no_clip,
            "hdr_preprocess": "log1p(mu * hdr) / log1p(mu)",
        },
        "environment": {
            "pyiqa": package_version(pyiqa),
            "torch": package_version(torch),
            "opencv": package_version(cv2),
            "device": args.device,
        },
    }


def make_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    set_deterministic(args.seed)
    pred_paths = list_pred_hdrs(args.pred_dir)
    gt_paths = list_gt_hdrs(args.gt_dir) if args.gt_dir else []
    if gt_paths and len(gt_paths) != len(pred_paths):
        raise ValueError(f"Prediction/GT count mismatch: {len(pred_paths)} predictions, {len(gt_paths)} GT files")

    device = torch.device(args.device)
    metrics = build_metrics(device, has_gt=bool(gt_paths))
    rows = []

    for idx, pred_path in enumerate(pred_paths):
        pred = read_hdr(pred_path)
        gt = None
        gt_path = None
        if gt_paths:
            gt_path = gt_paths[idx]
            gt = read_hdr(gt_path)
            if gt.shape != pred.shape:
                if not args.allow_resize:
                    raise ValueError(f"Shape mismatch for {pred_path.name}: pred={pred.shape}, gt={gt.shape}")
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

        pred_mu = range_compress(pred, args.mu)
        gt_mu = range_compress(gt, args.mu) if gt is not None else None
        pred_tensor = to_tensor(pred_mu, device, clip=not args.no_clip)
        gt_tensor = to_tensor(gt_mu, device, clip=not args.no_clip) if gt_mu is not None else None
        scores = evaluate_pair(metrics, pred_tensor, gt_tensor)
        row = {
            "index": idx,
            "prediction": str(pred_path),
            "ground_truth": str(gt_path) if gt_path is not None else "",
            **image_stats(pred_mu, "pred_mu"),
            **scores,
        }
        rows.append(row)
        score_text = ", ".join(f"{key}={value:.6f}" for key, value in scores.items())
        print(f"[{idx + 1}/{len(pred_paths)}] {pred_path.name}: {score_text}")

    metric_names = list(NO_REFERENCE_METRICS) + (["dists"] if gt_paths else [])
    summary = {
        "meta": build_metadata(args, metric_names),
        "scores": summarize(rows, metric_names),
    }
    make_parent_dir(args.output_csv)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "prediction",
                "ground_truth",
                "pred_mu_min",
                "pred_mu_max",
                "pred_mu_mean",
                *metric_names,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    make_parent_dir(args.summary_json)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved per-image metrics to {args.output_csv}")
    print(f"Saved summary metrics to {args.summary_json}")


if __name__ == "__main__":
    main()
