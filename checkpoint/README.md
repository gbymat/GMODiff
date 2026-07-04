# Checkpoints

This directory is reserved for locally downloaded or trained checkpoints.

Suggested layout:

```text
checkpoints/
  dareg_stage1/
    best_checkpoint.pth
    latest_checkpoint.pth
  dareg_stage2/
    best_checkpoint.pth
```

For GMODiff stage-3 training and inference, put pretrained release weights in:

```text
model_zoo/
  dareg_mask.pth
  gmodiff.pkl
```

These checkpoint files are kept with the release tree when present.
