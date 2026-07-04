# Baseline Inference

This folder keeps inference code for three comparison baselines:

- `HDR-Transformer`
- `AFUNet`
- `SCTNet`

Run one baseline with `1=HDR-Transformer`, `2=AFUNet`, `3=SCTNet`:

```bash
bash infer_baseline.sh 1 /path/to/dataset results/baselines/cavit
bash infer_baseline.sh 2 /path/to/dataset results/baselines/afunet
bash infer_baseline.sh 3 /path/to/dataset results/baselines/sctnet
```

Use patch inference with:

```bash
bash infer_baseline.sh 3 /path/to/dataset results/baselines/sctnet_patch patch 256
```

The dataset path should contain a `Test` folder. Each test scene should include
three `.tif` LDR images, `exposure.txt`, and `HDRImg.hdr`.
