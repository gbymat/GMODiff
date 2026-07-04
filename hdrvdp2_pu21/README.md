# HDR-VDP2 and PU21 Evaluation

This folder provides a MATLAB script for evaluating HDR reconstruction results with:

- `HDR-VDP2`
- `PU21-PSNR`
- `PU21-SSIM`

The entry point is `./run/test.m`. The script automatically adds the bundled HDR-VDP2 code and the PU21 MATLAB implementation to the MATLAB path.

## Requirements

- MATLAB
- Image Processing Toolbox, required by the PU21 SSIM implementation
- HDR images saved as `.hdr` files

## Expected Data Layout

By default, `test.m` expects the following directory structure:

```text
benchmark_dir/
  gt/
    1.hdr
    2.hdr
    ...
  MethodA/
    1.hdr
    2.hdr
    ...
  MethodB/
    1.hdr
    2.hdr
    ...
```

The prediction file names must match the ground-truth file names. For example, if the ground-truth file is `12.hdr`, each evaluated method should also provide a prediction named `12.hdr`.

If your predictions are stored under an additional subfolder, such as:

```text
benchmark_dir/
  gt/
    1.hdr
  MethodA/
    pred_hdr/
      1.hdr
```

use the optional third argument of `build_method_config`, as shown below.

## Configuration

Open `test.m` and edit the configuration block:

```matlab
benchmark_dir = 'path/to/benchmark';
method_names = {'MethodA', 'MethodB'};

cfg.ref_dir = fullfile(benchmark_dir, 'gt');
cfg.methods = build_method_config(benchmark_dir, method_names);
```

For predictions stored in `benchmark_dir/<method_name>/pred_hdr/`, use:

```matlab
cfg.methods = build_method_config(benchmark_dir, method_names, 'pred_hdr');
```

For a fully custom layout, set the paths directly:

```matlab
cfg.ref_dir = 'path/to/gt_hdr';
cfg.methods = [
    struct('name', 'MethodA', 'pred_dir', 'path/to/method_a_hdr')
    struct('name', 'MethodB', 'pred_dir', 'path/to/method_b_hdr')
];
```

## Running

Start MATLAB, change to this folder, and run:

```matlab
test
```

The script prints per-image scores and then reports the mean scores for each method:

![](../../figs/metrics.png)

## Evaluation Parameters

The default script maps the maximum value of each ground-truth HDR image to:

```matlab
cfg.peak_luminance = 1000; % cd/m^2
```

The prediction is scaled with the same factor as its corresponding ground-truth image. This keeps the evaluation protocol consistent across methods.

HDR-VDP2 also depends on viewing geometry:

```matlab
cfg.display_size_px = [1500 1000]; % [width height]
cfg.display_diagonal_in = 30;
cfg.viewing_distance_m = 0.5;
cfg.color_encoding = 'rgb-bt.709' or 'rgb-native';
```

Use the same parameters for all compared methods. Changing these parameters can change the absolute HDR-VDP2 score, so they should be reported when publishing results.

PU21-PSNR and PU21-SSIM do not use the display size or viewing distance, but they do depend on the luminance scaling protocol.

## Notes

- Missing prediction files are skipped by default. Set `cfg.skip_missing_predictions = false` if missing files should stop the evaluation.
- Ground-truth and prediction images must have the same resolution.
- For fair comparison, all methods should be evaluated with the same file list, luminance scaling, display parameters, and color encoding.
