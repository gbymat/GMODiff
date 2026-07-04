%% HDR-VDP2 and PU21 evaluation for HDR reconstruction results.
% Edit the configuration block below, then run this script from MATLAB.

clear;
clc;

%% Configuration
script_dir = fileparts(mfilename('fullpath'));
metric_root = fileparts(script_dir);

% Add HDR-VDP2 and PU21 MATLAB code to the search path.
addpath(metric_root);
addpath(fullfile(metric_root, 'pu21', 'matlab'));

% Benchmark layout:
%   benchmark_dir/gt/*.hdr
%   benchmark_dir/<method_name>/*.hdr
benchmark_dir = 'path/to/benchmark';
method_names = {'MethodA'};

cfg.ref_dir = fullfile(benchmark_dir, 'gt');
cfg.methods = build_method_config(benchmark_dir, method_names);

% HDR images are assumed to be in relative linear RGB. Each reference image
% is mapped to this display peak; the prediction is scaled by the same factor.
cfg.peak_luminance = 1000; % cd/m^2

% HDR-VDP2 viewing/display settings.
cfg.display_size_px = [1500 1000]; % [width height]
cfg.viewing_distance_m = 0.5;
cfg.display_diagonal_in = 30;
cfg.color_encoding = 'rgb-bt.709';
cfg.hdrvdp_options = {};

% Missing predictions are skipped by default so a partial benchmark can still
% report valid means for the files that exist.
cfg.skip_missing_predictions = true;

%% Evaluation
validate_inputs(cfg);

ppd = hdrvdp_pix_per_deg( ...
    cfg.display_diagonal_in, ...
    cfg.display_size_px, ...
    cfg.viewing_distance_m);

ref_files = dir(fullfile(cfg.ref_dir, '*.hdr'));
fprintf('Reference directory: %s\n', cfg.ref_dir);
fprintf('Reference images: %d\n\n', numel(ref_files));

all_results = struct([]);

for method_idx = 1:numel(cfg.methods)
    method = cfg.methods(method_idx);
    fprintf('=== %s ===\n', method.name);
    fprintf('Prediction directory: %s\n', method.pred_dir);

    metrics = evaluate_method(ref_files, cfg.ref_dir, method.pred_dir, cfg, ppd);
    all_results(method_idx).name = method.name; %#ok<SAGROW>
    all_results(method_idx).metrics = metrics; %#ok<SAGROW>

    print_method_summary(metrics);
end

%% Local functions
function methods = build_method_config(benchmark_dir, method_names, pred_subdir)
    if nargin < 3
        pred_subdir = '';
    end

    methods = repmat(struct('name', '', 'pred_dir', ''), 1, numel(method_names));

    for idx = 1:numel(method_names)
        methods(idx).name = method_names{idx};
        methods(idx).pred_dir = fullfile(benchmark_dir, method_names{idx}, pred_subdir);
    end
end

function validate_inputs(cfg)
    if ~exist('hdrvdp', 'file')
        error('HDR-VDP2 function hdrvdp.m is not on the MATLAB path.');
    end

    if ~exist('pu21_metric', 'file')
        error('PU21 function pu21_metric.m is not on the MATLAB path.');
    end

    if ~exist(cfg.ref_dir, 'dir')
        error('Reference directory does not exist: %s', cfg.ref_dir);
    end

    for idx = 1:numel(cfg.methods)
        if ~exist(cfg.methods(idx).pred_dir, 'dir')
            error('Prediction directory does not exist: %s', cfg.methods(idx).pred_dir);
        end
    end
end

function metrics = evaluate_method(ref_files, ref_dir, pred_dir, cfg, ppd)
    n_files = numel(ref_files);
    metrics.file = strings(n_files, 1);
    metrics.hdrvdp2 = nan(n_files, 1);
    metrics.pu21_psnr = nan(n_files, 1);
    metrics.pu21_ssim = nan(n_files, 1);
    metrics.used = false(n_files, 1);

    for file_idx = 1:n_files
        ref_name = ref_files(file_idx).name;
        ref_path = fullfile(ref_dir, ref_name);
        pred_path = fullfile(pred_dir, ref_name);
        metrics.file(file_idx) = string(ref_name);

        if ~exist(pred_path, 'file')
            message = sprintf('Missing prediction for %s', ref_name);
            if cfg.skip_missing_predictions
                warning('%s. Skipping.', message);
                continue;
            end
            error('%s: %s', message, pred_path);
        end

        fprintf('[%d/%d] %s\n', file_idx, n_files, ref_name);

        try
            [I_ref, I_pred] = read_and_normalize_pair(ref_path, pred_path, cfg.peak_luminance);

            res = hdrvdp(I_pred, I_ref, cfg.color_encoding, ppd, cfg.hdrvdp_options);
            metrics.hdrvdp2(file_idx) = res.Q;
            metrics.pu21_psnr(file_idx) = pu21_metric(I_pred, I_ref, 'PSNR');
            metrics.pu21_ssim(file_idx) = pu21_metric(I_pred, I_ref, 'SSIM');
            metrics.used(file_idx) = true;

            fprintf('  HDR-VDP2 %.4f | PU21-PSNR %.4f | PU21-SSIM %.4f\n', ...
                metrics.hdrvdp2(file_idx), ...
                metrics.pu21_psnr(file_idx), ...
                metrics.pu21_ssim(file_idx));
        catch ME
            warning('Failed on %s: %s', ref_name, ME.message);
        end
    end
end

function [I_ref, I_pred] = read_and_normalize_pair(ref_path, pred_path, peak_luminance)
    I_ref = hdrread(ref_path);
    I_pred = hdrread(pred_path);

    if ~isequal(size(I_ref), size(I_pred))
        error('Image sizes differ. Reference: [%s], prediction: [%s].', ...
            num2str(size(I_ref)), num2str(size(I_pred)));
    end

    max_ref = max(I_ref(:));
    if ~isfinite(max_ref) || max_ref <= 0
        error('Reference image has invalid maximum luminance: %g', max_ref);
    end

    scale = peak_luminance / max_ref;
    I_ref = I_ref * scale;
    I_pred = I_pred * scale;
end

function print_method_summary(metrics)
    valid = metrics.used;
    fprintf('---\n');
    fprintf('Valid image pairs: %d/%d\n', nnz(valid), numel(valid));

    if ~any(valid)
        fprintf('No valid image pairs were evaluated.\n\n');
        return;
    end

    fprintf('HDR-VDP2:  %.4f\n', mean(metrics.hdrvdp2(valid), 'omitnan'));
    fprintf('PU21-PSNR: %.4f\n', mean(metrics.pu21_psnr(valid), 'omitnan'));
    fprintf('PU21-SSIM: %.4f\n', mean(metrics.pu21_ssim(valid), 'omitnan'));
    fprintf('\n');
end
