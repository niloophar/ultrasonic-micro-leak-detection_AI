"""
Contrastive Encoder Pipeline CLI.

Entry point for Ansatz A: Single Encoder + Supervised Contrastive Loss
over all series, recycling the anomaly_detection CNNAutoEncoder backbone.

Usage:
    python -m src.contrastive.main --run train             # Single training run + full diagnostics
    python -m src.contrastive.main --run tune              # Optuna study (200 trials)
    python -m src.contrastive.main --run loso              # Leave-one-series-out eval
    python -m src.contrastive.main --run train --plot      # With diagnostic plots
"""
import argparse
import json
import os
import sys
from pathlib import Path


# Feature-group presets for the contrastive encoder (see CLAUDE.md §"Feature Groups").
# Swap via --feature-group at runtime; avoids editing selected_features in the config.
FEATURE_GROUPS = {
    'g5': [f'norm_interval_down_{i}' for i in range(9)]
          + [f'norm_interval_up_{i}' for i in range(9)],
    'g6': [f'norm_interval_diff_{i}' for i in range(9)],
    'g7': [f'echo_ratio_{i}' for i in range(9)],
    'g8': [f'norm_interval_down_{i}_tc' for i in range(9)]
          + [f'norm_interval_up_{i}_tc' for i in range(9)],
    # Gruppe 9 — total-length combinations (operate on period_sum_{up,down}
    # rather than per-ZC intervals). Designed for micro-leak + drift scenarios
    # where per-ZC features drown in noise; the total traversal length and its
    # symmetric / temperature-normalised combinations keep more usable signal.
    'g9': [
        'period_sum_total',
        'period_diff_signed',
        'flow_index',
        'period_sum_total_per_tpipe',
        'period_diff_per_tpipe',
        'flow_index_per_tpipe',
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="KILD Contrastive Encoder Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.contrastive.main --run tune
    python -m src.contrastive.main --run train --plot
    python -m src.contrastive.main --run loso
        """
    )

    parser.add_argument(
        '--run',
        nargs='+',
        choices=['train', 'tune', 'loso', 'chrono_extrap', 'chrono_extrap_temp', 'chrono_extrap_pooled', 'chrono_extrap_pooled_trainthresh', 'eval', 'predict', 'detector_eval'],
        default=['train'],
        help='Steps to run: train, tune, loso, chrono_extrap, chrono_extrap_temp, chrono_extrap_pooled, chrono_extrap_pooled_trainthresh, eval, predict, detector_eval',
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        default=os.getenv('PLOT_ENABLED', 'false').lower() == 'true',
        help='Enable plotting (default: disabled)',
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        default=False,
        help='Disable plotting',
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'contrastive'),
        help='Path to contrastive config directory (default: configs/contrastive)',
    )

    parser.add_argument(
        '--preparation-config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'preparation'),
        help='Path to preparation config directory',
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=os.getenv('CONTAINER_DATA_PATH', 'data'),
        help='Base data directory (default: data)',
    )

    parser.add_argument(
        '--img-path',
        type=str,
        default=os.getenv('CONTAINER_IMG_PATH', 'images'),
        help='Base image directory for plots (default: images)',
    )

    parser.add_argument(
        '--only-files',
        nargs='+',
        type=str,
        default=None,
        help='Only process these file stems',
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=int(os.getenv('MAX_WORKERS', 1)),
        help='Max number of parallel workers (default: 1)',
    )

    parser.add_argument(
        '--encoder-type',
        type=str,
        choices=['autoencoder', 'classifier'],
        default=None,
        help='Override encoder_type in encoder.json (autoencoder = MSE + SupCon, classifier = CE + SupCon).',
    )

    parser.add_argument(
        '--encoder-arch',
        type=str,
        choices=['cnn', 'transformer'],
        default=None,
        help=(
            "Override encoder_arch in encoder.json. 'cnn' (default) uses the CNN backbone "
            "(CNNAutoEncoder recycled from anomaly_detection). 'transformer' uses a "
            "Transformer-encoder + mirrored decoder; only valid with encoder_type='autoencoder'."
        ),
    )

    parser.add_argument(
        '--feature-group',
        type=str,
        choices=sorted(FEATURE_GROUPS.keys()),
        default=None,
        help='Override selected_features using a preset (g5/g6/g7/g8). See FEATURE_GROUPS in this module.',
    )

    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help='Override output_root in encoder.json (artefact folder; typically ./contrastive_g<N>_<ae|clf>).',
    )

    parser.add_argument(
        '--study-name',
        type=str,
        default=None,
        help='Override study_name in encoder.json (Optuna SQLite study name; pair with matching --output-root).',
    )

    parser.add_argument(
        '--contrastive-loss',
        type=str,
        choices=['supcon', 'rnc', 'both', 'deepsad', 'none'],
        default=None,
        help=(
            "Override contrastive_loss in encoder.json: supcon (class-based), "
            "rnc (continuous target via rnc_target), both (weighted sum), "
            "deepsad (asymmetric one-class, mutually exclusive with supcon/rnc), "
            "or none."
        ),
    )

    parser.add_argument(
        '--warmup-epochs',
        type=int,
        default=None,
        help=(
            'Override warmup_epochs in encoder.json. Linear LR warmup over N epochs '
            'before ReduceLROnPlateau kicks in; best-epoch tracking is gated post-warmup. '
            'For DeepSAD, 10 is recommended to prevent best_epoch=0/1 pathology. '
            'Default 0 (no warmup) for CNN; tuned by Optuna for Transformer.'
        ),
    )

    parser.add_argument(
        '--optuna-warmup-steps',
        type=int,
        default=None,
        help=(
            'Override optuna_warmup_steps in encoder.json (MedianPruner n_warmup_steps). '
            'For DeepSAD, 30 is recommended. Default: 15.'
        ),
    )

    parser.add_argument(
        '--optuna-n-startup-trials',
        type=int,
        default=None,
        help=(
            'Override optuna_n_startup_trials in encoder.json (TPE random phase). '
            'For DeepSAD, 30 is recommended. Default: 20.'
        ),
    )

    parser.add_argument(
        '--optuna-n-trials',
        type=int,
        default=None,
        help=(
            'Override optuna_n_trials in encoder.json (total Optuna trials). '
            'For DeepSAD, at least 150 is recommended (200 for final validation). '
            'Default: 200.'
        ),
    )

    parser.add_argument(
        '--rnc-target',
        type=str,
        choices=['temp_pipe', 'temp_amb', 'temp_diff', 'series_index'],
        default=None,
        help='Override rnc_target in encoder.json (which continuous signal drives the RNC loss).',
    )

    parser.add_argument(
        '--cnn-channel-mode',
        type=str,
        choices=['single', 'dual'],
        default=None,
        help="Override cnn_channel_mode in encoder.json. Use 'single' when the feature set is not up/down-symmetric (e.g. G9).",
    )

    # --- Encoder train / GMM eval split overrides ----------------------------
    # Decouple "which series feed the encoder" from "which series the GMM is
    # scored against". Default behaviour (no flag set) reproduces splits.json
    # exactly. Override either side to run Eval A / Eval B style configurations
    # without regenerating splits.json. See CLAUDE.md §"Eval recipes".
    parser.add_argument(
        '--encoder-train-series',
        nargs='+',
        type=str,
        default=None,
        help=(
            'Override which series train the encoder. Defaults to splits.json '
            'train_series (winter-4). Example: --encoder-train-series '
            '10.01.2026 24.01.2026 7.02.2026 7.03.2026 15.03.2026 5.04.2026 '
            '11.04.2026 (Eval B style — leaves 21.02 + 18.04 as edges).'
        ),
    )
    parser.add_argument(
        '--gmm-eval-series',
        nargs='+',
        type=str,
        default=None,
        help=(
            'Override which series the GMM is scored against. Defaults to '
            'splits.json test_series. Example for Eval B: --gmm-eval-series '
            '21.02.2026 18.04.2026.'
        ),
    )
    parser.add_argument(
        '--loso-held-out',
        nargs='+',
        type=str,
        default=None,
        help=(
            'Run only these series as held-out folds. Training pool for each '
            'fold is still the full loso_series from encoder.json. Results are '
            'appended to any existing loso_results.json in the output root. '
            'Example: --loso-held-out ha-6.06.2026 ha-2.06.2026'
        ),
    )
    parser.add_argument(
        '--chrono-min-train',
        type=int,
        default=None,
        help=(
            'chrono_extrap only. Minimum number of (chronologically earliest) '
            'series in the first training window, i.e. the split index the walk '
            'forward starts at. Default 4 — matches "train series 1-4, test '
            'series 5" as the first split. Splits then expand one series at a '
            'time (train 1-5/test 6, train 1-6/test 7, ...) through the full '
            'chrono_series list.'
        ),
    )
    parser.add_argument(
        '--chrono-test-series',
        nargs='+',
        type=str,
        default=None,
        help=(
            'chrono_extrap only. Run only the splits whose held-out (test) '
            'series is in this list — the training window for each is still '
            'everything chronologically before it. Results append to any '
            'existing chrono_extrap_results.json in the output root. '
            'Example: --chrono-test-series ha-2.06.2026 ha-20.06.2026'
        ),
    )
    parser.add_argument(
        '--gmm-eval-mode',
        type=str,
        choices=[
            'per_series', 'global', 'global_conditional',
            'global_hierarchical', 'both', 'all',
        ],
        default=None,
        help=(
            'GMM strategy: per_series (legacy — fresh GMM fitted per eval '
            "series on its own NoFlow), global (single GMM on encoder-train "
            'NoFlow, scored across all eval series — deployment-realistic), '
            'global_conditional (linear z~T regression on centroids + GMM on '
            'residuals — mean conditioning), global_hierarchical (per-series'
            '-centered Σ + T-extrapolated centroid — separates centroid '
            'model from scatter model, attacks the per-site/global gap '
            'directly). '
            "'both' = per_series + global (back-compat), 'all' = all four. "
            'Default: per_series.'
        ),
    )
    parser.add_argument(
        '--gmm-n-components',
        type=int,
        default=None,
        help=(
            'Number of Gaussian components for every GMM eval mode (global, '
            'per_series, etc). Default 1 (single blob). >1 lets the mixture '
            'discover sub-populations on its own — e.g. one component per '
            'installation site — without any per-site refit or manual '
            'assignment; useful for testing whether a single shared detector '
            'can cover multiple houses uniformly.'
        ),
    )
    parser.add_argument(
        '--eval-noflow-split-mode',
        type=str,
        default=None,
        choices=['fair', 'optimistic_upper_bound'],
        help=(
            "How per_series eval splits each series' NoFlow. 'fair' (default): "
            'earliest NoFlow files calibrate the GMM + q95 threshold, only later, '
            'disjoint NoFlow files + all Leak files are scored — calibration NoFlow '
            "never appears in any reported metric. 'optimistic_upper_bound': the "
            'OLD behaviour (fits AND scores on the same series NoFlow) — must be '
            'requested explicitly, never the default. See study note '
            '_mnt_d_projekte_KILD_inoson_kild_study_center_check_and_noflow_split.txt.'
        ),
    )
    parser.add_argument(
        '--eval-calibration-file-ratio',
        type=float,
        default=None,
        help=(
            'Fraction of each series\' NoFlow FILES (earliest, by date) used for '
            "calibration under --eval-noflow-split-mode=fair. Default 0.5. Always "
            'leaves >=1 file for test when >1 NoFlow file exists.'
        ),
    )
    parser.add_argument(
        '--eval-min-calibration-patches',
        type=int,
        default=None,
        help=(
            'Below this many calibration-NoFlow patches, fall back to a mean-only '
            '(no covariance) anomaly score instead of fitting a full GMM covariance '
            'from too little data. Default 500.'
        ),
    )
    parser.add_argument(
        '--gmm-temp-target',
        nargs='+',
        type=str,
        choices=['temp_pipe', 'temp_amb', 'temp_diff'],
        default=None,
        help=(
            'Temperature signal(s) used by global_conditional GMM as the '
            'covariate(s) in the centroid regression. Pass one name for a '
            'single-covariate regression (e.g. --gmm-temp-target temp_pipe) '
            'or two for a joint multi-covariate regression '
            '(e.g. --gmm-temp-target temp_pipe temp_amb). Default: temp_pipe.'
        ),
    )

    parser.add_argument(
        '--as-blind-series',
        nargs='+',
        type=str,
        default=None,
        help=(
            'Treat these already-labeled series as blind during --run predict '
            '(sanity test). Labels are ignored for prediction; the output CSV '
            'includes a true_class column for comparison. The series must NOT '
            'be in encoder_train_series.'
        ),
    )

    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help=(
            'Override data_root in encoder.json (relative path under '
            '--data-path where prep wrote splits.json + prepared/). '
            'Used for ablation runs that point at parallel prep outputs '
            'such as ./ablation/pw32_f0.'
        ),
    )

    args = parser.parse_args()

    plot_enabled = args.plot and not args.no_plot
    max_workers = args.max_workers

    # Resolve paths
    if os.getenv('DOCKER_ENV'):
        config_dir = Path(args.config_dir)
        prep_config_dir = Path(args.preparation_config_dir)
        data_path = Path(args.data_path)
        img_path = Path(args.img_path)
    else:
        root_path = Path(__file__).resolve().parent.parent.parent
        config_dir = root_path / args.config_dir
        prep_config_dir = root_path / args.preparation_config_dir
        data_path = root_path / args.data_path
        img_path = root_path / args.img_path

    # Load contrastive configs
    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    config = {'contrastive': {}}
    for module in ['encoder', 'training']:
        p = config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['contrastive'][module] = json.load(f)

    if not config['contrastive']:
        print(f"Error: No contrastive configs found in {config_dir}")
        sys.exit(1)

    # Apply CLI overrides on the encoder config — lets one configs/contrastive/
    # drive many (encoder_type × feature_group) variants without editing JSON.
    encoder_cfg = config['contrastive'].setdefault('encoder', {})
    overrides = []
    cli_overridden_keys: set = set()
    if args.encoder_type is not None:
        encoder_cfg['encoder_type'] = args.encoder_type
        cli_overridden_keys.add('encoder_type')
        overrides.append(f"encoder_type={args.encoder_type}")
    if args.encoder_arch is not None:
        encoder_cfg['encoder_arch'] = args.encoder_arch
        cli_overridden_keys.add('encoder_arch')
        overrides.append(f"encoder_arch={args.encoder_arch}")
    if args.feature_group is not None:
        encoder_cfg['selected_features'] = list(FEATURE_GROUPS[args.feature_group])
        cli_overridden_keys.add('selected_features')
        overrides.append(f"feature_group={args.feature_group} ({len(encoder_cfg['selected_features'])} features)")
    if args.output_root is not None:
        encoder_cfg['output_root'] = args.output_root
        overrides.append(f"output_root={args.output_root}")
    if args.study_name is not None:
        encoder_cfg['study_name'] = args.study_name
        overrides.append(f"study_name={args.study_name}")
    if args.contrastive_loss is not None:
        encoder_cfg['contrastive_loss'] = args.contrastive_loss
        cli_overridden_keys.add('contrastive_loss')
        overrides.append(f"contrastive_loss={args.contrastive_loss}")
    if args.warmup_epochs is not None:
        encoder_cfg['warmup_epochs'] = args.warmup_epochs
        overrides.append(f"warmup_epochs={args.warmup_epochs}")
    if args.optuna_warmup_steps is not None:
        encoder_cfg['optuna_warmup_steps'] = args.optuna_warmup_steps
        overrides.append(f"optuna_warmup_steps={args.optuna_warmup_steps}")
    if args.optuna_n_startup_trials is not None:
        encoder_cfg['optuna_n_startup_trials'] = args.optuna_n_startup_trials
        overrides.append(f"optuna_n_startup_trials={args.optuna_n_startup_trials}")
    if args.optuna_n_trials is not None:
        encoder_cfg['optuna_n_trials'] = args.optuna_n_trials
        overrides.append(f"optuna_n_trials={args.optuna_n_trials}")
    if args.rnc_target is not None:
        encoder_cfg['rnc_target'] = args.rnc_target
        cli_overridden_keys.add('rnc_target')
        overrides.append(f"rnc_target={args.rnc_target}")
    if args.cnn_channel_mode is not None:
        encoder_cfg['cnn_channel_mode'] = args.cnn_channel_mode
        cli_overridden_keys.add('cnn_channel_mode')
        overrides.append(f"cnn_channel_mode={args.cnn_channel_mode}")
    if args.encoder_train_series is not None:
        encoder_cfg['encoder_train_series'] = list(args.encoder_train_series)
        overrides.append(
            f"encoder_train_series={args.encoder_train_series}"
        )
    if args.gmm_eval_series is not None:
        encoder_cfg['gmm_eval_series'] = list(args.gmm_eval_series)
        overrides.append(f"gmm_eval_series={args.gmm_eval_series}")
    if args.loso_held_out is not None:
        encoder_cfg['loso_held_out'] = list(args.loso_held_out)
        overrides.append(f"loso_held_out={args.loso_held_out}")
    if args.chrono_min_train is not None:
        encoder_cfg['chrono_min_train'] = args.chrono_min_train
        overrides.append(f"chrono_min_train={args.chrono_min_train}")
    if args.chrono_test_series is not None:
        encoder_cfg['chrono_test_series'] = list(args.chrono_test_series)
        overrides.append(f"chrono_test_series={args.chrono_test_series}")
    if args.gmm_eval_mode is not None:
        encoder_cfg['gmm_eval_mode'] = args.gmm_eval_mode
        overrides.append(f"gmm_eval_mode={args.gmm_eval_mode}")
    if args.gmm_n_components is not None:
        encoder_cfg['gmm_n_components'] = args.gmm_n_components
        overrides.append(f"gmm_n_components={args.gmm_n_components}")
    if args.eval_noflow_split_mode is not None:
        encoder_cfg['eval_noflow_split_mode'] = args.eval_noflow_split_mode
        overrides.append(f"eval_noflow_split_mode={args.eval_noflow_split_mode}")
    if args.eval_calibration_file_ratio is not None:
        encoder_cfg['eval_calibration_file_ratio'] = args.eval_calibration_file_ratio
        overrides.append(f"eval_calibration_file_ratio={args.eval_calibration_file_ratio}")
    if args.eval_min_calibration_patches is not None:
        encoder_cfg['eval_min_calibration_patches'] = args.eval_min_calibration_patches
        overrides.append(f"eval_min_calibration_patches={args.eval_min_calibration_patches}")
    if args.gmm_temp_target is not None:
        encoder_cfg['gmm_temp_target'] = args.gmm_temp_target
        overrides.append(f"gmm_temp_target={args.gmm_temp_target}")
    if args.data_root is not None:
        encoder_cfg['data_root'] = args.data_root
        overrides.append(f"data_root={args.data_root}")

    # Stashed for train/loso so the variant block in optuna_best_params.json
    # can restore tune-time settings without stomping on explicit CLI choices.
    if cli_overridden_keys:
        encoder_cfg['_cli_overridden_keys'] = sorted(cli_overridden_keys)

    # Preparation configs (for splits.json path, consistency with AD pipeline)
    config['preparation'] = {}
    for module in ['data_preparation', 'splitting']:
        p = prep_config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['preparation'][module] = json.load(f)

    print("=" * 60)
    print("KILD CONTRASTIVE ENCODER PIPELINE")
    print("=" * 60)
    print(f"Config:    {config_dir}")
    print(f"Data:      {data_path}")
    print(f"Images:    {img_path}")
    print(f"Steps:     {', '.join(args.run)}")
    print(f"Plotting:  {'enabled' if plot_enabled else 'disabled'}")
    if overrides:
        print(f"Overrides: {'; '.join(overrides)}")
    print("=" * 60)

    img_path.mkdir(parents=True, exist_ok=True)

    for step in args.run:
        print(f"\n{'#' * 60}")
        print(f"# STEP: {step.upper()}")
        print(f"{'#' * 60}\n")

        if step == 'train':
            from .train import train_contrastive
            train_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'tune':
            from .train import tune_contrastive
            tune_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'loso':
            from .train import loso_contrastive
            loso_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'chrono_extrap':
            from .train import chronological_extrapolation_contrastive
            chronological_extrapolation_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'chrono_extrap_temp':
            from .train import chronological_extrapolation_temperature_contrastive
            chronological_extrapolation_temperature_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'chrono_extrap_pooled':
            from .train import chronological_extrapolation_pooled_contrastive
            chronological_extrapolation_pooled_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'chrono_extrap_pooled_trainthresh':
            from .train import chronological_extrapolation_pooled_trainthresh_contrastive
            chronological_extrapolation_pooled_trainthresh_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'eval':
            from .train import eval_contrastive
            eval_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )
        elif step == 'predict':
            from .train import predict_contrastive
            predict_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
                as_blind_series=args.as_blind_series,
            )
        elif step == 'detector_eval':
            from .train import detector_eval_contrastive
            detector_eval_contrastive(
                config, data_path, img_path, plot_enabled,
                only_files=args.only_files, max_workers=max_workers,
            )

    print("\n" + "=" * 60)
    print("CONTRASTIVE PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
