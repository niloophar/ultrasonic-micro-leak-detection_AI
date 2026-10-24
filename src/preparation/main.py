"""
Preparation Pipeline CLI - Milestone C.

Entry point for the data preparation pipeline:
  1. extract   - Extract equilibrium data for flow/noflow classes
  2. features  - Compute engineered features and save per-segment CSVs
  3. splitting - Assign segments to train/val/test splits (splits.json)
  4. prepare   - Build patches from feature CSVs (downsample, cut, center)
                 Overlap is only applied to train segments (augmentation).

Usage:
    python -m src.preparation.main --run extract features splitting prepare
    python -m src.preparation.main --run extract --plot
    python -m src.preparation.main --run splitting
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="KILD Preparation Pipeline (Milestone C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.preparation.main --run extract features splitting prepare
    python -m src.preparation.main --run features --plot
    python -m src.preparation.main --run splitting prepare
    python -m src.preparation.main --run prepare
        """
    )

    parser.add_argument(
        '--run',
        nargs='+',
        choices=['extract', 'features', 'splitting', 'prepare'],
        default=['extract', 'features', 'splitting', 'prepare'],
        help='Steps to run (in order): extract, features, splitting, prepare'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        default=os.getenv('PLOT_ENABLED', 'false').lower() == 'true',
        help='Enable plotting (default: disabled)'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        default=False,
        help='Disable plotting'
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'preparation'),
        help='Path to preparation config directory (default: configs/preparation)'
    )

    parser.add_argument(
        '--preprocessing-config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'preprocessing'),
        help='Path to preprocessing config directory (for label config)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default=os.getenv('CONTAINER_DATA_PATH', 'data'),
        help='Base data directory (default: data)'
    )

    parser.add_argument(
        '--img-path',
        type=str,
        default=os.getenv('CONTAINER_IMG_PATH', 'images'),
        help='Base image directory for plots (default: images)'
    )

    parser.add_argument(
        '--only-files',
        nargs='+',
        type=str,
        default=None,
        help='Only process these file stems'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=int(os.getenv('MAX_WORKERS', 1)),
        help='Max number of parallel workers (default: 1)'
    )

    # --- Ablation overrides ------------------------------------------------
    # Lets one prep config drive multiple sandbox copies (e.g. pw32 × pw64 ×
    # filter on/off) without editing the JSONs by hand. Each ablation cell
    # gets its own --output-root so the 4 prepared trees coexist on disk.
    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help=(
            "Override `output_root` in all four prep configs (extract / "
            "features / splitting / data_preparation). Lets parallel "
            "ablation cells write to separate directories without editing "
            "JSON. Example: --output-root ./ablation/pw32_f0."
        ),
    )
    parser.add_argument(
        '--patch-width',
        type=int,
        default=None,
        help=(
            'Override data_preparation.patch.patch_width (default 64). '
            'Used for pw ablation runs.'
        ),
    )
    parser.add_argument(
        '--noflow-time-filter',
        dest='noflow_time_filter',
        action='store_true',
        default=None,
        help=(
            'Force-enable the [00:00, 05:00) NoFlow time filter '
            '(data_preparation.patch.noflow_time_filter.enabled).'
        ),
    )
    parser.add_argument(
        '--no-noflow-time-filter',
        dest='noflow_time_filter',
        action='store_false',
        help='Force-disable the NoFlow time filter.',
    )

    args = parser.parse_args()

    # Determine plot setting
    plot_enabled = args.plot and not args.no_plot
    max_workers = args.max_workers

    # Resolve paths
    if os.getenv('DOCKER_ENV'):
        config_dir = Path(args.config_dir)
        preproc_config_dir = Path(args.preprocessing_config_dir)
        data_path = Path(args.data_path)
        img_path = Path(args.img_path) if plot_enabled else None
    else:
        root_path = Path(__file__).resolve().parent.parent.parent
        config_dir = root_path / args.config_dir
        preproc_config_dir = root_path / args.preprocessing_config_dir
        data_path = root_path / args.data_path
        img_path = root_path / args.img_path if plot_enabled else None

    # Load per-module preparation configs
    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    config = {'preparation': {}}

    for module in ['extract_equilibrium', 'feature_engineering', 'data_preparation',
                   'splitting']:
        p = config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['preparation'][module] = json.load(f)

    if not config['preparation']:
        print(f"Error: No preparation configs found in {config_dir}")
        sys.exit(1)

    # Apply CLI ablation overrides AFTER config load — keeps the JSONs as
    # canonical defaults but lets a single command line redirect outputs and
    # flip patch settings without on-disk edits.
    overrides = []
    if args.output_root is not None:
        for module, cfg in config['preparation'].items():
            cfg['output_root'] = args.output_root
        overrides.append(f"output_root={args.output_root}")
    if args.patch_width is not None:
        prep_dp = config['preparation'].setdefault('data_preparation', {})
        patch_cfg = prep_dp.setdefault('patch', {})
        patch_cfg['patch_width'] = args.patch_width
        overrides.append(f"patch_width={args.patch_width}")
    if args.noflow_time_filter is not None:
        prep_dp = config['preparation'].setdefault('data_preparation', {})
        patch_cfg = prep_dp.setdefault('patch', {})
        ntf = patch_cfg.setdefault('noflow_time_filter', {})
        ntf['enabled'] = bool(args.noflow_time_filter)
        if args.noflow_time_filter:
            ntf.setdefault('start_hour', 0)
            ntf.setdefault('end_hour', 5)
        overrides.append(
            f"noflow_time_filter={'on' if args.noflow_time_filter else 'off'}"
        )

    # Load label config from preprocessing (needed for leakage windows)
    label_path = preproc_config_dir / 'label.json'
    if label_path.exists():
        with open(label_path, 'r') as f:
            config['label'] = json.load(f)

    print("=" * 60)
    print("KILD PREPARATION PIPELINE")
    print("=" * 60)
    print(f"Config:    {config_dir}")
    print(f"Data:      {data_path}")
    print(f"Images:    {img_path if img_path else 'disabled'}")
    print(f"Steps:     {', '.join(args.run)}")
    print(f"Plotting:  {'enabled' if plot_enabled else 'disabled'}")
    if overrides:
        print(f"Overrides: {'; '.join(overrides)}")
    print("=" * 60)

    # Lazy imports
    from .extract_equilibrium import process_extract
    from .feature_engineering import process_features
    from .data_preparation import process_prepare
    from .splitting import process_splitting

    # Create image directory if plotting
    if img_path:
        img_path.mkdir(parents=True, exist_ok=True)

    # Run steps
    for step in args.run:
        print(f"\n{'#' * 60}")
        print(f"# STEP: {step.upper()}")
        print(f"{'#' * 60}\n")

        if step == 'extract':
            process_extract(config, data_path, img_path, plot_enabled,
                            only_files=args.only_files, max_workers=max_workers)
        elif step == 'features':
            process_features(config, data_path, img_path, plot_enabled,
                             only_files=args.only_files, max_workers=max_workers)
        elif step == 'prepare':
            process_prepare(config, data_path, img_path, plot_enabled,
                            only_files=args.only_files, max_workers=max_workers)
        elif step == 'splitting':
            process_splitting(config, data_path, img_path, plot_enabled,
                              only_files=args.only_files, max_workers=max_workers)

    print("\n" + "=" * 60)
    print("PREPARATION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
