"""
Classification Pipeline CLI - Milestone C.

Entry point for the classification (training) pipeline:
  1. train - Scale, train MLP/XGBoost, and evaluate
  2. analyze - Class distribution visualization
  3. latent - Extract CNN features, t-SNE + GMM visualization

Usage:
    python -m src.classification.main --run train
    python -m src.classification.main --run train --model xgboost
    python -m src.classification.main --run latent
    python -m src.classification.main --run analyze
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="KILD Classification Pipeline (Milestone C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.classification.main --run train
    python -m src.classification.main --run train --model xgboost
    python -m src.classification.main --run analyze
        """
    )

    parser.add_argument(
        '--run',
        nargs='+',
        choices=['train', 'tune', 'latent', 'analyze'],
        default=['train'],
        help='Steps to run: train, tune, latent, analyze'
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
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'classification'),
        help='Path to classification config directory (default: configs/classification)'
    )

    parser.add_argument(
        '--preparation-config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'preparation'),
        help='Path to preparation config directory (for data_preparation and splitting configs)'
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

    parser.add_argument(
        '--model',
        type=str,
        choices=['mlp', 'xgboost', 'transformer'],
        default=None,
        help='Model to train: mlp, xgboost, or transformer (default: from training.json or mlp)'
    )

    args = parser.parse_args()

    # Determine plot setting
    plot_enabled = args.plot and not args.no_plot
    max_workers = args.max_workers

    # Resolve paths
    if os.getenv('DOCKER_ENV'):
        config_dir = Path(args.config_dir)
        prep_config_dir = Path(args.preparation_config_dir)
        preproc_config_dir = Path(args.preprocessing_config_dir)
        data_path = Path(args.data_path)
        img_path = Path(args.img_path)
    else:
        root_path = Path(__file__).resolve().parent.parent.parent
        config_dir = root_path / args.config_dir
        prep_config_dir = root_path / args.preparation_config_dir
        preproc_config_dir = root_path / args.preprocessing_config_dir
        data_path = root_path / args.data_path
        img_path = root_path / args.img_path

    # Load per-module classification configs
    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    config = {'classification': {}}

    for module in ['mlp', 'xgboost', 'transformer', 'training']:
        p = config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['classification'][module] = json.load(f)

    if not config['classification']:
        print(f"Error: No classification configs found in {config_dir}")
        sys.exit(1)

    # Resolve model selection: CLI > training.json > default 'mlp'
    if args.model is None:
        training_cfg = config['classification'].get('training', {})
        args.model = training_cfg.get('model', 'mlp')

    # Load preparation configs (train needs data_preparation for patch dims,
    # splitting for splits.json path)
    config['preparation'] = {}
    for module in ['data_preparation', 'splitting']:
        p = prep_config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['preparation'][module] = json.load(f)

    # Load label config from preprocessing (needed for leakage windows)
    label_path = preproc_config_dir / 'label.json'
    if label_path.exists():
        with open(label_path, 'r') as f:
            config['label'] = json.load(f)

    # Load analysis config
    if os.getenv('DOCKER_ENV'):
        analysis_config_dir = Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'analysis'
    else:
        analysis_config_dir = root_path / 'configs' / 'analysis'

    analysis_path = analysis_config_dir / 'feature_importance.json'
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            config['analysis'] = {'feature_importance': json.load(f)}
    else:
        config['analysis'] = {'feature_importance': {}}

    print("=" * 60)
    print("KILD CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Config:    {config_dir}")
    print(f"Data:      {data_path}")
    print(f"Images:    {img_path}")
    print(f"Steps:     {', '.join(args.run)}")
    print(f"Plotting:  {'enabled' if plot_enabled else 'disabled'}")
    print("=" * 60)

    # Create image directory
    img_path.mkdir(parents=True, exist_ok=True)

    # Run steps
    for step in args.run:
        print(f"\n{'#' * 60}")
        print(f"# STEP: {step.upper()}")
        print(f"{'#' * 60}\n")

        if step == 'train':
            from .train import train_model, train_xgboost

            if args.model == 'xgboost':
                train_xgboost(config, data_path, img_path, plot_enabled,
                              only_files=args.only_files, max_workers=max_workers)
            elif args.model == 'transformer':
                train_model(config, data_path, img_path, plot_enabled,
                            only_files=args.only_files, max_workers=max_workers,
                            model_name='transformer')
            else:
                train_model(config, data_path, img_path, plot_enabled,
                            only_files=args.only_files, max_workers=max_workers)

        elif step == 'tune':
            if args.model == 'transformer':
                from .train import tune_transformer
                tune_transformer(config, data_path, img_path, plot_enabled,
                                 only_files=args.only_files, max_workers=max_workers)
            else:
                from .train import tune_cnn
                tune_cnn(config, data_path, img_path, plot_enabled,
                         only_files=args.only_files, max_workers=max_workers)

        elif step == 'latent':
            from .train import extract_latent

            extract_latent(config, data_path, img_path, plot_enabled,
                           only_files=args.only_files, max_workers=max_workers)

        elif step == 'analyze':
            from src.analysis.class_distribution import (
                compute_class_distribution, plot_class_distribution,
            )

            xgb_config = config['classification'].get('xgboost', {})
            output_root = xgb_config.get('output_root', './prepared')

            dist = compute_class_distribution(data_path, output_root)
            analysis_plot_dir = img_path / 'classification' / 'analysis'
            analysis_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_class_distribution(dist, analysis_plot_dir / 'class_distribution.png')

    print("\n" + "=" * 60)
    print("CLASSIFICATION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
