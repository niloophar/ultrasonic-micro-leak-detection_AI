"""
Anomaly Detection Pipeline CLI.

Entry point for the anomaly detection (AutoEncoder) pipeline:
  1. train - Train CNN AutoEncoder on NoFlow data, evaluate anomaly detection
  2. tune     - Optuna hyperparameter search (architecture, scoring, model type)
  3. ensemble  - Evaluate CNN Classifier + AE ensemble (OR, AND, score blend)

Usage:
    python -m src.anomaly_detection.main --run train
    python -m src.anomaly_detection.main --run tune
    python -m src.anomaly_detection.main --run ensemble
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="KILD Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.anomaly_detection.main --run train
        """
    )

    parser.add_argument(
        '--run',
        nargs='+',
        choices=['train', 'tune', 'ensemble', 'latent_ad'],
        default=['train'],
        help='Steps to run: train, tune, ensemble, latent_ad'
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
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'anomaly_detection'),
        help='Path to anomaly_detection config directory (default: configs/anomaly_detection)'
    )

    parser.add_argument(
        '--preparation-config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'preparation'),
        help='Path to preparation config directory'
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

    args = parser.parse_args()

    # Determine plot setting
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

    # Load per-module anomaly detection configs
    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    config = {'anomaly_detection': {}}

    for module in ['autoencoder', 'training']:
        p = config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['anomaly_detection'][module] = json.load(f)

    if not config['anomaly_detection']:
        print(f"Error: No anomaly_detection configs found in {config_dir}")
        sys.exit(1)

    # Load preparation configs (for splits.json path)
    config['preparation'] = {}
    for module in ['data_preparation', 'splitting']:
        p = prep_config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config['preparation'][module] = json.load(f)

    print("=" * 60)
    print("KILD ANOMALY DETECTION PIPELINE")
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
            from .train import train_autoencoder

            train_autoencoder(config, data_path, img_path, plot_enabled,
                              only_files=args.only_files,
                              max_workers=max_workers)

        elif step == 'tune':
            from .train import tune_autoencoder

            tune_autoencoder(config, data_path, img_path, plot_enabled,
                             only_files=args.only_files,
                             max_workers=max_workers)

        elif step == 'ensemble':
            from .train import ensemble_evaluate

            ensemble_evaluate(config, data_path, img_path, plot_enabled,
                              only_files=args.only_files,
                              max_workers=max_workers)

        elif step == 'latent_ad':
            from .train import latent_anomaly_detection

            latent_anomaly_detection(config, data_path, img_path, plot_enabled,
                                     only_files=args.only_files,
                                     max_workers=max_workers)

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
