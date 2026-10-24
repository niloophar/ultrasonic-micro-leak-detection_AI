import argparse
import json
import sys
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(
        description="KILD Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.preprocessing.main --run label clean relaxation
    python -m src.preprocessing.main --run label --no-plot
    python -m src.preprocessing.main --run clean relaxation --plot
        """
    )

    parser.add_argument(
        '--run',
        nargs='+',
        choices=['label', 'clean', 'relaxation', 'equilibrium'],
        default=os.getenv('RUN_STEPS', 'label clean relaxation').split(),
        help='Steps to run (in order): label, clean, relaxation, equilibrium'
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
        help='Disable plotting (default)'
    )

    parser.add_argument(
        '--config-dir',
        type=str,
        default=str(Path(os.getenv('CONFIG_DIR_PATH', 'configs')) / 'preprocessing'),
        help='Path to config directory (default: configs/preprocessing)'
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
        help='Only process these file stems (e.g. "2025_20_12 1800_1759_V")'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=int(os.getenv('MAX_WORKERS', 1)),
        help='Max number of parallel workers (default: 1 or from MAX_WORKERS env)'
    )

    args = parser.parse_args()

    # Determine plot setting
    plot_enabled = args.plot and not args.no_plot
    max_workers = args.max_workers

    # Resolve paths
    # Inside the container, we use absolute paths from the environment variables
    # For local execution, we construct them relative to the project root.
    if os.getenv('DOCKER_ENV'):
        config_dir = Path(args.config_dir)
        data_path = Path(args.data_path)
        img_path = Path(args.img_path) if plot_enabled else None
    else:
        root_path = Path(__file__).resolve().parent.parent.parent
        config_dir = root_path / args.config_dir
        data_path = root_path / args.data_path
        img_path = root_path / args.img_path if plot_enabled else None

    # Load per-module configs
    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    config = {}
    for module in ['label', 'clean', 'relaxation', 'equilibrium']:
        p = config_dir / f'{module}.json'
        if p.exists():
            with open(p, 'r') as f:
                config[module] = json.load(f)

    print("=" * 60)
    print("KILD PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Config:    {config_dir}")
    print(f"Data:      {data_path}")
    print(f"Images:    {img_path if img_path else 'disabled'}")
    print(f"Steps:     {', '.join(args.run)}")
    print(f"Plotting:  {'enabled' if plot_enabled else 'disabled'}")
    print(f"Workers:   {max_workers}")
    if args.only_files:
        print(f"Only files: {args.only_files}")
    print("=" * 60)

    # Import processing functions (lazy import to avoid circular dependencies)
    from .label import process_label
    from .clean import process_clean
    from .relaxation import process_relaxation
    from .equilibrium import process_equilibrium

    # Create image directory if plotting is enabled
    if img_path:
        img_path.mkdir(parents=True, exist_ok=True)

    # Run requested steps in order
    for step in args.run:
        print(f"\n{'#' * 60}")
        print(f"# STEP: {step.upper()}")
        print(f"{'#' * 60}\n")

        if step == 'label':
            process_label(config, data_path, img_path, plot_enabled, only_files=args.only_files, max_workers=max_workers)
        elif step == 'clean':
            process_clean(config, data_path, img_path, plot_enabled, only_files=args.only_files, max_workers=max_workers)
        elif step == 'relaxation':
            process_relaxation(config, data_path, img_path, plot_enabled, only_files=args.only_files, max_workers=max_workers)
        elif step == 'equilibrium':
            process_equilibrium(config, data_path, img_path, plot_enabled, only_files=args.only_files, max_workers=max_workers)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

        
        
if __name__ == '__main__':
    main()
