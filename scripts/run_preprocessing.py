#!/usr/bin/env python3
"""
Work Package 2: Data Preprocessing and Quality Assurance Pipeline

This script processes raw ultrasonic sensor data (K and V clusters) and:
1. Merges K and V data streams by timestamp
2. Detects valid Supreme Data Blocks (K: 64 samples, then V: 128×64 samples)
3. Performs intra-block outlier imputation
4. Filters blocks based on statistical outliers
5. Outputs cleaned data ready for further analysis

Usage:
    python scripts/run_wp02_preprocessing.py [--max-workers N] [--config PATH]
"""
import sys
from pathlib import Path
import argparse
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import (
    run_preprocessing_pipeline,
    print_pipeline_summary
)
from src.common.models import FileProcessingTask
from src.common.constants import DATA_RAW, DATA_PROCESSED, BLOCK_STRUCTURE_KV, BLOCK_STRUCTURE_V_ONLY


def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        'intra_block_iqr_factor': 3.0,
        'block_level_iqr_factor': 1.5,
        'block_filter_feature': 't_up_0 in ps',
        'block_stat_type': 'mean',
        'max_workers': None,
        'skip_existing': False
    }

    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            default_config.update(user_config)

    return default_config


def create_tasks_from_config(config: dict) -> list:
    """
    Create processing tasks from configuration.

    Expected config structure:
        files:
          noflow:
            - "2025_12_03 2200_2215h"
            - "2025_12_03 2215_2230h"
          flow:
            - "2025_12_04 2245_2300h"

    Expected directory structure:
        data/raw/
            noflow/
                <name> K.csv
                <name> V.csv
            flow/
                <name> K.csv
                <name> V.csv

    Output to:
        data/merged/
            noflow/
                <name>.csv
            flow/
                <name>.csv
    """
    tasks = []

    # Get block structure mode from config
    block_structure = config.get('block_structure', BLOCK_STRUCTURE_KV)

    # Get file lists from config
    files_config = config.get('files', {})

    # TODO Controll labels by config
    for sample_type in ['noflow', 'flow']:
        filenames = files_config.get(sample_type, [])

        # Handle None case (empty YAML list with only comments)
        if filenames is None:
            filenames = []

        for filename in filenames:
            # Define input paths based on block structure
            input_paths = {}

            if block_structure == BLOCK_STRUCTURE_KV:
                # K+V mode: Need both K and V files
                k_path = DATA_RAW / sample_type / f'{filename} K.csv'
                v_path = DATA_RAW / sample_type / f'{filename} V.csv'

                # Check if input files exist
                if not k_path.exists():
                    print(f'Warning: K file not found: {k_path}')
                    continue
                if not v_path.exists():
                    print(f'Warning: V file not found: {v_path}')
                    continue

                input_paths = {'K': str(k_path), 'V': str(v_path)}

            elif block_structure == BLOCK_STRUCTURE_V_ONLY:
                # V-only mode: Need only V file
                v_path = DATA_RAW / sample_type / f'{filename} V.csv'

                # Check if input file exists
                if not v_path.exists():
                    print(f'Warning: V file not found: {v_path}')
                    continue

                input_paths = {'V': str(v_path)}

            else:
                print(f'Error: Invalid block_structure: {block_structure}')
                continue

            # Output path (merged, without K/V postfix)
            output_path = DATA_PROCESSED / sample_type / f'{filename}.csv'

            # Skip if output exists and skip_existing is True
            if config.get('skip_existing', False) and output_path.exists():
                print(f'Skipping existing file: {output_path}')
                continue

            task = FileProcessingTask(
                sample_type=sample_type,
                filename=filename,
                input_paths=input_paths,
                output_path=str(output_path),
                config=config
            )
            tasks.append(task)

    return tasks


def main():
    """Main entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description='WP02: Data Preprocessing and Quality Assurance Pipeline'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=PROJECT_ROOT / 'configs' / 'preprocessing_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run sequentially instead of parallel'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that already exist in output'
    )

    args = parser.parse_args()

    # Load configuration
    print(f'Loading configuration from: {args.config}')
    config = load_config(args.config)

    # Override with command line arguments
    if args.max_workers is not None:
        config['max_workers'] = args.max_workers
    if args.sequential:
        config['max_workers'] = 1
    if args.skip_existing:
        config['skip_existing'] = True

    # Print configuration summary
    print('\n' + '=' * 80)
    print('PREPROCESSING CONFIGURATION')
    print('=' * 80)

    # Execution mode
    if config['max_workers'] == 1:
        print(f"Execution mode:       Sequential (single-threaded)")
    elif config['max_workers'] is None:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"Execution mode:       Parallel (auto-detect: {cpu_count} workers)")
    else:
        print(f"Execution mode:       Parallel ({config['max_workers']} workers)")

    # Block structure
    block_structure = config.get('block_structure', BLOCK_STRUCTURE_KV)
    print(f"Block structure:      {block_structure}")

    # Outlier handling strategy
    strategy = config.get('outlier_strategy', 'imputation')
    strategy_display = strategy if strategy is not None else "none"
    print(f"Outlier strategy:     {strategy_display}")

    if strategy == 'trimming':
        trim_start = config.get('trim_edge_start', 4)
        trim_end = config.get('trim_edge_end', 2)
        total_trimmed = trim_start + trim_end

        print(f"  - Edge trimming:    first {trim_start}, last {trim_end} points")

        # Calculate and display exact block structure after trimming
        if block_structure == BLOCK_STRUCTURE_KV:
            k_after = 64 - total_trimmed
            v_after = 8192 - total_trimmed
            print(f"  - Exact structure:  K (1×64 - {total_trimmed} = {k_after}) → V (128×64 - {total_trimmed} = {v_after})")
        elif block_structure == BLOCK_STRUCTURE_V_ONLY:
            v_after = 8192 - total_trimmed
            print(f"  - Exact structure:  V (128×64 - {total_trimmed} = {v_after})")

    elif strategy == 'imputation':
        print(f"  - IQR factor:       {config.get('intra_block_iqr_factor', 3.0)}")

        # Display block structure for imputation strategy
        if block_structure == BLOCK_STRUCTURE_KV:
            print(f"  - Exact structure:  K (1×64 = 64) → V (128×64 = 8192)")
        elif block_structure == BLOCK_STRUCTURE_V_ONLY:
            print(f"  - Exact structure:  V (128×64 = 8192)")

    elif strategy is None or strategy == 'none':
        print(f"  - No outlier handling applied")

        # Display block structure for none strategy
        if block_structure == BLOCK_STRUCTURE_KV:
            print(f"  - Exact structure:  K (1×64 = 64) → V (128×64 = 8192)")
        elif block_structure == BLOCK_STRUCTURE_V_ONLY:
            print(f"  - Exact structure:  V (128×64 = 8192)")

    # Zero-Crossing shift correction status
    enable_zc = config.get('enable_zc_shift_correction', False)
    print(f"ZC shift correction:  {'Enabled' if enable_zc else 'Disabled'}")
    if enable_zc:
        zc_period = config.get('zc_expected_period_ps', None)
        if zc_period is None:
            print(f"  - ZC period:        auto-detect from data")
        else:
            print(f"  - ZC period:        {zc_period:.0f} ps (manual)")
        print(f"  - ZC tolerance:     {config.get('zc_tolerance_factor', 0.1)*100:.0f}%")

    # Block-level filtering
    enable_filter = config.get('enable_block_filter', True)
    print(f"Block filter:         {'Enabled' if enable_filter else 'Disabled'}")
    if enable_filter:
        print(f"  - Feature:          {config.get('block_filter_feature', 't_up_0 in ps')}")
        print(f"  - IQR factor:       {config.get('block_level_iqr_factor', 1.5)}")
        print(f"  - Statistic:        {config.get('block_stat_type', 'mean')}")

    print('=' * 80)

    # Create processing tasks
    print('\nCreating processing tasks...')
    tasks = create_tasks_from_config(config)

    if not tasks:
        print('No tasks to process. Check your configuration and data files.')
        return

    print(f'Found {len(tasks)} file(s) to process\n')

    # Run pipeline
    results = run_preprocessing_pipeline(
        tasks=tasks,
        max_workers=config['max_workers'],
        show_progress=True
    )

    # Print summary
    print_pipeline_summary(results)

    # Exit with appropriate code
    failed = sum(1 for r in results if r['status'] == 'error')
    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
