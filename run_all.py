#!/usr/bin/env python3
"""
Run All Analysis Scripts
执行完整的ISAC分析流程

Usage:
    python run_all.py [config.yaml]

Author: Complete Pipeline Runner
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, description):
    """
    Execute a command and report status

    Args:
        cmd: Command to run (list or string)
        description: Description of the command

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.1f}s")
        print(f"Error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Script not found or Python not in PATH")
        return False


def main():
    """Main execution pipeline"""

    print("\n" + "=" * 80)
    print("COMPLETE ISAC ANALYSIS PIPELINE")
    print("Executing all analysis scripts in sequence")
    print("=" * 80)

    # Parse config file argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config.yaml'

    # Verify config file exists
    if not os.path.exists(config_file):
        print(f"\n✗ Error: Configuration file not found: {config_file}")
        print("Usage: python run_all.py [config.yaml]")
        sys.exit(1)

    print(f"\nUsing configuration file: {config_file}")

    # Get absolute path for config
    config_path = os.path.abspath(config_file)

    # Track results
    results = {}
    start_time = time.time()

    # Step 1: Main analysis (Pareto sweep)
    step_name = "Main Analysis (Pareto Sweep)"
    cmd = [sys.executable, "main.py", config_path]
    results[step_name] = run_command(cmd, step_name)

    # Step 2: SNR sweep
    step_name = "SNR Sweep"
    cmd = [sys.executable, "scan_snr_sweep.py", config_path]
    results[step_name] = run_command(cmd, step_name)

    # Step 3: Threshold sweep
    step_name = "Threshold Sweep"
    cmd = [sys.executable, "threshold_sweep.py", config_path]
    results[step_name] = run_command(cmd, step_name)

    # Step 4: Generate visualizations
    step_name = "Generate Visualizations"
    cmd = [sys.executable, "visualize_results.py"]
    results[step_name] = run_command(cmd, step_name)

    # # Step 5: Generate tables
    # step_name = "Generate Paper Tables"
    # cmd = [sys.executable, "make_paper_tables.py"]
    # results[step_name] = run_command(cmd, step_name)

    # Step 4: MIMO Analysis
    step_name = "MIMO Scaling Analysis"
    cmd = [sys.executable, "mimo_analysis.py", config_path]
    results[step_name] = run_command(cmd, step_name)

    # Step 5: Supplementary Figures
    step_name = "Supplementary Figures"
    cmd = [sys.executable, "generate_supplementary_figures.py", config_path]
    results[step_name] = run_command(cmd, step_name)


    # Summary
    total_time = time.time() - start_time
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    for step_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {step_name}")

    print(f"\n{'=' * 80}")
    print(f"Results: {success_count}/{total_count} steps completed successfully")
    print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"{'=' * 80}")

    if success_count == total_count:
        print("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nGenerated outputs:")
        print("  - Results: ./results/")
        print("  - Figures: ./figures/")
        print("  - Tables: Check console output from make_paper_tables.py")
        sys.exit(0)
    else:
        print(f"\n⚠ WARNING: Only {success_count}/{total_count} steps completed")
        print("\nPlease check error messages above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()