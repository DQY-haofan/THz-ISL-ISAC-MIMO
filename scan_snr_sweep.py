#!/usr/bin/env python3
"""
Results Generator: Capacity vs. SNR Sweep (FIXED VERSION)
DR-08 / P2-DR-01 / P2-DR-05 Special Scan with ADAPTIVE SNR RANGE

KEY FIX: SNR sweep now automatically centers around SNR_crit ± 30dB
This ensures we capture the full capacity curve from linear region to saturation.

This script runs a dedicated sweep of C_J and C_G vs. SNR at a
FIXED (and optimal) alpha, to generate data for:
1. The primary "Capacity vs SNR" plot (visualize_results.py)
2. The "Jensen Gap" validation table (make_paper_tables.py)

Usage:
    python scan_snr_sweep_fixed.py [config.yaml]

Author: Generated according to DR-08 Protocol v1.0 + Expert Fix
"""

import numpy as np
import pandas as pd
import yaml
import copy
import sys
import os
import warnings

# Import validated DR-08 engines
sys.path.insert(0, '/mnt/user-data/uploads')
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure physics_engine.py and limits_engine.py are in the same directory")
    sys.exit(1)


def run_snr_sweep(config_path: str = 'config.yaml'):
    """
    Performs a detailed C_J vs SNR sweep at a fixed alpha with ADAPTIVE SNR RANGE.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Path to generated CSV file, or None if failed
    """

    print("=" * 80)
    print("DEDICATED SNR SWEEP (ADAPTIVE RANGE - FIXED VERSION)")
    print("Automatically centers sweep around SNR_crit for optimal curve capture")
    print("=" * 80)

    # 1. Load and validate config
    print(f"\n[1/5] Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        validate_config(config)
        print("✓ Configuration loaded and validated")
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {config_path}")
        print("  Please provide a valid config.yaml file")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)

    # 2. Set fixed alpha (use default or 'optimal' from config)
    fixed_alpha = config.get('isac_model', {}).get('alpha', 0.05)
    config['isac_model']['alpha'] = fixed_alpha
    print(f"\n[2/5] Using fixed ISAC overhead: α = {fixed_alpha}")

    # ===================================================================
    # 3. PRE-SCAN: Calculate SNR_crit to determine optimal sweep range
    # ===================================================================
    print(f"\n[3/5] Pre-scan: Determining SNR_crit for adaptive range...")

    try:
        # Run simulation chain once to get SNR_crit
        g_factors_prescan = calc_g_sig_factors(config)
        n_outputs_prescan = calc_n_f_vector(config, g_factors_prescan)

        # Calculate SNR_crit (independent of actual SNR0 value)
        c_j_prescan = calc_C_J(
            config,
            g_factors_prescan,
            n_outputs_prescan,
            [0.0],  # SNR value doesn't affect SNR_crit calculation
            compute_C_G=False
        )

        snr_crit_db = float(c_j_prescan['SNR_crit_db'])
        c_sat = float(c_j_prescan['C_sat'])

        print(f"  ✓ Pre-scan complete:")
        print(f"    SNR_crit: {snr_crit_db:.2f} dB")
        print(f"    C_sat: {c_sat:.3f} bits/s/Hz")

    except Exception as e:
        print(f"  ✗ Pre-scan failed: {e}")
        print("  Falling back to config-specified range")
        import traceback
        traceback.print_exc()

        # Fallback to config range if pre-scan fails
        snr_config = config.get('simulation', {})
        snr_crit_db = 0.0  # Default guess

    # ===================================================================
    # 4. Define ADAPTIVE SNR sweep range centered on SNR_crit
    # ===================================================================
    print(f"\n[4/5] Defining adaptive SNR sweep range...")

    # Adaptive range: SNR_crit ± 30 dB to capture full curve
    margin_db = 30.0
    snr_min = snr_crit_db - margin_db
    snr_max = snr_crit_db + margin_db

    # Override with config if explicitly specified
    snr_config = config.get('simulation', {})
    if 'SNR_sweep_range_override' in snr_config:
        snr_min = snr_config['SNR_sweep_range_override'][0]
        snr_max = snr_config['SNR_sweep_range_override'][1]
        print(f"  ⚠ Using config override range: [{snr_min}, {snr_max}] dB")
    else:
        print(f"  ✓ Adaptive range: [{snr_min:.1f}, {snr_max:.1f}] dB")
        print(f"    (SNR_crit ± {margin_db} dB)")

    # Number of points
    n_points = snr_config.get('SNR_sweep_points', 100)
    snr_sweep = np.linspace(snr_min, snr_max, n_points)

    print(f"  Sweeping {len(snr_sweep)} points from {snr_sweep[0]:.1f} to {snr_sweep[-1]:.1f} dB")

    # ===================================================================
    # 5. Run full simulation chain with C_G computation
    # ===================================================================
    try:
        print("\n[5/5] Running full simulation chain (Physics + Limits)...")

        # Phase 1A: Multiplicative gains
        print("  - Calculating multiplicative gains...")
        g_factors = calc_g_sig_factors(config)

        # Phase 1B: Additive noise (depends on g_factors)
        print("  - Calculating additive noise sources...")
        n_outputs = calc_n_f_vector(config, g_factors)

        # Phase 2: Communication Capacity (C_J and C_G)
        # CRITICAL: Set compute_C_G=True to get Jensen Gap data
        print("  - Computing communication capacity (C_J and C_G)...")
        c_j_results = calc_C_J(
            config,
            g_factors,
            n_outputs,
            snr_sweep,
            compute_C_G=True  # Enable DR-05 validation
        )
        print("✓ Simulation complete")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ===================================================================
    # 6. Package and save results
    # ===================================================================
    print("\n[6/6] Packaging and saving results...")

    results_data = {
        'SNR0_db': snr_sweep,
        'C_J_bps_hz': c_j_results['C_J_vec'],
    }

    # Add C_G and Jensen gap if available
    if 'C_G_vec' in c_j_results:
        results_data['C_G_bps_hz'] = c_j_results['C_G_vec']
        results_data['Jensen_gap_bits'] = c_j_results['C_J_vec'] - c_j_results['C_G_vec']
    else:
        # Fill with NaN if C_G not computed
        results_data['C_G_bps_hz'] = np.nan
        results_data['Jensen_gap_bits'] = np.nan
        print("  ⚠ Warning: C_G not computed (Jensen gap unavailable)")

    # Add scalar metrics (constant for all SNR)
    scalar_metrics = {
        'C_sat': c_j_results['C_sat'],
        'SNR_crit_db': c_j_results['SNR_crit_db'],
        'alpha': fixed_alpha,
        'Gamma_eff_total': n_outputs['Gamma_eff_total'],
        'sigma_2_phi_c_res_rad2': n_outputs['sigma_2_phi_c_res'],
        'eta_bsq_avg': g_factors['eta_bsq_avg'],
        'B_hz': config['channel']['B_hz'],
        'f_c_hz': config['channel']['f_c_hz']
    }

    df = pd.DataFrame(results_data)
    for key, value in scalar_metrics.items():
        df[key] = value

    # Save to CSV
    output_config = config.get('outputs', {})
    save_path = output_config.get('save_path', '/mnt/user-data/outputs/')
    table_prefix = output_config.get('table_prefix', 'DR08_results')

    os.makedirs(save_path, exist_ok=True)
    csv_filename = os.path.join(save_path, f"{table_prefix}_snr_sweep.csv")

    df.to_csv(csv_filename, index=False, float_format='%.6e')
    print(f"✓ SNR sweep results saved to: {csv_filename}")

    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}")
    print(f"  Fixed alpha: {fixed_alpha}")
    print(f"  SNR sweep range: [{snr_sweep[0]:.1f}, {snr_sweep[-1]:.1f}] dB")
    print(f"  Number of points: {len(snr_sweep)}")
    print(f"\n  Saturation capacity: {c_j_results['C_sat']:.3f} bits/s/Hz")
    print(f"  Critical SNR: {c_j_results['SNR_crit_db']:.2f} dB")
    print(f"  Hardware quality (Γ_eff): {n_outputs['Gamma_eff_total']:.2e}")
    print(f"  Beam squint loss (avg): {-10 * np.log10(g_factors['eta_bsq_avg']):.2f} dB")

    if 'Jensen_gap_bits' in df.columns and not df['Jensen_gap_bits'].isnull().all():
        max_gap = df['Jensen_gap_bits'].max()
        mean_gap = df['Jensen_gap_bits'].mean()

        # Find SNR where gap is maximum
        max_gap_idx = df['Jensen_gap_bits'].idxmax()
        max_gap_snr = df.loc[max_gap_idx, 'SNR0_db']

        print(f"\n  Jensen Gap Statistics:")
        print(f"    Maximum gap: {max_gap:.4f} bits/s/Hz (at SNR={max_gap_snr:.1f} dB)")
        print(f"    Mean gap: {mean_gap:.4f} bits/s/Hz")
        print(f"    B/f_c ratio: {config['channel']['B_hz'] / config['channel']['f_c_hz']:.4f}")

    print(f"{'=' * 80}\n")

    return csv_filename


def main():
    """Main entry point"""

    if len(sys.argv) > 1:
        config_file_path = sys.argv[1]
    else:
        # Try to find config.yaml in common locations
        search_paths = [
            'config.yaml',
            '/mnt/user-data/uploads/config.yaml',
            '/mnt/user-data/outputs/config.yaml'
        ]

        config_file_path = None
        for path in search_paths:
            if os.path.exists(path):
                config_file_path = path
                break

        if config_file_path is None:
            print("Error: No config.yaml found")
            print("Usage: python scan_snr_sweep_fixed.py [config.yaml]")
            print("\nSearched locations:")
            for path in search_paths:
                print(f"  - {path}")
            sys.exit(1)

    try:
        output_file = run_snr_sweep(config_file_path)
        if output_file:
            print(f"✓ SNR sweep completed successfully")
            print(f"  Output: {output_file}")
            print(f"\nNext steps:")
            print(f"  1. Run visualize_results.py to generate plots")
            print(f"  2. Run make_paper_tables.py to generate LaTeX tables")
            print(f"\nKey improvement: SNR range now automatically centers on SNR_crit")
            print(f"  - This ensures full capture of capacity curve")
            print(f"  - From linear region (low SNR) → transition → saturation (high SNR)")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()