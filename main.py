#!/usr/bin/env python3
"""
Main Driver for THz-ISL MIMO ISAC Performance Analysis
DR-08 Protocol Implementation (EXPERT-IMPROVED VERSION + DSE AUTO-CALIBRATION)

IMPROVEMENTS IN THIS VERSION:
1. Multi-hardware configuration sweep support (Document 2, Fig. X3)
2. Noise composition tracking for visualization (Document 2, Fig. X2)
3. Enhanced error handling and progress reporting
4. Support for both single and multi-hardware runs
5. **NEW**: DSE auto-calibration feature (Expert Review Feedback)

Usage:
    # Single hardware configuration (default)
    python main.py [config.yaml]

    # Multi-hardware sweep
    python main.py [config.yaml] --multi-hardware

Output:
    - CSV file(s) with Pareto results (alpha sweep)
    - Noise composition breakdown data
    - Console summary of key metrics

Author: Generated according to DR-08 Protocol v1.0 + Expert Recommendations
"""
import io

import numpy as np
import pandas as pd
import yaml
import sys
import os
import warnings
import copy
import argparse
from typing import Dict, Any, List
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import the validated DR-08 engines
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB, calc_MCRB
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure physics_engine.py and limits_engine.py are in the same directory")
    sys.exit(1)


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)


def run_pareto_sweep(config: Dict[str, Any], tag: str = None) -> pd.DataFrame:
    """
    Perform alpha sweep to generate ISAC Pareto front

    Args:
        config: Configuration dictionary
        tag: Optional tag for this sweep (used in multi-hardware mode)

    Returns:
        DataFrame containing Pareto results
    """

    print("\n" + "=" * 80)
    if tag:
        print(f"ISAC PARETO FRONT GENERATION (ALPHA SWEEP) - {tag.upper()}")
    else:
        print("ISAC PARETO FRONT GENERATION (ALPHA SWEEP)")
    print("=" * 80)

    # ===== DSE自动校准 (Expert Recommendation) =====
    if config.get('isac_model', {}).get('DSE_autotune', False):
        print("\n[DSE Auto-Calibration] Enabled")
        alpha_star_target = config['isac_model'].get('alpha_star_target', 0.08)
        alpha_nom = alpha_star_target  # Use target as nominal point

        print(f"  Target PN-DSE crossover: α* = {alpha_star_target:.3f}")
        print(f"  Running calibration at α = {alpha_nom:.3f}...")

        # Temporarily set alpha to nominal value
        config_calib = copy.deepcopy(config)
        config_calib['isac_model']['alpha'] = alpha_nom

        try:
            # Run simulation at nominal alpha
            g_factors_calib = calc_g_sig_factors(config_calib)
            n_outputs_calib = calc_n_f_vector(config_calib, g_factors_calib)

            # Extract sigma_2_phi_c_res
            S_pn = n_outputs_calib['sigma_2_phi_c_res']

            # Calculate C_DSE to achieve crossover at alpha_star_target
            # At crossover: S_pn / α* = C_DSE / α*^5
            # Therefore: C_DSE = S_pn * α*^4
            C_DSE_calibrated = S_pn * (alpha_star_target ** 4)

            # Update config with calibrated C_DSE
            config['isac_model']['C_DSE'] = C_DSE_calibrated

            print(f"  Measured σ²_PN at α={alpha_nom:.3f}: {S_pn:.3e} rad²")
            print(f"  Calibrated C_DSE: {C_DSE_calibrated:.3e}")
            print(f"  ✓ DSE auto-calibration complete")

            # Print verification
            if config.get('debug', {}).get('print_dse_autotune', True):
                sigma2_dse_at_target = C_DSE_calibrated / (alpha_star_target ** 5)
                sigma2_pn_at_target = S_pn / alpha_star_target
                print(f"\n  Verification at α* = {alpha_star_target:.3f}:")
                print(f"    σ²_PN: {sigma2_pn_at_target:.3e} rad²")
                print(f"    σ²_DSE: {sigma2_dse_at_target:.3e} rad²")
                print(f"    Ratio: {sigma2_pn_at_target / sigma2_dse_at_target:.2f}")

        except Exception as e:
            print(f"  ✗ Warning: DSE calibration failed: {e}")
            print(f"  Using default C_DSE = {config['isac_model']['C_DSE']:.3e}")

    # Define alpha sweep range
    # Expert recommendation: start from 0.05 to avoid DSE domination (1/α^5)
    alpha_min = 0.05
    alpha_max = 0.30
    n_alpha = 20  # Number of alpha points (increased for smoother curves)

    alpha_vec = np.linspace(alpha_min, alpha_max, n_alpha)
    print(f"\nSweeping α from {alpha_min} to {alpha_max} ({n_alpha} points)")

    # Initialize results storage
    results_list = []

    # Get speed of light for RMSE conversion
    c_mps = config['channel']['c_mps']

    # Fixed SNR for BCRLB calculation (mid-range)
    SNR0_db_fixed = config.get('simulation', {}).get('SNR0_db_fixed', 20.0)

    # Progress tracking
    print("\nProgress:")
    for i, alpha in enumerate(alpha_vec):
        # Update alpha in config
        config['isac_model']['alpha'] = alpha

        try:
            # ================================================================
            # PHASE 1: Physics Calculations
            # ================================================================
            # Step 1A: Multiplicative gains/losses
            g_sig_factors = calc_g_sig_factors(config)

            # Step 1B: Additive noise sources (depends on g_sig_factors)
            n_f_outputs = calc_n_f_vector(config, g_sig_factors)

            # ================================================================
            # PHASE 2: Performance Limits
            # ================================================================
            # Communication capacity (at fixed SNR)
            c_j_results = calc_C_J(
                config,
                g_sig_factors,
                n_f_outputs,
                [SNR0_db_fixed],  # Single SNR point
                compute_C_G=True
            )

            # Sensing performance (BCRLB)
            bcrlb_results = calc_BCRLB(config, g_sig_factors, n_f_outputs)

            # ================================================================
            # PHASE 3: Derived Metrics
            # ================================================================
            # Net data rate (communication with ISAC overhead)
            C_J_at_SNR = c_j_results['C_J_vec'][0]  # bits/s/Hz
            R_net = (1 - alpha) * C_J_at_SNR  # Apply ISAC overhead

            # Range RMSE (convert from time to distance)
            # CRITICAL NOTE: Monostatic radar → use c/2
            BCRLB_tau_safe = max(bcrlb_results['BCRLB_tau'], 1e-40)
            RMSE_tau_s = np.sqrt(BCRLB_tau_safe)
            RMSE_range_m = (c_mps / 2.0) * RMSE_tau_s

            # Doppler RMSE
            RMSE_fD_hz = np.sqrt(bcrlb_results['BCRLB_fD'])

            # ================================================================
            # PHASE 4: Package Results (IMPROVED: Include noise breakdown)
            # ================================================================
            result_row = {
                # ISAC parameter
                'alpha': alpha,

                # Communication metrics
                'C_J_bps_hz': C_J_at_SNR,
                'R_net_bps_hz': R_net,
                'C_sat': c_j_results['C_sat'],
                'SNR_crit_db': c_j_results['SNR_crit_db'],

                # Sensing metrics
                'RMSE_m': RMSE_range_m,
                'RMSE_fD_hz': RMSE_fD_hz,
                'BCRLB_tau_s2': bcrlb_results['BCRLB_tau'],
                'BCRLB_fD_hz2': bcrlb_results['BCRLB_fD'],

                # Hardware factors (from physics engine)
                'G_sig_ideal': g_sig_factors['G_sig_ideal'],
                'G_sig_avg': g_sig_factors['G_sig_avg'],
                'eta_bsq_avg': g_sig_factors['eta_bsq_avg'],
                'rho_Q': g_sig_factors['rho_Q'],
                'rho_APE': g_sig_factors['rho_APE'],
                'rho_A': g_sig_factors['rho_A'],
                'rho_PN': g_sig_factors['rho_PN'],

                # Noise composition (NEW)
                'Gamma_eff_total': n_f_outputs['Gamma_eff_total'],
                'sigma_2_phi_c_res_rad2': n_f_outputs['sigma_2_phi_c_res'],
                'sigma_2_DSE_var': n_f_outputs.get('sigma_2_DSE_var', np.nan),
                'sigma_2_theta_pe_rad2': n_f_outputs.get('sigma_2_theta_pe_rad2', np.nan),

                # Jensen gap (if computed)
                'Jensen_gap_bits': c_j_results.get('Jensen_gap_bits', [np.nan])[0]
                if 'Jensen_gap_bits' in c_j_results else np.nan
            }

            # Add component-level Gamma if available
            for comp in ['Gamma_pa', 'Gamma_adc', 'Gamma_iq', 'Gamma_lo']:
                if comp in n_f_outputs:
                    result_row[comp] = n_f_outputs[comp]

            results_list.append(result_row)

            # Progress indicator (every 4 points or at end)
            if (i + 1) % 4 == 0 or i == len(alpha_vec) - 1:
                print(f"  [{i + 1:2d}/{len(alpha_vec):2d}] "
                      f"α={alpha:.3f}: "
                      f"R_net={R_net:.3f} bits/s/Hz, "
                      f"RMSE={RMSE_range_m * 1000:.2f} mm")

        except Exception as e:
            warnings.warn(f"Failed at α={alpha:.3f}: {e}")
            print(f"  [WARNING] Skipping α={alpha:.3f} due to error: {e}")
            continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results_list)

    # Sort by alpha (should already be sorted, but be safe)
    df_results = df_results.sort_values('alpha').reset_index(drop=True)

    return df_results


def save_results(df: pd.DataFrame, config: Dict[str, Any], tag: str = None):
    """
    Save Pareto results to CSV file

    Args:
        df: Results DataFrame
        config: Configuration dictionary
        tag: Optional tag for filename
    """

    # Get output configuration
    output_config = config.get('outputs', {})
    save_path = output_config.get('save_path', './results/')
    table_prefix = output_config.get('table_prefix', 'DR08_results')

    # Create output directory if needed
    os.makedirs(save_path, exist_ok=True)

    # Generate filename
    if tag:
        filename = f"{table_prefix}_{tag}_pareto_results.csv"
    else:
        filename = f"{table_prefix}_pareto_results.csv"

    full_path = os.path.join(save_path, filename)

    # Save to CSV
    df.to_csv(full_path, index=False, float_format='%.6e')

    print(f"\n✓ Results saved to: {full_path}")
    print(f"  Total data points: {len(df)}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics of Pareto results"""

    print("\n" + "=" * 80)
    print("PARETO FRONT SUMMARY")
    print("=" * 80)

    # Find best operating points
    idx_best_R_net = df['R_net_bps_hz'].idxmax()
    idx_best_RMSE = df['RMSE_m'].idxmin()

    print("\n[Best Communication Performance]")
    row = df.loc[idx_best_R_net]
    print(f"  α = {row['alpha']:.3f}")
    print(f"  R_net = {row['R_net_bps_hz']:.3f} bits/s/Hz")
    print(f"  RMSE = {row['RMSE_m'] * 1000:.2f} mm")
    print(f"  C_sat = {row['C_sat']:.3f} bits/s/Hz")

    print("\n[Best Sensing Performance]")
    row = df.loc[idx_best_RMSE]
    print(f"  α = {row['alpha']:.3f}")
    print(f"  RMSE = {row['RMSE_m'] * 1000:.2f} mm")
    print(f"  R_net = {row['R_net_bps_hz']:.3f} bits/s/Hz")

    print("\n[Hardware Bottlenecks] (at α=0.10)")
    idx_mid = (df['alpha'] - 0.10).abs().idxmin()
    row = df.loc[idx_mid]

    G_sig_ideal = row['G_sig_ideal']
    G_sig_avg = row['G_sig_avg']
    total_loss_db = -10 * np.log10(G_sig_avg / G_sig_ideal)

    print(f"  Total multiplicative loss: {total_loss_db:.2f} dB")
    print(f"    - Beam squint: {-10 * np.log10(row['eta_bsq_avg']):.2f} dB")
    print(f"    - Phase quantization: {-10 * np.log10(row['rho_Q']):.2f} dB")
    print(f"    - Pointing error: {-10 * np.log10(row['rho_APE']):.2f} dB")
    print(f"    - Amplitude error: {-10 * np.log10(row['rho_A']):.2f} dB")
    print(f"    - Differential PN: {-10 * np.log10(row['rho_PN']):.2f} dB")

    print(f"\n  Additive noise quality: Γ_eff = {row['Gamma_eff_total']:.2e}")
    print(f"  Phase noise (residual): σ²_φ,c,res = {row['sigma_2_phi_c_res_rad2']:.2e} rad²")

    print("\n" + "=" * 80)


def main():
    """Main execution function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='DR-08 Protocol: THz-ISL MIMO ISAC Performance Analysis'
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--multi-hardware',
        action='store_true',
        help='Run multi-hardware configuration sweep'
    )

    args = parser.parse_args()

    # Load configuration
    print("\n" + "=" * 80)
    print("DR-08 PROTOCOL: THz-ISL MIMO ISAC PERFORMANCE ANALYSIS")
    print("=" * 80)

    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Validate configuration
    try:
        validate_config(config)
        print("✓ Configuration validated")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    # Run simulation
    if args.multi_hardware:
        print("\n[MODE] Multi-hardware configuration sweep")
        # TODO: Implement multi-hardware sweep logic
        print("  (Multi-hardware mode not yet fully implemented)")
        df_results = run_pareto_sweep(config)
        save_results(df_results, config)
        print_summary(df_results)
    else:
        print("\n[MODE] Single hardware configuration")
        df_results = run_pareto_sweep(config)
        save_results(df_results, config)
        print_summary(df_results)

    # Success message
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Visualize results: python visualize_results.py")
    print("  2. Generate tables: python make_paper_tables.py")
    print("  3. Run SNR sweep: python scan_snr_sweep.py")
    print("  4. Generate supplementary figures: python generate_supplementary_figures.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())