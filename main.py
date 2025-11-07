#!/usr/bin/env python3
"""
Main Driver for THz-ISL MIMO ISAC Performance Analysis
DR-08 Protocol Implementation (COMPLETE VERSION)

This script performs alpha sweep to generate the ISAC Pareto front.

Usage:
    python main.py [config.yaml]

Output:
    - CSV file with Pareto results (alpha sweep)
    - Console summary of key metrics

Author: Generated according to DR-08 Protocol v1.0
"""

import numpy as np
import pandas as pd
import yaml
import sys
import os
import warnings
from typing import Dict, Any

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


def run_pareto_sweep(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Perform alpha sweep to generate ISAC Pareto front

    Args:
        config: Configuration dictionary

    Returns:
        DataFrame containing Pareto results
    """

    print("\n" + "=" * 80)
    print("ISAC PARETO FRONT GENERATION (ALPHA SWEEP)")
    print("=" * 80)

    # Define alpha sweep range
    # Expert recommendation: start from 0.05 to avoid DSE domination (1/α^5)
    alpha_min = 0.05
    alpha_max = 0.30
    n_alpha = 15  # Number of alpha points

    alpha_vec = np.linspace(alpha_min, alpha_max, n_alpha)
    print(f"\nSweeping α from {alpha_min} to {alpha_max} ({n_alpha} points)")

    # Initialize results storage
    results_list = []

    # Get speed of light for RMSE conversion
    c_mps = config['channel']['c_mps']

    # Fixed SNR for BCRLB calculation (mid-range)
    SNR0_db_fixed = 20.0  # dB

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
            # CRITICAL NOTE (Expert Review Item #2):
            # If τ is ROUND-TRIP delay: RMSE_range = (c/2) * sqrt(BCRLB_tau)
            # If τ is ONE-WAY delay: RMSE_range = c * sqrt(BCRLB_tau)
            # Here we assume MONOSTATIC (round-trip), so use c/2
            BCRLB_tau_safe = max(bcrlb_results['BCRLB_tau'], 1e-40)  # ✅ 防止下溢
            RMSE_tau_s = np.sqrt(bcrlb_results['BCRLB_tau'])
            RMSE_range_m = (c_mps / 2.0) * RMSE_tau_s

            if RMSE_range_m < 1e-5:  # < 10 micrometers
                print(f"  ⚠ Warning: RMSE suspiciously small ({RMSE_range_m:.2e} m), "
                      f"BCRLB_tau={bcrlb_results['BCRLB_tau']:.2e}")

            # Doppler RMSE
            RMSE_fD_hz = np.sqrt(bcrlb_results['BCRLB_fD'])

            # ================================================================
            # PHASE 4: Package Results
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

                # Noise factors
                'Gamma_eff_total': n_f_outputs['Gamma_eff_total'],
                'Gamma_pa': n_f_outputs['Gamma_pa'],
                'Gamma_adc': n_f_outputs['Gamma_adc'],
                'Gamma_iq': n_f_outputs['Gamma_iq'],
                'Gamma_lo': n_f_outputs['Gamma_lo'],
                'sigma_2_phi_c_res_rad2': n_f_outputs['sigma_2_phi_c_res'],
                'sigma_2_DSE_var': n_f_outputs['sigma2_DSE'],

                # System parameters
                'SNR0_db': SNR0_db_fixed,
                'B_hz': config['channel']['B_hz'],
                'f_c_hz': config['channel']['f_c_hz']
            }

            results_list.append(result_row)

            # Progress indicator
            progress_pct = (i + 1) / n_alpha * 100
            print(f"  [{i+1}/{n_alpha}] α={alpha:.3f}: R_net={R_net:.3f} bits/s/Hz, RMSE={RMSE_range_m*1000:.3f} mm ({progress_pct:.0f}%)")

        except Exception as e:
            print(f"  [ERROR] α={alpha:.3f} failed: {e}")
            # Continue with next alpha value
            continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results_list)

    print("\n✓ Alpha sweep completed successfully!")
    print(f"  Generated {len(df_results)} valid data points")

    return df_results


def save_results(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Save results to CSV file

    Args:
        df: Results DataFrame
        config: Configuration dictionary

    Returns:
        Path to saved CSV file
    """

    # Get output configuration
    output_config = config.get('outputs', {})
    save_path = output_config.get('save_path', './results/')
    table_prefix = output_config.get('table_prefix', 'DR08_results')

    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Generate filename
    csv_filename = os.path.join(save_path, f"{table_prefix}_pareto_results.csv")

    # Save to CSV
    df.to_csv(csv_filename, index=False, float_format='%.6e')

    print(f"\n✓ Results saved to: {csv_filename}")

    return csv_filename


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics of Pareto results

    Args:
        df: Results DataFrame
    """

    print("\n" + "=" * 80)
    print("PARETO FRONT SUMMARY")
    print("=" * 80)

    # Find best operating points
    best_R_net_idx = df['R_net_bps_hz'].idxmax()
    best_RMSE_idx = df['RMSE_m'].idxmin()

    print("\n1. Best Communication Performance:")
    print(f"   α = {df.loc[best_R_net_idx, 'alpha']:.3f}")
    print(f"   R_net = {df.loc[best_R_net_idx, 'R_net_bps_hz']:.3f} bits/s/Hz")
    print(f"   RMSE = {df.loc[best_R_net_idx, 'RMSE_m']*1000:.3f} mm")

    print("\n2. Best Sensing Performance:")
    print(f"   α = {df.loc[best_RMSE_idx, 'alpha']:.3f}")
    print(f"   R_net = {df.loc[best_RMSE_idx, 'R_net_bps_hz']:.3f} bits/s/Hz")
    print(f"   RMSE = {df.loc[best_RMSE_idx, 'RMSE_m']*1000:.3f} mm")

    print("\n3. System-Level Metrics (averaged):")
    print(f"   C_sat = {df['C_sat'].mean():.3f} bits/s/Hz")
    print(f"   SNR_crit = {df['SNR_crit_db'].mean():.2f} dB")
    print(f"   Γ_eff (total) = {df['Gamma_eff_total'].mean():.2e}")
    print(f"   η_bsq (avg) = {df['eta_bsq_avg'].mean():.4f}")

    print("\n4. Hardware Breakdown (first point):")
    if len(df) > 0:
        total_gamma = df['Gamma_eff_total'].iloc[0]
        print(f"   Gamma_PA:  {df['Gamma_pa'].iloc[0]:.2e} ({100*df['Gamma_pa'].iloc[0]/total_gamma:.1f}%)")
        print(f"   Gamma_ADC: {df['Gamma_adc'].iloc[0]:.2e} ({100*df['Gamma_adc'].iloc[0]/total_gamma:.1f}%)")
        print(f"   Gamma_I/Q: {df['Gamma_iq'].iloc[0]:.2e} ({100*df['Gamma_iq'].iloc[0]/total_gamma:.1f}%)")
        print(f"   Gamma_LO:  {df['Gamma_lo'].iloc[0]:.2e} ({100*df['Gamma_lo'].iloc[0]/total_gamma:.1f}%)")

    print("\n" + "=" * 80)


def main():
    """Main entry point"""

    print("=" * 80)
    print("THz-ISL MIMO ISAC PERFORMANCE ANALYSIS")
    print("DR-08 Protocol Implementation")
    print("=" * 80)

    # Parse command-line arguments
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default config path
        config_path = 'config.yaml'

    # Load configuration
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)

    # Validate configuration
    try:
        validate_config(config)
        print("✓ Configuration validated")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    # Suppress runtime warnings for cleaner output
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Run Pareto sweep
    try:
        df_results = run_pareto_sweep(config)
    except Exception as e:
        print(f"\n✗ Pareto sweep failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    try:
        csv_path = save_results(df_results, config)
    except Exception as e:
        print(f"\n✗ Failed to save results: {e}")
        sys.exit(1)

    # Print summary
    print_summary(df_results)

    # Next steps guidance
    print("\n📊 Next Steps:")
    print("  1. Run: python scan_snr_sweep.py config.yaml")
    print("  2. Run: python visualize_results.py")
    print("  3. Run: python make_paper_tables.py")
    print("\n✓ Pareto front generation complete!")


if __name__ == "__main__":
    main()