#!/usr/bin/env python3
"""
Main Driver for THz-ISL MIMO ISAC Performance Analysis
DR-08 Protocol Implementation (EXPERT-IMPROVED VERSION)

IMPROVEMENTS IN THIS VERSION:
1. Multi-hardware configuration sweep support (Document 2, Fig. X3)
2. Noise composition tracking for visualization (Document 2, Fig. X2)
3. Enhanced error handling and progress reporting
4. Support for both single and multi-hardware runs

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

import numpy as np
import pandas as pd
import yaml
import sys
import os
import warnings
import copy
import argparse
from typing import Dict, Any, List

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

                # Noise factors
                'Gamma_eff_total': n_f_outputs['Gamma_eff_total'],
                'Gamma_pa': n_f_outputs['Gamma_pa'],
                'Gamma_adc': n_f_outputs['Gamma_adc'],
                'Gamma_iq': n_f_outputs['Gamma_iq'],
                'Gamma_lo': n_f_outputs['Gamma_lo'],
                'sigma_2_phi_c_res_rad2': n_f_outputs['sigma_2_phi_c_res'],
                'sigma_2_DSE_var': n_f_outputs['sigma2_DSE'],

                # ⭐ NEW: Noise composition breakdown (for Fig. X2)
                'noise_white': n_f_outputs.get('noise_components', {}).get('white', 0),
                'noise_gamma': n_f_outputs.get('noise_components', {}).get('gamma', 0),
                'noise_rsm': n_f_outputs.get('noise_components', {}).get('rsm', 0),
                'noise_pn': n_f_outputs.get('noise_components', {}).get('pn', 0),
                'noise_dse': n_f_outputs.get('noise_components', {}).get('dse', 0),

                # System parameters
                'SNR0_db': SNR0_db_fixed,
                'B_hz': config['channel']['B_hz'],
                'f_c_hz': config['channel']['f_c_hz'],
                'N': config['simulation']['N']
            }

            results_list.append(result_row)

            # Progress indicator
            progress_pct = (i + 1) / n_alpha * 100
            print(
                f"  [{i + 1}/{n_alpha}] α={alpha:.3f}: R_net={R_net:.3f} bits/s/Hz, "
                f"RMSE={RMSE_range_m * 1000:.3f} mm ({progress_pct:.0f}%)")

        except Exception as e:
            print(f"  [ERROR] α={alpha:.3f} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Convert to DataFrame
    df_results = pd.DataFrame(results_list)

    print("\n✓ Alpha sweep completed successfully!")
    print(f"  Generated {len(df_results)} valid data points")

    return df_results


def run_multi_hardware_sweep(config: Dict[str, Any], hardware_profiles: List[Dict[str, Any]]) -> Dict[
    str, pd.DataFrame]:
    """
    Run Pareto sweep for multiple hardware configurations

    This implements Document 2, Fig. X3: Hardware Sensitivity Analysis

    Args:
        config: Base configuration dictionary
        hardware_profiles: List of hardware profile dictionaries, each containing:
            - 'name': Profile identifier (e.g., 'good', 'mid', 'poor')
            - Hardware parameters to override

    Returns:
        Dictionary mapping profile names to their results DataFrames
    """

    print("\n" + "=" * 80)
    print("MULTI-HARDWARE CONFIGURATION SWEEP")
    print("Document 2, Fig. X3: Hardware Sensitivity Analysis")
    print("=" * 80)
    print(f"\nRunning {len(hardware_profiles)} hardware configurations...")

    results_dict = {}

    for i, profile in enumerate(hardware_profiles):
        profile_name = profile.get('name', f'config_{i}')
        print(f"\n[{i + 1}/{len(hardware_profiles)}] Processing profile: {profile_name}")
        print("-" * 80)

        # Create a copy of config for this profile
        profile_config = copy.deepcopy(config)

        # Apply hardware overrides
        for key, value in profile.items():
            if key != 'name':
                # Support nested keys (e.g., 'hardware.gamma_adc_bits')
                if '.' in key:
                    sections = key.split('.')
                    target = profile_config
                    for section in sections[:-1]:
                        target = target[section]
                    target[sections[-1]] = value
                else:
                    # Assume hardware section by default
                    if key in profile_config.get('hardware', {}):
                        profile_config['hardware'][key] = value

        # Print key hardware parameters for this profile
        print(f"  Hardware parameters:")
        for key, value in profile.items():
            if key != 'name':
                print(f"    {key}: {value}")

        # Run Pareto sweep for this configuration
        try:
            df_results = run_pareto_sweep(profile_config, tag=profile_name)
            results_dict[profile_name] = df_results

            # Quick summary
            best_R_net = df_results['R_net_bps_hz'].max()
            best_RMSE = df_results['RMSE_m'].min()
            print(f"  ✓ Profile '{profile_name}' complete:")
            print(f"    Best R_net: {best_R_net:.3f} bits/s/Hz")
            print(f"    Best RMSE: {best_RMSE * 1000:.3f} mm")

        except Exception as e:
            print(f"  ✗ Profile '{profile_name}' failed: {e}")
            continue

    print("\n" + "=" * 80)
    print(f"✓ Multi-hardware sweep completed: {len(results_dict)}/{len(hardware_profiles)} profiles successful")
    print("=" * 80)

    return results_dict


def get_default_hardware_profiles() -> List[Dict[str, Any]]:
    """
    Define default hardware profiles for multi-hardware sweep

    Returns three tiers: Good, Mid, Poor

    Returns:
        List of hardware profile dictionaries
    """
    return [
        {
            'name': 'good',
            'gamma_adc_bits': 12,  # 12-bit ENOB
            'gamma_lo_jitter_s': 10e-15,  # 10 fs
            'gamma_pa_floor': 0.001,
            'gamma_iq_irr_dbc': -35.0
        },
        {
            'name': 'mid',
            'gamma_adc_bits': 10,  # 10-bit ENOB
            'gamma_lo_jitter_s': 20e-15,  # 20 fs
            'gamma_pa_floor': 0.005,
            'gamma_iq_irr_dbc': -30.0
        },
        {
            'name': 'poor',
            'gamma_adc_bits': 8,  # 8-bit ENOB
            'gamma_lo_jitter_s': 50e-15,  # 50 fs
            'gamma_pa_floor': 0.02,
            'gamma_iq_irr_dbc': -25.0
        }
    ]


def save_results(df: pd.DataFrame, config: Dict[str, Any], tag: str = None) -> str:
    """
    Save results to CSV file

    Args:
        df: Results DataFrame
        config: Configuration dictionary
        tag: Optional tag to append to filename

    Returns:
        Path to saved CSV file
    """

    output_config = config.get('outputs', {})
    save_path = output_config.get('save_path', './results/')
    table_prefix = output_config.get('table_prefix', 'DR08_results')

    os.makedirs(save_path, exist_ok=True)

    # Generate filename
    if tag:
        csv_filename = os.path.join(save_path, f"{table_prefix}_pareto_{tag}.csv")
    else:
        csv_filename = os.path.join(save_path, f"{table_prefix}_pareto_results.csv")

    # Save to CSV
    df.to_csv(csv_filename, index=False, float_format='%.6e')

    print(f"\n✓ Results saved to: {csv_filename}")

    return csv_filename


def save_multi_hardware_results(results_dict: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> List[str]:
    """
    Save multiple hardware configuration results

    Args:
        results_dict: Dictionary mapping profile names to DataFrames
        config: Configuration dictionary

    Returns:
        List of saved file paths
    """

    saved_files = []

    for profile_name, df in results_dict.items():
        csv_path = save_results(df, config, tag=profile_name)
        saved_files.append(csv_path)

    return saved_files


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
    print(f"   RMSE = {df.loc[best_R_net_idx, 'RMSE_m'] * 1000:.3f} mm")

    print("\n2. Best Sensing Performance:")
    print(f"   α = {df.loc[best_RMSE_idx, 'alpha']:.3f}")
    print(f"   R_net = {df.loc[best_RMSE_idx, 'R_net_bps_hz']:.3f} bits/s/Hz")
    print(f"   RMSE = {df.loc[best_RMSE_idx, 'RMSE_m'] * 1000:.3f} mm")

    print("\n3. System-Level Metrics (averaged):")
    print(f"   C_sat = {df['C_sat'].mean():.3f} bits/s/Hz")
    print(f"   SNR_crit = {df['SNR_crit_db'].mean():.2f} dB")
    print(f"   Γ_eff (total) = {df['Gamma_eff_total'].mean():.2e}")
    print(f"   η_bsq (avg) = {df['eta_bsq_avg'].mean():.4f}")

    print("\n4. Hardware Breakdown (first point):")
    if len(df) > 0:
        total_gamma = df['Gamma_eff_total'].iloc[0]
        print(f"   Gamma_PA:  {df['Gamma_pa'].iloc[0]:.2e} ({100 * df['Gamma_pa'].iloc[0] / total_gamma:.1f}%)")
        print(f"   Gamma_ADC: {df['Gamma_adc'].iloc[0]:.2e} ({100 * df['Gamma_adc'].iloc[0] / total_gamma:.1f}%)")
        print(f"   Gamma_I/Q: {df['Gamma_iq'].iloc[0]:.2e} ({100 * df['Gamma_iq'].iloc[0] / total_gamma:.1f}%)")
        print(f"   Gamma_LO:  {df['Gamma_lo'].iloc[0]:.2e} ({100 * df['Gamma_lo'].iloc[0] / total_gamma:.1f}%)")

    print("\n5. Noise Composition Range (min-max across α):")
    if 'noise_white' in df.columns:
        print(f"   White:  [{df['noise_white'].min():.2e}, {df['noise_white'].max():.2e}] W/Hz")
        print(f"   Gamma:  [{df['noise_gamma'].min():.2e}, {df['noise_gamma'].max():.2e}] W/Hz")
        print(f"   PN:     [{df['noise_pn'].min():.2e}, {df['noise_pn'].max():.2e}] W/Hz")
        print(f"   DSE:    [{df['noise_dse'].min():.2e}, {df['noise_dse'].max():.2e}] W/Hz")

    print("\n" + "=" * 80)


def main():
    """Main entry point"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='THz-ISL MIMO ISAC Performance Analysis (DR-08 Protocol)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to configuration YAML file (default: config.yaml)')
    parser.add_argument('--multi-hardware', action='store_true',
                        help='Run multi-hardware configuration sweep')
    parser.add_argument('--hardware-profiles', type=str, default=None,
                        help='Path to custom hardware profiles YAML file')

    args = parser.parse_args()

    print("=" * 80)
    print("THz-ISL MIMO ISAC PERFORMANCE ANALYSIS")
    print("DR-08 Protocol Implementation (Expert-Improved Version)")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Validate configuration
    try:
        validate_config(config)
        print("✓ Configuration validated")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    # Suppress runtime warnings for cleaner output
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Decide whether to run single or multi-hardware sweep
    if args.multi_hardware:
        # Multi-hardware mode
        if args.hardware_profiles:
            # Load custom profiles
            with open(args.hardware_profiles, 'r') as f:
                hardware_profiles = yaml.safe_load(f)['profiles']
        else:
            # Use default profiles
            hardware_profiles = get_default_hardware_profiles()

        print(f"\n📊 Running multi-hardware sweep with {len(hardware_profiles)} configurations")

        try:
            results_dict = run_multi_hardware_sweep(config, hardware_profiles)
            saved_files = save_multi_hardware_results(results_dict, config)

            print("\n" + "=" * 80)
            print("MULTI-HARDWARE SWEEP SUMMARY")
            print("=" * 80)
            print(f"\nGenerated {len(saved_files)} result files:")
            for file in saved_files:
                print(f"  - {file}")

            # Print summary for each profile
            for profile_name, df in results_dict.items():
                print(f"\n--- Profile: {profile_name} ---")
                print_summary(df)

        except Exception as e:
            print(f"\n✗ Multi-hardware sweep failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # Single hardware mode (default)
        try:
            df_results = run_pareto_sweep(config)
            csv_path = save_results(df_results, config)
            print_summary(df_results)

        except Exception as e:
            print(f"\n✗ Pareto sweep failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Next steps guidance
    print("\n📊 Next Steps:")
    print("  1. Run: python scan_snr_sweep.py config.yaml")
    print("  2. Run: python threshold_sweep.py config.yaml")
    print("  3. Run: python visualize_results.py")
    print("  4. Run: python make_paper_tables.py")
    print("\n✓ Performance analysis complete!")


if __name__ == "__main__":
    main()