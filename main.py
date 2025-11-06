#!/usr/bin/env python3
"""
Results Generator for THz-ISL MIMO ISAC System
DR-08 Protocol Implementation - Main Execution Script

This module implements the final Results Generator that orchestrates the complete
ISAC Pareto front analysis and produces publication-ready output tables.

Functions:
    calc_Pareto_Front(config): Execute ISAC trade-off analysis via alpha sweep
    main(config_path): Main execution entry point with full pipeline

Author: Generated according to DR-08 Protocol v1.0
Date: November 2025
"""

import numpy as np
import pandas as pd
import yaml
import copy
import sys
import os
from typing import Dict, Any, List
import warnings

# Import validated DR-08 engines
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB, calc_MCRB
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure physics_engine.py and limits_engine.py are in the same directory")
    sys.exit(1)


def calc_Pareto_Front(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate ISAC Pareto Front via Direct Parameter Sweep

    This function implements the core ISAC trade-off analysis by sweeping over
    the alpha parameter space, computing both communication (C_J) and sensing
    (BCRLB) performance at each operating point.

    Architecture:
        - Direct Line Search over alpha_vec (DR-08, Sec 5.4, L15-48)
        - Mixed aperture: sigma_2_phi_c_res (scalar) → C_J (communication)
                         N_k_psd (vector) → BCRLB (sensing)
        - Stores complete internal state for each alpha value

    Args:
        config: Master configuration dictionary with alpha_vec defined

    Returns:
        List of dictionaries, each containing complete results for one alpha value:
            - alpha: ISAC overhead parameter
            - R_net_bps_hz: Net data rate [bits/s/Hz]
            - C_J_bps_hz: Jensen capacity at reference SNR [bits/s/Hz]
            - RMSE_m: Range RMSE [meters]
            - sigma_2_phi_c_res_rad2: Residual phase noise variance [rad²]
            - sigma_2_DSE_var: DSE residual variance
            - Gamma_eff_total: Hardware quality factor
            - eta_bsq_avg: Average beam squint factor
            - Nt, Nr, B_hz, f_c_hz: System parameters
            - seed: Random seed for reproducibility

    Reference:
        DR-08 Protocol Sec 5.4 [L1420-1440]: Pareto Front Computation
        P2-DR-04: ISAC Pareto Boundary Theory
    """

    print("\n" + "=" * 80)
    print("ISAC PARETO FRONT ANALYSIS")
    print("=" * 80)

    # Extract alpha sweep parameters
    try:
        alpha_vec = config['simulation']['alpha_vec']
        if not isinstance(alpha_vec, (list, np.ndarray)):
            raise ValueError("alpha_vec must be a list or array")
        if len(alpha_vec) == 0:
            raise ValueError("alpha_vec cannot be empty")
    except KeyError:
        raise KeyError("config.simulation.alpha_vec is required for Pareto front calculation")

    # Extract reference SNR for capacity reporting (default: 30 dB)
    SNR_ref_db = config['simulation'].get('SNR_ref_db', 30.0)

    # Extract TTD overhead if architecture requires it
    alpha_TTD = config.get('isac_model', {}).get('alpha_TTD', 0.0)

    # Speed of light for RMSE conversion
    c_mps = config['channel']['c_mps']

    # Initialize results storage
    pareto_data = []

    print(f"\nSweeping alpha over {len(alpha_vec)} points: {alpha_vec}")
    print(f"Reference SNR for capacity: {SNR_ref_db} dB")
    print(f"TTD overhead: {alpha_TTD:.4f}")
    print("\nProgress:")

    # === MAIN PARETO SWEEP LOOP ===
    # DR-08 Sec 5.4, Lines 15-48

    for idx, alpha_val in enumerate(alpha_vec):
        print(f"  [{idx + 1}/{len(alpha_vec)}] α = {alpha_val:.4f}...", end=" ", flush=True)

        try:
            # Step 1: Create loop-specific configuration (DR-08 Sec 5.4, L20)
            # CRITICAL: Use deepcopy to avoid cross-contamination between loop iterations
            loop_config = copy.deepcopy(config)

            # Override alpha value for this iteration
            loop_config['isac_model']['alpha'] = alpha_val

            # Step 2: Execute Physics Engine cascade (DR-08 Sec 5.4, L23-25)
            # Phase 1A: Multiplicative gains/losses
            g_sig_factors = calc_g_sig_factors(loop_config)

            # Phase 1B: Additive noise sources (causally depends on g_sig_factors)
            n_f_outputs = calc_n_f_vector(loop_config, g_sig_factors)

            # === CRITICAL: Apply safety clamping to N_k_psd ===
            # DR-08 Sec 5.4 safety requirement: prevent division by zero in FIM
            N_k_psd = n_f_outputs['N_k_psd']
            eps = np.finfo(float).eps
            N_k_psd_safe = np.maximum(N_k_psd, eps)
            n_f_outputs['N_k_psd'] = N_k_psd_safe  # Update with clamped version

            if np.any(N_k_psd <= 0):
                warnings.warn(f"α={alpha_val:.4f}: Clamped {np.sum(N_k_psd <= 0)} non-positive N_k_psd values")

            # Step 3: Calculate Communication Performance (DR-08 Sec 5.4, L28)
            # Communication Aperture: Uses SCALAR sigma_2_phi_c_res
            SNR_sweep = [SNR_ref_db - 10, SNR_ref_db, SNR_ref_db + 10]  # Local sweep for robustness
            c_j_results = calc_C_J(
                loop_config,
                g_sig_factors,
                n_f_outputs,
                SNR_sweep,
                compute_C_G=False  # Skip Jensen gap calculation in production
            )

            # Extract capacity at reference SNR
            C_J_ref_idx = 1  # Middle point is SNR_ref_db
            C_J_bps_hz = c_j_results['C_J_vec'][C_J_ref_idx]

            # Step 4: Calculate Net Data Rate (DR-08 Sec 5.4, L35)
            # R_net = (1 - α - α_TTD) * C_J * Bandwidth_prefix
            # Note: Bandwidth_prefix handles units conversion if needed
            overhead_total = alpha_val + alpha_TTD
            if overhead_total >= 1.0:
                warnings.warn(f"α={alpha_val:.4f}: Total overhead ≥ 1.0, setting R_net=0")
                R_net_bps_hz = 0.0
            else:
                R_net_bps_hz = (1.0 - overhead_total) * C_J_bps_hz

            # Step 5: Calculate Sensing Performance (DR-08 Sec 5.4, L31)
            # Sensing Aperture: Uses VECTOR N_k_psd (already clamped)
            bcrlb_results = calc_BCRLB(loop_config, g_sig_factors, n_f_outputs)

            # Extract range BCRLB and convert to RMSE in meters
            BCRLB_tau_s2 = bcrlb_results['BCRLB_tau']  # [seconds²]
            RMSE_tau_s = np.sqrt(BCRLB_tau_s2)  # [seconds]
            RMSE_m = RMSE_tau_s * c_mps  # [meters]

            # Step 6: Calculate DSE variance (DR-08 Sec 5.4, L37)
            # DSE noise component: sigma²_DSE = C_DSE / α⁵
            C_DSE = loop_config['isac_model']['C_DSE']
            if alpha_val > 0:
                sigma_2_DSE_var = C_DSE / (alpha_val ** 5)
            else:
                sigma_2_DSE_var = np.inf  # Infinite penalty for α=0

            # Step 7: Package complete results (DR-08 Sec 5.4, L40-48)
            result_dict = {
                # Primary performance metrics
                'alpha': alpha_val,
                'R_net_bps_hz': R_net_bps_hz,
                'C_J_bps_hz': C_J_bps_hz,
                'RMSE_m': RMSE_m,

                # Internal state variables (for analysis)
                'sigma_2_phi_c_res_rad2': n_f_outputs['sigma_2_phi_c_res'],
                'sigma_2_DSE_var': sigma_2_DSE_var,
                'Gamma_eff_total': n_f_outputs['Gamma_eff_total'],
                'eta_bsq_avg': g_sig_factors['eta_bsq_avg'],

                # System configuration parameters
                'Nt': loop_config['array']['Nt'],
                'Nr': loop_config['array']['Nr'],
                'B_hz': loop_config['channel']['B_hz'],
                'f_c_hz': loop_config['channel']['f_c_hz'],
                'seed': loop_config.get('seed', 42),

                # Additional derived metrics
                'G_sig_avg': g_sig_factors['G_sig_avg'],
                'C_sat': c_j_results['C_sat'],
                'SNR_crit_db': c_j_results['SNR_crit_db'],
                'BCRLB_tau_s2': BCRLB_tau_s2,
                'BCRLB_fD_Hz2': bcrlb_results['BCRLB_fD'],

                # Overhead tracking
                'overhead_total': overhead_total,
                'alpha_TTD': alpha_TTD
            }

            pareto_data.append(result_dict)
            print("✓")

        except Exception as e:
            print(f"✗ Failed: {e}")
            warnings.warn(f"α={alpha_val:.4f}: Calculation failed - {e}")
            # Continue with next alpha value
            continue

    print(f"\nCompleted: {len(pareto_data)}/{len(alpha_vec)} points successful")

    if len(pareto_data) == 0:
        raise RuntimeError("Pareto front calculation failed for all alpha values")

    return pareto_data


def main(config_path: str = 'config.yaml'):
    """
    Main execution entry point for THz-ISL MIMO ISAC Results Generator

    This function orchestrates the complete simulation pipeline:
    1. Load and validate configuration
    2. Set reproducibility seed
    3. Execute Pareto front analysis
    4. Generate publication-ready output tables
    5. Save results to CSV

    Workflow (DR-08 Sec 6.1):
        Line 1-5:   Load config and set seed
        Line 10:    Call calc_Pareto_Front
        Line 13:    Convert to DataFrame
        Line 15-20: Save to CSV with metadata

    Args:
        config_path: Path to YAML configuration file (default: 'config.yaml')

    Returns:
        None (writes results to file)

    Output Files:
        - {output_prefix}_pareto_results.csv: Main results table
        - {output_prefix}_summary.txt: Human-readable summary

    Reference:
        DR-08 Protocol Sec 6.1-6.2 [L2984-3011]: Main Execution Pipeline
    """

    print("\n" + "=" * 80)
    print("THz-ISL MIMO ISAC RESULTS GENERATOR")
    print("DR-08 Protocol Implementation")
    print("=" * 80)

    # Step 1: Load configuration file
    print(f"\n[1/5] Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"✗ Error parsing YAML file: {e}")
        sys.exit(1)

    # Step 2: Validate configuration
    print("\n[2/5] Validating configuration...")
    try:
        validate_config(config)
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    # Step 3: Set reproducibility seed (CRITICAL for reproducibility)
    # DR-08 Sec 6.1, L3-5
    seed = config.get('seed', 42)
    np.random.seed(seed)
    print(f"\n[3/5] Random seed set: {seed}")
    print("✓ Reproducibility enabled")

    # Step 4: Execute Pareto Front Analysis
    print("\n[4/5] Executing ISAC Pareto Front Analysis...")
    try:
        pareto_data = calc_Pareto_Front(config)
        print(f"✓ Pareto analysis completed: {len(pareto_data)} data points")
    except Exception as e:
        print(f"✗ Pareto analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Generate Output Tables
    print("\n[5/5] Generating output tables...")

    # Convert to pandas DataFrame (DR-08 Sec 6.1, L13)
    df = pd.DataFrame(pareto_data)

    # Ensure required columns are present (DR-08 Sec 6.2.2)
    required_columns = [
        'alpha', 'R_net_bps_hz', 'C_J_bps_hz', 'RMSE_m',
        'sigma_2_phi_c_res_rad2', 'sigma_2_DSE_var', 'Gamma_eff_total',
        'eta_bsq_avg', 'Nt', 'Nr', 'B_hz', 'f_c_hz', 'seed'
    ]

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        warnings.warn(f"Missing required columns: {missing_cols}")

    # Reorder columns for clarity (primary metrics first)
    column_order = [
        'alpha', 'R_net_bps_hz', 'C_J_bps_hz', 'RMSE_m',
        'Gamma_eff_total', 'sigma_2_phi_c_res_rad2', 'sigma_2_DSE_var',
        'eta_bsq_avg', 'G_sig_avg', 'C_sat', 'SNR_crit_db',
        'BCRLB_tau_s2', 'BCRLB_fD_Hz2', 'overhead_total', 'alpha_TTD',
        'Nt', 'Nr', 'B_hz', 'f_c_hz', 'seed'
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    # Prepare output paths
    output_config = config.get('outputs', {})
    save_path = output_config.get('save_path', '/mnt/user-data/outputs/')
    table_prefix = output_config.get('table_prefix', 'DR08_results')

    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)

    # Generate filenames
    csv_filename = os.path.join(save_path, f"{table_prefix}_pareto_results.csv")
    summary_filename = os.path.join(save_path, f"{table_prefix}_summary.txt")

    # Save main results table to CSV
    df.to_csv(csv_filename, index=False, float_format='%.6e')
    print(f"✓ Results saved to: {csv_filename}")

    # Generate human-readable summary
    with open(summary_filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THz-ISL MIMO ISAC PARETO FRONT ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Carrier Frequency: {config['channel']['f_c_hz'] / 1e9:.1f} GHz\n")
        f.write(f"  Bandwidth: {config['channel']['B_hz'] / 1e9:.1f} GHz\n")
        f.write(f"  Antenna Array: Nt={config['array']['Nt']}, Nr={config['array']['Nr']}\n")
        f.write(f"  Random Seed: {seed}\n")
        f.write(f"  Alpha Sweep: {len(pareto_data)} points\n\n")

        f.write("Performance Summary:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Max R_net: {df['R_net_bps_hz'].max():.3f} bits/s/Hz\n")
        f.write(f"  Min RMSE: {df['RMSE_m'].min() * 1e3:.3f} mm\n")
        f.write(f"  Hardware Quality: Γ_eff = {df['Gamma_eff_total'].iloc[0]:.2e}\n")
        f.write(f"  Beam Squint Loss: η_bsq = {df['eta_bsq_avg'].iloc[0]:.4f}\n\n")

        f.write("Pareto Front Points:\n")
        f.write("-" * 40 + "\n")
        f.write(df[['alpha', 'R_net_bps_hz', 'C_J_bps_hz', 'RMSE_m']].to_string(index=False))
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Summary saved to: {summary_filename}")

    # Display key results
    print("\n" + "=" * 80)
    print("KEY RESULTS")
    print("=" * 80)
    print(f"\nTop 5 Performance Points (by R_net):")
    print(df.nlargest(5, 'R_net_bps_hz')[['alpha', 'R_net_bps_hz', 'RMSE_m']])

    print(f"\nTop 5 Performance Points (by RMSE):")
    print(df.nsmallest(5, 'RMSE_m')[['alpha', 'R_net_bps_hz', 'RMSE_m']])

    print("\n" + "=" * 80)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {csv_filename}")
    print(f"  - {summary_filename}")
    print("\nResults generation complete!")


def create_default_config(output_path: str = 'config_default.yaml'):
    """
    Create a default configuration file for testing

    This helper function generates a complete configuration with reasonable
    defaults for THz-ISL MIMO ISAC systems.

    Args:
        output_path: Path where to save the default config

    Returns:
        Dictionary containing the default configuration
    """

    default_config = {
        # Random seed for reproducibility
        'seed': 42,

        # Array configuration
        'array': {
            'geometry': 'ULA',
            'Nt': 64,
            'Nr': 64,
            'L_ap_m': 0.05,
            'theta_0_deg': 15.0
        },

        # Channel configuration
        'channel': {
            'f_c_hz': 140e9,
            'B_hz': 5e9,
            'c_mps': 299792458.0
        },

        # Hardware impairments
        'hardware': {
            'gamma_pa_floor': 0.005,
            'gamma_adc_bits': 6,
            'gamma_iq_irr_dbc': -30.0,
            'gamma_lo_jitter_s': 50e-15,
            'rho_q_bits': 4,
            'rho_a_error_rms': 0.02,
            'papr_db': 0.1,
            'ibo_db': 0.5
        },

        # Platform dynamics
        'platform': {
            'sigma_theta_rad': 1e-6
        },

        # Phase noise model
        'pn_model': {
            'S_phi_c_model_type': 'Wiener',
            'S_phi_c_K2': 200.0,
            'S_phi_c_K0': 1e-15,
            'B_loop_hz': 1e6,
            'H_err_model_type': 'FirstOrderHPF',
            'sigma_rel_sq_rad2': 0.01
        },

        # ISAC parameters
        'isac_model': {
            'alpha': 0.05,  # Default value (will be overridden in sweep)
            'alpha_TTD': 0.01,
            'L_TTD_db': 2.0,
            'C_PN': 1e-3,
            'C_DSE': 1e-9
        },

        # Waveform parameters
        'waveform': {
            'S_RSM_path': None,
            'Phi_q': 0.0
        },

        # Simulation control
        'simulation': {
            'N': 2048,
            'FIM_MODE': 'Whittle',
            'SNR0_db_vec': [10, 20, 30, 40, 50],
            'SNR_ref_db': 30.0,
            'alpha_vec': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]  # Pareto sweep
        },

        # Output configuration
        'outputs': {
            'save_path': '/mnt/user-data/outputs/',
            'table_prefix': 'DR08_results'
        }
    }

    # Save to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    print(f"Default configuration saved to: {output_path}")
    return default_config


if __name__ == "__main__":
    """
    Command-line interface for results generator

    Usage:
        python main.py [config_path]

    If no config path provided, creates and uses default configuration.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description='THz-ISL MIMO ISAC Results Generator (DR-08 Protocol)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Use default config
    python main.py config.yaml              # Use custom config
    python main.py --create-default         # Create default config only
        """
    )

    parser.add_argument(
        'config_path',
        nargs='?',
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--create-default',
        action='store_true',
        help='Create default configuration file and exit'
    )

    args = parser.parse_args()

    if args.create_default:
        # Create default config and exit
        create_default_config('config_default.yaml')
        print("\nDefault configuration created. You can now run:")
        print("  python main.py config_default.yaml")
        sys.exit(0)

    # Check if config file exists, if not create default
    if not os.path.exists(args.config_path):
        print(f"Configuration file '{args.config_path}' not found.")
        print("Creating default configuration...")
        create_default_config(args.config_path)
        print(f"\nUsing default configuration: {args.config_path}")

    # Run main pipeline
    try:
        main(args.config_path)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)