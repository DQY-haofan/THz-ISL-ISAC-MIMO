#!/usr/bin/env python3
"""
Results Tabulation: Generate Tables for IEEE Paper
DR-08 / P2-DR-04 / P2-DR-01 Table Generation Script

Reads the output CSV from main.py and scan_snr_sweep.py to generate
publication-ready tables for the "Results & Discussion" section.

Usage:
    python make_paper_tables.py [pareto_csv] [snr_csv]

Author: Generated according to DR-08 Protocol v1.0
"""

import pandas as pd
import numpy as np
import sys
import os


def format_scientific(value, precision=2):
    """Format number in scientific notation for LaTeX"""
    if pd.isna(value) or np.isinf(value):
        return '---'
    exp = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
    mantissa = value / (10 ** exp)
    return f"{mantissa:.{precision}f}e{exp:+d}"


def make_tables(pareto_csv_path: str, snr_csv_path: str = None):
    """
    Loads results and generates formatted tables.

    Args:
        pareto_csv_path: Path to Pareto results CSV
        snr_csv_path: Path to SNR sweep results CSV (optional)
    """

    print(f"\n{'=' * 80}")
    print("IEEE PAPER TABLE GENERATION")
    print(f"{'=' * 80}\n")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print(f"Loading Pareto data from: {pareto_csv_path}")
    if not os.path.exists(pareto_csv_path):
        print(f"✗ Error: File not found {pareto_csv_path}")
        print("  Please run main.py first to generate Pareto results")
        return False

    try:
        df_pareto = pd.read_csv(pareto_csv_path)
        print(f"✓ Loaded {len(df_pareto)} Pareto data points")
    except Exception as e:
        print(f"✗ Error loading Pareto CSV: {e}")
        return False

    # ========================================================================
    # TABLE A: Communication Benchmarks vs. ISAC Overhead (α)
    # Corresponds to P2-DR-01 Table 1/2 spec [DR-08, Sec 6.2.1]
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("TABLE A: Communication Benchmarks vs. ISAC Overhead (α)")
    print("Corresponds to P2-DR-01 Table 1/2 spec [DR-08, Sec 6.2.1]")
    print(f"{'=' * 80}\n")

    table_A_cols = [
        'alpha',
        'C_sat',
        'SNR_crit_db',
        'sigma_2_phi_c_res_rad2'
    ]

    if all(col in df_pareto.columns for col in table_A_cols):
        df_table_A = df_pareto[table_A_cols].copy()

        # Format for readability
        df_table_A['C_sat'] = df_table_A['C_sat'].map('{:.3f}'.format)
        df_table_A['SNR_crit_db'] = df_table_A['SNR_crit_db'].map('{:.2f}'.format)
        df_table_A['sigma_2_phi_c_res_rad2'] = df_table_A['sigma_2_phi_c_res_rad2'].map('{:.2e}'.format)

        # Rename columns for publication
        df_table_A.columns = ['α', 'C_sat (bits/s/Hz)', 'SNR_crit (dB)', 'σ²_φ,c,res (rad²)']

        print(df_table_A.to_string(index=False))
        print("\n")

        # Generate LaTeX table
        print("LaTeX Format:")
        print("-" * 80)
        latex_str = df_table_A.to_latex(index=False, escape=False,
                                        column_format='c' * len(df_table_A.columns))
        print(latex_str)
    else:
        print("⚠ Warning: Required columns not found for Table A")
        print(f"  Available columns: {df_pareto.columns.tolist()}")

    # ========================================================================
    # TABLE B: ISAC Pareto Front Key Operating Points
    # Corresponds to P2-DR-04 Table 1/2/3 spec [DR-08, Sec 6.2.2]
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("TABLE B: ISAC Pareto Front Key Operating Points")
    print("Corresponds to P2-DR-04 Table 1/2/3 spec [DR-08, Sec 6.2.2]")
    print(f"{'=' * 80}\n")

    table_B_cols = [
        'alpha',
        'R_net_bps_hz',
        'RMSE_m',
        'sigma_2_phi_c_res_rad2',
        'sigma_2_DSE_var'
    ]

    if all(col in df_pareto.columns for col in table_B_cols):
        df_table_B = df_pareto[table_B_cols].copy()

        # Format for readability
        df_table_B['R_net_bps_hz'] = df_table_B['R_net_bps_hz'].map('{:.3f}'.format)
        df_table_B['RMSE_m'] = (df_table_B['RMSE_m'].astype(float) * 1000).map('{:.3f}'.format)  # to mm
        df_table_B['sigma_2_phi_c_res_rad2'] = df_table_B['sigma_2_phi_c_res_rad2'].map('{:.2e}'.format)
        df_table_B['sigma_2_DSE_var'] = df_table_B['sigma_2_DSE_var'].map('{:.2e}'.format)

        # Rename columns
        df_table_B.columns = ['α', 'R_net (bits/s/Hz)', 'RMSE (mm)',
                              'σ²_φ,c,res (rad²)', 'σ²_DSE']

        print("--- Full Pareto Sweep Table ---")
        print(df_table_B.to_string(index=False))
        print("\n")

        # Summary points
        try:
            df_pareto_numeric = df_pareto.copy()
            best_R_net_idx = df_pareto_numeric['R_net_bps_hz'].idxmax()
            best_RMSE_idx = df_pareto_numeric['RMSE_m'].idxmin()

            df_summary = pd.concat([
                df_pareto_numeric.loc[[best_R_net_idx]],
                df_pareto_numeric.loc[[best_RMSE_idx]]
            ])
            df_summary.index = ['Best R_net', 'Best RMSE']

            print("\n--- Key Operating Points Summary ---")
            summary_display = df_summary[table_B_cols].copy()
            summary_display['R_net_bps_hz'] = summary_display['R_net_bps_hz'].map('{:.3f}'.format)
            summary_display['RMSE_m'] = (summary_display['RMSE_m'].astype(float) * 1000).map('{:.3f}'.format)
            summary_display['sigma_2_phi_c_res_rad2'] = summary_display['sigma_2_phi_c_res_rad2'].map('{:.2e}'.format)
            summary_display['sigma_2_DSE_var'] = summary_display['sigma_2_DSE_var'].map('{:.2e}'.format)
            summary_display.columns = ['α', 'R_net (bits/s/Hz)', 'RMSE (mm)',
                                       'σ²_φ,c,res (rad²)', 'σ²_DSE']
            print(summary_display.to_string())
            print("\n")

            # LaTeX format for summary
            print("LaTeX Format (Summary Points):")
            print("-" * 80)
            latex_summary = summary_display.to_latex(escape=False,
                                                     column_format='l' + 'c' * (len(summary_display.columns)))
            print(latex_summary)

        except (ValueError, KeyError) as e:
            print(f"\n⚠ Could not generate summary points: {e}")
    else:
        print("⚠ Warning: Required columns not found for Table B")
        print(f"  Available columns: {df_pareto.columns.tolist()}")

    # ========================================================================
    # TABLE C: Jensen Gap Summary (from expert review, DR-05)
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("TABLE C: Jensen Gap (C_J vs C_G) Summary (DR-05)")
    print(f"{'=' * 80}\n")

    if snr_csv_path and os.path.exists(snr_csv_path):
        try:
            df_snr = pd.read_csv(snr_csv_path)
            print(f"✓ Loaded SNR sweep data: {len(df_snr)} points\n")

            if 'Jensen_gap_bits' in df_snr.columns and not df_snr['Jensen_gap_bits'].isnull().all():
                max_gap = df_snr['Jensen_gap_bits'].max()
                mean_gap = df_snr['Jensen_gap_bits'].mean()
                min_gap = df_snr['Jensen_gap_bits'].min()

                # Find gap at SNR_crit
                snr_crit_db = df_snr['SNR_crit_db'].iloc[0]
                idx_crit = (df_snr['SNR0_db'] - snr_crit_db).abs().argmin()
                gap_at_crit = df_snr.iloc[idx_crit]['Jensen_gap_bits']

                B_over_fc = df_snr['B_hz'].iloc[0] / df_snr['f_c_hz'].iloc[0]
                alpha_val = df_snr['alpha'].iloc[0]

                # Create summary DataFrame
                jensen_summary = pd.DataFrame({
                    'Metric': [
                        'Configuration',
                        'α (ISAC overhead)',
                        'B/f_c ratio',
                        'Maximum Gap',
                        'Mean Gap',
                        'Minimum Gap',
                        'Gap at SNR_crit'
                    ],
                    'Value': [
                        f"{df_snr['B_hz'].iloc[0] / 1e9:.1f} GHz / {df_snr['f_c_hz'].iloc[0] / 1e9:.0f} GHz",
                        f"{alpha_val:.3f}",
                        f"{B_over_fc:.4f}",
                        f"{max_gap:.4f} bits/s/Hz",
                        f"{mean_gap:.4f} bits/s/Hz",
                        f"{min_gap:.4f} bits/s/Hz",
                        f"{gap_at_crit:.4f} bits/s/Hz"
                    ]
                })

                print(jensen_summary.to_string(index=False))
                print("\n")

                # Assessment
                print("Jensen Gap Assessment:")
                if max_gap < 0.05:
                    print(f"  ✓ EXCELLENT: Max gap {max_gap:.4f} < 0.05 bits/s/Hz (DR-05 tight threshold)")
                elif max_gap < 0.5:
                    print(f"  ✓ GOOD: Max gap {max_gap:.4f} < 0.5 bits/s/Hz (acceptable for THz)")
                elif max_gap < 3.0:
                    print(f"  ⚠ MODERATE: Max gap {max_gap:.4f} < 3.0 bits/s/Hz (within limits)")
                else:
                    print(f"  ✗ LARGE: Max gap {max_gap:.4f} >= 3.0 bits/s/Hz (may need investigation)")
                print("\n")

                # LaTeX format
                print("LaTeX Format:")
                print("-" * 80)
                latex_jensen = jensen_summary.to_latex(index=False, escape=False,
                                                       column_format='lc')
                print(latex_jensen)

            else:
                print("⚠ SNR sweep file found, but 'Jensen_gap_bits' column missing.")
                print("  Run scan_snr_sweep.py with compute_C_G=True")

        except Exception as e:
            print(f"⚠ Error processing SNR sweep file: {e}")
    else:
        print("⚠ SNR sweep file not found or not provided.")
        print("  Run scan_snr_sweep.py to generate data for this table")
        if snr_csv_path:
            print(f"  Searched path: {snr_csv_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("TABLE GENERATION COMPLETE")
    print(f"{'=' * 80}\n")
    print("Generated tables:")
    print("  - Table A: Communication benchmarks vs α")
    print("  - Table B: ISAC Pareto front operating points")
    if snr_csv_path and os.path.exists(snr_csv_path):
        print("  - Table C: Jensen gap validation")
    print("\nAll tables are formatted for direct inclusion in IEEE papers.")
    print("LaTeX code has been provided for easy integration.")

    return True


def main():
    """Main entry point"""

    # Parse command-line arguments
    if len(sys.argv) > 1:
        pareto_file_path = sys.argv[1]
    else:
        # Try to find pareto results automatically (Windows-compatible)
        search_paths = [
            'results/DR08_results_pareto_results.csv',  # Most common location
            'DR08_results_pareto_results.csv',  # Current directory
            './results/DR08_results_pareto_results.csv',  # Explicit relative
        ]

        pareto_file_path = None
        for path in search_paths:
            if os.path.exists(path):
                pareto_file_path = path
                break

        if pareto_file_path is None:
            print("Error: No Pareto results CSV found")
            print("Usage: python make_paper_tables.py [pareto_csv] [snr_csv]")
            print("\nSearched locations:")
            for path in search_paths:
                print(f"  - {path}")
            sys.exit(1)

    # Optional second argument for SNR sweep file
    snr_file_path = None
    if len(sys.argv) > 2:
        snr_file_path = sys.argv[2]
    else:
        # Try to infer SNR file path from pareto path
        default_snr_path = pareto_file_path.replace('_pareto_results.csv', '_snr_sweep.csv')
        if os.path.exists(default_snr_path):
            snr_file_path = default_snr_path
        else:
            # Search in common locations (Windows-compatible)
            search_paths = [
                'results/DR08_results_snr_sweep.csv',  # Most common location
                'DR08_results_snr_sweep.csv',  # Current directory
                './results/DR08_results_snr_sweep.csv',  # Explicit relative
            ]
            for path in search_paths:
                if os.path.exists(path):
                    snr_file_path = path
                    break

    try:
        success = make_tables(pareto_file_path, snr_file_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()