#!/usr/bin/env python3
"""
Comprehensive Results Visualization for ISAC System
DR-08 / P2-DR-04 / P2-DR-01 Unified Visualization Script
FIXED VERSION - Addresses Expert Review Items #3 and #4

This script provides a complete visualization suite for ISAC performance analysis:
1. Pareto Front Plot (R_net vs RMSE, color-mapped by alpha)
2. Internal States Plot (Hardware factors, noise sources vs alpha) + Gamma breakdown + Alpha crossover
3. Capacity vs SNR Plot (C_J, C_G, C_sat, SNR_crit)
4. Jensen Gap Analysis Plot

New in this version:
- Gamma component breakdown (stacked bar chart) - Expert Item #3
- Alpha crossover point annotation (PN vs DSE intersection) - Expert Item #4

Usage:
    python visualize_results.py --pareto <pareto_csv> --snr <snr_csv>
    python visualize_results.py  # Uses default paths

Author: Generated according to DR-08 Protocol v1.0 + Expert Review
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse

# Matplotlib a-la-IEEE style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,
    'figure.figsize': (7, 5),
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


def find_alpha_crossover(df: pd.DataFrame) -> float:
    """
    Find the alpha value where PN and DSE curves cross.

    Args:
        df: DataFrame with columns 'alpha', 'sigma_2_phi_c_res_rad2', 'sigma_2_DSE_var'

    Returns:
        Alpha value at crossover, or None if not found
    """
    try:
        # Get PN and DSE curves
        alpha_vals = df['alpha'].values
        pn_vals = df['sigma_2_phi_c_res_rad2'].values
        dse_vals = df['sigma_2_DSE_var'].values

        # Find where DSE crosses PN (DSE decreases as alpha increases)
        # Look for sign change in (PN - DSE)
        diff = pn_vals - dse_vals

        # Find zero crossing using linear interpolation
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:  # Sign change detected
                # Linear interpolation
                alpha_cross = alpha_vals[i] - diff[i] * (alpha_vals[i + 1] - alpha_vals[i]) / (diff[i + 1] - diff[i])
                return alpha_cross

        # If no crossing found, return the alpha where they're closest
        min_diff_idx = np.argmin(np.abs(diff))
        return alpha_vals[min_diff_idx]

    except Exception as e:
        print(f"  Warning: Could not find alpha crossover: {e}")
        return None


def plot_pareto_front(csv_path: str, output_dir: str = None):
    """
    Loads Pareto data and generates the R_net vs RMSE plot.

    Args:
        csv_path: Path to Pareto results CSV
        output_dir: Directory to save output figures (defaults to same as CSV)
    """

    print(f"\n{'=' * 80}")
    print("PARETO FRONT VISUALIZATION")
    print(f"{'=' * 80}")
    print(f"Loading results from {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        print("Please run main.py first to generate results.")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False

    print(f"Loaded {len(df)} data points.")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Primary Pareto Plot (R_net vs RMSE) ---
    fig, ax = plt.subplots()

    # Scatter plot, color-mapped by 'alpha'
    sc = ax.scatter(df['RMSE_m'] * 1000,  # Convert to mm
                    df['R_net_bps_hz'],
                    c=df['alpha'],
                    cmap='viridis',
                    s=100,  # marker size
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.5,
                    zorder=10)

    # Optional: Add a connecting line
    df_sorted = df.sort_values(by='alpha')
    ax.plot(df_sorted['RMSE_m'] * 1000,
            df_sorted['R_net_bps_hz'],
            'k--',  # dashed black line
            alpha=0.4,
            linewidth=1.5,
            zorder=5)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'ISAC Overhead ($\alpha$)', fontsize=12)

    # Labels and Scaling
    ax.set_xlabel('Range RMSE (mm)', fontsize=13)
    ax.set_ylabel('Net Data Rate (bits/s/Hz)', fontsize=13)
    ax.set_title('ISAC Pareto Front: $R_{net}$ vs. RMSE', fontsize=14, fontweight='bold')

    # Use log scale for RMSE for better visibility
    ax.set_xscale('log')

    # Set grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # Save figure
    output_filename = os.path.join(output_dir, 'fig_pareto_front.png')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Pareto front plot saved to {output_filename}")
    plt.close(fig)

    # ========================================================================
    # FIXED: Enhanced Internal States Plot with Gamma Breakdown and Alpha Crossover
    # Addresses Expert Review Items #3 and #4
    # ========================================================================
    fig_states, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    # Plot 1: R_net and RMSE vs Alpha
    ax1 = axes[0]
    color1 = 'tab:blue'
    ax1.plot(df['alpha'], df['R_net_bps_hz'],
             color=color1, marker='o', markersize=6,
             linewidth=2, label='$R_{net}$')
    ax1.set_ylabel('Net Data Rate (bits/s/Hz)', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.legend(loc='upper left', fontsize=10)

    ax1b = ax1.twinx()
    color2 = 'tab:red'
    ax1b.plot(df['alpha'], df['RMSE_m'] * 1000,
              color=color2, marker='s', markersize=6,
              linewidth=2, label='RMSE')
    ax1b.set_ylabel('Range RMSE (mm)', color=color2, fontsize=12)
    ax1b.tick_params(axis='y', labelcolor=color2)
    ax1b.set_yscale('log')
    ax1b.legend(loc='upper right', fontsize=10)
    ax1.set_title(r'Performance vs. ISAC Overhead ($\alpha$)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 2: PN vs DSE scaling WITH CROSSOVER ANNOTATION (Expert Item #4)
    # ========================================================================
    ax2 = axes[1]
    ax2.plot(df['alpha'], df['sigma_2_phi_c_res_rad2'],
             marker='^', markersize=6, linewidth=2,
             color='tab:purple', label=r'$\sigma^2_{\phi,c,res}$ (PN)')
    ax2.set_ylabel(r'PN Variance (rad$^2$)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', fontsize=10)

    ax2b = ax2.twinx()
    ax2b.plot(df['alpha'], df['sigma_2_DSE_var'],
              marker='v', markersize=6, linewidth=2,
              color='tab:orange', label=r'$\sigma^2_{DSE}$')
    ax2b.set_ylabel('DSE Variance', fontsize=12, color='tab:orange')
    ax2b.tick_params(axis='y', labelcolor='tab:orange')
    ax2b.set_yscale('log')
    ax2b.legend(loc='upper right', fontsize=10)

    # FIXED: Find and annotate alpha crossover point (Expert Item #4)
    alpha_cross = find_alpha_crossover(df)
    if alpha_cross is not None:
        ax2.axvline(x=alpha_cross, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax2.text(alpha_cross, ax2.get_ylim()[1] * 0.5,
                 f'Crossover\n$\\alpha^* = {alpha_cross:.3f}$',
                 color='red', fontsize=10, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        print(f"  ✓ Alpha crossover point: α* = {alpha_cross:.3f}")

    ax2.set_title(r'Noise/Mismatch Variance vs. $\alpha$ (with Crossover)',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # Plot 3: Hardware Factors (Original)
    # ========================================================================
    ax3 = axes[2]
    ax3.plot(df['alpha'], df['Gamma_eff_total'],
             marker='p', markersize=6, linewidth=2,
             color='tab:brown', label=r'$\Gamma_{eff, total}$')
    ax3.set_ylabel(r'$\Gamma_{eff, total}$', fontsize=12)
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', fontsize=10)

    ax3b = ax3.twinx()
    ax3b.plot(df['alpha'], df['eta_bsq_avg'],
              marker='D', markersize=6, linewidth=2,
              color='tab:green', label=r'$\eta_{bsq, avg}$')
    ax3b.set_ylabel(r'$\eta_{bsq, avg}$', fontsize=12, color='tab:green')
    ax3b.tick_params(axis='y', labelcolor='tab:green')
    ax3b.legend(loc='upper right', fontsize=10)

    ax3.set_title(r'Hardware Factors vs. $\alpha$',
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # NEW Plot 4: Gamma Component Breakdown (Expert Item #3)
    # Stacked bar chart showing contribution of each Gamma component
    # ========================================================================
    ax4 = axes[3]

    # Check if Gamma breakdown columns exist
    if all(col in df.columns for col in ['Gamma_pa', 'Gamma_adc', 'Gamma_iq', 'Gamma_lo']):
        # Select a subset of alpha values for clarity (e.g., every other point)
        plot_indices = range(0, len(df), max(1, len(df) // 8))
        alpha_subset = df['alpha'].iloc[plot_indices].values

        # Extract component values
        gamma_pa = df['Gamma_pa'].iloc[plot_indices].values
        gamma_adc = df['Gamma_adc'].iloc[plot_indices].values
        gamma_iq = df['Gamma_iq'].iloc[plot_indices].values
        gamma_lo = df['Gamma_lo'].iloc[plot_indices].values

        # Create stacked bar chart
        width = 0.015 if len(alpha_subset) > 5 else 0.03
        x_pos = np.arange(len(alpha_subset))

        p1 = ax4.bar(alpha_subset, gamma_pa, width, label='PA', color='tab:red', alpha=0.8)
        p2 = ax4.bar(alpha_subset, gamma_adc, width, bottom=gamma_pa,
                     label='ADC', color='tab:blue', alpha=0.8)
        p3 = ax4.bar(alpha_subset, gamma_iq, width,
                     bottom=gamma_pa + gamma_adc,
                     label='I/Q', color='tab:green', alpha=0.8)
        p4 = ax4.bar(alpha_subset, gamma_lo, width,
                     bottom=gamma_pa + gamma_adc + gamma_iq,
                     label='LO', color='tab:orange', alpha=0.8)

        ax4.set_ylabel(r'$\Gamma$ Components (linear)', fontsize=12)
        ax4.set_yscale('log')
        ax4.legend(loc='upper right', fontsize=10, ncol=2)
        ax4.set_title(r'Hardware Distortion Breakdown vs. $\alpha$ (NEW)',
                      fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Print breakdown for first point
        if len(df) > 0:
            total = df['Gamma_eff_total'].iloc[0]
            print(f"\n  Gamma Breakdown (α={df['alpha'].iloc[0]:.3f}):")
            print(f"    PA:  {df['Gamma_pa'].iloc[0]:.2e} ({100 * df['Gamma_pa'].iloc[0] / total:.1f}%)")
            print(f"    ADC: {df['Gamma_adc'].iloc[0]:.2e} ({100 * df['Gamma_adc'].iloc[0] / total:.1f}%)")
            print(f"    I/Q: {df['Gamma_iq'].iloc[0]:.2e} ({100 * df['Gamma_iq'].iloc[0] / total:.1f}%)")
            print(f"    LO:  {df['Gamma_lo'].iloc[0]:.2e} ({100 * df['Gamma_lo'].iloc[0] / total:.1f}%)")
    else:
        ax4.text(0.5, 0.5, 'Gamma breakdown data not available\n(Run with updated physics_engine.py)',
                 ha='center', va='center', transform=ax4.transAxes,
                 fontsize=12, color='red')
        ax4.set_title(r'Hardware Distortion Breakdown (Data Missing)',
                      fontsize=13, fontweight='bold')

    ax4.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=13)

    plt.tight_layout()
    output_states_fig = os.path.join(output_dir, 'fig_internal_states.png')
    plt.savefig(output_states_fig, dpi=300, bbox_inches='tight')
    print(f"✓ Internal states plot saved to {output_states_fig}")
    plt.close(fig_states)

    return True


def plot_snr_sweep(csv_path: str, output_dir: str = None):
    """
    Loads SNR sweep data and generates the C_J/C_G vs. SNR plot.

    Args:
        csv_path: Path to SNR sweep results CSV
        output_dir: Directory to save output figures (defaults to same as CSV)
    """

    print(f"\n{'=' * 80}")
    print("SNR SWEEP VISUALIZATION")
    print(f"{'=' * 80}")
    print(f"Loading results from {csv_path}...")

    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        print("Please run scan_snr_sweep.py first to generate results.")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False

    print(f"Loaded {len(df)} data points.")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Extract key metrics from the first row (they are constant)
    C_sat = df['C_sat'].iloc[0]
    SNR_crit_db = df['SNR_crit_db'].iloc[0]
    fixed_alpha = df['alpha'].iloc[0]

    # --- Primary Plot: Capacity vs. SNR ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot C_J (Jensen Upper Bound)
    ax.plot(df['SNR0_db'], df['C_J_bps_hz'],
            label=r'$C_J$ (Jensen Upper Bound)',
            color='tab:blue',
            linewidth=2.5,
            marker='o',
            markersize=4,
            markevery=3)

    # Plot C_G (Exact Gaussian) if available
    if 'C_G_bps_hz' in df.columns and not df['C_G_bps_hz'].isnull().all():
        ax.plot(df['SNR0_db'], df['C_G_bps_hz'],
                label=r'$C_G$ (Exact Gaussian)',
                color='tab:green',
                linestyle='--',
                linewidth=2.5,
                marker='s',
                markersize=4,
                markevery=3)

    # Plot C_sat (Hardware Ceiling)
    ax.axhline(y=C_sat, color='tab:red', linestyle=':',
               linewidth=2,
               label=f'$C_{{sat}}$ = {C_sat:.3f} bits/s/Hz')

    # Plot SNR_crit (Hardware Knee)
    ax.axvline(x=SNR_crit_db, color='tab:purple', linestyle=':',
               linewidth=2,
               label=f'$SNR_{{crit}}$ = {SNR_crit_db:.2f} dB')

    # Add shaded region showing hardware-limited regime
    ax.axvspan(-50, SNR_crit_db, alpha=0.1, color='gray',
               label='SNR-limited regime')
    ax.axvspan(SNR_crit_db, 60, alpha=0.1, color='red',
               label='Hardware-limited regime')

    # Labels and Limits
    ax.set_xlabel(r'SNR$_0$ (dB)', fontsize=13)
    ax.set_ylabel('Capacity (bits/s/Hz)', fontsize=13)
    ax.set_title(f'Communication Capacity vs. SNR (at $\\alpha={fixed_alpha}$)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)

    # Save figure
    output_filename = os.path.join(output_dir, 'fig_capacity_vs_snr.png')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Capacity vs. SNR plot saved to {output_filename}")
    plt.close(fig)

    # --- Secondary Plot: Jensen Gap ---
    if 'Jensen_gap_bits' in df.columns and not df['Jensen_gap_bits'].isnull().all():
        fig_gap, ax_gap = plt.subplots(figsize=(8, 5))

        ax_gap.plot(df['SNR0_db'], df['Jensen_gap_bits'],
                    label='Jensen Gap ($C_J - C_G$)',
                    color='tab:orange',
                    linewidth=2.5,
                    marker='o',
                    markersize=5)

        # Add horizontal line at zero for reference
        ax_gap.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

        # Highlight SNR_crit
        ax_gap.axvline(x=SNR_crit_db, color='tab:purple', linestyle=':',
                       linewidth=2, alpha=0.6,
                       label=f'$SNR_{{crit}}$ = {SNR_crit_db:.2f} dB')

        ax_gap.set_xlabel(r'SNR$_0$ (dB)', fontsize=13)
        ax_gap.set_ylabel('Jensen Gap (bits/s/Hz)', fontsize=13)
        ax_gap.set_title('Jensen Gap vs. SNR (DR-05 Validation)',
                         fontsize=14, fontweight='bold')
        ax_gap.legend(loc='best', fontsize=11)
        ax_gap.grid(True, which='both', linestyle='--', alpha=0.7)

        output_gap_fig = os.path.join(output_dir, 'fig_jensen_gap.png')
        plt.tight_layout()
        plt.savefig(output_gap_fig, dpi=300, bbox_inches='tight')
        print(f"✓ Jensen Gap plot saved to {output_gap_fig}")
        plt.close(fig_gap)

    return True


def generate_all_visualizations(pareto_csv: str = None, snr_csv: str = None,
                                output_dir: str = None):
    """
    Generate all visualizations from available data files.

    Args:
        pareto_csv: Path to Pareto results CSV (optional)
        snr_csv: Path to SNR sweep results CSV (optional)
        output_dir: Output directory for all figures
    """

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ISAC VISUALIZATION SUITE (FIXED VERSION)")
    print("=" * 80)

    # Use default paths if not specified
    default_output_dir = 'figures/'
    if output_dir is None:
        output_dir = default_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Try to find data files automatically
    if pareto_csv is None:
        search_paths = [
            'results/DR08_results_pareto_results.csv',
            'DR08_results_pareto_results.csv',
            './results/DR08_results_pareto_results.csv',
        ]
        for path in search_paths:
            if os.path.exists(path):
                pareto_csv = path
                break

    if snr_csv is None:
        search_paths = [
            'results/DR08_results_snr_sweep.csv',
            'DR08_results_snr_sweep.csv',
            './results/DR08_results_snr_sweep.csv',
        ]
        for path in search_paths:
            if os.path.exists(path):
                snr_csv = path
                break

    success_count = 0
    total_count = 0

    # Generate Pareto visualizations
    if pareto_csv:
        total_count += 1
        if plot_pareto_front(pareto_csv, output_dir):
            success_count += 1
    else:
        print("\n⚠ Warning: Pareto results CSV not found")
        print("  Run main.py to generate Pareto data")

    # Generate SNR sweep visualizations
    if snr_csv:
        total_count += 1
        if plot_snr_sweep(snr_csv, output_dir):
            success_count += 1
    else:
        print("\n⚠ Warning: SNR sweep CSV not found")
        print("  Run scan_snr_sweep.py to generate SNR sweep data")

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    print(f"Generated {success_count}/{total_count} visualization sets successfully")
    print(f"Output directory: {output_dir}")

    if success_count == total_count and total_count > 0:
        print("\n✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print("\nGenerated figures:")
        for fname in os.listdir(output_dir):
            if fname.endswith('.png'):
                print(f"  - {fname}")
        return True
    else:
        print("\n⚠ Some visualizations could not be generated")
        print("  Please ensure all required data files are available")
        return False


def main():
    """Main entry point with command-line argument parsing"""

    parser = argparse.ArgumentParser(
        description='Comprehensive ISAC Results Visualization (FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default paths
  python visualize_results.py

  # Specify custom paths
  python visualize_results.py --pareto results/pareto.csv --snr results/snr.csv

  # Specify output directory
  python visualize_results.py --output-dir ./figures/
        '''
    )

    parser.add_argument('--pareto', type=str, default=None,
                        help='Path to Pareto results CSV file')
    parser.add_argument('--snr', type=str, default=None,
                        help='Path to SNR sweep results CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures')

    args = parser.parse_args()

    success = generate_all_visualizations(
        pareto_csv=args.pareto,
        snr_csv=args.snr,
        output_dir=args.output_dir
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()