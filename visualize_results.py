#!/usr/bin/env python3
"""
Complete ISAC Visualization Suite - EXPERT-IMPROVED VERSION
IEEE Publication Style with Enhanced Noise Analysis

NEW FEATURES IN THIS VERSION:
1. Noise Composition vs Alpha (Document 2, Fig. X2) - CRITICAL
2. Multi-Hardware Comparison (Document 2, Fig. Y1)
3. Improved color schemes and layout
4. Better error handling

Generated Figures:
1. fig_pn_dse_crossover      [PN vs DSE crossover, semi-log]
2. fig_capacity_vs_snr       [Communication capacity]
3. fig_pareto_front          [ISAC Pareto front]
4. fig_performance_vs_alpha  [R_net and RMSE vs alpha]
5. fig_hardware_factors      [Hardware quality factors vs alpha]
6. fig_gamma_breakdown       [Gamma noise component breakdown]
7. fig_jensen_gap            [Jensen gap validation]
8. fig_noise_composition     [Noise composition vs alpha - NEW!]
9. fig_multi_hardware        [Multiple hardware tiers - NEW!]

Author: IEEE publication version + Expert Recommendations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import warnings
import glob

warnings.filterwarnings('ignore')


def setup_ieee_style():
    """
    Configure matplotlib for IEEE journal publication standards.
    Matching advisor's style: Helvetica/Arial fonts, uniform 8pt, non-bold.
    """

    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (3.5, 2.625),  # IEEE single column width
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font settings (Helvetica/Arial, sans-serif)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'text.usetex': False,

        # Line and marker settings
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # Axes settings
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'axes.axisbelow': True,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.borderpad': 0.3,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.5,

        # Tick settings
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
    })

    # Color scheme
    colors = {
        'pn': '#8B008B',  # Purple for PN
        'dse': '#FF8C00',  # Orange for DSE
        'blue': '#0072BD',  # Blue
        'orange': '#D95319',  # Orange
        'green': '#77AC30',  # Green
        'red': '#A2142F',  # Red
        'purple': '#7E2F8E',  # Purple
        'yellow': '#EDB120',  # Yellow
        'black': '#000000',  # Black
    }

    return colors


def plot_noise_composition_vs_alpha(df: pd.DataFrame, output_dir: Path, colors: dict):
    """
    ⭐ NEW CRITICAL FIGURE: Noise Composition vs Alpha

    This implements Document 2, Fig. X2: Noise Composition Analysis
    Shows how different noise sources (white, gamma, RSM, PN, DSE) vary with alpha.

    This is the KEY FIGURE for explaining RMSE behavior!
    """

    print(f"\n{'=' * 70}")
    print("FIGURE X2: NOISE COMPOSITION vs ALPHA (NEW - CRITICAL!)")
    print(f"{'=' * 70}")

    # Check if noise component columns exist
    noise_cols = ['noise_white', 'noise_gamma', 'noise_rsm', 'noise_pn', 'noise_dse']

    # Find actual column names (might have different formatting)
    actual_cols = {}
    for search_name in noise_cols:
        for col in df.columns:
            if search_name in col.lower():
                actual_cols[search_name] = col
                break

    if len(actual_cols) < 3:
        print(f"  ⚠ Warning: Insufficient noise component columns")
        print(f"  Available: {df.columns.tolist()}")
        print(f"  Needed: {noise_cols}")
        print(f"  Skipping this plot")
        return False

    print(f"  Found {len(actual_cols)} noise components")

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    alpha_vals = df['alpha'].values

    # Define plot properties
    labels = {
        'noise_white': 'Thermal',
        'noise_gamma': 'HW Distortion',
        'noise_rsm': 'RSM',
        'noise_pn': 'Phase Noise',
        'noise_dse': 'DSE'
    }

    plot_colors = {
        'noise_white': '#1f77b4',  # Blue
        'noise_gamma': '#ff7f0e',  # Orange
        'noise_rsm': '#2ca02c',  # Green
        'noise_pn': '#d62728',  # Red
        'noise_dse': '#9467bd'  # Purple
    }

    markers = {
        'noise_white': 'o',
        'noise_gamma': 's',
        'noise_rsm': '^',
        'noise_pn': 'v',
        'noise_dse': 'D'
    }

    # Plot each noise component
    for key, col in actual_cols.items():
        if col in df.columns:
            # Skip if all zeros or NaN
            if df[col].sum() == 0 or df[col].isnull().all():
                continue

            ax.plot(alpha_vals, df[col].values,
                    label=labels.get(key, key),
                    color=plot_colors.get(key, '#888888'),
                    linewidth=1.5,
                    marker=markers.get(key, 'o'),
                    markersize=4,
                    markevery=2)

    # Set log scale for y-axis (noise PSDs span many orders of magnitude)
    ax.set_yscale('log')

    # Labels
    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax.set_ylabel('Noise PSD (W/Hz)', fontsize=8)

    # Grid
    ax.grid(True, which='both', alpha=0.3, linewidth=0.5)

    # Legend
    ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=1)

    plt.tight_layout()

    # Save both PNG and PDF
    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_noise_composition.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")

    plt.close()
    return True


def plot_multi_hardware_comparison(results_dict: dict, output_dir: Path, colors: dict):
    """
    ⭐ NEW FIGURE: Multi-Hardware Configuration Comparison

    This implements Document 2, Fig. Y1: Three Hardware Tiers Comparison
    Shows how hardware quality affects the Pareto front.
    """

    print(f"\n{'=' * 70}")
    print("FIGURE Y1: MULTI-HARDWARE COMPARISON (NEW)")
    print(f"{'=' * 70}")

    if not results_dict or len(results_dict) < 2:
        print("  ⚠ Need at least 2 hardware profiles for comparison")
        return False

    print(f"  Comparing {len(results_dict)} hardware configurations")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.625))

    # Color map for different profiles
    profile_colors = plt.cm.Set1(np.linspace(0, 1, len(results_dict)))

    for idx, (profile_name, df) in enumerate(results_dict.items()):
        alpha_vals = df['alpha'].values
        R_net = df['R_net_bps_hz'].values
        RMSE_mm = df['RMSE_m'].values * 1000  # Convert to mm

        # Plot 1: R_net vs alpha
        ax1.plot(alpha_vals, R_net,
                 label=profile_name,
                 color=profile_colors[idx],
                 linewidth=1.5, marker='o', markersize=4,
                 markevery=2)

        # Plot 2: RMSE vs alpha (log scale)
        ax2.semilogy(alpha_vals, RMSE_mm,
                     label=profile_name,
                     color=profile_colors[idx],
                     linewidth=1.5, marker='o', markersize=4,
                     markevery=2)

    # Configure Plot 1
    ax1.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax1.set_ylabel(r'$R_{\mathrm{net}}$ (bits/s/Hz)', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, loc='best')

    # Configure Plot 2
    ax2.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax2.set_ylabel('Range RMSE (mm)', fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=7, loc='best')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_multi_hardware.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")

    plt.close()
    return True


def find_alpha_crossover(df: pd.DataFrame) -> tuple:
    """Find the alpha value where PN and DSE curves cross."""
    try:
        alpha_vals = df['alpha'].values
        pn_vals = df['sigma_2_phi_c_res_rad2'].values
        dse_vals = df['sigma_2_DSE_var'].values

        diff = pn_vals - dse_vals

        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:
                alpha_cross = alpha_vals[i] - diff[i] * (alpha_vals[i + 1] - alpha_vals[i]) / (diff[i + 1] - diff[i])
                pn_cross = np.interp(alpha_cross, alpha_vals, pn_vals)
                dse_cross = np.interp(alpha_cross, alpha_vals, dse_vals)
                return alpha_cross, pn_cross, dse_cross

        min_diff_idx = np.argmin(np.abs(diff))
        return alpha_vals[min_diff_idx], pn_vals[min_diff_idx], dse_vals[min_diff_idx]

    except Exception as e:
        return None, None, None



def plot_performance_vs_alpha(df: pd.DataFrame, output_dir: Path, colors: dict):
    """Figure 2: R_net and RMSE vs Alpha (dual Y-axis)"""

    print(f"\n{'=' * 70}")
    print("FIGURE 2: PERFORMANCE vs ALPHA")
    print(f"{'=' * 70}")

    fig, ax1 = plt.subplots()

    # Left Y-axis: R_net
    color1 = colors['blue']
    ax1.plot(df['alpha'], df['R_net_bps_hz'],
             color=color1, marker='o', markersize=4,
             linewidth=1.0, label=r'$R_{\mathrm{net}}$')
    ax1.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax1.set_ylabel('$R_{\mathrm{net}}$ (bits/s/Hz)', color=color1, fontsize=8)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=8)
    ax1.legend(loc='upper left', fontsize=8)

    # Right Y-axis: RMSE
    ax2 = ax1.twinx()
    color2 = colors['red']
    ax2.plot(df['alpha'], df['RMSE_m'] * 1000,
             color=color2, marker='s', markersize=4,
             linewidth=1.0, label='RMSE')
    ax2.set_ylabel('RMSE (mm)', color=color2, fontsize=8)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=8)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=8)

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_performance_vs_alpha.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")

    plt.close()
    return True


def plot_capacity_vs_snr(csv_path: Path, output_dir: Path, colors: dict):
    """Figure 3: Capacity vs SNR"""

    print(f"\n{'=' * 70}")
    print("FIGURE 3: CAPACITY vs SNR")
    print(f"{'=' * 70}")

    if not csv_path.exists():
        print(f"  Error: File not found {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} SNR points")
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return False

    C_sat = df['C_sat'].iloc[0]
    SNR_crit_db = df['SNR_crit_db'].iloc[0]

    fig, ax = plt.subplots()

    # Plot C_J
    ax.plot(df['SNR0_db'], df['C_J_bps_hz'],
            label=r'$C_J$',
            color=colors['blue'],
            linewidth=1.0,
            marker='o',
            markersize=3,
            markevery=4)

    # Plot C_G
    if 'C_G_bps_hz' in df.columns and not df['C_G_bps_hz'].isnull().all():
        ax.plot(df['SNR0_db'], df['C_G_bps_hz'],
                label=r'$C_G$',
                color=colors['red'],
                linestyle='--',
                linewidth=1.0,
                marker='s',
                markersize=3,
                markevery=5)

    # C_sat line
    ax.axhline(y=C_sat, color=colors['green'], linestyle=':',
               linewidth=1.5, label=f'$C_{{\\mathrm{{sat}}}}$ = {C_sat:.2f}')

    # SNR_crit line
    ax.axvline(x=SNR_crit_db, color=colors['purple'], linestyle=':',
               linewidth=1.5, label=f'$\\mathrm{{SNR}}_{{\\mathrm{{crit}}}}$ = {SNR_crit_db:.1f} dB')

    ax.set_xlabel(r'$\mathrm{SNR}_0$ (dB)', fontsize=8)
    ax.set_ylabel('Capacity (bits/s/Hz)', fontsize=8)
    ax.legend(loc='lower right', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_capacity_vs_snr.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")

    plt.close()
    return True


def plot_pareto_front(csv_path: Path, output_dir: Path, colors: dict):
    """Figure 4: ISAC Pareto Front"""

    print(f"\n{'=' * 70}")
    print("FIGURE 4: PARETO FRONT")
    print(f"{'=' * 70}")

    if not csv_path.exists():
        print(f"  Error: File not found {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} data points")
    except Exception as e:
        print(f"  Error loading CSV: {e}")
        return False

    fig, ax = plt.subplots()

    # Scatter plot
    sc = ax.scatter(df['RMSE_m'] * 1000,
                    df['R_net_bps_hz'],
                    c=df['alpha'],
                    cmap='viridis',
                    s=30,
                    alpha=0.85,
                    edgecolors='black',
                    linewidth=0.5)

    # Connecting line
    df_sorted = df.sort_values(by='alpha')
    ax.plot(df_sorted['RMSE_m'] * 1000,
            df_sorted['R_net_bps_hz'],
            'k--',
            alpha=0.35,
            linewidth=0.8)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\alpha$', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel('Range RMSE (mm)', fontsize=8)
    ax.set_ylabel('$R_{\mathrm{net}}$ (bits/s/Hz)', fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_pareto_front.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")

    plt.close()
    return True


def generate_all_visualizations(pareto_csv: str = None, snr_csv: str = None,
                                output_dir: str = None,
                                multi_hardware_dir: str = None):
    """Generate all visualizations including new noise composition figure"""

    print("\n" + "=" * 70)
    print("COMPLETE IEEE PUBLICATION VISUALIZATION SUITE")
    print("Expert-Improved Version with Noise Analysis")
    print("=" * 70)

    # Setup style
    colors = setup_ieee_style()

    # Setup output directory
    if output_dir is None:
        output_dir = 'figures'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Find data files
    if pareto_csv is None:
        search_paths = [
            Path('results/DR08_pareto_results.csv'),
            Path('results/DR08_results_pareto_results.csv'),
            Path('DR08_pareto_results.csv'),
            Path('DR08_results_pareto_results.csv'),
        ]
        for path in search_paths:
            if path.exists():
                pareto_csv = path
                break
    else:
        pareto_csv = Path(pareto_csv)

    if snr_csv is None:
        search_paths = [
            Path('results/DR08_snr_sweep.csv'),
            Path('results/DR08_results_snr_sweep.csv'),
            Path('DR08_snr_sweep.csv'),
            Path('DR08_results_snr_sweep.csv'),
        ]
        for path in search_paths:
            if path.exists():
                snr_csv = path
                break
    else:
        snr_csv = Path(snr_csv)

    success_count = 0
    total_count = 9  # Updated count with new figures

    # Generate all figures from pareto data
    if pareto_csv and pareto_csv.exists():
        df_pareto = pd.read_csv(pareto_csv)

        # Figure 1: PN vs DSE crossover


        # Figure 2: Performance vs alpha
        if plot_performance_vs_alpha(df_pareto, output_dir, colors):
            success_count += 1

        # ⭐ Figure X2: Noise composition (NEW!)
        if plot_noise_composition_vs_alpha(df_pareto, output_dir, colors):
            success_count += 1

        # Figure 4: Pareto front
        if plot_pareto_front(pareto_csv, output_dir, colors):
            success_count += 1
    else:
        print(f"\n  Warning: Pareto CSV not found")
        print(f"    Searched: {pareto_csv}")
        print(f"    Run: python main.py config.yaml")

    # Generate figures from SNR data
    if snr_csv and snr_csv.exists():
        # Figure 3: Capacity vs SNR
        if plot_capacity_vs_snr(snr_csv, output_dir, colors):
            success_count += 1
    else:
        print(f"\n  Warning: SNR CSV not found")
        print(f"    Run: python scan_snr_sweep.py config.yaml")

    # ⭐ Multi-hardware comparison (NEW!)
    if multi_hardware_dir:
        results_dict = {}
        pattern = f"{multi_hardware_dir}/*_pareto_*.csv"
        for csv_file in glob.glob(pattern):
            profile_name = Path(csv_file).stem.split('_pareto_')[-1]
            try:
                results_dict[profile_name] = pd.read_csv(csv_file)
            except Exception as e:
                print(f"  Warning: Could not load {csv_file}: {e}")

        if len(results_dict) >= 2:
            if plot_multi_hardware_comparison(results_dict, output_dir, colors):
                success_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Generated {success_count}/{total_count} figures")
    print(f"\nOutput files in {output_dir.absolute()}:")

    for file in sorted(output_dir.glob('fig_*.png')):
        pdf_file = file.with_suffix('.pdf')
        status = "✓" if pdf_file.exists() else "✗"
        print(f"  {status} {file.name} (+ PDF)")

    if success_count >= 4:  # At least core figures
        print("\n✓ CORE FIGURES GENERATED SUCCESSFULLY")
        return True
    else:
        print(f"\n⚠ Generated only {success_count} figures")
        return False


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='Complete IEEE Publication Visualization (Expert-Improved)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--pareto', type=str, default=None,
                        help='Path to Pareto results CSV')
    parser.add_argument('--snr', type=str, default=None,
                        help='Path to SNR sweep CSV')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Output directory')
    parser.add_argument('--multi-hardware', type=str, default=None,
                        help='Directory containing multi-hardware CSV files')

    args = parser.parse_args()

    success = generate_all_visualizations(
        pareto_csv=args.pareto,
        snr_csv=args.snr,
        output_dir=args.output_dir,
        multi_hardware_dir=args.multi_hardware
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()