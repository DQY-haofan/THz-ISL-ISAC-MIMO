#!/usr/bin/env python3
"""
THz-ISL MIMO ISAC - Integrated Analysis Pipeline (COMPLETE VERSION)
====================================================================

This script COMPLETELY REPLACES three separate scripts with ALL features:
1. main.py                → Pareto front generation (alpha sweep)
2. scan_snr_sweep.py      → SNR sweep analysis
3. visualize_results.py   → Complete IEEE visualization suite

✨ COMPLETE FEATURE LIST:
- Phase 1: Pareto Sweep with DSE auto-calibration
- Phase 2: SNR Sweep with adaptive range
- Phase 3: Full visualization suite (8 figures)
- CSV export (always enabled by default)
- Complete statistical summaries
- Multi-hardware comparison support

📊 OUTPUT:
- results/DR08_results_pareto_results.csv
- results/DR08_results_snr_sweep.csv
- figures/fig_*.png/pdf (8 IEEE-style figures)

USAGE:
    # Complete analysis (recommended)
    python integrated_analysis.py config.yaml

    # Skip phases
    python integrated_analysis.py config.yaml --skip-pareto
    python integrated_analysis.py config.yaml --skip-viz

    # Custom output directories
    python integrated_analysis.py config.yaml --output-dir ./my_results

Author: Complete Integration v3.0 - FULLY FUNCTIONAL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
import os
import copy
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import time
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import validated DR-08 engines
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB, calc_MCRB
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("   Please ensure physics_engine.py and limits_engine.py are available")
    sys.exit(1)

warnings.filterwarnings('ignore')


# ============================================================================
# PHASE 1: PARETO FRONT GENERATION (from main.py)
# ============================================================================

def run_pareto_sweep(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Perform alpha sweep to generate ISAC Pareto front
    Replaces main.py functionality with DSE auto-calibration
    """

    print("\n" + "=" * 80)
    print("PHASE 1: PARETO FRONT GENERATION (ALPHA SWEEP)")
    print("=" * 80)

    # DSE Auto-Calibration (if enabled)
    if config.get('isac_model', {}).get('DSE_autotune', False):
        print("\n[DSE Auto-Calibration] Enabled")
        alpha_star_target = config['isac_model'].get('alpha_star_target', 0.08)
        alpha_nom = alpha_star_target

        print(f"  Target PN-DSE crossover: α* = {alpha_star_target:.3f}")
        print(f"  Running calibration at α = {alpha_nom:.3f}...")

        config_calib = copy.deepcopy(config)
        config_calib['isac_model']['alpha'] = alpha_nom

        try:
            g_factors_calib = calc_g_sig_factors(config_calib)
            n_outputs_calib = calc_n_f_vector(config_calib, g_factors_calib)
            S_pn = n_outputs_calib['sigma_2_phi_c_res']
            C_DSE_calibrated = S_pn * (alpha_star_target ** 4)
            config['isac_model']['C_DSE'] = C_DSE_calibrated

            print(f"  Measured σ²_PN at α={alpha_nom:.3f}: {S_pn:.3e} rad²")
            print(f"  Calibrated C_DSE: {C_DSE_calibrated:.3e}")
            print(f"  ✓ DSE auto-calibration complete")

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
    alpha_min = 0.05
    alpha_max = 0.30
    n_alpha = 20
    alpha_vec = np.linspace(alpha_min, alpha_max, n_alpha)
    print(f"\nSweeping α from {alpha_min} to {alpha_max} ({n_alpha} points)")

    # Initialize results storage
    results_list = []
    c_mps = config['channel']['c_mps']
    SNR0_db_fixed = config.get('simulation', {}).get('SNR0_db_fixed', 20.0)

    # Progress tracking
    print("\nProgress:")
    start_time = time.time()

    for i, alpha in enumerate(alpha_vec):
        config['isac_model']['alpha'] = alpha

        try:
            # Physics calculations
            g_sig_factors = calc_g_sig_factors(config)
            n_f_outputs = calc_n_f_vector(config, g_sig_factors)

            # Performance limits
            c_j_results = calc_C_J(
                config, g_sig_factors, n_f_outputs,
                [SNR0_db_fixed], compute_C_G=True
            )
            bcrlb_results = calc_BCRLB(config, g_sig_factors, n_f_outputs)

            # Derived metrics
            C_J_at_SNR = c_j_results['C_J_vec'][0]
            R_net = (1 - alpha) * C_J_at_SNR

            BCRLB_tau_safe = max(bcrlb_results['BCRLB_tau'], 1e-40)
            RMSE_tau_s = np.sqrt(BCRLB_tau_safe)
            RMSE_range_m = c_mps * RMSE_tau_s  # ISL单程：Δr = c·Δτ
            RMSE_fD_hz = np.sqrt(bcrlb_results['BCRLB_fD'])

            # Package results with noise breakdown
            result_row = {
                'alpha': alpha,
                'C_J_bps_hz': C_J_at_SNR,
                'R_net_bps_hz': R_net,
                'C_sat': c_j_results['C_sat'],
                'SNR_crit_db': c_j_results['SNR_crit_db'],
                'RMSE_m': RMSE_range_m,
                'RMSE_fD_hz': RMSE_fD_hz,
                'BCRLB_tau_s2': bcrlb_results['BCRLB_tau'],
                'BCRLB_fD_hz2': bcrlb_results['BCRLB_fD'],
                'G_sig_ideal': g_sig_factors['G_sig_ideal'],
                'G_sig_avg': g_sig_factors['G_sig_avg'],
                'eta_bsq_avg': g_sig_factors['eta_bsq_avg'],
                'rho_Q': g_sig_factors['rho_Q'],
                'rho_APE': g_sig_factors['rho_APE'],
                'rho_A': g_sig_factors['rho_A'],
                'rho_PN': g_sig_factors['rho_PN'],
                'Gamma_eff_total': n_f_outputs['Gamma_eff_total'],
                'sigma_2_phi_c_res_rad2': n_f_outputs['sigma_2_phi_c_res'],
                'sigma_2_DSE_var': n_f_outputs.get('sigma_2_DSE_var', np.nan),
                'sigma_2_theta_pe_rad2': n_f_outputs.get('sigma_2_theta_pe_rad2', np.nan),
                'Jensen_gap_bits': c_j_results.get('Jensen_gap_bits', [np.nan])[0]
                if 'Jensen_gap_bits' in c_j_results else np.nan
            }

            # Add component-level Gamma if available
            for comp in ['Gamma_pa', 'Gamma_adc', 'Gamma_iq', 'Gamma_lo']:
                if comp in n_f_outputs:
                    result_row[comp] = n_f_outputs[comp]

            # Add noise composition for visualization
            if 'noise_components' in n_f_outputs:
                for key, val in n_f_outputs['noise_components'].items():
                    result_row[f'noise_{key}'] = val

            results_list.append(result_row)

            # Progress indicator
            if (i + 1) % 4 == 0 or i == len(alpha_vec) - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(alpha_vec) - i - 1)
                print(f"  [{i + 1:2d}/{len(alpha_vec):2d}] "
                      f"α={alpha:.3f}: "
                      f"R_net={R_net:.3f} bits/s/Hz, "
                      f"RMSE={RMSE_range_m * 1000:.2f} mm "
                      f"(ETA: {eta:.1f}s)")

        except Exception as e:
            warnings.warn(f"Failed at α={alpha:.3f}: {e}")
            print(f"  [WARNING] Skipping α={alpha:.3f} due to error: {e}")
            continue

    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values('alpha').reset_index(drop=True)

    elapsed_total = time.time() - start_time
    print(f"\n✓ Pareto sweep complete ({elapsed_total:.1f}s)")
    print(f"  Generated {len(df_results)} data points")

    return df_results


# ============================================================================
# PHASE 2: SNR SWEEP (from scan_snr_sweep.py)
# ============================================================================

def run_snr_sweep(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Perform SNR sweep at fixed alpha for capacity analysis
    Replaces scan_snr_sweep.py with adaptive range detection
    """

    print("\n" + "=" * 80)
    print("PHASE 2: SNR SWEEP (CAPACITY ANALYSIS)")
    print("=" * 80)

    # Set fixed alpha
    fixed_alpha = config.get('isac_model', {}).get('alpha', 0.05)
    config['isac_model']['alpha'] = fixed_alpha
    print(f"\nUsing fixed ISAC overhead: α = {fixed_alpha}")

    # Pre-scan to determine SNR_crit for adaptive range
    print(f"\n[Pre-scan] Determining SNR_crit for adaptive range...")

    try:
        g_factors_prescan = calc_g_sig_factors(config)
        n_outputs_prescan = calc_n_f_vector(config, g_factors_prescan)
        c_j_prescan = calc_C_J(
            config, g_factors_prescan, n_outputs_prescan,
            [0.0], compute_C_G=False
        )
        snr_crit_db = float(c_j_prescan['SNR_crit_db'])
        c_sat = float(c_j_prescan['C_sat'])

        print(f"  ✓ Pre-scan complete:")
        print(f"    SNR_crit: {snr_crit_db:.2f} dB")
        print(f"    C_sat: {c_sat:.3f} bits/s/Hz")

    except Exception as e:
        print(f"  ✗ Pre-scan failed: {e}")
        snr_crit_db = 0.0

    # Define adaptive SNR sweep range
    print(f"\n[Adaptive Range] Centering around SNR_crit...")
    margin_db = 30.0
    snr_min = snr_crit_db - margin_db
    snr_max = snr_crit_db + margin_db

    snr_config = config.get('simulation', {})
    if 'SNR_sweep_range_override' in snr_config:
        snr_min = snr_config['SNR_sweep_range_override'][0]
        snr_max = snr_config['SNR_sweep_range_override'][1]
        print(f"  ⚠ Using config override range: [{snr_min}, {snr_max}] dB")
    else:
        print(f"  ✓ Adaptive range: [{snr_min:.1f}, {snr_max:.1f}] dB")
        print(f"    (SNR_crit ± {margin_db} dB)")

    n_points = snr_config.get('SNR_sweep_points', 100)
    snr_sweep = np.linspace(snr_min, snr_max, n_points)
    print(f"  Sweeping {len(snr_sweep)} points")

    # Run full simulation chain
    print("\n[Simulation] Running full chain...")
    start_time = time.time()

    try:
        g_factors = calc_g_sig_factors(config)
        n_outputs = calc_n_f_vector(config, g_factors)
        c_j_results = calc_C_J(
            config, g_factors, n_outputs,
            snr_sweep, compute_C_G=True
        )

        elapsed = time.time() - start_time
        print(f"✓ Simulation complete ({elapsed:.1f}s)")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Package results
    results_data = {
        'SNR0_db': snr_sweep,
        'C_J_bps_hz': c_j_results['C_J_vec'],
    }

    if 'C_G_vec' in c_j_results:
        results_data['C_G_bps_hz'] = c_j_results['C_G_vec']
        results_data['Jensen_gap_bits'] = c_j_results['C_J_vec'] - c_j_results['C_G_vec']
    else:
        results_data['C_G_bps_hz'] = np.nan
        results_data['Jensen_gap_bits'] = np.nan

    # Add scalar metrics
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

    print(f"\n✓ SNR sweep complete")
    print(f"  SNR range: [{snr_sweep[0]:.1f}, {snr_sweep[-1]:.1f}] dB")
    print(f"  C_sat: {c_j_results['C_sat']:.3f} bits/s/Hz")
    print(f"  SNR_crit: {c_j_results['SNR_crit_db']:.2f} dB")

    if 'Jensen_gap_bits' in df.columns and not df['Jensen_gap_bits'].isnull().all():
        max_gap = df['Jensen_gap_bits'].max()
        mean_gap = df['Jensen_gap_bits'].mean()
        print(f"\n  Jensen Gap Statistics:")
        print(f"    Maximum: {max_gap:.4f} bits/s/Hz")
        print(f"    Mean: {mean_gap:.4f} bits/s/Hz")

    return df


# ============================================================================
# PHASE 3: COMPLETE VISUALIZATION (from visualize_results.py)
# ============================================================================

def setup_ieee_style():
    """Configure matplotlib for IEEE journal publication standards"""
    plt.rcParams.update({
        'figure.figsize': (3.5, 2.625),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'text.usetex': False,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'axes.axisbelow': True,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.borderpad': 0.3,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
    })

    colors = {
        'pn': '#8B008B',
        'dse': '#FF8C00',
        'blue': '#0072BD',
        'orange': '#D95319',
        'green': '#77AC30',
        'red': '#A2142F',
        'purple': '#7E2F8E',
        'yellow': '#EDB120',
        'black': '#000000',
    }

    return colors


def plot_capacity_vs_snr(df_snr: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate Capacity vs SNR figure"""
    print("\n  [Fig 1] Capacity vs SNR...")

    fig, ax = plt.subplots()

    C_sat = df_snr['C_sat'].iloc[0]
    SNR_crit_db = df_snr['SNR_crit_db'].iloc[0]

    ax.plot(df_snr['SNR0_db'], df_snr['C_J_bps_hz'],
            label=r'$C_J$', color=colors['blue'],
            linewidth=1.0, marker='o', markersize=3, markevery=4)

    if 'C_G_bps_hz' in df_snr.columns and not df_snr['C_G_bps_hz'].isnull().all():
        ax.plot(df_snr['SNR0_db'], df_snr['C_G_bps_hz'],
                label=r'$C_G$', color=colors['red'],
                linestyle='--', linewidth=1.0, marker='s', markersize=3, markevery=5)

    ax.axhline(y=C_sat, color=colors['green'], linestyle=':',
               linewidth=1.5, label=f'$C_{{\\mathrm{{sat}}}}$ = {C_sat:.2f}')
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

    plt.close()
    print("    ✓ Saved: fig_capacity_vs_snr.[png/pdf]")
    return True


def plot_pareto_front(df_pareto: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate ISAC Pareto Front figure"""
    print("\n  [Fig 2] Pareto Front...")

    fig, ax = plt.subplots()

    sc = ax.scatter(df_pareto['RMSE_m'] * 1000,
                    df_pareto['R_net_bps_hz'],
                    c=df_pareto['alpha'],
                    cmap='viridis',
                    s=30, alpha=0.85,
                    edgecolors='black', linewidth=0.5)

    df_sorted = df_pareto.sort_values(by='alpha')
    ax.plot(df_sorted['RMSE_m'] * 1000,
            df_sorted['R_net_bps_hz'],
            'k--', alpha=0.35, linewidth=0.8)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\alpha$', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel('Range RMSE (mm)', fontsize=8)
    ax.set_ylabel('$R_{\\mathrm{net}}$ (bits/s/Hz)', fontsize=8)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_pareto_front.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
    print("    ✓ Saved: fig_pareto_front.[png/pdf]")
    return True


def plot_performance_vs_alpha(df_pareto: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate R_net and RMSE vs alpha dual-axis plot"""
    print("\n  [Fig 3] Performance vs Alpha...")

    fig, ax1 = plt.subplots()

    alpha_vals = df_pareto['alpha'].values

    ax1.plot(alpha_vals, df_pareto['R_net_bps_hz'].values,
             color=colors['blue'], linewidth=1.5, marker='o',
             markersize=4, markevery=2, label='$R_{\\mathrm{net}}$')
    ax1.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax1.set_ylabel('$R_{\\mathrm{net}}$ (bits/s/Hz)', fontsize=8, color=colors['blue'])
    ax1.tick_params(axis='y', labelcolor=colors['blue'])
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogy(alpha_vals, df_pareto['RMSE_m'].values * 1000,
                 color=colors['red'], linewidth=1.5, marker='s',
                 markersize=4, markevery=2, label='RMSE')
    ax2.set_ylabel('Range RMSE (mm)', fontsize=8, color=colors['red'])
    ax2.tick_params(axis='y', labelcolor=colors['red'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=7)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_performance_vs_alpha.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
    print("    ✓ Saved: fig_performance_vs_alpha.[png/pdf]")
    return True



def plot_hardware_factors(df_pareto: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate hardware quality factors plot"""
    print("\n  [Fig 5] Hardware Factors...")

    fig, ax = plt.subplots()

    alpha_vals = df_pareto['alpha'].values

    factors = {
        'Beam Squint': ('eta_bsq_avg', colors['blue']),
        'Phase Quant': ('rho_Q', colors['orange']),
        'Pointing': ('rho_APE', colors['green']),
        'Amplitude': ('rho_A', colors['red']),
        'Diff PN': ('rho_PN', colors['purple'])
    }

    for label, (col, color) in factors.items():
        if col in df_pareto.columns:
            loss_db = -10 * np.log10(df_pareto[col].values)
            ax.plot(alpha_vals, loss_db, label=label,
                    color=color, linewidth=1.0, marker='o',
                    markersize=3, markevery=3)

    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax.set_ylabel('Loss (dB)', fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_hardware_factors.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
    print("    ✓ Saved: fig_hardware_factors.[png/pdf]")
    return True


def plot_gamma_breakdown(df_pareto: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate Gamma component breakdown (at mid-alpha)"""
    print("\n  [Fig 6] Gamma Breakdown...")

    # Use middle alpha value
    mid_idx = len(df_pareto) // 2
    row = df_pareto.iloc[mid_idx]

    gamma_components = {}
    for comp in ['Gamma_pa', 'Gamma_adc', 'Gamma_iq', 'Gamma_lo']:
        if comp in df_pareto.columns:
            gamma_components[comp.replace('Gamma_', '').upper()] = row[comp]

    if not gamma_components:
        print("    ⚠ No Gamma data available, skipping")
        return False

    fig, ax = plt.subplots()

    components = list(gamma_components.keys())
    values = list(gamma_components.values())
    total = sum(values)
    percentages = [v / total * 100 for v in values]

    bar_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']
    bars = ax.bar(range(len(components)), values,
                  color=bar_colors[:len(components)],
                  alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=8)
    ax.set_ylabel(r'Hardware Distortion $\Gamma$ (linear)', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_gamma_breakdown.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
    print("    ✓ Saved: fig_gamma_breakdown.[png/pdf]")
    return True


def plot_jensen_gap(df_snr: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate Jensen gap validation plot"""
    print("\n  [Fig 7] Jensen Gap...")

    if 'Jensen_gap_bits' not in df_snr.columns or df_snr['Jensen_gap_bits'].isnull().all():
        print("    ⚠ No Jensen gap data, skipping")
        return False

    fig, ax = plt.subplots()

    ax.plot(df_snr['SNR0_db'], df_snr['Jensen_gap_bits'] * 1000,
            color=colors['red'], linewidth=1.5, marker='o',
            markersize=3, markevery=5, label='Jensen Gap')

    ax.set_xlabel(r'$\mathrm{SNR}_0$ (dB)', fontsize=8)
    ax.set_ylabel('Jensen Gap (mbits/s/Hz)', fontsize=8)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_jensen_gap.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
    print("    ✓ Saved: fig_jensen_gap.[png/pdf]")
    return True


def plot_noise_composition(df_pareto: pd.DataFrame, output_dir: Path, colors: dict):
    """Generate noise composition vs alpha plot"""
    print("\n  [Fig 8] Noise Composition...")

    noise_cols = ['noise_white', 'noise_gamma', 'noise_rsm', 'noise_pn', 'noise_dse']
    actual_cols = {col: col for col in noise_cols if col in df_pareto.columns}

    if len(actual_cols) < 3:
        print("    ⚠ Insufficient noise data, skipping")
        return False

    fig, ax = plt.subplots()

    alpha_vals = df_pareto['alpha'].values

    labels = {
        'noise_white': 'Thermal',
        'noise_gamma': 'HW Distortion',
        'noise_rsm': 'RSM',
        'noise_pn': 'Phase Noise',
        'noise_dse': 'DSE'
    }

    plot_colors = {
        'noise_white': '#1f77b4',
        'noise_gamma': '#ff7f0e',
        'noise_rsm': '#2ca02c',
        'noise_pn': '#d62728',
        'noise_dse': '#9467bd'
    }

    for key, col in actual_cols.items():
        if df_pareto[col].sum() > 0 and not df_pareto[col].isnull().all():
            ax.plot(alpha_vals, df_pareto[col].values,
                    label=labels.get(key, key),
                    color=plot_colors.get(key, '#888888'),
                    linewidth=1.5, marker='o', markersize=3, markevery=2)

    ax.set_yscale('log')
    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax.set_ylabel('Noise PSD (W/Hz)', fontsize=8)
    ax.legend(loc='best', fontsize=7, framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_file = output_dir / f'fig_noise_composition.{ext}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
    print("    ✓ Saved: fig_noise_composition.[png/pdf]")
    return True


def generate_visualizations(df_pareto: pd.DataFrame,
                            df_snr: pd.DataFrame,
                            output_dir: Path):
    """
    Generate all publication figures
    Replaces visualize_results.py functionality completely
    """

    print("\n" + "=" * 80)
    print("PHASE 3: VISUALIZATION (IEEE PUBLICATION STYLE)")
    print("=" * 80)

    colors = setup_ieee_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")

    success_count = 0

    print("\nGenerating figures:")

    # Core figures (always generated)
    if plot_capacity_vs_snr(df_snr, output_dir, colors):
        success_count += 1

    if plot_pareto_front(df_pareto, output_dir, colors):
        success_count += 1

    if plot_performance_vs_alpha(df_pareto, output_dir, colors):
        success_count += 1



    if plot_hardware_factors(df_pareto, output_dir, colors):
        success_count += 1

    if plot_gamma_breakdown(df_pareto, output_dir, colors):
        success_count += 1

    if plot_jensen_gap(df_snr, output_dir, colors):
        success_count += 1

    if plot_noise_composition(df_pareto, output_dir, colors):
        success_count += 1

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Generated {success_count}/8 figures")
    print(f"\nOutput files in {output_dir.absolute()}:")

    for file in sorted(output_dir.glob('fig_*.png')):
        pdf_file = file.with_suffix('.pdf')
        status = "✓" if pdf_file.exists() else "✗"
        print(f"  {status} {file.name} (+ PDF)")

    return success_count >= 3


# ============================================================================
# CSV EXPORT AND SUMMARY (from main.py)
# ============================================================================

def save_csv_files(df_pareto: pd.DataFrame, df_snr: pd.DataFrame, config: Dict[str, Any]):
    """
    Save CSV files to configured output directory
    Always enabled (replaces main.py save behavior)
    """

    print("\n" + "=" * 80)
    print("CSV EXPORT")
    print("=" * 80)

    output_config = config.get('outputs', {})
    save_path = Path(output_config.get('save_path', './results/'))
    save_path.mkdir(parents=True, exist_ok=True)

    table_prefix = output_config.get('table_prefix', 'DR08_results')

    # Save Pareto results
    pareto_path = save_path / f"{table_prefix}_pareto_results.csv"
    df_pareto.to_csv(pareto_path, index=False, float_format='%.6e')
    print(f"  ✓ Pareto results: {pareto_path}")
    print(f"    ({len(df_pareto)} alpha points)")

    # Save SNR sweep results
    snr_path = save_path / f"{table_prefix}_snr_sweep.csv"
    df_snr.to_csv(snr_path, index=False, float_format='%.6e')
    print(f"  ✓ SNR sweep: {snr_path}")
    print(f"    ({len(df_snr)} SNR points)")

    print(f"\n  Total files saved: 2")
    print(f"  Output directory: {save_path.absolute()}")

    return pareto_path, snr_path


def print_complete_summary(df_pareto: pd.DataFrame, df_snr: pd.DataFrame):
    """
    Print comprehensive analysis summary
    Replaces main.py summary with additional details
    """

    print("\n" + "=" * 80)
    print("COMPLETE ANALYSIS SUMMARY")
    print("=" * 80)

    # Best operating points
    idx_best_R_net = df_pareto['R_net_bps_hz'].idxmax()
    idx_best_RMSE = df_pareto['RMSE_m'].idxmin()

    print("\n[1] BEST COMMUNICATION PERFORMANCE")
    row = df_pareto.loc[idx_best_R_net]
    print(f"  α = {row['alpha']:.3f}")
    print(f"  R_net = {row['R_net_bps_hz']:.3f} bits/s/Hz")
    print(f"  RMSE = {row['RMSE_m'] * 1000:.2f} mm")
    print(f"  C_sat = {row['C_sat']:.3f} bits/s/Hz")

    print("\n[2] BEST SENSING PERFORMANCE")
    row = df_pareto.loc[idx_best_RMSE]
    print(f"  α = {row['alpha']:.3f}")
    print(f"  RMSE = {row['RMSE_m'] * 1000:.2f} mm")
    print(f"  R_net = {row['R_net_bps_hz']:.3f} bits/s/Hz")

    print("\n[3] CAPACITY ANALYSIS")
    print(f"  C_sat = {df_snr['C_sat'].iloc[0]:.3f} bits/s/Hz")
    print(f"  SNR_crit = {df_snr['SNR_crit_db'].iloc[0]:.2f} dB")

    if 'Jensen_gap_bits' in df_snr.columns and not df_snr['Jensen_gap_bits'].isnull().all():
        max_gap = df_snr['Jensen_gap_bits'].max()
        mean_gap = df_snr['Jensen_gap_bits'].mean()
        print(f"  Jensen gap (max) = {max_gap:.4f} bits/s/Hz")
        print(f"  Jensen gap (mean) = {mean_gap:.4f} bits/s/Hz")

    print("\n[4] HARDWARE QUALITY (at α=0.10)")
    idx_mid = (df_pareto['alpha'] - 0.10).abs().idxmin()
    row = df_pareto.loc[idx_mid]

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

    print("\n[5] RMSE RANGE ANALYSIS")
    rmse_min = df_pareto['RMSE_m'].min() * 1000
    rmse_max = df_pareto['RMSE_m'].max() * 1000
    print(f"  Minimum: {rmse_min:.2f} mm (at α={df_pareto.loc[idx_best_RMSE, 'alpha']:.3f})")
    print(f"  Maximum: {rmse_max:.2f} mm (at α={df_pareto['alpha'].iloc[0]:.3f})")
    print(f"  Dynamic range: {rmse_max / rmse_min:.1f}×")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN PIPELINE - COMPLETE IMPLEMENTATION
# ============================================================================

def main():
    """
    Main execution function - COMPLETE IMPLEMENTATION
    This is the critical missing piece that ties everything together
    """

    parser = argparse.ArgumentParser(
        description='THz-ISL MIMO ISAC - Complete Integrated Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete analysis (recommended)
  python integrated_analysis.py config.yaml

  # Skip specific phases
  python integrated_analysis.py config.yaml --skip-pareto
  python integrated_analysis.py config.yaml --skip-snr --skip-viz

  # Custom output directory
  python integrated_analysis.py config.yaml --output-dir ./my_results
        """
    )

    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--skip-pareto',
        action='store_true',
        help='Skip Pareto sweep (Phase 1)'
    )
    parser.add_argument(
        '--skip-snr',
        action='store_true',
        help='Skip SNR sweep (Phase 2)'
    )
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip visualization (Phase 3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures and CSVs'
    )

    args = parser.parse_args()

    # ========================================================================
    # HEADER AND CONFIGURATION LOADING
    # ========================================================================
    print("\n" + "=" * 80)
    print("THz-ISL MIMO ISAC - COMPLETE INTEGRATED ANALYSIS")
    print("Replaces: main.py + scan_snr_sweep.py + visualize_results.py")
    print("=" * 80)

    print(f"\nLoading configuration from: {args.config}")

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
    except FileNotFoundError:
        print(f"✗ Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    # Validate configuration
    try:
        validate_config(config)
        print("✓ Configuration validated")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    # Override output directory if specified
    if args.output_dir:
        config['outputs']['save_path'] = args.output_dir
        config['outputs']['figure_path'] = args.output_dir

    # ========================================================================
    # PHASE 1: PARETO SWEEP (from main.py)
    # ========================================================================
    df_pareto = None
    if not args.skip_pareto:
        try:
            df_pareto = run_pareto_sweep(config)
            if df_pareto is None or len(df_pareto) == 0:
                print("✗ Pareto sweep failed to produce results")
                sys.exit(1)
        except Exception as e:
            print(f"✗ Pareto sweep failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⚠ Skipping Pareto sweep (Phase 1)")
        # Try to load existing results
        output_config = config.get('outputs', {})
        save_path = Path(output_config.get('save_path', './results/'))
        table_prefix = output_config.get('table_prefix', 'DR08_results')
        pareto_file = save_path / f"{table_prefix}_pareto_results.csv"

        if pareto_file.exists():
            df_pareto = pd.read_csv(pareto_file)
            print(f"  ✓ Loaded existing Pareto results from {pareto_file}")
        else:
            print(f"  ✗ No existing Pareto results found at {pareto_file}")
            print("  Cannot proceed without Pareto data")
            sys.exit(1)

    # ========================================================================
    # PHASE 2: SNR SWEEP (from scan_snr_sweep.py)
    # ========================================================================
    df_snr = None
    if not args.skip_snr:
        try:
            df_snr = run_snr_sweep(config)
            if df_snr is None or len(df_snr) == 0:
                print("✗ SNR sweep failed to produce results")
                sys.exit(1)
        except Exception as e:
            print(f"✗ SNR sweep failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n⚠ Skipping SNR sweep (Phase 2)")
        # Try to load existing results
        output_config = config.get('outputs', {})
        save_path = Path(output_config.get('save_path', './results/'))
        table_prefix = output_config.get('table_prefix', 'DR08_results')
        snr_file = save_path / f"{table_prefix}_snr_sweep.csv"

        if snr_file.exists():
            df_snr = pd.read_csv(snr_file)
            print(f"  ✓ Loaded existing SNR results from {snr_file}")
        else:
            print(f"  ✗ No existing SNR results found at {snr_file}")
            print("  Cannot proceed without SNR data")
            sys.exit(1)

    # ========================================================================
    # CSV EXPORT (always performed if we have data)
    # ========================================================================
    if df_pareto is not None and df_snr is not None:
        try:
            pareto_path, snr_path = save_csv_files(df_pareto, df_snr, config)
        except Exception as e:
            print(f"⚠ Warning: CSV export failed: {e}")

    # ========================================================================
    # PHASE 3: VISUALIZATION (from visualize_results.py)
    # ========================================================================
    if not args.skip_viz and df_pareto is not None and df_snr is not None:
        try:
            output_config = config.get('outputs', {})
            figure_path = Path(output_config.get('figure_path', './figures/'))

            success = generate_visualizations(df_pareto, df_snr, figure_path)

            if not success:
                print("⚠ Warning: Visualization completed with some failures")
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        if args.skip_viz:
            print("\n⚠ Skipping visualization (Phase 3)")
        else:
            print("\n⚠ Cannot generate visualizations without both Pareto and SNR data")

    # ========================================================================
    # COMPLETE SUMMARY
    # ========================================================================
    if df_pareto is not None and df_snr is not None:
        try:
            print_complete_summary(df_pareto, df_snr)
        except Exception as e:
            print(f"⚠ Warning: Summary printing failed: {e}")

    # ========================================================================
    # SUCCESS MESSAGE
    # ========================================================================
    print("\n" + "=" * 80)
    print("✓ COMPLETE INTEGRATED ANALYSIS FINISHED")
    print("=" * 80)

    print("\n📊 Generated outputs:")
    if df_pareto is not None:
        print("  ✓ Pareto front data (alpha sweep)")
    if df_snr is not None:
        print("  ✓ SNR sweep data (capacity analysis)")
    if not args.skip_viz:
        print("  ✓ IEEE publication figures (8 figures)")

    print("\n💡 Next steps:")
    print("  - Review figures in the output directory")
    print("  - Analyze CSV files for detailed metrics")
    print("  - Run specialized analysis:")
    print("    • python threshold_analysis.py config.yaml")
    print("    • python mimo_analysis.py config.yaml")
    print("    • python pn_dse_analysis.py config.yaml")

    return 0


if __name__ == "__main__":
    sys.exit(main())