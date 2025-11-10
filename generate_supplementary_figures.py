#!/usr/bin/env python3
"""
Supplementary Figure Generation Script
IEEE Publication Style - Matching Main Visualization Suite

This script generates supplementary figures with the same style as the main figures:
- IEEE journal publication standards
- Helvetica/Arial fonts, uniform 8pt, non-bold
- No titles, only axis labels
- Single plots per figure

Generated Figures:
1. fig_gamma_breakdown       - Hardware distortion breakdown
2. fig_ablation_alpha_policy - Alpha policy ablation study

Usage:
    python generate_supplementary_figures.py [config.yaml]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import copy
import sys
import os
from pathlib import Path

try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def setup_ieee_style():
    """
    Configure matplotlib for IEEE journal publication standards.
    Matching main visualization style: Helvetica/Arial fonts, uniform 8pt, non-bold.
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


def generate_figure_gamma_breakdown(config, output_dir='figures'):
    """
    Gamma Breakdown Figure
    Shows hardware distortion components with magnified inset view
    """
    print("\n" + "=" * 80)
    print("FIGURE: Gamma Breakdown")
    print("=" * 80)

    # Calculate Gamma components
    cfg = copy.deepcopy(config)
    cfg['isac_model']['alpha'] = 0.1

    g_factors = calc_g_sig_factors(cfg)
    n_outputs = calc_n_f_vector(cfg, g_factors)

    # Extract components
    gamma_components = {
        'PA': n_outputs.get('Gamma_pa', 0),
        'ADC': n_outputs.get('Gamma_adc', 0),
        'IQ': n_outputs.get('Gamma_iq', 0),
        'LO': n_outputs.get('Gamma_lo', 0)
    }

    total_gamma = sum(gamma_components.values())
    gamma_percentages = {k: v / total_gamma * 100 for k, v in gamma_components.items()}

    print(f"  Gamma breakdown:")
    for comp, val in gamma_components.items():
        pct = gamma_percentages[comp]
        print(f"    {comp}: {val:.3e} ({pct:.2f}%)")

    # Create figure with inset
    fig = plt.figure(figsize=(3.5, 2.625))
    ax_main = plt.subplot(111)

    components = list(gamma_components.keys())
    values = [gamma_components[c] for c in components]
    percentages = [gamma_percentages[c] for c in components]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']
    bars = ax_main.bar(range(len(components)), values, color=colors,
                       alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add percentage labels
    for i, (bar, pct, val) in enumerate(zip(bars, percentages, values)):
        height = bar.get_height()

        if pct > 10:  # Large components
            ax_main.text(bar.get_x() + bar.get_width() / 2., height / 2,
                         f'{pct:.1f}%',
                         ha='center', va='center',
                         fontsize=8, color='white')
        else:  # Small components
            ax_main.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{pct:.2f}%',
                         ha='center', va='bottom',
                         fontsize=7)

        # Absolute value labels
        ax_main.text(bar.get_x() + bar.get_width() / 2., 0,
                     f'{val:.2e}',
                     ha='center', va='bottom',
                     fontsize=6, rotation=0)

    ax_main.set_xticks(range(len(components)))
    ax_main.set_xticklabels(components, fontsize=8)
    ax_main.set_ylabel(r'Hardware Distortion $\Gamma$ (linear)', fontsize=8)
    ax_main.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

    # Magnified inset for small components
    small_components = ['ADC', 'LO']
    small_indices = [components.index(c) for c in small_components if c in components]

    if len(small_indices) >= 2:
        ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.35])

        small_values = [values[i] for i in small_indices]
        small_colors = [colors[i] for i in small_indices]
        small_labels = [components[i] for i in small_indices]
        small_pcts = [percentages[i] for i in small_indices]

        bars_inset = ax_inset.bar(range(len(small_labels)), small_values,
                                  color=small_colors, alpha=0.8,
                                  edgecolor='black', linewidth=0.5)

        for bar, pct, val in zip(bars_inset, small_pcts, small_values):
            height = bar.get_height()
            ax_inset.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{pct:.3f}%\n{val:.2e}',
                          ha='center', va='bottom',
                          fontsize=7)

        ax_inset.set_xticks(range(len(small_labels)))
        ax_inset.set_xticklabels(small_labels, fontsize=8)
        ax_inset.set_ylabel(r'$\Gamma$ (linear)', fontsize=7)
        ax_inset.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ['png', 'pdf']:
        output_path = output_dir / f'fig_gamma_breakdown.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")

    plt.close()
    return True


def generate_figure_alpha_policy_rnet(config, output_dir='figures'):
    """
    Alpha Policy Ablation - R_net Plot (Single Figure)
    """
    print("\n" + "=" * 80)
    print("FIGURE: Alpha Policy - Communication Performance")
    print("=" * 80)

    alpha_vec = np.linspace(0.01, 0.50, 30)

    results = {
        'CONST_POWER': {'R_net': [], 'alpha': []},
        'CONST_ENERGY': {'R_net': [], 'alpha': []}
    }

    for policy in ['CONST_POWER', 'CONST_ENERGY']:
        print(f"\n[{policy}] Running alpha sweep...")

        for i, alpha in enumerate(alpha_vec):
            try:
                cfg = copy.deepcopy(config)
                cfg['isac_model']['alpha'] = alpha
                cfg['isac_model']['alpha_model'] = policy

                g_factors = calc_g_sig_factors(cfg)
                n_outputs = calc_n_f_vector(cfg, g_factors)

                # 在读取 SNR0 固定值的位置加兜底  :contentReference[oaicite:12]{index=12}
                snr0_db_fixed = cfg['simulation'].get('SNR0_db_fixed')
                if snr0_db_fixed is None:
                    vec = cfg['simulation'].get('SNR0_db_vec', [0.0])
                    snr0_db_fixed = float(vec[0])

                c_j_results = calc_C_J(cfg, g_factors, n_outputs, [snr0_db_fixed], compute_C_G=False)

                # FIX: Use actual capacity at given SNR, not saturation capacity
                R_net = (1 - alpha) * c_j_results['C_J_vec'][0]

                results[policy]['alpha'].append(alpha)
                results[policy]['R_net'].append(R_net)

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(alpha_vec)}")

            except Exception as e:
                print(f"  Warning: Failed at α={alpha:.3f}: {e}")
                continue

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    for policy, color, marker in [('CONST_POWER', '#0072BD', 'o'),
                                  ('CONST_ENERGY', '#D95319', 's')]:
        data = results[policy]
        if len(data['alpha']) > 0:
            ax.plot(data['alpha'], data['R_net'], marker=marker, markersize=4,
                    linewidth=1.0, label=policy, color=color, alpha=0.8,
                    markevery=3)

            # Mark optimal point
            idx_max = np.argmax(data['R_net'])
            ax.plot(data['alpha'][idx_max], data['R_net'][idx_max],
                    '*', markersize=12, color=color,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax.set_ylabel(r'Net Rate $R_{\mathrm{net}}$ (bits/s/Hz)', fontsize=8)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim([0, 0.5])

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ['png', 'pdf']:
        output_path = output_dir / f'fig_ablation_alpha_rnet.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")

    plt.close()
    return True


def generate_figure_alpha_policy_rmse(config, output_dir='figures'):
    """
    Alpha Policy Ablation - RMSE Plot (Single Figure)
    """
    print("\n" + "=" * 80)
    print("FIGURE: Alpha Policy - Sensing Performance")
    print("=" * 80)

    alpha_vec = np.linspace(0.01, 0.50, 30)

    results = {
        'CONST_POWER': {'RMSE': [], 'alpha': []},
        'CONST_ENERGY': {'RMSE': [], 'alpha': []}
    }

    for policy in ['CONST_POWER', 'CONST_ENERGY']:
        print(f"\n[{policy}] Running alpha sweep...")

        for i, alpha in enumerate(alpha_vec):
            try:
                cfg = copy.deepcopy(config)
                cfg['isac_model']['alpha'] = alpha
                cfg['isac_model']['alpha_model'] = policy

                g_factors = calc_g_sig_factors(cfg)
                n_outputs = calc_n_f_vector(cfg, g_factors)

                bcrlb_results = calc_BCRLB(cfg, g_factors, n_outputs)
                RMSE = np.sqrt(bcrlb_results['BCRLB_tau']) * (cfg['channel']['c_mps'] / 2.0)

                results[policy]['alpha'].append(alpha)
                results[policy]['RMSE'].append(RMSE * 1000)  # to mm

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(alpha_vec)}")

            except Exception as e:
                print(f"  Warning: Failed at α={alpha:.3f}: {e}")
                continue

    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    for policy, color, marker in [('CONST_POWER', '#0072BD', 'o'),
                                  ('CONST_ENERGY', '#D95319', 's')]:
        data = results[policy]
        if len(data['alpha']) > 0:
            ax.semilogy(data['alpha'], data['RMSE'], marker=marker, markersize=4,
                        linewidth=1.0, label=policy, color=color, alpha=0.8,
                        markevery=3)

            # Mark optimal point (minimum RMSE)
            idx_min = np.argmin(data['RMSE'])
            ax.plot(data['alpha'][idx_min], data['RMSE'][idx_min],
                    '*', markersize=12, color=color,
                    markeredgecolor='black', markeredgewidth=0.5)

    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax.set_ylabel(r'Range RMSE (mm, log scale)', fontsize=8)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    ax.set_xlim([0, 0.5])

    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ext in ['png', 'pdf']:
        output_path = output_dir / f'fig_ablation_alpha_rmse.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")

    # Save data
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    for policy in ['CONST_POWER', 'CONST_ENERGY']:
        df = pd.DataFrame(results[policy])
        csv_path = results_dir / f'fig_ablation_{policy}_data.csv'
        df.to_csv(csv_path, index=False)

    plt.close()
    return True


def main():
    """Main function"""

    print("=" * 80)
    print("SUPPLEMENTARY FIGURES GENERATOR")
    print("IEEE Publication Style - Matching Main Visualization Suite")
    print("=" * 80)

    # Setup IEEE style
    colors = setup_ieee_style()

    # Load configuration
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config.yaml'

    if not os.path.exists(config_path):
        print(f"\n✗ Config file not found: {config_path}")
        sys.exit(1)

    print(f"\nLoading configuration: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    validate_config(config)
    print("✓ Configuration validated")

    # Generate figures
    success_count = 0
    total_count = 3

    try:
        if generate_figure_gamma_breakdown(config):
            success_count += 1
    except Exception as e:
        print(f"\n✗ Gamma breakdown failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        if generate_figure_alpha_policy_rnet(config):
            success_count += 1
    except Exception as e:
        print(f"\n✗ Alpha policy R_net failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        if generate_figure_alpha_policy_rmse(config):
            success_count += 1
    except Exception as e:
        print(f"\n✗ Alpha policy RMSE failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print(f"COMPLETE: {success_count}/{total_count} supplementary figures generated")
    print("=" * 80)

    if success_count == total_count:
        print("\n✓ All supplementary figures generated successfully!")
        print("\nStyle improvements:")
        print("  - IEEE publication standards")
        print("  - Matching main figure style")
        print("  - No titles, only axis labels")
        print("  - Non-bold fonts, uniform 8pt")
        print("  - Single plots per figure")
        sys.exit(0)
    else:
        print(f"\n⚠ Only {success_count}/{total_count} figures generated")
        sys.exit(1)


if __name__ == "__main__":
    main()