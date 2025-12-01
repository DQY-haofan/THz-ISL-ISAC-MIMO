#!/usr/bin/env python3
"""
Sensitivity Analysis Script (FINAL FIX v2)
Fixes the C_DSE calibration power law bug (Power 5 vs 4)
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import copy
import io
from pathlib import Path

# 防止 Windows 控制台打印中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import existing engines
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J
except ImportError as e:
    print(f"Error importing engines: {e}")
    sys.exit(1)


def setup_ieee_style():
    """
    Standardized Matplotlib configuration for IEEE Transactions.
    Size: 3.5 inches (single column)
    Font: Arial/Helvetica, 8pt
    """
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (3.5, 2.625),  # 3.5" width, 4:3 aspect ratio
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,          # Main text size
        'axes.titlesize': 8,     # Should ideally be empty (use caption)
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,    # Legend slightly smaller
        'text.usetex': False,    # Better compatibility, use mathtext

        # Line and marker settings
        'lines.linewidth': 1.0,  # Thin, precise lines
        'lines.markersize': 4,
        'lines.markeredgewidth': 0.5,

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'axes.axisbelow': True,  # Grid behind data

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fancybox': False, # Square corners preferred
        'legend.edgecolor': 'black',
        'legend.borderpad': 0.2,
        'legend.labelspacing': 0.2, # Compact spacing

        # Tick settings
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'in', # Ticks inside is often cleaner
        'ytick.direction': 'in',
    })

    # Standard Color Palette (IEEE/Matlab style)
    colors = {
        'blue':    '#0072BD',
        'orange':  '#D95319',
        'yellow':  '#EDB120',
        'purple':  '#7E2F8E',
        'green':   '#77AC30',
        'cyan':    '#4DBEEE',
        'red':     '#A2142F',
        'black':   '#000000',
        'gray':    '#7F7F7F',
    }
    return colors

def perform_autotune(config):
    """
    Replicate the autotune logic from physics_engine.py EXACTLY.
    """
    if config.get('isac_model', {}).get('DSE_autotune', False):
        target = config['isac_model'].get('alpha_star_target', 0.09)  # Default to 0.09 if missing

        # Temp run to get PN variance at target alpha
        cfg_temp = copy.deepcopy(config)
        cfg_temp['isac_model']['alpha'] = target

        # Need to ensure DSE is not overwriting things yet, so set dummy
        cfg_temp['dse_model']['C_DSE'] = 1e-9

        g = calc_g_sig_factors(cfg_temp)
        n = calc_n_f_vector(cfg_temp, g)
        S_pn = n['sigma_2_phi_c_res']

        # === CRITICAL FIX: Power must be 5, not 4 ===
        # Physics: sigma2_dse = C_DSE / alpha^5
        # Condition: sigma2_dse = S_pn at alpha=target
        # C_DSE / target^5 = S_pn  =>  C_DSE = S_pn * target^5
        C_DSE = S_pn * (target ** 5)

        config['isac_model']['C_DSE'] = C_DSE
        print(f"  [Auto-Tune] Calibrated C_DSE for alpha*={target:.3f}")
        print(f"              S_pn={S_pn:.2e}, C_DSE={C_DSE:.2e}")
    return config


def run_sensitivity_pa(base_config, output_dir):
    print("\nRunning PA Sensitivity Analysis...")

    # Sweep range
    gamma_db_vec = np.linspace(-5, -35, 31)
    c_sat_vec = []

    snr_high = [50.0]

    # Get baseline value
    baseline_linear = base_config['hardware']['gamma_pa_floor']
    baseline_db = 10 * np.log10(baseline_linear)

    # Calculate baseline C_sat
    g_base = calc_g_sig_factors(base_config)
    n_base = calc_n_f_vector(base_config, g_base)
    res_base = calc_C_J(base_config, g_base, n_base, snr_high)
    baseline_csat = res_base['C_sat']

    print(f"  PA Baseline: {baseline_db:.1f} dB -> C_sat: {baseline_csat:.2f}")

    for g_db in gamma_db_vec:
        cfg = copy.deepcopy(base_config)
        cfg['hardware']['gamma_pa_floor'] = 10 ** (g_db / 10.0)

        g_factors = calc_g_sig_factors(cfg)
        n_outputs = calc_n_f_vector(cfg, g_factors)
        res = calc_C_J(cfg, g_factors, n_outputs, snr_high)
        c_sat_vec.append(res['C_sat'])

    fig, ax = plt.subplots()
    ax.plot(gamma_db_vec, c_sat_vec, 'o-', color='#D95319', markersize=4)
    ax.plot(baseline_db, baseline_csat, 's', color='black', label='Baseline', markersize=6, zorder=10)

    ax.set_xlabel(r'PA Distortion Power $\Gamma_{\mathrm{PA}}$ (dB)', fontsize=8)
    ax.set_ylabel(r'Saturation Capacity $C_{\mathrm{sat}}$ (bits/s/Hz)', fontsize=8)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_sensitivity_pa.png')
    plt.savefig(output_dir / 'fig_sensitivity_pa.pdf')


def run_sensitivity_jerk(base_config, output_dir):
    print("\nRunning Jerk Noise Sensitivity Analysis...")

    # 1. Calibrate C_DSE first
    config = perform_autotune(copy.deepcopy(base_config))

    scale_vec = np.logspace(-2, 1, 20)
    alpha_star_vec = []

    # Constants from calibrated config
    cfg = copy.deepcopy(config)

    # We need a reference alpha to extract C_PN
    # Because physics_engine scales PN inside calc_n_f_vector
    alpha_ref = 0.1  # Arbitrary reference point
    cfg['isac_model']['alpha'] = alpha_ref
    g_factors = calc_g_sig_factors(cfg)
    n_outputs = calc_n_f_vector(cfg, g_factors)

    # Extract C_PN
    sigma2_pn_ref = n_outputs['sigma_2_phi_c_res']
    pn_exp = cfg['pn_model'].get('alpha_exponent', -1.0)
    C_PN = sigma2_pn_ref / (alpha_ref ** pn_exp)

    # Extract C_DSE
    # Note: calc_n_f_vector might interpret C_DSE differently if we don't pass it explicitly
    # But here we updated config['isac_model']['C_DSE'] in perform_autotune
    C_DSE_base = config['isac_model']['C_DSE']

    # Verify baseline alpha*
    # alpha* = (C_DSE/C_PN)^(1/4) if scaling is -1 vs -5
    # (Power difference is 4)
    baseline_alpha = (C_DSE_base / C_PN) ** 0.25
    print(f"  Jerk Baseline check: alpha* = {baseline_alpha:.4f} (Should be close to target)")

    for scale in scale_vec:
        # Scale C_DSE (simulating q_j scaling)
        C_DSE_scaled = C_DSE_base * scale
        alpha_star = (C_DSE_scaled / C_PN) ** 0.25
        alpha_star_vec.append(alpha_star)

    fig, ax = plt.subplots()
    ax.semilogx(scale_vec, alpha_star_vec, 'o-', color='#7E2F8E', markersize=4)

    # Mark baseline
    ax.plot(1.0, baseline_alpha, 's', color='black', label='Baseline', markersize=6, zorder=10)

    ax.set_xlabel(r'Jerk Noise Scaling Factor ($q_j/q_{j,0}$)', fontsize=8)
    ax.set_ylabel(r'Optimal Overhead $\alpha^*$', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_sensitivity_jerk.png')
    plt.savefig(output_dir / 'fig_sensitivity_jerk.pdf')


def main():
    setup_ieee_style()
    config_path = 'config.yaml'

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    try:
        run_sensitivity_pa(config, output_dir)
        run_sensitivity_jerk(config, output_dir)
        print("\n✓ Sensitivity plots generated successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()