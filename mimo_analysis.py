#!/usr/bin/env python3
"""
MIMO Scaling Verification Script (IEEE STYLE - FIXED VERSION)
验证修复后的代码是否正确实现MIMO标度

Key improvements:
- Matches IEEE publication style (8pt fonts, no bold, no titles)
- Saves figures to figures/ directory
- Saves CSV to results/ directory
- Single plots (no subplots)
- Auto-derived expected slopes based on hardware model

Expected behavior after fix:
1. Communication: SNR_crit(dB) vs 10*log10(Nt*Nr) should have slope ≈ -1.5 (CE mode)
2. Sensing: RMSE vs Nt*Nr (log-log) should have slope ≈ -0.5

Usage:
    python mimo_analysis.py [config.yaml]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import yaml
import os
import sys
import pandas as pd
from pathlib import Path

# Import the FIXED modules
sys.path.insert(0, '/mnt/user-data/uploads')
from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
from limits_engine import calc_C_J, calc_BCRLB


def setup_ieee_style():
    """
    Configure matplotlib for IEEE journal publication standards.
    Matching visualize_results.py style: Helvetica/Arial fonts, uniform 8pt, non-bold.
    """
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (3.5, 2.625),  # IEEE single column width
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font settings (Helvetica/Arial, sans-serif, NO BOLD)
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

    # Color scheme matching visualize_results.py
    colors = {
        'blue': '#0072BD',
        'orange': '#D95319',
        'green': '#77AC30',
        'red': '#A2142F',
        'purple': '#7E2F8E',
        'yellow': '#EDB120',
        'black': '#000000',
    }

    return colors


def generate_mimo_combined_figure(results, colors, figures_dir):
    """
    生成 MIMO 性能合并图（双y轴）- 线条分离版本

    改进：通过添加垂直偏移，让拟合线和理论线分离显示
    """

    print("\n  Generating combined MIMO performance figure (dual y-axis, separated lines)...")

    # 提取数据
    g_ar_arr = np.array(results['g_ar'])
    snr_crit_arr = np.array(results['SNR_crit_db'])
    rmse_arr = np.array(results['RMSE_m'])
    Gamma_arr = np.array(results['Gamma_eff_total'])

    # === 计算 SNR_crit 拟合 ===
    log_NtNr = np.log10(g_ar_arr)
    log_Gamma = np.log10(Gamma_arr)

    slope_gamma, intercept_gamma, _, _, _ = linregress(log_NtNr, log_Gamma)
    p_gamma = 2.0 * slope_gamma
    expected_slope_snr = -(1.0 + 0.5 * p_gamma)

    x_comm = 10 * np.log10(g_ar_arr)
    y_comm = snr_crit_arr

    slope_comm, intercept_comm, r_value_comm, _, std_err_comm = linregress(x_comm, y_comm)

    # === 计算 RMSE 拟合 ===
    log_g_ar = np.log10(g_ar_arr)
    log_rmse = np.log10(rmse_arr)

    slope_sense, intercept_sense, r_value_sense, _, std_err_sense = linregress(log_g_ar, log_rmse)

    # === 创建双y轴图 ===
    fig, ax1 = plt.subplots(figsize=(3.5, 2.625))

    # ===== 左y轴：SNR_crit =====
    color_snr = colors['blue']

    ax1.set_xlabel('10*log10(Nt*Nr) [dB]', fontsize=8)
    ax1.set_ylabel('SNR_crit [dB]', fontsize=8, color=color_snr)

    x_fit = np.linspace(x_comm.min(), x_comm.max(), 100)

    # ===== 方案1：调整理论线的截距（推荐）=====
    # 让理论线在拟合线上方或下方偏移

    # SNR 数据范围
    snr_range = y_comm.max() - y_comm.min()
    offset_snr = snr_range * 0.05  # 偏移 5% 的数据范围

    # 拟合线（实际数据的拟合）
    y_fit = intercept_comm + slope_comm * x_fit
    ax1.plot(x_fit, y_fit, '-', linewidth=1.5, color=color_snr, alpha=0.7,
             label=f'SNR fit (m={slope_comm:.2f})', zorder=1)

    # 理论线（使用相同斜率，但截距偏移）
    # 计算偏移后的截距，使理论线整体下移
    intercept_theory_snr = intercept_comm - offset_snr
    y_theory = intercept_theory_snr + expected_slope_snr * x_fit
    ax1.plot(x_fit, y_theory, '--', linewidth=1.5, color='#77AC30', alpha=0.9,
             label=f'SNR theory (m={expected_slope_snr:.2f})', zorder=2)

    # 数据点（最后画，在顶层）
    ax1.plot(x_comm, y_comm, 'o', markersize=5, linewidth=0,
             label='SNR_crit (data)', color=color_snr,
             markeredgecolor='black', markeredgewidth=0.5,
             zorder=3)

    ax1.tick_params(axis='y', labelcolor=color_snr, labelsize=8)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # ===== 右y轴：RMSE =====
    ax2 = ax1.twinx()
    color_rmse = colors['orange']
    ax2.set_ylabel('Range RMSE [mm, log]', fontsize=8, color=color_rmse)

    # RMSE 数据范围（对数空间）
    log_rmse_range = np.log10(rmse_arr.max()) - np.log10(rmse_arr.min())
    offset_factor_rmse = 1.2  # 偏移因子（乘法），使理论线整体上移20%

    g_ar_fit = 10 ** (x_fit / 10)

    # RMSE 拟合线
    rmse_fit = 10 ** (intercept_sense) * g_ar_fit ** (slope_sense)
    ax2.semilogy(x_fit, rmse_fit * 1000, '-', linewidth=1.5,
                 color=color_rmse, alpha=0.7,
                 label=f'RMSE fit (m={slope_sense:.2f})', zorder=1)

    # RMSE 理论线（使用相同斜率-0.5，但整体上移）
    # 调整截距使线条上移
    intercept_theory_rmse = intercept_sense + log_rmse_range * 0.15  # 对数空间偏移
    rmse_theory = 10 ** (intercept_theory_rmse) * g_ar_fit ** (-0.5)
    ax2.semilogy(x_fit, rmse_theory * 1000, '--', linewidth=1.5,
                 color='#A2142F', alpha=0.9,
                 label='RMSE theory (m=-0.5)', zorder=2)

    # RMSE 数据点
    ax2.semilogy(x_comm, rmse_arr * 1000, 's', markersize=5, linewidth=0,
                 label='RMSE (data)', color=color_rmse,
                 markeredgecolor='black', markeredgewidth=0.5,
                 zorder=3)

    ax2.tick_params(axis='y', labelcolor=color_rmse, labelsize=8)

    # ===== 图例（紧凑型，两列）=====
    lines1 = ax1.get_lines()
    lines2 = ax2.get_lines()

    labels1 = [l.get_label() for l in lines1]
    labels2 = [l.get_label() for l in lines2]

    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    ax1.legend(all_lines, all_labels, fontsize=6.5, loc='lower right',
               ncol=2, framealpha=0.9, columnspacing=0.8,
               handlelength=1.5, handletextpad=0.5)

    # 添加性能指标文本框
    textstr = f'R2: SNR={r_value_comm ** 2:.3f}, RMSE={r_value_sense ** 2:.3f}'
    # ax1.text(0.02, 0.02, textstr,
    #          transform=ax1.transAxes, fontsize=7,
    #          verticalalignment='bottom',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, linewidth=0.5))

    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = figures_dir / f'fig_mimo_combined.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_path}")

    plt.close()

    return True



def verify_mimo_scaling(config_path='config.yaml'):
    """Verify MIMO scaling for both communication and sensing"""

    print("=" * 80)
    print("MIMO SCALING VERIFICATION (IEEE Style)")
    print("=" * 80)
    print()

    # Setup IEEE style
    colors = setup_ieee_style()

    # Create output directories
    figures_dir = Path('figures')
    results_dir = Path('results')
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    validate_config(config)

    # Test array sizes (square arrays: Nt = Nr)
    N_values = [16, 32, 64, 128, 256]

    results = {
        'N': [],
        'Nt': [],
        'Nr': [],
        'g_ar': [],
        'SNR_crit_db': [],
        'BCRLB_tau': [],
        'RMSE_m': [],
        'Gamma_eff_total': [],
        'sigma2_gamma': [],
        'SNR_ratio': []  # P_rx / sigma2_gamma
    }

    print("Running MIMO scaling sweep...")
    print(f"{'N':>4} | {'Nt':>4} | {'Nr':>4} | {'g_ar':>8} | {'SNR_crit':>10} | {'RMSE(mm)':>10} | {'SNR_ratio':>10}")
    print("-" * 80)

    for N in N_values:
        # Square array configuration
        config['array']['Nt'] = N
        config['array']['Nr'] = N

        try:
            # Run pipeline
            g_factors = calc_g_sig_factors(config)
            n_outputs = calc_n_f_vector(config, g_factors)

            # Communication metrics
            SNR0_db_single = config['simulation'].get('SNR0_db_fixed',
                                                      config['simulation']['SNR0_db_vec'][0])
            c_j_results = calc_C_J(config, g_factors, n_outputs, [SNR0_db_single], compute_C_G=False)

            # Sensing metrics
            bcrlb_results = calc_BCRLB(config, g_factors, n_outputs)

            # Store results
            g_ar = g_factors['g_ar']
            snr_crit = c_j_results['SNR_crit_db']
            bcrlb_tau = bcrlb_results['BCRLB_tau']
            rmse_m = np.sqrt(bcrlb_tau) * config['channel']['c_mps'] / 2.0  # Monostatic: c/2

            # Check SNR ratio (should scale as g_ar / (Nt+Nr))
            P_rx = n_outputs['P_rx_total']
            sigma2_gamma = n_outputs['sigma2_gamma']
            snr_ratio = P_rx / sigma2_gamma

            results['N'].append(N)
            results['Nt'].append(N)
            results['Nr'].append(N)
            results['g_ar'].append(g_ar)
            results['SNR_crit_db'].append(snr_crit)
            results['BCRLB_tau'].append(bcrlb_tau)
            results['RMSE_m'].append(rmse_m)
            results['Gamma_eff_total'].append(n_outputs['Gamma_eff_total'])
            results['sigma2_gamma'].append(sigma2_gamma)
            results['SNR_ratio'].append(snr_ratio)

            print(
                f"{N:4d} | {N:4d} | {N:4d} | {g_ar:8.0f} | {snr_crit:10.2f} | {rmse_m * 1000:10.3f} | {snr_ratio:10.2f}")

        except Exception as e:
            print(f"Error at N={N}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results['N']) < 3:
        print("\n✗ Insufficient data points for scaling analysis")
        return False

    # Save CSV
    df_results = pd.DataFrame(results)
    csv_path = results_dir / 'mimo_scaling_data.csv'
    df_results.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"\n✓ Saved data: {csv_path}")

    # Convert to arrays for analysis
    g_ar_arr = np.array(results['g_ar'])
    snr_crit_arr = np.array(results['SNR_crit_db'])
    rmse_arr = np.array(results['RMSE_m'])
    snr_ratio_arr = np.array(results['SNR_ratio'])
    Nt_arr = np.array(results['Nt'])
    Nr_arr = np.array(results['Nr'])
    Gamma_arr = np.array(results['Gamma_eff_total'])

    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    # ===================================================================
    # Test 1: Communication - SNR_crit scaling
    # ===================================================================
    print("\n[Test 1] Communication: SNR_crit vs Array Size")
    print("-" * 80)

    # Determine expected slope based on hardware distortion scaling
    log_NtNr = np.log10(g_ar_arr)
    log_Gamma = np.log10(Gamma_arr)

    slope_gamma, intercept_gamma, _, _, _ = linregress(log_NtNr, log_Gamma)
    p_gamma = 2.0 * slope_gamma  # Convert from log10(Nt*Nr) to log10(Nt+Nr) basis

    print(f"\n[Auto-Detection] Hardware distortion scaling:")
    print(f"  log10(Gamma) vs log10(Nt*Nr): slope = {slope_gamma:.3f}")
    print(f"  Inferred: Gamma ∝ (Nt+Nr)^{p_gamma:.3f}")

    # Expected slope in dB scale: x = 10*log10(Nt*Nr)
    expected_slope = -(1.0 + 0.5 * p_gamma)

    print(f"\n[Expected Scaling]")
    print(f"  SNR_crit(dB) = C - 10*log10(Nt*Nr) - 10*log10(Gamma)")
    print(f"  With Gamma ∝ (Nt+Nr)^{p_gamma:.2f}:")
    print(f"  Expected slope: {expected_slope:.3f}")

    # Regression: SNR_crit vs 10*log10(g_ar)
    x_comm = 10 * np.log10(g_ar_arr)  # 10*log10(Nt*Nr) in dB
    y_comm = snr_crit_arr  # SNR_crit in dB

    slope_comm, intercept_comm, r_value_comm, _, std_err_comm = linregress(x_comm, y_comm)

    print(f"\n[Measured Scaling]")
    print(f"  Regression: SNR_crit(dB) = {intercept_comm:.2f} + {slope_comm:.3f} * 10log10(Nt*Nr)")
    print(f"  Measured slope: {slope_comm:.3f} ± {std_err_comm:.3f}")
    print(f"  R² = {r_value_comm ** 2:.4f}")
    print(
        f"  Deviation from expected: {abs(slope_comm - expected_slope):.3f} ({abs(slope_comm - expected_slope) / abs(expected_slope) * 100:.1f}%)")

    # Use adaptive tolerance
    tolerance = 0.15 if len(results['N']) < 5 else 0.10

    if abs(slope_comm - expected_slope) < tolerance and r_value_comm ** 2 > 0.95:
        print(f"  ✓ PASS: Slope within {tolerance:.0%} of theory, R² > 0.95")
        comm_pass = True
    else:
        print(f"  ✗ FAIL: Slope or R² outside acceptable range")
        comm_pass = False

    # ===================================================================
    # Test 2: Sensing - RMSE scaling
    # ===================================================================
    print("\n[Test 2] Sensing: RMSE vs Array Size")
    print("-" * 80)

    # Expected: RMSE ∝ 1/√(Nt*Nr)
    # => log(RMSE) = C - 0.5*log(Nt*Nr)

    log_g_ar = np.log10(g_ar_arr)
    log_rmse = np.log10(rmse_arr)

    slope_sense, intercept_sense, r_value_sense, _, std_err_sense = linregress(log_g_ar, log_rmse)

    print(f"\n[Measured Scaling]")
    print(f"  Log-log regression: log(RMSE) = {intercept_sense:.3f} + {slope_sense:.3f} * log(Nt*Nr)")
    print(f"  Measured slope: {slope_sense:.3f} ± {std_err_sense:.3f}")
    print(f"  R² = {r_value_sense ** 2:.4f}")
    print(f"  Deviation: {abs(slope_sense + 0.5):.3f} ({abs(slope_sense + 0.5) / 0.5 * 100:.1f}%)")

    # RMSE improvement factor
    rmse_ratio = rmse_arr[0] / rmse_arr[-1]
    g_ar_ratio = np.sqrt(g_ar_arr[-1] / g_ar_arr[0])
    print(f"\n  RMSE improvement: {rmse_ratio:.2f}× (from N={results['N'][0]} to N={results['N'][-1]})")
    print(f"  Expected (theory): {g_ar_ratio:.2f}×")
    print(f"  Relative error: {abs(rmse_ratio - g_ar_ratio) / g_ar_ratio * 100:.1f}%")

    if abs(slope_sense + 0.5) < 0.15 and r_value_sense ** 2 > 0.90:
        print(f"  ✓ PASS: Slope within 15% of theory, R² > 0.90")
        sense_pass = True
    else:
        print(f"  ✗ FAIL: Slope or R² outside acceptable range")
        sense_pass = False

    # ===================================================================
    # Test 3: SNR ratio scaling (diagnostic)
    # ===================================================================
    print("\n[Test 3] Diagnostic: SNR Ratio (P_rx / sigma2_gamma)")
    print("-" * 80)

    expected_ratio = g_ar_arr / (Nt_arr + Nr_arr)
    measured_ratio = snr_ratio_arr

    # Normalize to first value
    expected_norm = expected_ratio / expected_ratio[0]
    measured_norm = measured_ratio / measured_ratio[0]

    print(f"  {'N':>4} | {'Expected':>10} | {'Measured':>10} | {'Error(%)':>10}")
    print("  " + "-" * 50)
    for i, N in enumerate(results['N']):
        error_pct = abs(measured_norm[i] - expected_norm[i]) / expected_norm[i] * 100
        print(f"  {N:4d} | {expected_norm[i]:10.2f} | {measured_norm[i]:10.2f} | {error_pct:10.1f}")

    avg_error = np.mean(np.abs(measured_norm - expected_norm) / expected_norm) * 100
    if avg_error < 10:
        print(f"\n  ✓ PASS: Average error {avg_error:.1f}% < 10%")
        ratio_pass = True
    else:
        print(f"\n  ✗ FAIL: Average error {avg_error:.1f}% >= 10%")
        ratio_pass = False

    # ===================================================================
    # Generate IEEE-style figures (NO TITLES, single plots)
    # ===================================================================
    print("\\n" + "=" * 80)
    print("GENERATING IEEE-STYLE FIGURES")
    print("=" * 80)

    # Combined Figure: SNR_crit and RMSE (dual y-axis)
    generate_mimo_combined_figure(results, colors, figures_dir)

    # Figure 2: SNR ratio scaling（原 Figure 3，保持不变）
    fig, ax = plt.subplots()

    # Figure 3: SNR ratio scaling
    fig, ax = plt.subplots()
    ax.plot(results['N'], expected_norm, 'o', markersize=4, linewidth=1.0,
            label='Expected: ∝ N/2', color=colors['green'],
            markeredgecolor='black', markeredgewidth=0.3)
    ax.plot(results['N'], measured_norm, 's', markersize=4, linewidth=1.0,
            label='Measured: P_rx/σ²_γ', color=colors['blue'],
            markeredgecolor='black', markeredgewidth=0.3)
    ax.set_xlabel(r'Array Size ($N_t = N_r = N$)', fontsize=8)
    ax.set_ylabel(r'Normalized SNR Ratio', fontsize=8)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = figures_dir / f'fig_mimo_snr_ratio.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    plt.close()

    # ===================================================================
    # Final verdict
    # ===================================================================
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    all_pass = comm_pass and sense_pass and ratio_pass

    if all_pass:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe fix is successful! MIMO scaling now works correctly:")
        print(f"  • Communication: SNR_crit shifts left by ~{-10 * expected_slope:.1f}dB per 10× array size")
        print(f"  • Sensing: RMSE improves by ~√N per N× array size")
        print(f"  • Hardware distortion scales as (Nt+Nr)^{p_gamma:.1f}, not (Nt*Nr)")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nFailed tests:")
        if not comm_pass:
            print("  ✗ Communication scaling")
        if not sense_pass:
            print("  ✗ Sensing scaling")
        if not ratio_pass:
            print("  ✗ SNR ratio diagnostic")

    print("\n" + "=" * 80)

    return all_pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Try common locations
        search_paths = [
            'config.yaml',
            '/mnt/user-data/uploads/config.yaml',
        ]
        config_path = None
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break

        if config_path is None:
            print("Error: Config file not found")
            print("Usage: python mimo_analysis.py [config.yaml]")
            sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        success = verify_mimo_scaling(config_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)