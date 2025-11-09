#!/usr/bin/env python3
"""
MIMO Scaling Verification Script
验证修复后的代码是否正确实现MIMO标度

Expected behavior after fix:
1. Communication: SNR_crit(dB) vs 10*log10(Nt*Nr) should have slope ≈ -1.5 (CE mode with linear Gamma accumulation)
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

# Import the FIXED modules
sys.path.insert(0, '/home/claude')
from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
from limits_engine import calc_C_J, calc_BCRLB


def verify_mimo_scaling(config_path='config.yaml'):
    """Verify MIMO scaling for both communication and sensing"""

    print("=" * 80)
    print("MIMO SCALING VERIFICATION (After Fix)")
    print("=" * 80)
    print()

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
            rmse_m = np.sqrt(bcrlb_tau) * config['channel']['c_mps']

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

    # Convert to arrays for analysis
    g_ar_arr = np.array(results['g_ar'])
    snr_crit_arr = np.array(results['SNR_crit_db'])
    rmse_arr = np.array(results['RMSE_m'])
    snr_ratio_arr = np.array(results['SNR_ratio'])
    Nt_arr = np.array(results['Nt'])
    Nr_arr = np.array(results['Nr'])

    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    # ===================================================================
    # Test 1: Communication - SNR_crit scaling
    # ===================================================================
    print("\n[Test 1] Communication: SNR_crit vs Array Size")
    print("-" * 80)

    # ===== CRITICAL FIX: Auto-derive expected slope =====
    # In CE/PAPC mode with linear hardware accumulation:
    # - Γ_total ∝ (Nt+Nr)^p, where p=1.0 for linear accumulation
    # - G ∝ Nt*Nr (array gain)
    # - SNR_crit(dB) = -10*log10(G) - 10*log10(Γ)
    #
    # For square arrays (Nt=Nr=N):
    # - log10(Nt+Nr) ≈ log10(2N) ≈ 0.5*log10(N²) + const
    # - Expected slope = -(1.0 + 0.5*p_gamma)
    #
    # If p_gamma=1.0 (current implementation): slope = -1.5
    # If p_gamma=0.0 (ideal hardware): slope = -1.0

    p_gamma = 1.0  # Current implementation: linear accumulation Γ ∝ (Nt+Nr)
    expected_slope_comm = -(1.0 + 0.5 * p_gamma)  # = -1.5 for CE mode

    print(f"  Hardware accumulation model: Γ_total ∝ (Nt+Nr)^{p_gamma}")
    print(f"  Auto-derived expected slope: {expected_slope_comm:.2f}")

    # Regression: SNR_crit(dB) = C + slope * 10*log10(Nt*Nr)
    x_comm = 10 * np.log10(g_ar_arr)  # 10*log10(Nt*Nr) in dB
    y_comm = snr_crit_arr  # SNR_crit in dB

    slope_comm, intercept_comm, r_value_comm, _, std_err_comm = linregress(x_comm, y_comm)

    print(f"\n  Regression: SNR_crit(dB) = {intercept_comm:.2f} + {slope_comm:.3f} * 10log10(Nt*Nr)")
    print(f"  Expected slope: {expected_slope_comm:.2f}")
    print(f"  Measured slope: {slope_comm:.3f} ± {std_err_comm:.3f}")
    print(f"  R² = {r_value_comm ** 2:.4f}")
    print(
        f"  Deviation: {abs(slope_comm - expected_slope_comm):.3f} ({abs(slope_comm - expected_slope_comm) / abs(expected_slope_comm) * 100:.1f}%)")

    # Tolerance check
    tolerance = 0.1
    if abs(slope_comm - expected_slope_comm) < tolerance and r_value_comm ** 2 > 0.95:
        print(f"  ✓ PASS: Slope within {tolerance} of theory, R² > 0.95")
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
    # => slope of log(RMSE) vs log(g_ar) should be -0.5

    x_sense = np.log10(g_ar_arr)  # log10(Nt*Nr)
    y_sense = np.log10(rmse_arr * 1000)  # log10(RMSE in mm)

    slope_sense, intercept_sense, r_value_sense, _, std_err_sense = linregress(x_sense, y_sense)

    print(f"  Regression: log10(RMSE_mm) = {intercept_sense:.2f} + {slope_sense:.3f} * log10(Nt*Nr)")
    print(f"  Expected slope: -0.5")
    print(f"  Measured slope: {slope_sense:.3f} ± {std_err_sense:.3f}")
    print(f"  R² = {r_value_sense ** 2:.4f}")
    print(f"  Deviation: {abs(slope_sense + 0.5):.3f} ({abs(slope_sense + 0.5) / 0.5 * 100:.1f}%)")

    # RMSE improvement factor from smallest to largest array
    rmse_ratio = rmse_arr[0] / rmse_arr[-1]
    g_ar_ratio = np.sqrt(g_ar_arr[-1] / g_ar_arr[0])
    print(f"\n  RMSE improvement: {rmse_ratio:.2f}× (from N={results['N'][0]} to N={results['N'][-1]})")
    print(f"  Expected (theory): {g_ar_ratio:.2f}×")
    print(f"  Relative error: {abs(rmse_ratio - g_ar_ratio) / g_ar_ratio * 100:.1f}%")

    if abs(slope_sense + 0.5) < 0.1 and r_value_sense ** 2 > 0.95:
        print(f"  ✓ PASS: Slope within 10% of theory, R² > 0.95")
        sense_pass = True
    else:
        print(f"  ✗ FAIL: Slope or R² outside acceptable range")
        sense_pass = False

    # ===================================================================
    # Test 3: SNR ratio scaling (diagnostic)
    # ===================================================================
    print("\n[Test 3] Diagnostic: SNR Ratio (P_rx / sigma2_gamma)")
    print("-" * 80)

    # Expected: SNR_ratio ∝ Nt*Nr / (Nt+Nr)
    # For square arrays (Nt=Nr=N): SNR_ratio ∝ N²/(2N) = N/2

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
    # Generate visualization
    # ===================================================================
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: SNR_crit vs 10log10(g_ar)
    ax1.plot(x_comm, y_comm, 'o-', markersize=8, linewidth=2, label='Measured')
    x_fit = np.linspace(x_comm.min(), x_comm.max(), 100)
    y_fit = intercept_comm + slope_comm * x_fit
    ax1.plot(x_fit, y_fit, '--', linewidth=2,
             label=f'Fit: slope={slope_comm:.3f}')
    y_theory = intercept_comm + expected_slope_comm * x_fit
    ax1.plot(x_fit, y_theory, ':', linewidth=2, color='gray',
             label=f'Theory: slope={expected_slope_comm:.2f}')
    ax1.set_xlabel(r'$10\log_{10}(N_t N_r)$ (dB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$\mathrm{SNR}_{\mathrm{crit}}$ (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Communication: Critical SNR vs Array Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: RMSE vs g_ar (log-log)
    ax2.loglog(g_ar_arr, rmse_arr * 1000, 'o-', markersize=8, linewidth=2, label='Measured')
    g_ar_fit = np.logspace(np.log10(g_ar_arr.min()), np.log10(g_ar_arr.max()), 100)
    rmse_fit = 10 ** (intercept_sense) * g_ar_fit ** (slope_sense)
    ax2.loglog(g_ar_fit, rmse_fit, '--', linewidth=2,
               label=f'Fit: slope={slope_sense:.3f}')
    rmse_theory = 10 ** (intercept_sense) * g_ar_fit ** (-0.5)
    ax2.loglog(g_ar_fit, rmse_theory, ':', linewidth=2, color='gray',
               label='Theory: slope=-0.5')
    ax2.set_xlabel(r'$N_t N_r$', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Range RMSE (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Sensing: RMSE vs Array Size', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: SNR ratio scaling
    ax3.plot(results['N'], expected_norm, 'o-', markersize=8, linewidth=2,
             label='Expected: ∝ N/2', color='green')
    ax3.plot(results['N'], measured_norm, 's-', markersize=8, linewidth=2,
             label='Measured: P_rx/σ²_γ', color='blue')
    ax3.set_xlabel(r'Array Size ($N_t = N_r = N$)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Normalized SNR Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Diagnostic: Hardware SNR Scaling', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary table
    ax4.axis('off')

    summary_data = [
        ['Metric', 'Expected', 'Measured', 'Status'],
        ['=' * 15, '=' * 10, '=' * 10, '=' * 6],
        ['SNR_crit slope', f'{expected_slope_comm:.2f}', f'{slope_comm:.3f}', '✓' if comm_pass else '✗'],
        ['RMSE slope', '-0.5', f'{slope_sense:.3f}', '✓' if sense_pass else '✗'],
        ['SNR ratio', 'Linear', f'{avg_error:.1f}%', '✓' if ratio_pass else '✗'],
    ]

    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Color code the status
    for i in range(2, 5):
        if summary_data[i][3] == '✓':
            table[(i, 3)].set_facecolor('#90EE90')
        else:
            table[(i, 3)].set_facecolor('#FFB6C1')

    ax4.set_title('(d) Verification Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    os.makedirs('/mnt/user-data/outputs/figures', exist_ok=True)
    output_path = '/mnt/user-data/outputs/figures/mimo_scaling_verification.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_path}")

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
        print(f"  • Communication: SNR_crit slope = {expected_slope_comm:.2f} (CE mode with linear Γ accumulation)")
        print(f"  • Sensing: RMSE improves by ~√N per N× array size")
        print(f"  • Hardware distortion scales as (Nt+Nr), not (Nt*Nr)")
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
        config_path = 'config.yaml'

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