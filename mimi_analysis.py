#!/usr/bin/env python3
"""
生成MIMO扩展性关键图（专家建议）

根据专家文档，必须生成以下4张图：
1. MIMO通信扩展性：SNR_{0,β} vs 10log10(NtNr)
2. MIMO感知扩展性：RMSE_tau vs NtNr (log-log)
3. Whittle离散化收敛性：误差 vs N
4. Beam-squint有效性热图：B/fc vs L_ap/λ

Usage:
    python generate_mimo_scaling_figures.py config.yaml
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
from typing import Dict, Any
import warnings

# 假设physics_engine和limits_engine已在同目录
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB
except ImportError:
    print("Error: 请确保physics_engine.py和limits_engine.py在同目录")
    sys.exit(1)


def generate_fig1_mimo_comm_scaling(config: Dict[str, Any], output_dir='figures'):
    """
    图1: MIMO通信扩展性
    画 SNR_{0,β} vs 10log10(NtNr)，β=0.9/0.95
    理论参考线斜率 = -1
    """
    print("\n" + "=" * 80)
    print("图1: MIMO通信扩展性 (Communication Scaling)")
    print("=" * 80)

    # 扫描天线数
    Nt_Nr_values = [16, 32, 64, 128]
    beta_values = [0.9, 0.95]

    results = {beta: {'Nt_Nr': [], 'SNR_0_beta_dB': [], 'g_ar': []} for beta in beta_values}

    for Nt_Nr in Nt_Nr_values:
        print(f"\n[Nt=Nr={Nt_Nr}] 计算中...")

        cfg = config.copy()
        cfg['array']['Nt'] = Nt_Nr
        cfg['array']['Nr'] = Nt_Nr

        try:
            g_factors = calc_g_sig_factors(cfg)
            n_outputs = calc_n_f_vector(cfg, g_factors)

            # 扫描SNR找到达到β*C_sat的最小SNR0
            SNR_sweep = np.linspace(-30, 50, 200)
            c_results = calc_C_J(cfg, g_factors, n_outputs, SNR_sweep, compute_C_G=False)

            C_sat = c_results['C_sat']
            C_J_vec = c_results['C_J_vec']

            for beta in beta_values:
                target_capacity = beta * C_sat
                idx = np.where(C_J_vec >= target_capacity)[0]
                if len(idx) > 0:
                    SNR_0_beta = SNR_sweep[idx[0]]
                    results[beta]['Nt_Nr'].append(Nt_Nr)
                    results[beta]['SNR_0_beta_dB'].append(SNR_0_beta)
                    results[beta]['g_ar'].append(g_factors['g_ar'])
                    print(f"  β={beta}: SNR_{{0,{beta}}} = {SNR_0_beta:.2f} dB")

        except Exception as e:
            print(f"  失败: {e}")
            continue

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))

    for beta, color, marker in [(0.9, 'blue', 'o'), (0.95, 'red', 's')]:
        data = results[beta]
        if len(data['Nt_Nr']) > 0:
            g_ar_array = np.array(data['g_ar'])
            SNR_array = np.array(data['SNR_0_beta_dB'])

            # 数据点
            ax.plot(10 * np.log10(g_ar_array), SNR_array,
                    marker=marker, markersize=10, linewidth=2.5,
                    label=f'β = {beta}', color=color, alpha=0.8)

            # 理论参考线 (斜率 = -1)
            x_theory = 10 * np.log10(g_ar_array)
            # 拟合得到截距
            from scipy.stats import linregress
            slope, intercept, r_val, _, _ = linregress(x_theory, SNR_array)
            y_theory = intercept - x_theory
            ax.plot(x_theory, y_theory, '--', color=color, linewidth=2,
                    alpha=0.6, label=f'Theory (slope=-1), β={beta}')

            print(f"\nβ={beta} 拟合结果:")
            print(f"  测量斜率: {slope:.3f}")
            print(f"  理论斜率: -1.0")
            print(f"  误差: {abs(slope + 1.0):.3f}")
            print(f"  R²: {r_val ** 2:.4f}")

    ax.set_xlabel(r'$10\log_{10}(N_t N_r)$ (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\mathrm{SNR}_{0,\beta}$ (dB)', fontsize=14, fontweight='bold')
    ax.set_title('MIMO Communication Scaling\n' +
                 r'(SNR to reach $\beta \cdot C_{\mathrm{sat}}$)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_mimo_comm_scaling.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ 图1已保存: {output_path}")
    plt.close()


def generate_fig2_mimo_sense_scaling(config: Dict[str, Any], output_dir='figures'):
    """
    图2: MIMO感知扩展性
    画 RMSE_tau vs NtNr (log-log)，斜率≈-0.5
    """
    print("\n" + "=" * 80)
    print("图2: MIMO感知扩展性 (Sensing Scaling)")
    print("=" * 80)

    Nt_Nr_values = [16, 32, 64, 128, 256]

    results = {'Nt_Nr': [], 'RMSE_m': [], 'g_ar': []}

    for Nt_Nr in Nt_Nr_values:
        print(f"\n[Nt=Nr={Nt_Nr}] 计算中...")

        cfg = config.copy()
        cfg['array']['Nt'] = Nt_Nr
        cfg['array']['Nr'] = Nt_Nr

        try:
            g_factors = calc_g_sig_factors(cfg)
            n_outputs = calc_n_f_vector(cfg, g_factors)
            bcrlb_results = calc_BCRLB(cfg, g_factors, n_outputs)

            RMSE_tau = np.sqrt(bcrlb_results['BCRLB_tau'])
            c_mps = cfg['channel']['c_mps']
            RMSE_m = RMSE_tau * c_mps

            results['Nt_Nr'].append(Nt_Nr)
            results['RMSE_m'].append(RMSE_m * 1000)  # to mm
            results['g_ar'].append(g_factors['g_ar'])

            print(f"  RMSE = {RMSE_m * 1000:.3f} mm")
            print(f"  RMSE * sqrt(NtNr) = {RMSE_m * 1000 * np.sqrt(g_factors['g_ar']):.1f}")

        except Exception as e:
            print(f"  失败: {e}")
            continue

    # 绘图 (log-log)
    fig, ax = plt.subplots(figsize=(10, 7))

    if len(results['Nt_Nr']) > 0:
        g_ar_array = np.array(results['g_ar'])
        RMSE_array = np.array(results['RMSE_m'])

        # 数据点
        ax.loglog(g_ar_array, RMSE_array, 'o', markersize=10,
                  color='purple', alpha=0.8, label='Measured RMSE')

        # 拟合直线 (log-log space)
        from scipy.stats import linregress
        log_g = np.log10(g_ar_array)
        log_rmse = np.log10(RMSE_array)
        slope, intercept, r_val, _, _ = linregress(log_g, log_rmse)

        # 理论参考线 (斜率 = -0.5)
        g_ar_theory = np.logspace(np.log10(g_ar_array.min()),
                                  np.log10(g_ar_array.max()), 100)
        RMSE_theory = 10 ** (intercept) * g_ar_theory ** (-0.5)
        ax.loglog(g_ar_theory, RMSE_theory, '--', linewidth=2.5,
                  color='gray', label='Theory (slope = -0.5)')

        # 拟合线
        RMSE_fit = 10 ** (intercept) * g_ar_array ** slope
        ax.loglog(g_ar_array, RMSE_fit, '-', linewidth=2,
                  color='darkviolet', alpha=0.7,
                  label=f'Fit (slope = {slope:.3f}, R²={r_val ** 2:.4f})')

        print(f"\n拟合结果:")
        print(f"  测量斜率: {slope:.3f}")
        print(f"  理论斜率: -0.5")
        print(f"  误差: {abs(slope + 0.5):.3f}")
        print(f"  R²: {r_val ** 2:.4f}")

        if abs(slope + 0.5) < 0.05:
            print("  ✓ EXCELLENT: 斜率误差 < 5%")
        elif abs(slope + 0.5) < 0.1:
            print("  ✓ GOOD: 斜率误差 < 10%")
        else:
            print("  ⚠ WARNING: 斜率偏差较大")

    ax.set_xlabel(r'$N_t N_r$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Range RMSE (mm)', fontsize=14, fontweight='bold')
    ax.set_title('MIMO Sensing Scaling (BCRLB)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'fig_mimo_sense_scaling.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ 图2已保存: {output_path}")
    plt.close()


def generate_fig3_whittle_convergence(config: Dict[str, Any], output_dir='figures'):
    """
    图3: Whittle离散化收敛性
    画相对误差 vs N，找N_min使误差<1%
    """
    print("\n" + "=" * 80)
    print("图3: Whittle离散化收敛性 (Discretization Convergence)")
    print("=" * 80)

    N_values = [1024, 2048, 4096, 8192, 16384, 32768]

    results = {'N': [], 'BCRLB_tau': [], 'C_J': [], 'rel_err_BCRLB': [], 'rel_err_C': []}

    # 使用最大N作为参考值
    print(f"\n[计算参考值] N={N_values[-1]}...")
    cfg_ref = config.copy()
    cfg_ref['simulation']['N'] = N_values[-1]

    try:
        g_ref = calc_g_sig_factors(cfg_ref)
        n_ref = calc_n_f_vector(cfg_ref, g_ref)
        bcrlb_ref = calc_BCRLB(cfg_ref, g_ref, n_ref)
        c_ref = calc_C_J(cfg_ref, g_ref, n_ref, [20.0], compute_C_G=False)

        BCRLB_tau_ref = bcrlb_ref['BCRLB_tau']
        C_J_ref = c_ref['C_sat']

        print(f"  参考BCRLB_tau = {BCRLB_tau_ref:.6e}")
        print(f"  参考C_sat = {C_J_ref:.6f}")

    except Exception as e:
        print(f"  失败: {e}")
        return

    # 扫描N
    for N in N_values[:-1]:  # 不包括参考值
        print(f"\n[N={N}] 计算中...")

        cfg = config.copy()
        cfg['simulation']['N'] = N

        try:
            g_factors = calc_g_sig_factors(cfg)
            n_outputs = calc_n_f_vector(cfg, g_factors)
            bcrlb_results = calc_BCRLB(cfg, g_factors, n_outputs)
            c_results = calc_C_J(cfg, g_factors, n_outputs, [20.0], compute_C_G=False)

            BCRLB_tau = bcrlb_results['BCRLB_tau']
            C_J = c_results['C_sat']

            rel_err_BCRLB = abs(BCRLB_tau - BCRLB_tau_ref) / BCRLB_tau_ref * 100
            rel_err_C = abs(C_J - C_J_ref) / C_J_ref * 100

            results['N'].append(N)
            results['BCRLB_tau'].append(BCRLB_tau)
            results['C_J'].append(C_J)
            results['rel_err_BCRLB'].append(rel_err_BCRLB)
            results['rel_err_C'].append(rel_err_C)

            print(f"  BCRLB相对误差: {rel_err_BCRLB:.3f}%")
            print(f"  C_sat相对误差: {rel_err_C:.3f}%")

        except Exception as e:
            print(f"  失败: {e}")
            continue

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if len(results['N']) > 0:
        N_array = np.array(results['N'])

        # 子图1: BCRLB误差
        ax = ax1
        ax.plot(N_array, results['rel_err_BCRLB'], 'o-', markersize=8,
                linewidth=2, color='blue', label='BCRLB relative error')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                   label='1% threshold')
        ax.set_xlabel('FFT Points (N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Relative Error (%)', fontsize=13, fontweight='bold')
        ax.set_title('(a) BCRLB Convergence', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        # 找到N_min
        idx_good = np.where(np.array(results['rel_err_BCRLB']) < 1.0)[0]
        if len(idx_good) > 0:
            N_min = N_array[idx_good[0]]
            ax.axvline(x=N_min, color='green', linestyle=':', linewidth=2,
                       label=f'N_min = {N_min}')
            print(f"\n✓ BCRLB收敛: N ≥ {N_min} (误差 < 1%)")

        # 子图2: C_sat误差
        ax = ax2
        ax.plot(N_array, results['rel_err_C'], 's-', markersize=8,
                linewidth=2, color='green', label='C_sat relative error')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                   label='1% threshold')
        ax.set_xlabel('FFT Points (N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Relative Error (%)', fontsize=13, fontweight='bold')
        ax.set_title('(b) Capacity Convergence', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'fig_whittle_convergence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ 图3已保存: {output_path}")
    plt.close()


def generate_fig4_squint_validity(config: Dict[str, Any], output_dir='figures'):
    """
    图4: Beam-squint有效性热图
    画 Jensen gap vs (B/fc, L_ap/λ)
    """
    print("\n" + "=" * 80)
    print("图4: Beam-squint有效性热图 (Validity Map)")
    print("=" * 80)

    # 参数网格
    B_over_fc_vec = np.linspace(0.01, 0.2, 15)
    L_over_lambda_vec = np.linspace(10, 200, 15)

    gap_matrix = np.zeros((len(B_over_fc_vec), len(L_over_lambda_vec)))

    print(f"\n扫描 {len(B_over_fc_vec)} × {len(L_over_lambda_vec)} = {gap_matrix.size} 个点...")

    total = gap_matrix.size
    count = 0

    for i, B_over_fc in enumerate(B_over_fc_vec):
        for j, L_over_lambda in enumerate(L_over_lambda_vec):
            count += 1
            if count % 20 == 0:
                print(f"  进度: {count}/{total} ({100 * count / total:.1f}%)")

            cfg = config.copy()

            # 调整参数
            f_c = cfg['channel']['f_c_hz']
            cfg['channel']['B_hz'] = B_over_fc * f_c
            c_mps = cfg['channel']['c_mps']
            lambda_c = c_mps / f_c
            cfg['array']['L_ap_m'] = L_over_lambda * lambda_c

            try:
                g_factors = calc_g_sig_factors(cfg)
                n_outputs = calc_n_f_vector(cfg, g_factors)
                c_results = calc_C_J(cfg, g_factors, n_outputs, [20.0], compute_C_G=True)

                if 'Jensen_gap_bits' in c_results:
                    gap = c_results['Jensen_gap_bits'][0]
                else:
                    gap = np.nan

                gap_matrix[i, j] = gap

            except Exception as e:
                gap_matrix[i, j] = np.nan

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 热力图
    extent = [L_over_lambda_vec.min(), L_over_lambda_vec.max(),
              B_over_fc_vec.min(), B_over_fc_vec.max()]

    im = ax.imshow(gap_matrix, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', interpolation='bilinear')

    # 添加"安全区域"等高线 (gap < 0.001 bits/s/Hz)
    contour = ax.contour(L_over_lambda_vec, B_over_fc_vec, gap_matrix,
                         levels=[0.001], colors='red', linewidths=2.5,
                         linestyles='--')
    ax.clabel(contour, inline=True, fontsize=11, fmt='Gap=%.3f')

    ax.set_xlabel(r'Aperture Size ($L_{\mathrm{ap}}/\lambda$)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Bandwidth Ratio ($B/f_c$)',
                  fontsize=14, fontweight='bold')
    ax.set_title('Beam-Squint Validity Map (Jensen Gap)',
                 fontsize=15, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Jensen Gap (bits/s/Hz)', fontsize=12, fontweight='bold')

    # 添加文本说明
    textstr = 'Safe region: Gap < 0.001\n(inside red dashed line)'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'fig_squint_validity_map.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n✓ 图4已保存: {output_path}")
    plt.close()


def main():
    """主函数"""

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config.yaml'

    print("=" * 80)
    print("生成MIMO扩展性关键图 (专家建议)")
    print("=" * 80)

    if not os.path.exists(config_path):
        print(f"\n✗ 配置文件未找到: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:  # 添加encoding
        config = yaml.safe_load(f)

    validate_config(config)

    # 生成4张图
    try:
        generate_fig1_mimo_comm_scaling(config)
    except Exception as e:
        print(f"\n✗ 图1失败: {e}")

    try:
        generate_fig2_mimo_sense_scaling(config)
    except Exception as e:
        print(f"\n✗ 图2失败: {e}")

    try:
        generate_fig3_whittle_convergence(config)
    except Exception as e:
        print(f"\n✗ 图3失败: {e}")

    try:
        generate_fig4_squint_validity(config)
    except Exception as e:
        print(f"\n✗ 图4失败: {e}")

    print("\n" + "=" * 80)
    print("✓ 完成! 请查看 figures/ 目录")
    print("=" * 80)


if __name__ == "__main__":
    main()