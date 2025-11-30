#!/usr/bin/env python3
"""
Multi-Variate Analysis for THz-ISL MIMO ISAC
"Family of Curves" Generator for IEEE Transactions

功能：
1. 生成不同阵列规模 (Nt) 下的 Pareto 边界族
2. 生成不同硬件质量 (Hardware Quality) 下的 Pareto 边界族
3. 生成 IEEE 风格的高信息密度对比图

Usage:
    python multi_variate_analysis.py [config.yaml]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
import copy
import io
from pathlib import Path

# 防止中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入核心引擎
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB
except ImportError as e:
    print(f"❌ Error importing engines: {e}")
    sys.exit(1)


def setup_ieee_style():
    """IEEE 期刊绘图风格"""
    plt.rcParams.update({
        'figure.figsize': (3.5, 2.625),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'font.size': 8,
        'axes.labelsize': 8,
        'legend.fontsize': 7,
        'lines.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    return {
        'blue': '#0072BD', 'orange': '#D95319', 'green': '#77AC30',
        'purple': '#7E2F8E', 'red': '#A2142F', 'cyan': '#4DBEEE'
    }


def run_pareto_for_config(config, label, tag):
    """为特定配置计算完整的 Pareto 边界"""
    alpha_vec = np.linspace(0.05, 0.30, 15)  # 聚焦于感兴趣区域
    results = []

    # 预计算固定参数
    SNR0_db = config['simulation'].get('SNR0_db_fixed', 20.0)
    c_mps = config['channel']['c_mps']

    for alpha in alpha_vec:
        cfg = copy.deepcopy(config)
        cfg['isac_model']['alpha'] = alpha

        try:
            # 物理计算
            g = calc_g_sig_factors(cfg)
            n = calc_n_f_vector(cfg, g)

            # 性能极限
            c_res = calc_C_J(cfg, g, n, [SNR0_db], compute_C_G=False)
            b_res = calc_BCRLB(cfg, g, n)

            # 提取指标
            R_net = (1 - alpha) * c_res['C_J_vec'][0]
            RMSE_m = np.sqrt(b_res['BCRLB_tau']) * c_mps / 2.0

            results.append({
                'alpha': alpha,
                'R_net': R_net,
                'RMSE_mm': RMSE_m * 1000.0,
                'label': label,
                'tag': tag
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def sweep_array_size(base_config, colors, output_dir):
    """
    场景 A: 阵列规模扫描 (Array Size Sweep)
    展示 Nt 对 Pareto 边界的影响
    """
    print("\n[Scenario A] Sweeping Array Sizes...")

    Nt_values = [16, 64, 256]  # 对数级增长
    all_data = []

    # 颜色映射
    color_map = {16: colors['blue'], 64: colors['orange'], 256: colors['purple']}

    fig, ax = plt.subplots()

    for Nt in Nt_values:
        print(f"  Simulating Nt = Nr = {Nt}...")
        cfg = copy.deepcopy(base_config)
        cfg['array']['Nt'] = Nt
        cfg['array']['Nr'] = Nt  # 假设对称阵列

        df = run_pareto_for_config(cfg, label=f'$N={Nt}$', tag=Nt)
        all_data.append(df)

        # 绘图
        ax.plot(df['RMSE_mm'], df['R_net'], 'o-',
                label=f'$N_t=N_r={Nt}$',
                color=color_map[Nt], markersize=3)

        # 标注 alpha 方向
        # ax.arrow(...) 可以添加箭头指示 alpha 增大方向

    ax.set_xlabel('Range RMSE (mm) [Log Scale]')
    ax.set_ylabel('Net Rate $R_{net}$ (bits/s/Hz)')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left')

    # 添加物理洞察标注
    ax.text(0.95, 0.95, "Larger Array $\\rightarrow$\nBetter Sensing & Comms",
            transform=ax.transAxes, ha='right', va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_multi_array_size.pdf')
    plt.savefig(output_dir / 'fig_multi_array_size.png')
    print("  ✓ Saved fig_multi_array_size")


def sweep_hardware_quality(base_config, colors, output_dir):
    """
    场景 B: 硬件质量扫描 (Hardware Quality Sweep)
    展示 硬件参数 对 Pareto 边界的压缩作用
    """
    print("\n[Scenario B] Sweeping Hardware Quality...")

    scenarios = {
        'Ideal': {'pa': 0.0, 'adc': 100, 'iq': -100, 'pn': 0.0},  # 近似理想
        'SoA': {'pa': 0.005, 'adc': 10, 'iq': -40, 'pn': 0.01},  # 高端
        'Low-Cost': {'pa': 0.05, 'adc': 6, 'iq': -20, 'pn': 0.1}  # 低端
    }

    scenario_colors = {'Ideal': 'black', 'SoA': colors['green'], 'Low-Cost': colors['red']}
    line_styles = {'Ideal': '--', 'SoA': '-', 'Low-Cost': '-.'}

    fig, ax = plt.subplots()

    for name, params in scenarios.items():
        print(f"  Simulating {name} Hardware...")
        cfg = copy.deepcopy(base_config)

        # 应用硬件参数
        cfg['hardware']['gamma_pa_floor'] = params['pa']
        cfg['hardware']['gamma_adc_bits'] = params['adc']
        cfg['hardware']['gamma_iq_irr_dbc'] = params['iq']
        cfg['pn_model']['sigma_rel_sq_rad2'] = params['pn']

        df = run_pareto_for_config(cfg, label=name, tag=name)

        ax.plot(df['RMSE_mm'], df['R_net'],
                linestyle=line_styles[name],
                marker='o' if name != 'Ideal' else None,
                label=name, color=scenario_colors[name], markersize=3)

    ax.set_xlabel('Range RMSE (mm) [Log Scale]')
    ax.set_ylabel('Net Rate $R_{net}$ (bits/s/Hz)')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    # 标注 Gap
    ax.text(0.05, 0.1, "Hardware Gap:\nLow-Cost suffers\nsevere degradation",
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7, color='red')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_multi_hardware.pdf')
    plt.savefig(output_dir / 'fig_multi_hardware.png')
    print("  ✓ Saved fig_multi_hardware")


def main():
    print("=" * 80)
    print("MULTI-VARIATE ANALYSIS: FAMILY OF CURVES GENERATOR")
    print("=" * 80)

    colors = setup_ieee_style()
    output_dir = Path('figures_multi')
    output_dir.mkdir(exist_ok=True)

    # Load Config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Run Sweeps
    try:
        sweep_array_size(config, colors, output_dir)
        sweep_hardware_quality(config, colors, output_dir)
        print("\n✓ All multi-variate figures generated in 'figures_multi/'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()