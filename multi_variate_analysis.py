#!/usr/bin/env python3
"""
Multi-Variate Analysis for THz-ISL MIMO ISAC
"Family of Curves" Generator for IEEE Transactions

功能：
1. 生成不同阵列规模 (Nt) 下的 Pareto 边界族
2. 生成不同硬件质量 (Hardware Quality) 下的 Pareto 边界族
3. 生成不同带宽 (Bandwidth) 下的 Pareto 边界族 (New!)
4. 自动保存 PNG, PDF 和 CSV 数据

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
import os
from pathlib import Path

# 防止 Windows 控制台中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入核心物理引擎
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB
except ImportError as e:
    print(f"❌ Error importing engines: {e}")
    print("   Please ensure physics_engine.py and limits_engine.py are in the same directory.")
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

def run_pareto_for_config(config, label, tag_col_name, tag_value):
    """为特定配置计算完整的 Pareto 边界"""
    alpha_vec = np.linspace(0.05, 0.30, 20)  # 聚焦于感兴趣区域
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
                'R_net_bps_hz': R_net,
                'RMSE_mm': RMSE_m * 1000.0,
                'label': label,
                tag_col_name: tag_value
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def save_plot_and_data(fig, df, filename_base, output_dir):
    """同时保存 PNG, PDF 和 CSV"""
    # Save CSV
    csv_path = output_dir / f"{filename_base}.csv"
    df.to_csv(csv_path, index=False)

    # Save Figures
    png_path = output_dir / f"{filename_base}.png"
    pdf_path = output_dir / f"{filename_base}.pdf"

    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)

    print(f"  ✓ Saved: {filename_base} (.png, .pdf, .csv)")


def sweep_array_size(base_config, colors, output_dir):
    """
    场景 A: 阵列规模扫描 (Nt Scaling)
    展示: 增大 N 对感知收益巨大，但对通信收益受限（甚至受损）
    """
    print("\n[Scenario A] Sweeping Array Sizes (Nt)...")

    Nt_values = [16, 64, 256]
    all_dfs = []

    color_map = {16: colors['blue'], 64: colors['orange'], 256: colors['purple']}

    fig, ax = plt.subplots()

    for Nt in Nt_values:
        print(f"  Simulating Nt = Nr = {Nt}...")
        cfg = copy.deepcopy(base_config)
        cfg['array']['Nt'] = Nt
        cfg['array']['Nr'] = Nt

        # 调整 N (观测点数) 以保持处理增益一致性？(可选，这里保持 N 不变以控制变量)

        df = run_pareto_for_config(cfg, label=f'$N={Nt}$', tag_col_name='Nt', tag_value=Nt)
        if not df.empty:
            all_dfs.append(df)

            ax.plot(df['RMSE_mm'], df['R_net_bps_hz'], 'o-',
                    label=f'$N_t=N_r={Nt}$',
                    color=color_map[Nt], markersize=3, linewidth=1.2)

    ax.set_xlabel('Range RMSE (mm) [Log Scale]')
    ax.set_ylabel('Net Rate $R_{net}$ (bits/s/Hz)')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left')



    if all_dfs:
        full_df = pd.concat(all_dfs)
        save_plot_and_data(fig, full_df, 'fig_multi_array_size', output_dir)
    plt.close()


def sweep_hardware_quality(base_config, colors, output_dir):
    """
    场景 B: 硬件质量扫描 (Hardware Sensitivity)
    展示: 廉价硬件如何显著压缩 Pareto 前沿
    """
    print("\n[Scenario B] Sweeping Hardware Quality...")

    scenarios = {
        'Ideal': {'pa': 0.0, 'adc': 100, 'iq': -120, 'pn': 0.0},  # 近似完美
        'SoA': {'pa': 0.005, 'adc': 10, 'iq': -40, 'pn': 0.01},  # 高端商用
        'Low-Cost': {'pa': 0.05, 'adc': 6, 'iq': -20, 'pn': 0.1}  # 低成本/立方星
    }

    scenario_colors = {'Ideal': 'black', 'SoA': colors['green'], 'Low-Cost': colors['red']}
    line_styles = {'Ideal': '--', 'SoA': '-', 'Low-Cost': '-.'}

    fig, ax = plt.subplots()
    all_dfs = []

    for name, params in scenarios.items():
        print(f"  Simulating {name} Hardware...")
        cfg = copy.deepcopy(base_config)

        cfg['hardware']['gamma_pa_floor'] = params['pa']
        cfg['hardware']['gamma_adc_bits'] = params['adc']
        cfg['hardware']['gamma_iq_irr_dbc'] = params['iq']
        cfg['pn_model']['sigma_rel_sq_rad2'] = params['pn']

        df = run_pareto_for_config(cfg, label=name, tag_col_name='Quality', tag_value=name)
        if not df.empty:
            all_dfs.append(df)

            ax.plot(df['RMSE_mm'], df['R_net_bps_hz'],
                    linestyle=line_styles[name],
                    marker='o' if name != 'Ideal' else None,
                    label=name, color=scenario_colors[name], markersize=3)

    ax.set_xlabel('Range RMSE (mm) [Log Scale]')
    ax.set_ylabel('Net Rate $R_{net}$ (bits/s/Hz)')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()



    if all_dfs:
        full_df = pd.concat(all_dfs)
        save_plot_and_data(fig, full_df, 'fig_multi_hardware', output_dir)
    plt.close()


def sweep_bandwidth(base_config, colors, output_dir):
    """
    场景 C: 带宽扫描 (Bandwidth Trade-off)
    展示: B 增大 -> RMSE 改善, 但 Beam Squint 导致 Rate 下降
    """
    print("\n[Scenario C] Sweeping Bandwidth (B)...")

    B_values = [5e9, 10e9, 20e9]  # 5, 10, 20 GHz
    B_labels = ["5 GHz", "10 GHz", "20 GHz"]

    color_map = {5e9: colors['blue'], 10e9: colors['green'], 20e9: colors['red']}

    fig, ax = plt.subplots()
    all_dfs = []

    for B, label in zip(B_values, B_labels):
        print(f"  Simulating B = {label}...")
        cfg = copy.deepcopy(base_config)
        cfg['channel']['B_hz'] = float(B)

        # 注意: 改变 B 时保持 N 不变意味着频率分辨率改变

        df = run_pareto_for_config(cfg, label=label, tag_col_name='Bandwidth_Hz', tag_value=B)
        if not df.empty:
            all_dfs.append(df)

            ax.plot(df['RMSE_mm'], df['R_net_bps_hz'], 'o-',
                    label=label,
                    color=color_map[B], markersize=3)

    ax.set_xlabel('Range RMSE (mm) [Log Scale]')
    ax.set_ylabel('Net Rate $R_{net}$ (bits/s/Hz)')
    ax.set_xscale('log')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left')



    if all_dfs:
        full_df = pd.concat(all_dfs)
        save_plot_and_data(fig, full_df, 'fig_multi_bandwidth', output_dir)
    plt.close()


def main():
    print("=" * 80)
    print("MULTI-VARIATE ANALYSIS: FAMILY OF CURVES GENERATOR")
    print("IEEE Transactions Grade Visualization")
    print("=" * 80)

    colors = setup_ieee_style()
    output_dir = Path('figures_multi')
    output_dir.mkdir(exist_ok=True)

    # Load Config with UTF-8 encoding fix
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    try:
        # ✅ 关键修正: 指定 encoding='utf-8'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    # Run Sweeps
    try:
        sweep_array_size(config, colors, output_dir)
        sweep_hardware_quality(config, colors, output_dir)
        sweep_bandwidth(config, colors, output_dir)
        print(f"\n✓ Analysis Complete. Outputs saved to '{output_dir}/'")
        print("  - fig_multi_array_size.{png,pdf,csv}")
        print("  - fig_multi_hardware.{png,pdf,csv}")
        print("  - fig_multi_bandwidth.{png,pdf,csv}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()