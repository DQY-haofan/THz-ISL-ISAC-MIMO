#!/usr/bin/env python3
"""
改进版补充图生成脚本 - 修复显示问题

修复:
1. Gamma分解图 - 使用对数坐标或插入小图显示ADC/LO
2. α策略消融图 - 扩大α范围和改进可视化

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

try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_C_J, calc_BCRLB
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)


def generate_figure_C_gamma_breakdown(config, output_dir='figures'):
    """
    改进版图C: Γ分解 - 修复ADC/LO不可见问题

    改进:
    1. 主图显示所有组件
    2. 使用嵌入式放大图显示小组件
    3. 添加百分比和绝对值标注
    """
    print("\n" + "=" * 80)
    print("FIGURE C (IMPROVED): Gamma Breakdown with Inset")
    print("=" * 80)

    # 计算Gamma
    cfg = copy.deepcopy(config)
    cfg['isac_model']['alpha'] = 0.1

    g_factors = calc_g_sig_factors(cfg)
    n_outputs = calc_n_f_vector(cfg, g_factors)

    # 提取组件
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

    # === 创建图表（主图+嵌入放大图） ===
    fig = plt.figure(figsize=(10, 7))

    # 主图
    ax_main = plt.subplot(111)

    components = list(gamma_components.keys())
    values = [gamma_components[c] for c in components]
    percentages = [gamma_percentages[c] for c in components]

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']
    bars = ax_main.bar(range(len(components)), values, color=colors,
                       alpha=0.8, edgecolor='black', linewidth=1.5)

    # 添加百分比标签（在柱子上方或内部）
    for i, (bar, pct, val) in enumerate(zip(bars, percentages, values)):
        height = bar.get_height()

        # 主要标签（百分比）
        if pct > 10:  # 大的组件，标签在内部
            ax_main.text(bar.get_x() + bar.get_width() / 2., height / 2,
                         f'{pct:.1f}%',
                         ha='center', va='center',
                         fontsize=14, fontweight='bold', color='white')
        else:  # 小的组件，标签在上方
            ax_main.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{pct:.2f}%',
                         ha='center', va='bottom',
                         fontsize=10, fontweight='bold')

        # 绝对值标签（在底部）
        ax_main.text(bar.get_x() + bar.get_width() / 2., 0,
                     f'{val:.2e}',
                     ha='center', va='bottom',
                     fontsize=8, rotation=0)

    ax_main.set_xticks(range(len(components)))
    ax_main.set_xticklabels(components, fontsize=14, fontweight='bold')
    ax_main.set_ylabel(r'Hardware Distortion $\Gamma$ (linear)',
                       fontsize=13, fontweight='bold')
    ax_main.set_title(r'Gamma Breakdown at $\alpha=0.1$',
                      fontsize=15, fontweight='bold', pad=15)
    ax_main.grid(True, alpha=0.3, axis='y', linestyle='--')

    # === 嵌入放大图（显示ADC和LO） ===
    # 找到ADC和LO的索引
    small_components = ['ADC', 'LO']
    small_indices = [components.index(c) for c in small_components if c in components]

    if len(small_indices) >= 2:
        # 创建嵌入子图（放大显示小组件）
        ax_inset = fig.add_axes([0.55, 0.55, 0.35, 0.35])  # [left, bottom, width, height]

        small_values = [values[i] for i in small_indices]
        small_colors = [colors[i] for i in small_indices]
        small_labels = [components[i] for i in small_indices]
        small_pcts = [percentages[i] for i in small_indices]

        bars_inset = ax_inset.bar(range(len(small_labels)), small_values,
                                  color=small_colors, alpha=0.8,
                                  edgecolor='black', linewidth=1.5)

        # 添加标签
        for bar, pct, val in zip(bars_inset, small_pcts, small_values):
            height = bar.get_height()
            ax_inset.text(bar.get_x() + bar.get_width() / 2., height,
                          f'{pct:.3f}%\n{val:.2e}',
                          ha='center', va='bottom',
                          fontsize=9, fontweight='bold')

        ax_inset.set_xticks(range(len(small_labels)))
        ax_inset.set_xticklabels(small_labels, fontsize=11, fontweight='bold')
        ax_inset.set_ylabel(r'$\Gamma$ (linear)', fontsize=10)
        ax_inset.set_title('Magnified View', fontsize=11, fontweight='bold')
        ax_inset.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 添加连接线指示放大区域
        from matplotlib.patches import ConnectionPatch
        # 从主图的ADC/LO区域连接到嵌入图
        con1 = ConnectionPatch(xyA=(small_indices[0], values[small_indices[0]]),
                               coordsA=ax_main.transData,
                               xyB=(0, max(small_values)), coordsB=ax_inset.transData,
                               color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        fig.add_artist(con1)

    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_gamma_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Figure saved: {output_path}")

    plt.close()
    return True


def generate_figure_A_alpha_policy(config, output_dir='figures'):
    """
    改进版图A: α策略消融 - 扩大范围和改进显示

    改进:
    1. 扩大α范围: 0.01-0.50（而不是0.05-0.30）
    2. 使用对数坐标显示RMSE（更清楚）
    3. 标注关键点
    """
    print("\n" + "=" * 80)
    print("FIGURE A (IMPROVED): Alpha Policy Ablation - Extended Range")
    print("=" * 80)

    # 扩大α范围
    alpha_vec = np.linspace(0.01, 0.50, 30)  # 更大范围，更多点

    results = {
        'CONST_POWER': {'R_net': [], 'RMSE': [], 'alpha': []},
        'CONST_ENERGY': {'R_net': [], 'RMSE': [], 'alpha': []}
    }

    for policy in ['CONST_POWER', 'CONST_ENERGY']:
        print(f"\n[{policy}] Running extended alpha sweep...")

        for i, alpha in enumerate(alpha_vec):
            try:
                cfg = copy.deepcopy(config)
                cfg['isac_model']['alpha'] = alpha
                cfg['isac_model']['alpha_model'] = policy

                g_factors = calc_g_sig_factors(cfg)
                n_outputs = calc_n_f_vector(cfg, g_factors)

                c_j_results = calc_C_J(cfg, g_factors, n_outputs,
                                       [cfg['simulation']['SNR0_db_fixed']],
                                       compute_C_G=False)
                bcrlb_results = calc_BCRLB(cfg, g_factors, n_outputs)

                R_net = (1 - alpha) * c_j_results['C_sat']
                RMSE = np.sqrt(bcrlb_results['BCRLB_tau']) * cfg['channel']['c_mps']

                results[policy]['alpha'].append(alpha)
                results[policy]['R_net'].append(R_net)
                results[policy]['RMSE'].append(RMSE * 1000)  # to mm

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(alpha_vec)}")

            except Exception as e:
                print(f"  Warning: Failed at α={alpha:.3f}: {e}")
                continue

    # === 生成改进后的图表 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: R_net vs α (线性坐标)
    ax = ax1
    for policy, color, marker in [('CONST_POWER', 'blue', 'o'),
                                  ('CONST_ENERGY', 'red', 's')]:
        data = results[policy]
        if len(data['alpha']) > 0:
            ax.plot(data['alpha'], data['R_net'], marker=marker, markersize=5,
                    linewidth=2.5, label=policy, color=color, alpha=0.8,
                    markevery=3)  # 每3个点标记一次

            # 标出最优点
            idx_max = np.argmax(data['R_net'])
            ax.plot(data['alpha'][idx_max], data['R_net'][idx_max],
                    '*', markersize=18, color=color,
                    markeredgecolor='black', markeredgewidth=1.5)

            # 添加标注
            ax.annotate(f"α*={data['alpha'][idx_max]:.3f}",
                        xy=(data['alpha'][idx_max], data['R_net'][idx_max]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=11, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor=color))

    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Net Rate $R_{\mathrm{net}}$ (bits/s/Hz)', fontsize=14, fontweight='bold')
    ax.set_title('(a) Communication Performance', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 0.5])

    # 子图2: RMSE vs α (对数坐标)
    ax = ax2
    for policy, color, marker in [('CONST_POWER', 'blue', 'o'),
                                  ('CONST_ENERGY', 'red', 's')]:
        data = results[policy]
        if len(data['alpha']) > 0:
            ax.semilogy(data['alpha'], data['RMSE'], marker=marker, markersize=5,
                        linewidth=2.5, label=policy, color=color, alpha=0.8,
                        markevery=3)

            # 标出最优点（最小RMSE）
            idx_min = np.argmin(data['RMSE'])
            ax.plot(data['alpha'][idx_min], data['RMSE'][idx_min],
                    '*', markersize=18, color=color,
                    markeredgecolor='black', markeredgewidth=1.5)

            # 添加标注
            ax.annotate(f"α={data['alpha'][idx_min]:.3f}\nRMSE={data['RMSE'][idx_min]:.0f}mm",
                        xy=(data['alpha'][idx_min], data['RMSE'][idx_min]),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=10, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor=color))

    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Range RMSE (mm, log scale)', fontsize=14, fontweight='bold')
    ax.set_title('(b) Sensing Performance', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim([0, 0.5])

    # 添加文本说明
    textstr = 'Log scale used\nto show detail'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_ablation_alpha_policy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Figure saved: {output_path}")

    # 保存数据
    for policy in ['CONST_POWER', 'CONST_ENERGY']:
        df = pd.DataFrame(results[policy])
        csv_path = os.path.join('results', f'fig_A_{policy}_data.csv')
        os.makedirs('results', exist_ok=True)
        df.to_csv(csv_path, index=False)

    plt.close()
    return True


def main():
    """主函数"""

    print("=" * 80)
    print("IMPROVED SUPPLEMENTARY FIGURES GENERATOR")
    print("修复Gamma和α策略图的显示问题")
    print("=" * 80)

    # 加载配置
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

    # 生成改进后的图
    success_count = 0

    try:
        if generate_figure_C_gamma_breakdown(config):
            success_count += 1
    except Exception as e:
        print(f"\n✗ Figure C failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        if generate_figure_A_alpha_policy(config):
            success_count += 1
    except Exception as e:
        print(f"\n✗ Figure A failed: {e}")
        import traceback
        traceback.print_exc()

    # 总结
    print("\n" + "=" * 80)
    print(f"COMPLETE: {success_count}/2 improved figures generated")
    print("=" * 80)

    if success_count == 2:
        print("\n✓ All improved figures generated successfully!")
        print("\n改进点:")
        print("  1. Gamma图: 添加了放大嵌入图显示ADC/LO")
        print("  2. α策略图: 扩大范围(0.01-0.50)，使用对数坐标")
        sys.exit(0)
    else:
        print(f"\n⚠ Only {success_count}/2 figures generated")
        sys.exit(1)


if __name__ == "__main__":
    main()