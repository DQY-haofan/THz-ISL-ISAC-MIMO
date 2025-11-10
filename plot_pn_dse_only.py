#!/usr/bin/env python3
"""
PN-DSE Crossover独立绘图脚本
直接读取pn_dse_crossover.csv并绘制IEEE风格图

不依赖visualize_results.py
专门用于绘制PN-DSE交叉图

Usage:
    python plot_pn_dse_only.py --csv results/pn_dse_crossover.csv --output figures

Author: IEEE Style
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys


def setup_ieee_style():
    """IEEE出版风格"""
    plt.rcParams.update({
        'figure.figsize': (3.5, 2.625),
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.linewidth': 0.5,
        'grid.alpha': 0.3,
        'text.usetex': False,
    })


def find_crossover(df):
    """找到PN和DSE的交叉点"""

    alpha = df['alpha'].values
    pn = df['sigma2_pn_res'].values
    dse = df['sigma2_DSE'].values

    # 找到最接近的点
    ratio = dse / pn
    dist = np.abs(ratio - 1.0)
    idx = np.argmin(dist)

    return alpha[idx], pn[idx], dse[idx], ratio[idx]


def plot_pn_dse(csv_path, output_dir='./figures/'):
    """
    绘制PN-DSE crossover图

    Args:
        csv_path: pn_dse_crossover.csv路径
        output_dir: 输出目录
    """

    print("=" * 80)
    print("PN-DSE CROSSOVER PLOT (IEEE Style)")
    print("=" * 80)

    setup_ieee_style()

    # 读取数据
    print(f"\n[1/3] Loading data: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"  ✗ File not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} points")

    # 检查列名
    required_cols = ['alpha', 'sigma2_pn_res', 'sigma2_DSE']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return False

    # 找交叉点
    alpha_cross, pn_cross, dse_cross, ratio = find_crossover(df)
    print(f"\n  Crossover point:")
    print(f"    α* = {alpha_cross:.4f}")
    print(f"    PN = {pn_cross:.3e} rad²")
    print(f"    DSE = {dse_cross:.3e} rad²")
    print(f"    Ratio = {ratio:.3f}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 绘图
    print(f"\n[2/3] Generating plot...")

    fig, ax = plt.subplots()

    # 颜色
    color_pn = '#8B008B'  # Purple
    color_dse = '#FF8C00'  # Orange
    color_cross = '#A2142F'  # Red

    # 绘制PN曲线
    ax.plot(df['alpha'], df['sigma2_pn_res'],
            marker='o', markersize=3, linewidth=1.0,
            color=color_pn, label=r'$\sigma^2_{\phi,c,\mathrm{res}}$ (PN)',
            markeredgecolor='black', markeredgewidth=0.3)

    # 绘制DSE曲线
    ax.plot(df['alpha'], df['sigma2_DSE'],
            marker='s', markersize=3, linewidth=1.0,
            color=color_dse, label=r'$\sigma^2_{\mathrm{DSE}}$ (DSE)',
            markeredgecolor='black', markeredgewidth=0.3)

    # 标记交叉点
    ax.axvline(x=alpha_cross, color=color_cross, linestyle='--',
               linewidth=1.0, alpha=0.7)
    ax.plot(alpha_cross, pn_cross, '*', markersize=10,
            color=color_cross, markeredgecolor='darkred', markeredgewidth=0.5)

    # 标注交叉点
    ax.text(alpha_cross + 0.02, pn_cross * 0.3,
            f'$\\alpha^* = {alpha_cross:.3f}$',
            color=color_cross, fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      edgecolor=color_cross,
                      linewidth=0.8,
                      alpha=0.95))

    # 设置
    ax.set_xlabel(r'ISAC Overhead ($\alpha$)', fontsize=8)
    ax.set_ylabel(r'Variance (rad$^2$)', fontsize=8)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    # 保存
    print(f"\n[3/3] Saving figures...")
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'fig_pn_dse_crossover.{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")

    plt.close()

    print(f"\n✓ Plot generation complete")
    return True


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description='Plot PN-DSE Crossover (IEEE Style)'
    )
    parser.add_argument('--csv', default='results/pn_dse_crossover.csv',
                        help='CSV file with PN-DSE data')
    parser.add_argument('--output', default='figures',
                        help='Output directory for figures')

    args = parser.parse_args()

    success = plot_pn_dse(args.csv, args.output)

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)
        print(f"  Figures saved to: {args.output}/")
        print(f"  - fig_pn_dse_crossover.png")
        print(f"  - fig_pn_dse_crossover.pdf")
        return 0
    else:
        print("\n" + "=" * 80)
        print("FAILED")
        print("=" * 80)
        print("  Check error messages above")
        return 1


if __name__ == "__main__":
    sys.exit(main())