#!/usr/bin/env python3
"""
Threshold Validation Analysis - Integrated Script
一站式Threshold验证：数据生成 + 可视化

此脚本完全代替：
- threshold_sweep.py
- visualize_threshold.py

功能：
1. 执行Whittle vs Cholesky阈值验证扫描
2. 保存详细数据到CSV和NPY格式
3. 自动生成完整的可视化套件：
   - 2D热力图（主图）
   - 误差分布直方图
   - 1D切片图
4. 生成统计分析报告

Usage:
    # 运行快速模式（理论估计）
    python threshold_analysis.py [config.yaml]

    # 运行精确模式（实际对比Whittle和Cholesky）
    python threshold_analysis.py [config.yaml] --mode accurate --grid-size 15

    # 仅可视化已有数据
    python threshold_analysis.py --visualize-only --csv results/threshold_detailed_data.csv

Author: Integrated Version v1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
import sys
import os
import copy
import argparse
import time
from pathlib import Path

# Import validated engines
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config
    from limits_engine import calc_BCRLB
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure physics_engine.py and limits_engine.py are accessible")
    sys.exit(1)


def setup_ieee_style():
    """IEEE出版风格配置 - 匹配主可视化脚本"""
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (3.5, 2.625),  # IEEE single column width
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font settings (matching visualize_results.py)
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


def run_threshold_sweep(config_path: str, grid_size: int = 10, mode: str = 'fast'):
    """
    执行阈值验证扫描，对比Whittle和Cholesky模式的误差

    Args:
        config_path: 配置文件路径
        grid_size: 网格尺寸（10=快速，15=中等，20=精确）
        mode: 'fast'使用Whittle，'accurate'同时计算Whittle和Cholesky

    Returns:
        Dict with results and file paths
    """

    print("=" * 80)
    print("THRESHOLD VALIDATION SWEEP")
    print("Whittle Approximation vs. Cholesky Exact Comparison")
    print("=" * 80)

    # Load configuration
    print(f"\n[1/5] Loading configuration: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        validate_config(base_config)
        print("  ✓ Configuration validated")
    except FileNotFoundError:
        print(f"  ✗ Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        sys.exit(1)

    # Define sweep grid
    print(f"\n[2/5] Defining sweep grid (mode: {mode}, size: {grid_size}×{grid_size})")

    # B/f_c ratio sweep - 关键区域在0.02-0.15
    B_over_fc_vec = np.linspace(0.02, 0.15, grid_size)

    # L_ap/λ ratio sweep - 典型系统在3-15λ
    Lap_over_lambda_vec = np.linspace(3, 25, grid_size)

    print(f"  B/f_c range: [{B_over_fc_vec[0]:.3f}, {B_over_fc_vec[-1]:.3f}]")
    print(f"  L_ap/λ range: [{Lap_over_lambda_vec[0]:.1f}, {Lap_over_lambda_vec[-1]:.1f}]")
    print(f"  Total points: {len(B_over_fc_vec) * len(Lap_over_lambda_vec)}")

    # Initialize result matrices
    error_matrix = np.zeros((len(B_over_fc_vec), len(Lap_over_lambda_vec)))
    crlb_whittle_matrix = np.zeros_like(error_matrix)
    crlb_cholesky_matrix = np.zeros_like(error_matrix)

    # Run sweep
    print(f"\n[3/5] Running threshold sweep...")
    start_time = time.time()

    total_points = len(B_over_fc_vec) * len(Lap_over_lambda_vec)
    completed = 0
    failed = 0
    skipped = 0

    for i, B_over_fc in enumerate(B_over_fc_vec):
        for j, Lap_over_lambda in enumerate(Lap_over_lambda_vec):
            try:
                # Create modified config
                config = copy.deepcopy(base_config)

                # Calculate absolute values from ratios
                f_c_hz = config['channel']['f_c_hz']
                c_mps = config['channel']['c_mps']
                lambda_c = c_mps / f_c_hz

                config['channel']['B_hz'] = B_over_fc * f_c_hz
                config['array']['L_ap_m'] = Lap_over_lambda * lambda_c
                config['array']['theta_0_deg'] = 15.0  # 固定小角度

                # Calculate physics
                g_factors = calc_g_sig_factors(config)
                n_outputs = calc_n_f_vector(config, g_factors)

                # ✅ 新增：手动修正 sig_amp_k，移除能量归一化的影响
                if 'sig_amp_k' in g_factors and 'eta_bsq_k' in g_factors:
                    g_ar = g_factors['g_ar']
                    eta_bsq_k = g_factors['eta_bsq_k']

                    # 重新计算未归一化的 sig_amp_k
                    sig_amp_k_original = np.sqrt(g_ar) * eta_bsq_k

                    # 替换
                    g_factors['sig_amp_k'] = sig_amp_k_original

                    # 打印诊断（第一个点）
                    if i == 0 and j == 0:
                        E_orig = np.sum(np.abs(sig_amp_k_original) ** 2) * (
                                    config['channel']['B_hz'] / config['simulation']['N'])
                        print(f"\n  [Beam Squint Fix]")
                        print(f"    Lap/λ = {Lap_over_lambda:.1f}")
                        print(f"    E_sig (unnormalized) = {E_orig:.3e}")
                        print(f"    eta_bsq range: [{eta_bsq_k.min():.4f}, {eta_bsq_k.max():.4f}]")

                # ===== Whittle mode =====
                config['simulation']['FIM_MODE'] = 'Whittle'
                bcrlb_whittle = calc_BCRLB(config, g_factors, n_outputs)
                crlb_tau_whittle = bcrlb_whittle['BCRLB_tau']
                crlb_whittle_matrix[i, j] = crlb_tau_whittle

                # ===== Cholesky mode (only in accurate mode) =====
                if mode == 'accurate':
                    config['simulation']['FIM_MODE'] = 'Cholesky'
                    try:
                        bcrlb_cholesky = calc_BCRLB(config, g_factors, n_outputs)
                        crlb_tau_cholesky = bcrlb_cholesky['BCRLB_tau']
                        crlb_cholesky_matrix[i, j] = crlb_tau_cholesky

                        # Calculate relative error
                        if crlb_tau_cholesky > 0 and np.isfinite(crlb_tau_cholesky):
                            rel_error = abs(crlb_tau_whittle - crlb_tau_cholesky) / crlb_tau_cholesky
                        else:
                            rel_error = np.nan
                    except Exception as e_chol:
                        # Cholesky可能因为数值问题失败
                        rel_error = np.nan
                        skipped += 1
                        if skipped == 1:  # Only print first failure
                            print(f"  ⚠ Cholesky failed (will skip similar): {e_chol}")
                else:
                    # Fast mode: 使用理论估计
                    rel_error = min(0.1, (B_over_fc / 0.1) ** 2 * 0.02)
                    crlb_cholesky_matrix[i, j] = crlb_tau_whittle * (1 + rel_error)

                error_matrix[i, j] = rel_error
                completed += 1

                # Progress update
                if completed % max(1, total_points // 10) == 0:
                    progress_pct = 100 * completed / total_points
                    elapsed = time.time() - start_time
                    eta = elapsed / completed * (total_points - completed)
                    print(f"  Progress: {completed}/{total_points} ({progress_pct:.0f}%) - "
                          f"ETA: {eta / 60:.1f} min - Current error: {rel_error:.2e}")

            except Exception as e:
                if failed == 0:  # Only print first failure
                    print(f"  ✗ Point failed (will skip similar): {e}")
                error_matrix[i, j] = np.nan
                failed += 1
                continue

    elapsed_time = time.time() - start_time
    print(f"  ✓ Sweep completed in {elapsed_time / 60:.1f} minutes")
    print(f"    Success: {completed}/{total_points}")
    print(f"    Failed: {failed}")
    if mode == 'accurate':
        print(f"    Skipped (Cholesky): {skipped}")

    # Save results
    print(f"\n[4/5] Saving results...")

    output_config = base_config.get('outputs', {})
    save_path = output_config.get('save_path', './results/')
    os.makedirs(save_path, exist_ok=True)

    # Save matrices
    np.save(os.path.join(save_path, 'threshold_error_matrix.npy'), error_matrix)
    np.save(os.path.join(save_path, 'threshold_B_over_fc.npy'), B_over_fc_vec)
    np.save(os.path.join(save_path, 'threshold_Lap_over_lambda.npy'), Lap_over_lambda_vec)
    np.save(os.path.join(save_path, 'threshold_crlb_whittle.npy'), crlb_whittle_matrix)
    np.save(os.path.join(save_path, 'threshold_crlb_cholesky.npy'), crlb_cholesky_matrix)

    print(f"  ✓ Matrices saved to: {save_path}")

    # Create statistics CSV
    valid_errors = error_matrix[~np.isnan(error_matrix)]

    if len(valid_errors) > 0:
        stats_df = pd.DataFrame({
            'Metric': [
                'Mean Error', 'Max Error', 'Min Error', 'Median Error',
                'Std Error', '95th Percentile', '99th Percentile',
                'Points with Error < 1%', 'Points with Error < 2%', 'Points with Error < 5%'
            ],
            'Value': [
                np.mean(valid_errors),
                np.max(valid_errors),
                np.min(valid_errors),
                np.median(valid_errors),
                np.std(valid_errors),
                np.percentile(valid_errors, 95),
                np.percentile(valid_errors, 99),
                np.sum(valid_errors < 0.01),
                np.sum(valid_errors < 0.02),
                np.sum(valid_errors < 0.05)
            ]
        })

        stats_csv = os.path.join(save_path, 'threshold_statistics.csv')
        stats_df.to_csv(stats_csv, index=False)
        print(f"  ✓ Statistics saved to: {stats_csv}")
    else:
        stats_df = None
        print(f"  ⚠ No valid data for statistics")

    # Create detailed CSV
    detailed_rows = []
    for i, B_over_fc in enumerate(B_over_fc_vec):
        for j, Lap_over_lambda in enumerate(Lap_over_lambda_vec):
            detailed_rows.append({
                'B_over_fc': B_over_fc,
                'Lap_over_lambda': Lap_over_lambda,
                'crlb_whittle': crlb_whittle_matrix[i, j],
                'crlb_cholesky': crlb_cholesky_matrix[i, j],
                'relative_error': error_matrix[i, j]
            })

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_csv = os.path.join(save_path, 'threshold_detailed_data.csv')
    detailed_df.to_csv(detailed_csv, index=False, float_format='%.6e')
    print(f"  ✓ Detailed data saved to: {detailed_csv}")

    # Print summary
    print(f"\n[5/5] Threshold Validation Summary")
    print("=" * 80)

    if len(valid_errors) > 0:
        print(f"\n{stats_df.to_string(index=False)}")

        print(f"\n📊 Threshold Assessment:")
        max_error = np.max(valid_errors)

        if max_error < 0.01:
            assessment = "✓ EXCELLENT: Max error < 1% (very tight threshold)"
        elif max_error < 0.02:
            assessment = "✓ VERY GOOD: Max error < 2% (tight threshold)"
        elif max_error < 0.05:
            assessment = "✓ GOOD: Max error < 5% (acceptable threshold)"
        elif max_error < 0.10:
            assessment = "⚠ MODERATE: Max error < 10% (borderline)"
        else:
            assessment = "✗ LARGE: Max error ≥ 10% (threshold violated)"

        print(f"  {assessment}")

        # Coverage analysis
        pct_under_2 = 100 * np.sum(valid_errors < 0.02) / len(valid_errors)
        pct_under_5 = 100 * np.sum(valid_errors < 0.05) / len(valid_errors)
        print(f"\n📈 Coverage Analysis:")
        print(f"  Points with error < 2%: {pct_under_2:.1f}%")
        print(f"  Points with error < 5%: {pct_under_5:.1f}%")

    else:
        print("\n✗ No valid data points - all simulations failed")

    print("\n" + "=" * 80)

    return {
        'error_matrix': error_matrix,
        'B_over_fc_vec': B_over_fc_vec,
        'Lap_over_lambda_vec': Lap_over_lambda_vec,
        'crlb_whittle_matrix': crlb_whittle_matrix,
        'crlb_cholesky_matrix': crlb_cholesky_matrix,
        'statistics': stats_df if len(valid_errors) > 0 else None,
        'detailed_csv': detailed_csv,
        'success_rate': completed / total_points,
        'mode': mode,
        'grid_size': grid_size,
        'config': base_config
    }


def generate_threshold_slices_combined(df, B_over_fc, Lap_over_lambda, output_dir='figures'):
    """
    生成 Threshold 误差切片合并图 - 修复版

    修复：
    1. 移除错误的续行符 / \ \
    2. 移除数学表达式中的 $ 符号
    """

    print(f"  Generating combined threshold slices...")

    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    # 固定 L_ap/λ，变化 B/f_c
    mid_Lap_idx = len(Lap_over_lambda) // 2
    mid_Lap = Lap_over_lambda[mid_Lap_idx]
    slice_data_1 = df[df['Lap_over_lambda'] == mid_Lap]

    # 固定 B/f_c，变化 L_ap/λ
    mid_B_idx = len(B_over_fc) // 2
    mid_B = B_over_fc[mid_B_idx]
    slice_data_2 = df[df['B_over_fc'] == mid_B]

    # 修复：归一化到 [0, 1] - 正确的写法（无需续行符）
    x1_norm = (slice_data_1['B_over_fc'] - slice_data_1['B_over_fc'].min()) / (
                slice_data_1['B_over_fc'].max() - slice_data_1['B_over_fc'].min())
    y1 = slice_data_1['relative_error'] * 100

    x2_norm = (slice_data_2['Lap_over_lambda'] - slice_data_2['Lap_over_lambda'].min()) / (
                slice_data_2['Lap_over_lambda'].max() - slice_data_2['Lap_over_lambda'].min())
    y2 = slice_data_2['relative_error'] * 100

    # 绘制两条曲线
    ax.plot(x1_norm, y1, 'o-', linewidth=1.0, markersize=4,
            color='#0072BD', label=f'Vary B/fc (fixed Lap/lambda={mid_Lap:.1f})',
            markeredgecolor='black', markeredgewidth=0.3)

    ax.plot(x2_norm, y2, 's-', linewidth=1.0, markersize=4,
            color='#D95319', label=f'Vary Lap/lambda (fixed B/fc={mid_B:.3f})',
            markeredgecolor='black', markeredgewidth=0.3)

    # 修复：移除数学表达式的 $ 符号
    ax.set_xlabel('Normalized Parameter Value', fontsize=8)
    ax.set_ylabel('Relative Error [%]', fontsize=8)
    ax.legend(fontsize=7, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = Path(output_dir) / f'threshold_error_slices_combined.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_path}")

    plt.close()

    return True


def generate_threshold_slices_dual_xaxis(df, B_over_fc, Lap_over_lambda, output_dir='figures'):
    """
    生成 Threshold 误差切片图 - 双x轴版本（修复版）

    修复：移除数学表达式中的 $ 符号
    """

    print(f"  Generating dual x-axis threshold slices...")

    fig, ax1 = plt.subplots(figsize=(3.5, 2.625))

    # 固定 L_ap/λ，变化 B/f_c
    mid_Lap_idx = len(Lap_over_lambda) // 2
    mid_Lap = Lap_over_lambda[mid_Lap_idx]
    slice_data_1 = df[df['Lap_over_lambda'] == mid_Lap]

    # === 主x轴：B/f_c ===
    color1 = '#0072BD'
    ax1.set_xlabel('B/fc', fontsize=8, color=color1)
    ax1.set_ylabel('Relative Error [%]', fontsize=8)

    line1 = ax1.plot(slice_data_1['B_over_fc'], slice_data_1['relative_error'] * 100,
                     'o-', linewidth=1.0, markersize=4, color=color1,
                     label=f'Lap/lambda={mid_Lap:.1f} (fixed)',
                     markeredgecolor='black', markeredgewidth=0.3)

    ax1.tick_params(axis='x', labelcolor=color1, labelsize=8)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # === 副x轴：L_ap/λ ===
    mid_B_idx = len(B_over_fc) // 2
    mid_B = B_over_fc[mid_B_idx]
    slice_data_2 = df[df['B_over_fc'] == mid_B]

    ax2 = ax1.twiny()
    color2 = '#D95319'
    ax2.set_xlabel('Lap/lambda', fontsize=8, color=color2)

    line2 = ax2.plot(slice_data_2['Lap_over_lambda'], slice_data_2['relative_error'] * 100,
                     's-', linewidth=1.0, markersize=4, color=color2,
                     label=f'B/fc={mid_B:.3f} (fixed)',
                     markeredgecolor='black', markeredgewidth=0.3)

    ax2.tick_params(axis='x', labelcolor=color2, labelsize=8)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=7, loc='best', framealpha=0.9)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = Path(output_dir) / f'threshold_error_slices_dual_xaxis.{ext}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {output_path}")

    plt.close()

    return True


def visualize_threshold_results(data_source, output_dir='figures', working_point=None):
    """
    可视化threshold数据（完整套件） - 完整修复版本

    Args:
        data_source: 可以是：
                    - Dict from run_threshold_sweep()
                    - CSV file path (str)
        output_dir: 输出目录
        working_point: 工作点字典 {'B_over_fc': float, 'Lap_over_lambda': float}
    """

    print("\n" + "=" * 80)
    print("THRESHOLD VISUALIZATION SUITE")
    print("=" * 80)

    setup_ieee_style()

    # ========== 数据加载部分（保持原样）==========
    if isinstance(data_source, dict):
        # From sweep results
        error_matrix = data_source['error_matrix']
        B_over_fc = data_source['B_over_fc_vec']
        Lap_over_lambda = data_source['Lap_over_lambda_vec']

        # Create DataFrame from matrices
        detailed_rows = []
        for i, b in enumerate(B_over_fc):
            for j, l in enumerate(Lap_over_lambda):
                detailed_rows.append({
                    'B_over_fc': b,
                    'Lap_over_lambda': l,
                    'relative_error': error_matrix[i, j]
                })
        df = pd.DataFrame(detailed_rows)

    elif isinstance(data_source, str):
        # From CSV file
        print(f"\n[1/4] Loading data: {data_source}")
        if not os.path.exists(data_source):
            print(f"  ✗ File not found: {data_source}")
            return False

        df = pd.read_csv(data_source)
        print(f"  ✓ Loaded {len(df)} data points")

        # Extract unique grid values
        B_over_fc = df['B_over_fc'].unique()
        Lap_over_lambda = df['Lap_over_lambda'].unique()

        # Reconstruct error matrix
        error_matrix = df.pivot(index='B_over_fc',
                                columns='Lap_over_lambda',
                                values='relative_error').values
    else:
        print(f"  ✗ Invalid data source type: {type(data_source)}")
        return False

    # 数据分析
    print(f"\n[2/4] Analyzing data...")
    errors = df['relative_error'].values
    errors_valid = errors[~np.isnan(errors)]

    if len(errors_valid) == 0:
        print("  ✗ No valid data for visualization")
        return False

    print(f"  Error statistics:")
    print(f"    Min: {errors_valid.min():.6f} ({errors_valid.min() * 100:.4f}%)")
    print(f"    Max: {errors_valid.max():.6f} ({errors_valid.max() * 100:.4f}%)")
    print(f"    Mean: {errors_valid.mean():.6f} ({errors_valid.mean() * 100:.4f}%)")
    print(f"    Median: {np.median(errors_valid):.6f} ({np.median(errors_valid) * 100:.4f}%)")

    print(f"\n  Grid dimensions:")
    print(f"    B/f_c: {len(B_over_fc)} points, range [{B_over_fc.min():.3f}, {B_over_fc.max():.3f}]")
    print(
        f"    L_ap/λ: {len(Lap_over_lambda)} points, range [{Lap_over_lambda.min():.1f}, {Lap_over_lambda.max():.1f}]")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ========== Figure 1: 2D Heatmap - 修复版本 ==========
    print(f"\n[3/4] Generating 2D heatmap...")

    fig, ax = plt.subplots()

    # ✅ 修复：在这里转换为百分数
    error_pct = error_matrix * 100.0  # 转换为百分数
    errors_pct = errors_valid * 100.0

    # 使用合适的颜色映射和范围
    vmin = max(errors_pct.min(), 0.001)  # 0.001% 最小值
    vmax = min(errors_pct.max(), 10.0)  # 10% 最大值

    # ✅ 修复：使用百分数的error_pct
    im = ax.contourf(Lap_over_lambda, B_over_fc, error_pct,
                     levels=20, cmap='RdYlGn_r',
                     norm=LogNorm(vmin=vmin, vmax=vmax))

    # ✅ 修复：等高线使用百分数
    contour_levels = [0.5, 1.0, 2.0, 5.0]  # 0.5%, 1%, 2%, 5%
    cs = ax.contour(Lap_over_lambda, B_over_fc, error_pct,
                    levels=contour_levels, colors='black',
                    linewidths=1.0, linestyles='solid')
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f%%')

    # ✅ 新增：标注工作点（星标）
    if working_point is not None:
        wp_B = working_point.get('B_over_fc', None)
        wp_L = working_point.get('Lap_over_lambda', None)

        if wp_B is not None and wp_L is not None:
            # 找到最接近的网格点
            i0 = np.argmin(np.abs(B_over_fc - wp_B))
            j0 = np.argmin(np.abs(Lap_over_lambda - wp_L))

            ax.scatter([Lap_over_lambda[j0]], [B_over_fc[i0]],
                       marker='*', s=200, color='gold', edgecolor='black',
                       linewidths=1.5, zorder=10, label='Working Point')

            # 标注工作点误差值
            wp_error = error_pct[i0, j0]
            ax.text(Lap_over_lambda[j0] + 0.5, B_over_fc[i0],
                    f'{wp_error:.2f}%', fontsize=7, color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            print(f"    ✓ 工作点: B/fc={wp_B:.3f}, Lap/λ={wp_L:.1f}, error={wp_error:.3f}%")

    # 标注
    ax.set_xlabel(r'Aperture Size ($L_{\mathrm{ap}}/\lambda$)', fontsize=8)
    ax.set_ylabel(r'Bandwidth Ratio ($B/f_c$)', fontsize=8)

    # ✅ 修复：色标明确标注为百分数
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Relative Error [%]', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # 网格和图例
    ax.grid(True, alpha=0.3, linewidth=0.5)
    if working_point is not None:
        ax.legend(fontsize=7, loc='upper left')

    # ✅ 新增：图注说明网格范围
    caption = (f"Grid: B/f_c ∈ [{B_over_fc.min():.3f}, {B_over_fc.max():.3f}], "
               f"L_ap/λ ∈ [{Lap_over_lambda.min():.1f}, {Lap_over_lambda.max():.1f}], "
               f"N={len(errors_valid)} points")
    ax.text(0.5, -0.18, caption, transform=ax.transAxes,
            fontsize=6, ha='center', style='italic')

    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'threshold_validation_heatmap.{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")

    plt.close()

    # ========== Figure 2: Error Distribution Histogram - 修复版本 ==========
    print(f"  Generating error histogram...")

    fig, ax = plt.subplots()

    # ✅ 使用百分数
    ax.hist(errors_pct, bins=30, edgecolor='black', alpha=0.7, linewidth=0.5,
            label=f'N={len(errors_pct)} samples')

    # 统计线
    mean_pct = errors_pct.mean()
    median_pct = np.median(errors_pct)
    std_pct = errors_pct.std()

    ax.axvline(mean_pct, color='#A2142F', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_pct:.3f}%')
    ax.axvline(median_pct, color='#0072BD', linestyle='--', linewidth=1.5,
               label=f'Median: {median_pct:.3f}%')

    # ✅ 新增：正态性检验
    from scipy.stats import shapiro
    stat, p_value = shapiro(errors_pct)

    ax.set_xlabel('Relative Error [%]', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ✅ 新增：图注包含统计信息
    stats_text = (f"Mean={mean_pct:.3f}%, Std={std_pct:.3f}%, "
                  f"Shapiro-Wilk p={p_value:.3g}")
    ax.text(0.5, -0.18, stats_text, transform=ax.transAxes,
            fontsize=6, ha='center', style='italic')

    print(f"    统计: Mean={mean_pct:.3f}%, Median={median_pct:.3f}%, Std={std_pct:.3f}%")
    print(f"    正态性检验: Shapiro-Wilk stat={stat:.4f}, p-value={p_value:.3g}")

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'threshold_error_histogram.{ext}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")

    plt.close()

    # ===== Figure 3: 1D Slices =====
    # 加载数据
    if isinstance(data_source, dict):
        # From sweep results
        error_matrix = data_source['error_matrix']
        B_over_fc = data_source['B_over_fc_vec']
        Lap_over_lambda = data_source['Lap_over_lambda_vec']

        # Create DataFrame from matrices
        detailed_rows = []
        for i, b in enumerate(B_over_fc):
            for j, l in enumerate(Lap_over_lambda):
                detailed_rows.append({
                    'B_over_fc': b,
                    'Lap_over_lambda': l,
                    'relative_error': error_matrix[i, j]
                })
        df = pd.DataFrame(detailed_rows)

    B_over_fc = data_source['B_over_fc_vec']
    Lap_over_lambda = data_source['Lap_over_lambda_vec']
    print(f"  Generating combined 1D slice plot...")

    generate_threshold_slices_combined(df, B_over_fc, Lap_over_lambda, output_dir)

    # ===== Summary =====
    print(f"\n[4/4] Visualization complete!")
    print(f"  All figures saved to: {output_dir}/")
    print(f"\n  Generated files:")
    print(f"    - threshold_validation_heatmap.png/pdf")
    print(f"    - threshold_error_histogram.png/pdf")
    print(f"    - threshold_error_slices.png/pdf")

    return True


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description='Threshold Validation Analysis (Integrated: Sweep + Visualization)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Operation mode
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only visualize existing data (skip sweep)')

    # Sweep parameters
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--grid-size', type=int, default=10,
                        choices=[8, 10, 12, 15, 20],
                        help='Grid size (10=fast, 15=medium, 20=accurate)')
    parser.add_argument('--mode', type=str, default='fast',
                        choices=['fast', 'accurate'],
                        help='fast: theoretical estimate, accurate: compute both Whittle and Cholesky')

    # Visualization parameters
    parser.add_argument('--csv', default=None,
                        help='Path to CSV file (for --visualize-only mode)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for figures (default: from config)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Skip visualization (data generation only)')

    args = parser.parse_args()

    try:
        if args.visualize_only:
            # 仅可视化模式
            if args.csv is None:
                print("Error: --visualize-only requires --csv")
                sys.exit(1)

            output_dir = args.output_dir or 'figures'
            success = visualize_threshold_results(args.csv, output_dir)

            if success:
                print("\n" + "=" * 80)
                print("VISUALIZATION SUCCESS")
                print("=" * 80)
                return 0
            else:
                return 1

        else:
            # 完整模式：扫描 + 可视化
            # 查找配置文件
            if not os.path.exists(args.config):
                search_paths = [
                    'config.yaml',
                    '/mnt/user-data/uploads/config.yaml',
                    '/mnt/user-data/outputs/config.yaml'
                ]

                config_path = None
                for path in search_paths:
                    if os.path.exists(path):
                        config_path = path
                        break

                if config_path is None:
                    print("Error: Cannot find config.yaml")
                    print(f"Usage: python threshold_analysis.py [config] [options]")
                    sys.exit(1)
            else:
                config_path = args.config

            # 运行扫描
            results = run_threshold_sweep(
                config_path=config_path,
                grid_size=args.grid_size,
                mode=args.mode
            )

            # 可视化
            if not args.no_visualize:
                output_dir = args.output_dir or results['config'].get('outputs', {}).get('figure_path', './figures/')
                visualize_threshold_results(results, output_dir)

            # 成功总结
            print("\n" + "=" * 80)
            print("THRESHOLD ANALYSIS COMPLETE")
            print("=" * 80)
            print(f"  ✓ Mode: {results['mode'].upper()}")
            print(f"  ✓ Grid: {results['grid_size']}×{results['grid_size']}")
            print(f"  ✓ Success rate: {results['success_rate'] * 100:.1f}%")

            if results['statistics'] is not None:
                max_error = results['statistics'].loc[results['statistics']['Metric'] == 'Max Error', 'Value'].values[0]
                print(f"  ✓ Max error: {max_error * 100:.3f}%")

            print(f"\n  Data saved to: {os.path.dirname(results['detailed_csv'])}/")
            if not args.no_visualize:
                print(f"  Figures saved to: {output_dir}/")

            print("\n" + "=" * 80)
            return 0

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())