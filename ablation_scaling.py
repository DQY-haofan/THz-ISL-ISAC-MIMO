#!/usr/bin/env python3
"""
Ablation Study: PN vs DSE Scaling Verification
消融分析: 验证 σ²_PN ∝ 1/α 和 σ²_DSE ∝ 1/α⁵

生成图表用于"结果与讨论"章节
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import sys


def extract_scaling_from_pareto(pareto_csv='results/DR08_pareto_results.csv',
                                output_dir='figures'):
    """
    从已有的Pareto结果中提取PN/DSE标度关系

    Args:
        pareto_csv: Pareto结果CSV文件路径
        output_dir: 输出图表目录
    """

    print("=" * 80)
    print("ABLATION STUDY: PN vs DSE Scaling Verification")
    print("=" * 80)
    print()

    # 1. 加载数据
    print(f"[1/4] Loading data from: {pareto_csv}")

    if not os.path.exists(pareto_csv):
        print(f"✗ Error: File not found: {pareto_csv}")
        print("  Please run: python main.py config.yaml first")
        return False

    try:
        df = pd.read_csv(pareto_csv)
        print(f"✓ Loaded {len(df)} data points")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False

    # 2. 检查必需列
    required_cols = ['alpha', 'sigma_2_phi_c_res_rad2', 'sigma_2_DSE_var']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"✗ Error: Missing columns: {missing_cols}")
        print(f"  Available columns: {df.columns.tolist()}")
        return False

    print(f"✓ Required columns found")
    print()

    # 3. 提取数据
    print("[2/4] Extracting PN and DSE components...")

    alpha = df['alpha'].values
    sigma2_PN = df['sigma_2_phi_c_res_rad2'].values
    sigma2_DSE = df['sigma_2_DSE_var'].values

    # 移除无效值
    valid_mask = (alpha > 0) & (sigma2_PN > 0) & (sigma2_DSE > 0) & \
                 np.isfinite(sigma2_PN) & np.isfinite(sigma2_DSE)

    alpha = alpha[valid_mask]
    sigma2_PN = sigma2_PN[valid_mask]
    sigma2_DSE = sigma2_DSE[valid_mask]

    print(f"  Valid data points: {len(alpha)}")
    print(f"  α range: {alpha.min():.3f} to {alpha.max():.3f}")
    print(f"  σ²_PN range: {sigma2_PN.min():.2e} to {sigma2_PN.max():.2e} rad²")
    print(f"  σ²_DSE range: {sigma2_DSE.min():.2e} to {sigma2_DSE.max():.2e} rad²")
    print()

    # 4. 对数线性拟合 (验证幂律)
    print("[3/4] Performing log-log linear regression...")
    print()

    # 转换到对数空间
    log_alpha = np.log10(alpha)
    log_sigma2_PN = np.log10(sigma2_PN)
    log_sigma2_DSE = np.log10(sigma2_DSE)

    # PN拟合: σ²_PN = C_PN / α  →  log(σ²_PN) = log(C_PN) - log(α)
    slope_PN, intercept_PN, r_value_PN, p_value_PN, std_err_PN = \
        linregress(log_alpha, log_sigma2_PN)

    # DSE拟合: σ²_DSE = C_DSE / α⁵  →  log(σ²_DSE) = log(C_DSE) - 5*log(α)
    slope_DSE, intercept_DSE, r_value_DSE, p_value_DSE, std_err_DSE = \
        linregress(log_alpha, log_sigma2_DSE)

    print("PN Scaling Analysis:")
    print(f"  Theoretical slope: -1.0")
    print(f"  Measured slope:    {slope_PN:.3f} ± {std_err_PN:.3f}")
    print(f"  R² = {r_value_PN ** 2:.4f}")
    print(f"  Deviation: {abs(slope_PN + 1.0):.3f} ({abs(slope_PN + 1.0) / 1.0 * 100:.1f}%)")
    print()

    print("DSE Scaling Analysis:")
    print(f"  Theoretical slope: -5.0")
    print(f"  Measured slope:    {slope_DSE:.3f} ± {std_err_DSE:.3f}")
    print(f"  R² = {r_value_DSE ** 2:.4f}")
    print(f"  Deviation: {abs(slope_DSE + 5.0):.3f} ({abs(slope_DSE + 5.0) / 5.0 * 100:.1f}%)")
    print()

    # 5. 生成图表
    print("[4/4] Generating ablation figure...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建2×1子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # === 子图1: PN标度验证 ===
    ax = ax1

    # 散点图
    ax.loglog(alpha, sigma2_PN, 'o', markersize=8, color='purple',
              alpha=0.7, label='Measured $\\sigma^2_{\\phi,c,\\mathrm{res}}$')

    # 拟合线
    alpha_fit = np.logspace(np.log10(alpha.min()), np.log10(alpha.max()), 100)
    sigma2_PN_fit = 10 ** (intercept_PN) * alpha_fit ** (slope_PN)
    ax.loglog(alpha_fit, sigma2_PN_fit, '--', linewidth=2, color='darkviolet',
              label=f'Fit: $\\propto \\alpha^{{{slope_PN:.2f}}}$ (R²={r_value_PN ** 2:.4f})')

    # 理论线 (参考)
    C_PN_theory = sigma2_PN[len(sigma2_PN) // 2] * alpha[len(alpha) // 2]
    sigma2_PN_theory = C_PN_theory / alpha_fit
    ax.loglog(alpha_fit, sigma2_PN_theory, ':', linewidth=2, color='gray',
              label='Theory: $\\propto \\alpha^{-1}$')

    # 标注
    ax.set_xlabel('ISAC Overhead ($\\alpha$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$\\sigma^2_{\\phi,c,\\mathrm{res}}$ (rad²)', fontsize=14, fontweight='bold')
    ax.set_title('(a) Phase Noise Scaling Verification', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

    # 添加文本框显示结果
    textstr = f'Slope: {slope_PN:.3f} ± {std_err_PN:.3f}\n' \
              f'Theory: -1.0\n' \
              f'Error: {abs(slope_PN + 1.0) / 1.0 * 100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)

    # === 子图2: DSE标度验证 ===
    ax = ax2

    # 散点图
    ax.loglog(alpha, sigma2_DSE, 's', markersize=8, color='orange',
              alpha=0.7, label='Measured $\\sigma^2_{\\mathrm{DSE}}$')

    # 拟合线
    sigma2_DSE_fit = 10 ** (intercept_DSE) * alpha_fit ** (slope_DSE)
    ax.loglog(alpha_fit, sigma2_DSE_fit, '--', linewidth=2, color='darkorange',
              label=f'Fit: $\\propto \\alpha^{{{slope_DSE:.2f}}}$ (R²={r_value_DSE ** 2:.4f})')

    # 理论线 (参考)
    C_DSE_theory = sigma2_DSE[len(sigma2_DSE) // 2] * (alpha[len(alpha) // 2] ** 5)
    sigma2_DSE_theory = C_DSE_theory / (alpha_fit ** 5)
    ax.loglog(alpha_fit, sigma2_DSE_theory, ':', linewidth=2, color='gray',
              label='Theory: $\\propto \\alpha^{-5}$')

    # 标注
    ax.set_xlabel('ISAC Overhead ($\\alpha$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('$\\sigma^2_{\\mathrm{DSE}}$ (rad²)', fontsize=14, fontweight='bold')
    ax.set_title('(b) Dynamic Scan Error Scaling Verification', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)

    # 添加文本框显示结果
    textstr = f'Slope: {slope_DSE:.3f} ± {std_err_DSE:.3f}\n' \
              f'Theory: -5.0\n' \
              f'Error: {abs(slope_DSE + 5.0) / 5.0 * 100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, 'fig_scaling_ablation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

    print(f"✓ Figure saved: {output_path}")
    print()

    # 6. 生成LaTeX表格
    print("=" * 80)
    print("SCALING VERIFICATION SUMMARY (for paper)")
    print("=" * 80)
    print()

    print("LaTeX Table:")
    print("-" * 80)
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Power-Law Scaling Verification of PN and DSE Components}")
    print(r"  \label{tab:scaling_verification}")
    print(r"  \begin{tabular}{lccc}")
    print(r"    \hline")
    print(r"    Component & Theoretical Slope & Measured Slope & $R^2$ \\")
    print(r"    \hline")
    print(f"    $\\sigma^2_{{\\phi,c,\\mathrm{{res}}}}$ & $-1.0$ & "
          f"${slope_PN:.3f} \\pm {std_err_PN:.3f}$ & ${r_value_PN ** 2:.4f}$ \\\\")
    print(f"    $\\sigma^2_{{\\mathrm{{DSE}}}}$ & $-5.0$ & "
          f"${slope_DSE:.3f} \\pm {std_err_DSE:.3f}$ & ${r_value_DSE ** 2:.4f}$ \\\\")
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print()

    # 7. 评估结果
    print("=" * 80)
    print("ASSESSMENT")
    print("=" * 80)
    print()

    # PN评估
    pn_error_pct = abs(slope_PN + 1.0) / 1.0 * 100
    if pn_error_pct < 5.0 and r_value_PN ** 2 > 0.95:
        pn_status = "✓ EXCELLENT"
    elif pn_error_pct < 10.0 and r_value_PN ** 2 > 0.90:
        pn_status = "✓ GOOD"
    elif pn_error_pct < 20.0 and r_value_PN ** 2 > 0.80:
        pn_status = "⚠ ACCEPTABLE"
    else:
        pn_status = "✗ POOR"

    print(f"PN Scaling: {pn_status}")
    print(f"  Error: {pn_error_pct:.1f}% from theoretical -1.0")
    print(f"  R²: {r_value_PN ** 2:.4f}")
    print()

    # DSE评估
    dse_error_pct = abs(slope_DSE + 5.0) / 5.0 * 100
    if dse_error_pct < 5.0 and r_value_DSE ** 2 > 0.95:
        dse_status = "✓ EXCELLENT"
    elif dse_error_pct < 10.0 and r_value_DSE ** 2 > 0.90:
        dse_status = "✓ GOOD"
    elif dse_error_pct < 20.0 and r_value_DSE ** 2 > 0.80:
        dse_status = "⚠ ACCEPTABLE"
    else:
        dse_status = "✗ POOR"

    print(f"DSE Scaling: {dse_status}")
    print(f"  Error: {dse_error_pct:.1f}% from theoretical -5.0")
    print(f"  R²: {r_value_DSE ** 2:.4f}")
    print()

    if pn_error_pct < 10.0 and dse_error_pct < 10.0:
        print("✓ Overall: Theoretical scaling laws are VERIFIED")
        print("  Both PN and DSE follow expected power-law behavior")
    else:
        print("⚠ Overall: Some deviation from theory observed")
        print("  May indicate:")
        print("    - Numerical precision issues")
        print("    - Model approximations")
        print("    - Need for parameter tuning")

    print()
    print("=" * 80)

    # 保存结果到CSV
    results_df = pd.DataFrame({
        'Component': ['PN', 'DSE'],
        'Theoretical_Slope': [-1.0, -5.0],
        'Measured_Slope': [slope_PN, slope_DSE],
        'Std_Error': [std_err_PN, std_err_DSE],
        'R_Squared': [r_value_PN ** 2, r_value_DSE ** 2],
        'Error_Percent': [pn_error_pct, dse_error_pct]
    })

    csv_path = os.path.join('results', 'scaling_verification.csv')
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")

    return True


def main():
    """主函数"""

    if len(sys.argv) > 1:
        pareto_csv = sys.argv[1]
    else:
        # 尝试常见路径
        search_paths = [
            'results/DR08_pareto_results.csv',
            'results/DR08_results_pareto_results.csv',
            'DR08_pareto_results.csv',
        ]

        pareto_csv = None
        for path in search_paths:
            if os.path.exists(path):
                pareto_csv = path
                break

        if pareto_csv is None:
            print("Error: No Pareto results CSV found")
            print("Usage: python ablation_scaling.py [pareto_csv_path]")
            print()
            print("Searched locations:")
            for path in search_paths:
                print(f"  - {path}")
            print()
            print("Please run: python main.py config.yaml first")
            sys.exit(1)

    try:
        success = extract_scaling_from_pareto(pareto_csv)
        if success:
            print("\n✓ Scaling verification completed successfully!")
            print("  Use fig_scaling_ablation.pdf in your paper's Results section")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()