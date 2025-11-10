#!/usr/bin/env python3
"""
PN-DSE Crossover Data Generator
专门生成 PN vs DSE 方差交叉数据

根据专家意见：
- physics_engine 已经计算了 sigma2_DSE
- 只需扫描 alpha 并保存两个方差到 CSV

Usage:
    python scan_pn_dse_crossover.py [config.yaml]

Author: Expert Fix v1.0
"""

import numpy as np
import pandas as pd
import yaml
import copy
import sys
import os

sys.path.insert(0, '/mnt/user-data/uploads')
from physics_engine import calc_g_sig_factors, calc_n_f_vector, validate_config


def generate_pn_dse_crossover(config_path: str = 'config.yaml'):
    """
    生成 PN-DSE 交叉数据

    Args:
        config_path: 配置文件路径

    Returns:
        DataFrame with columns: alpha, sigma2_pn_res, sigma2_DSE
    """

    print("=" * 80)
    print("PN-DSE CROSSOVER DATA GENERATOR")
    print("=" * 80)

    # 1. 加载配置
    print(f"\n[1/4] 加载配置: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        validate_config(config)
        print("  ✓ 配置加载成功")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        sys.exit(1)

    # 2. 定义 alpha 扫描范围
    print("\n[2/4] 定义 alpha 扫描范围")
    alpha_vec = np.linspace(0.05, 0.30, 26)  # 26个点，从0.05到0.30
    print(f"  扫描范围: [{alpha_vec[0]:.3f}, {alpha_vec[-1]:.3f}]")
    print(f"  数据点数: {len(alpha_vec)}")

    # 3. 扫描并收集数据
    print("\n[3/4] 扫描 PN 和 DSE 方差...")
    rows = []

    for i, alpha in enumerate(alpha_vec):
        # 复制配置并设置当前 alpha
        cfg = copy.deepcopy(config)
        cfg['isac_model']['alpha'] = float(alpha)

        try:
            # 计算物理量
            g_factors = calc_g_sig_factors(cfg)
            n_outputs = calc_n_f_vector(cfg, g_factors)

            # 提取两个关键方差
            sigma2_pn_res = float(n_outputs['sigma_2_phi_c_res'])
            sigma2_DSE = float(n_outputs['sigma2_DSE'])

            rows.append({
                'alpha': alpha,
                'sigma2_pn_res': sigma2_pn_res,
                'sigma2_DSE': sigma2_DSE
            })

            # 进度显示
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  进度: {i + 1}/{len(alpha_vec)} " +
                      f"(α={alpha:.3f}, PN={sigma2_pn_res:.2e}, DSE={sigma2_DSE:.2e})")

        except Exception as e:
            print(f"  ⚠️  α={alpha:.3f} 计算失败: {e}")
            rows.append({
                'alpha': alpha,
                'sigma2_pn_res': np.nan,
                'sigma2_DSE': np.nan
            })

    print(f"  ✓ 完成 {len(rows)} 个数据点")

    # 4. 保存到 CSV
    print("\n[4/4] 保存数据...")
    df = pd.DataFrame(rows)

    # 确保输出目录存在
    output_config = config.get('outputs', {})
    save_path = output_config.get('save_path', './results/')
    os.makedirs(save_path, exist_ok=True)

    # 保存文件
    csv_filename = os.path.join(save_path, 'pn_dse_crossover.csv')
    df.to_csv(csv_filename, index=False, float_format='%.6e')
    print(f"  ✓ 数据保存到: {csv_filename}")

    # 5. 分析交叉点
    print("\n" + "=" * 80)
    print("交叉点分析")
    print("=" * 80)

    # 找到交叉点（PN 和 DSE 最接近的地方）
    df_valid = df.dropna()
    if len(df_valid) > 0:
        df_valid['ratio'] = df_valid['sigma2_DSE'] / df_valid['sigma2_pn_res']

        # 找最接近1的点
        df_valid['dist_to_1'] = np.abs(df_valid['ratio'] - 1.0)
        crossover_idx = df_valid['dist_to_1'].idxmin()

        alpha_cross = df_valid.loc[crossover_idx, 'alpha']
        pn_cross = df_valid.loc[crossover_idx, 'sigma2_pn_res']
        dse_cross = df_valid.loc[crossover_idx, 'sigma2_DSE']
        ratio_cross = df_valid.loc[crossover_idx, 'ratio']

        print(f"\n  交叉点估计:")
        print(f"    α* = {alpha_cross:.4f}")
        print(f"    σ²_PN = {pn_cross:.3e} rad²")
        print(f"    σ²_DSE = {dse_cross:.3e} rad²")
        print(f"    比值 (DSE/PN) = {ratio_cross:.3f}")

        # 检查是否与配置中的 alpha_star_target 接近
        alpha_star_target = config.get('isac_model', {}).get('alpha_star_target', None)
        if alpha_star_target is not None:
            error = abs(alpha_cross - alpha_star_target)
            print(f"\n  配置中的目标: α* = {alpha_star_target:.4f}")
            print(f"  偏差: {error:.4f} ({error / alpha_star_target * 100:.1f}%)")
            if error < 0.02:
                print(f"  ✓ DSE 自动校准工作正常!")
            else:
                print(f"  ⚠️  偏差较大，可能需要调整 config.yaml 中的 DSE_autotune")
    else:
        print("  ⚠️  没有有效数据点，无法分析交叉点")

    print("\n" + "=" * 80)

    return df


def main():
    """主函数"""

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # 搜索配置文件
        search_paths = [
            'config.yaml',
            '/mnt/user-data/uploads/config.yaml',
        ]

        config_file = None
        for path in search_paths:
            if os.path.exists(path):
                config_file = path
                break

        if config_file is None:
            print("Error: 找不到 config.yaml")
            print("Usage: python scan_pn_dse_crossover.py [config.yaml]")
            sys.exit(1)

    try:
        df = generate_pn_dse_crossover(config_file)

        print("\n✓ PN-DSE 交叉数据生成成功")
        print(f"  数据行数: {len(df)}")
        print(f"\n下一步:")
        print(f"  1. 运行 visualize_results.py 生成交叉图")
        print(f"  2. 检查图像 figures/fig_pn_dse_crossover.png")

        return 0

    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())