#!/usr/bin/env python3
"""
快速测试：验证RMSE对α的敏感性
"""

import numpy as np
import yaml
from physics_engine import calc_g_sig_factors, calc_n_f_vector
from limits_engine import calc_BCRLB


def test_alpha_sensitivity(config_path='config.yaml'):
    """测试RMSE是否随α变化"""

    print("=" * 80)
    print("α 敏感性测试")
    print("=" * 80)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 测试几个关键α值
    alpha_test = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    print(f"\n{'α':<8} {'RMSE(mm)':<12} {'σ²_PN':<15} {'σ²_DSE':<15} {'主导因素'}")
    print("-" * 70)

    results = []

    for alpha in alpha_test:
        config['isac_model']['alpha'] = alpha

        # 重新计算物理量
        g_sig = calc_g_sig_factors(config)
        n_f = calc_n_f_vector(config, g_sig)
        bcrlb = calc_BCRLB(config, g_sig, n_f)

        # 计算RMSE
        c = config['channel']['c_mps']
        RMSE_m = (c / 2) * np.sqrt(bcrlb['BCRLB_tau'])
        RMSE_mm = RMSE_m * 1000

        # 获取PN和DSE方差
        sigma2_pn = n_f.get('sigma_2_phi_c_res', 0)
        sigma2_dse = n_f.get('sigma_2_DSE_var', 0)

        # 判断主导因素
        if sigma2_pn > sigma2_dse * 2:
            dominant = "PN"
        elif sigma2_dse > sigma2_pn * 2:
            dominant = "DSE"
        else:
            dominant = "交叉"

        print(f"{alpha:<8.2f} {RMSE_mm:<12.3f} {sigma2_pn:<15.3e} {sigma2_dse:<15.3e} {dominant}")

        results.append({
            'alpha': alpha,
            'RMSE_mm': RMSE_mm,
            'sigma2_pn': sigma2_pn,
            'sigma2_dse': sigma2_dse
        })

    # 分析变化
    print("\n" + "=" * 80)
    print("变化分析")
    print("=" * 80)

    rmse_values = [r['RMSE_mm'] for r in results]
    rmse_min = min(rmse_values)
    rmse_max = max(rmse_values)
    rmse_range = rmse_max - rmse_min

    print(f"\nRMSE范围：")
    print(f"  最小值（α={alpha_test[np.argmin(rmse_values)]:.2f}）: {rmse_min:.3f} mm")
    print(f"  最大值（α={alpha_test[np.argmax(rmse_values)]:.2f}）: {rmse_max:.3f} mm")
    print(f"  变化范围: {rmse_range:.3f} mm")
    print(f"  变化比例: {rmse_max / rmse_min:.2f}x")

    # 判断敏感性
    if rmse_max / rmse_min > 2:
        print(f"\n✅ RMSE对α高度敏感（变化 {rmse_max / rmse_min:.1f}倍）")
        print("   修复成功！")
    elif rmse_max / rmse_min > 1.5:
        print(f"\n⚠️  RMSE对α中度敏感（变化 {rmse_max / rmse_min:.1f}倍）")
        print("   可能需要检查DSE和PN的设置")
    else:
        print(f"\n❌ RMSE对α不敏感（变化仅 {rmse_max / rmse_min:.1f}倍）")
        print("   需要检查循环内是否重新计算物理量")

    # 检查PN-DSE交叉点
    print(f"\nPN-DSE交叉点检查：")
    for r in results:
        if abs(r['sigma2_pn'] - r['sigma2_dse']) < 0.5 * (r['sigma2_pn'] + r['sigma2_dse']):
            print(f"  交叉点在 α ≈ {r['alpha']:.2f}")
            break

    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'

    results = test_alpha_sensitivity(config_path)

    print("\n提示：如果RMSE不随α变化，请检查：")
    print("1. 是否在循环内调用 calc_g_sig_factors(config)")
    print("2. 是否在循环内调用 calc_n_f_vector(config, g_sig)")
    print("3. 是否在循环内调用 calc_BCRLB(config, g_sig, n_f)")