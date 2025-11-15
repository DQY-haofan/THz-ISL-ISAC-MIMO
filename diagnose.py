#!/usr/bin/env python3
"""
硬件损伤影响诊断脚本
====================================
目的：找出为什么硬件参数修改不影响RMSE结果

诊断重点：
1. 噪声功率分解（N0 vs σ²_γ）
2. 硬件失真各组件贡献
3. 发射功率设置是否合理
4. SNR vs 硬件失真比例
5. 参数敏感性分析

Author: Diagnostic Expert
Date: 2025-11-15
"""

import numpy as np
import yaml
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# 导入原始engine
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector
    from limits_engine import calc_BCRLB

    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ 错误: 无法导入engine模块")
    print(f"详情: {e}")
    sys.exit(1)


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def diagnose_noise_components(config: dict) -> Dict[str, Any]:
    """
    诊断1: 噪声组成分析

    关键问题：σ²_γ (硬件失真) 相对于 N₀ (热噪声) 的比例
    """
    print_section("诊断1: 噪声功率分解")

    # 计算物理量
    g_factors = calc_g_sig_factors(config)
    n_outputs = calc_n_f_vector(config, g_factors)

    # 提取关键参数
    B_hz = config['channel']['B_hz']
    f_c_hz = config['channel']['f_c_hz']
    Nt = config['array']['Nt']
    Nr = config['array']['Nr']

    # 噪声组件
    N0_white = n_outputs['N0']  # 热噪声功率谱密度 (W/Hz)
    sigma2_gamma = n_outputs['sigma2_gamma']  # 硬件失真总功率 (W)
    sigma2_gamma_psd = sigma2_gamma / B_hz  # 硬件失真PSD (W/Hz)

    # ⚠️ 关键：提取 PN 和 DSE
    S_phi_c_res_k = n_outputs.get('S_phi_c_res_k', np.zeros(1))
    S_DSE_k = n_outputs.get('S_DSE_k', np.zeros(1))
    S_RSM_k = n_outputs.get('S_RSM_k', np.zeros(1))

    PN_psd_mean = float(np.mean(S_phi_c_res_k)) if len(S_phi_c_res_k) > 0 else 0
    DSE_psd_mean = float(np.mean(S_DSE_k)) if len(S_DSE_k) > 0 else 0
    RSM_psd_mean = float(np.mean(S_RSM_k)) if len(S_RSM_k) > 0 else 0

    # 硬件失真分量
    Gamma_pa = n_outputs['Gamma_pa']
    Gamma_adc = n_outputs['Gamma_adc']
    Gamma_iq = n_outputs['Gamma_iq']
    Gamma_lo = n_outputs['Gamma_lo']
    Gamma_eff_per_elem = n_outputs['Gamma_eff_per_element']

    P_tx_per_elem = n_outputs['P_tx_per_element']
    P_rx_total = n_outputs['P_rx_total']

    print(f"\n基础参数:")
    print(f"  频率 f_c     = {f_c_hz / 1e9:.1f} GHz")
    print(f"  带宽 B       = {B_hz / 1e9:.1f} GHz")
    print(f"  阵列尺寸     = {Nt}×{Nr} = {Nt * Nr}")
    print(f"  发射功率/单元 = {P_tx_per_elem:.3e} W")
    print(f"  接收总功率   = {P_rx_total:.3e} W")

    print(f"\n📊 完整噪声功率谱密度 (PSD) 分解:")
    print(f"  N₀ (热噪声)        = {N0_white:.3e} W/Hz  ({10 * np.log10(N0_white / N0_white):+.1f} dB)")
    print(f"  σ²_γ/B (硬件失真)  = {sigma2_gamma_psd:.3e} W/Hz  ({10 * np.log10(sigma2_gamma_psd / N0_white):+.1f} dB)")
    print(
        f"  PN (相位噪声)      = {PN_psd_mean:.3e} W/Hz  ({10 * np.log10(PN_psd_mean / N0_white) if PN_psd_mean > 0 else -np.inf:+.1f} dB)")
    print(
        f"  DSE (双边谱展宽)   = {DSE_psd_mean:.3e} W/Hz  ({10 * np.log10(DSE_psd_mean / N0_white) if DSE_psd_mean > 0 else -np.inf:+.1f} dB)")
    print(
        f"  RSM (旁瓣调制)     = {RSM_psd_mean:.3e} W/Hz  ({10 * np.log10(RSM_psd_mean / N0_white) if RSM_psd_mean > 0 else -np.inf:+.1f} dB)")

    total_noise_psd = N0_white + sigma2_gamma_psd + PN_psd_mean + DSE_psd_mean + RSM_psd_mean
    print(f"\n  总噪声PSD          = {total_noise_psd:.3e} W/Hz")

    print(f"\n📌 各噪声源占总噪声的比例:")
    print(f"  N₀       : {100 * N0_white / total_noise_psd:.2f}%")
    print(f"  σ²_γ/B   : {100 * sigma2_gamma_psd / total_noise_psd:.2f}%")
    print(f"  PN       : {100 * PN_psd_mean / total_noise_psd:.2f}%  ⚠️")
    print(f"  DSE      : {100 * DSE_psd_mean / total_noise_psd:.2f}%")
    print(f"  RSM      : {100 * RSM_psd_mean / total_noise_psd:.2f}%")

    print(f"\n硬件失真总功率:")
    print(f"  σ²_γ (总)          = {sigma2_gamma:.3e} W")
    print(f"  σ²_γ / P_rx        = {sigma2_gamma / P_rx_total:.6f}")

    print(f"\n硬件失真系数 (每单元):")
    print(f"  Γ_PA  = {Gamma_pa:.3e}  ({100 * Gamma_pa / Gamma_eff_per_elem:.1f}%)")
    print(f"  Γ_ADC = {Gamma_adc:.3e}  ({100 * Gamma_adc / Gamma_eff_per_elem:.1f}%)")
    print(f"  Γ_IQ  = {Gamma_iq:.3e}  ({100 * Gamma_iq / Gamma_eff_per_elem:.1f}%)")
    print(f"  Γ_LO  = {Gamma_lo:.3e}  ({100 * Gamma_lo / Gamma_eff_per_elem:.1f}%)")
    print(f"  Γ_eff = {Gamma_eff_per_elem:.3e}  (总和)")

    print(f"\n缩放验证:")
    print(f"  σ²_γ = Γ_eff × P_tx × (Nt+Nr)")
    print(f"       = {Gamma_eff_per_elem:.3e} × {P_tx_per_elem:.3e} × {Nt + Nr}")
    print(f"       = {Gamma_eff_per_elem * P_tx_per_elem * (Nt + Nr):.3e} W")
    print(f"  实际值 = {sigma2_gamma:.3e} W")
    print(f"  匹配? {'✓' if abs(Gamma_eff_per_elem * P_tx_per_elem * (Nt + Nr) - sigma2_gamma) < 1e-15 else '✗'}")

    # ⚠️ 关键诊断 - 比较 HW vs PN
    ratio_hw_to_N0_db = 10 * np.log10(sigma2_gamma_psd / N0_white)
    ratio_pn_to_N0_db = 10 * np.log10(PN_psd_mean / N0_white) if PN_psd_mean > 0 else -np.inf
    ratio_hw_to_pn_db = 10 * np.log10(sigma2_gamma_psd / PN_psd_mean) if PN_psd_mean > 0 else np.inf

    print(f"\n⚠️  关键诊断 - 为什么 HW 曲线与 AWGN 重叠？")
    print(f"  硬件失真 / 热噪声   = {ratio_hw_to_N0_db:+.1f} dB")
    print(f"  相位噪声 / 热噪声   = {ratio_pn_to_N0_db:+.1f} dB")
    print(f"  硬件失真 / 相位噪声 = {ratio_hw_to_pn_db:+.1f} dB  ⚠️⚠️⚠️")

    if PN_psd_mean > sigma2_gamma_psd:
        pn_dominance = PN_psd_mean / sigma2_gamma_psd
        print(f"\n  ❌❌❌ 问题根源找到了！")
        print(f"  相位噪声主导了系统，是硬件失真的 {pn_dominance:.1f}× !!!")
        print(f"  即使硬件失真很大 ({ratio_hw_to_N0_db:+.1f} dB)，")
        print(f"  它仍然被相位噪声淹没 ({ratio_pn_to_N0_db:+.1f} dB)。")
        print(f"\n  这就是为什么:")
        print(f"  - AWGN曲线: 仅包含N₀ (热噪声)")
        print(f"  - HW曲线:   包含N₀ + σ²_γ")
        print(f"  - 两者几乎相同: 因为σ²_γ << PN，主导因素是PN")
        print(f"\n  在消融研究中:")
        print(f"  - AWGN vs HW: 差异很小（因为都被PN主导）")
        print(f"  - HW vs HW+PN: 差异巨大（PN突然加入）")
    elif ratio_hw_to_N0_db < -20:
        print(f"\n  ❌ 硬件失真太小！({ratio_hw_to_N0_db:.1f} dB < -20 dB)")
        print(f"     硬件失真比热噪声小 {-ratio_hw_to_N0_db:.1f} dB")
    else:
        print(f"\n  ✓ 硬件失真足够显著")

    return {
        'N0_white': N0_white,
        'sigma2_gamma': sigma2_gamma,
        'sigma2_gamma_psd': sigma2_gamma_psd,
        'PN_psd_mean': PN_psd_mean,
        'DSE_psd_mean': DSE_psd_mean,
        'RSM_psd_mean': RSM_psd_mean,
        'total_noise_psd': total_noise_psd,
        'ratio_linear': sigma2_gamma_psd / N0_white,
        'ratio_db': ratio_hw_to_N0_db,
        'ratio_hw_to_pn_db': ratio_hw_to_pn_db,
        'Gamma_eff_per_elem': Gamma_eff_per_elem,
        'P_tx_per_elem': P_tx_per_elem,
        'P_rx_total': P_rx_total,
        'hardware_components': {
            'Gamma_pa': Gamma_pa,
            'Gamma_adc': Gamma_adc,
            'Gamma_iq': Gamma_iq,
            'Gamma_lo': Gamma_lo
        }
    }


def diagnose_bcrlb_calculation(config: dict, noise_diag: Dict) -> Dict[str, Any]:
    """
    诊断2: BCRLB 计算中的噪声使用

    验证 N_k_psd 是否正确包含了硬件失真
    """
    print_section("诊断2: BCRLB 噪声PSD构建")

    g_factors = calc_g_sig_factors(config)
    n_outputs = calc_n_f_vector(config, g_factors)

    # 获取BCRLB结果
    bcrlb_results = calc_BCRLB(config, g_factors, n_outputs)

    # 提取诊断信息
    diag = bcrlb_results.get('diagnostics', {})

    if diag:
        print(f"\nBCRLB 诊断信息:")
        print(f"  N_k_psd 均值     = {diag.get('N_k_mean', 0):.3e} W/Hz")
        print(f"  N₀              = {diag.get('N0_white', 0):.3e} W/Hz")
        print(f"  σ²_γ/B          = {diag.get('sigma2_gamma_psd', 0):.3e} W/Hz")
        print(f"  PN 贡献         = {diag.get('S_phi_mean', 0):.3e} W/Hz")
        print(f"  DSE 贡献        = {diag.get('S_DSE_mean', 0):.3e} W/Hz")

        # 验证 N_k 的组成
        N0 = diag.get('N0_white', 0)
        gamma_psd = diag.get('sigma2_gamma_psd', 0)
        N_k_est = N0 + gamma_psd
        N_k_actual = diag.get('N_k_mean', 0)

        print(f"\n验证 N_k_psd 组成:")
        print(f"  N₀ + σ²_γ/B (估算)  = {N_k_est:.3e} W/Hz")
        print(f"  N_k_psd (实际均值) = {N_k_actual:.3e} W/Hz")

        if N_k_actual > 0:
            if abs(N_k_est - N_k_actual) / N_k_actual < 0.1:
                print(f"  ✓ 匹配良好 (差异 < 10%)")
            else:
                print(f"  ⚠️  差异较大: {abs(N_k_est - N_k_actual) / N_k_actual * 100:.1f}%")
                print(f"     可能包含了额外的噪声源 (PN, DSE, RSM)")
        else:
            print(f"  ⚠️  N_k_psd为零！诊断信息未正确返回")
    else:
        print(f"  ⚠️  未找到 diagnostics 字段")
        print(f"     limits_engine.py 可能未启用诊断输出")

    # BCRLB 结果
    BCRLB_tau = bcrlb_results['BCRLB_tau']
    RMSE_m = np.sqrt(BCRLB_tau) * (3e8 / 2)
    RMSE_mm = RMSE_m * 1000

    print(f"\nBCRLB 结果:")
    print(f"  BCRLB_τ  = {BCRLB_tau:.3e} s²")
    print(f"  RMSE     = {RMSE_mm:.4f} mm")

    return {
        'BCRLB_tau': BCRLB_tau,
        'RMSE_mm': RMSE_mm,
        'diagnostics': diag
    }


def diagnose_parameter_sensitivity(config: dict) -> Dict[str, Any]:
    """
    诊断3: 参数敏感性分析

    测试修改硬件参数时，RMSE 的变化幅度
    """
    print_section("诊断3: 参数敏感性分析")

    # 基准配置
    g_factors_base = calc_g_sig_factors(config)
    n_outputs_base = calc_n_f_vector(config, g_factors_base)
    bcrlb_base = calc_BCRLB(config, g_factors_base, n_outputs_base)
    RMSE_base_mm = np.sqrt(bcrlb_base['BCRLB_tau']) * (3e8 / 2) * 1000

    print(f"\n基准配置:")
    print(f"  gamma_pa_floor   = {config['hardware']['gamma_pa_floor']:.4f}")
    print(f"  gamma_adc_bits   = {config['hardware']['gamma_adc_bits']}")
    print(f"  gamma_iq_irr_dbc = {config['hardware']['gamma_iq_irr_dbc']:.1f} dBc")
    print(f"  gamma_lo_jitter  = {config['hardware']['gamma_lo_jitter_s'] * 1e15:.1f} fs")
    print(f"  P_tx_fixed       = {config['isac_model'].get('P_tx_fixed', 1.0):.3e} W")
    print(f"  → RMSE_base      = {RMSE_base_mm:.4f} mm")

    # 测试1: 增大 PA 失真
    print(f"\n测试1: 增大 PA 失真 10倍")
    config_test = config.copy()
    config_test['hardware'] = config['hardware'].copy()
    config_test['hardware']['gamma_pa_floor'] = config['hardware']['gamma_pa_floor'] * 10

    g_factors_test = calc_g_sig_factors(config_test)
    n_outputs_test = calc_n_f_vector(config_test, g_factors_test)
    bcrlb_test = calc_BCRLB(config_test, g_factors_test, n_outputs_test)
    RMSE_test_mm = np.sqrt(bcrlb_test['BCRLB_tau']) * (3e8 / 2) * 1000

    print(f"  gamma_pa_floor → {config_test['hardware']['gamma_pa_floor']:.4f}")
    print(f"  RMSE           = {RMSE_test_mm:.4f} mm")
    print(f"  变化幅度       = {(RMSE_test_mm / RMSE_base_mm - 1) * 100:+.2f}%")

    if abs(RMSE_test_mm - RMSE_base_mm) / RMSE_base_mm < 0.01:
        print(f"  ❌ 几乎无变化 (<1%)！硬件失真不影响结果")

    # 测试2: 增大发射功率
    print(f"\n测试2: 增大发射功率 1000倍")
    config_test2 = config.copy()
    config_test2['isac_model'] = config['isac_model'].copy()
    P_tx_base = config['isac_model'].get('P_tx_fixed', 1.0)
    config_test2['isac_model']['P_tx_fixed'] = P_tx_base * 1000

    g_factors_test2 = calc_g_sig_factors(config_test2)
    n_outputs_test2 = calc_n_f_vector(config_test2, g_factors_test2)
    bcrlb_test2 = calc_BCRLB(config_test2, g_factors_test2, n_outputs_test2)
    RMSE_test2_mm = np.sqrt(bcrlb_test2['BCRLB_tau']) * (3e8 / 2) * 1000

    print(f"  P_tx_fixed → {config_test2['isac_model']['P_tx_fixed']:.3e} W")
    print(f"  RMSE       = {RMSE_test2_mm:.4f} mm")
    print(f"  变化幅度   = {(RMSE_test2_mm / RMSE_base_mm - 1) * 100:+.2f}%")

    # 测试3: 同时增大功率和硬件失真
    print(f"\n测试3: 功率×1000 + PA失真×10")
    config_test3 = config.copy()
    config_test3['hardware'] = config['hardware'].copy()
    config_test3['isac_model'] = config['isac_model'].copy()
    config_test3['hardware']['gamma_pa_floor'] = config['hardware']['gamma_pa_floor'] * 10
    config_test3['isac_model']['P_tx_fixed'] = P_tx_base * 1000

    g_factors_test3 = calc_g_sig_factors(config_test3)
    n_outputs_test3 = calc_n_f_vector(config_test3, g_factors_test3)
    bcrlb_test3 = calc_BCRLB(config_test3, g_factors_test3, n_outputs_test3)
    RMSE_test3_mm = np.sqrt(bcrlb_test3['BCRLB_tau']) * (3e8 / 2) * 1000

    print(f"  RMSE       = {RMSE_test3_mm:.4f} mm")
    print(f"  变化幅度   = {(RMSE_test3_mm / RMSE_base_mm - 1) * 100:+.2f}%")

    return {
        'RMSE_base': RMSE_base_mm,
        'RMSE_PA_10x': RMSE_test_mm,
        'RMSE_Ptx_1000x': RMSE_test2_mm,
        'RMSE_both': RMSE_test3_mm
    }


def recommend_parameters(noise_diag: Dict) -> None:
    """
    诊断4: 参数推荐

    基于诊断结果，给出使 HW 可见的参数建议
    """
    print_section("诊断4: 参数推荐")

    ratio_db = noise_diag['ratio_db']
    ratio_hw_to_pn_db = noise_diag.get('ratio_hw_to_pn_db', 0)
    P_tx = noise_diag['P_tx_per_elem']
    Gamma_eff = noise_diag['Gamma_eff_per_elem']
    N0 = noise_diag['N0_white']
    PN_psd = noise_diag.get('PN_psd_mean', 0)
    HW_psd = noise_diag['sigma2_gamma_psd']

    print(f"\n当前状态:")
    print(f"  硬件失真/热噪声     = {ratio_db:.1f} dB")
    print(f"  硬件失真/相位噪声   = {ratio_hw_to_pn_db:.1f} dB ⚠️")
    print(f"  发射功率/单元       = {P_tx:.3e} W")
    print(f"  Γ_eff (每单元)      = {Gamma_eff:.3e}")

    # 判断主要问题
    if PN_psd > HW_psd:
        # PN 主导的情况
        print(f"\n🔍 问题诊断:")
        print(f"  相位噪声主导系统 (PN/HW = {PN_psd / HW_psd:.1f}×)")
        print(f"  这导致消融研究中:")
        print(f"    • AWGN: 仅热噪声 N₀ = {N0:.2e} W/Hz")
        print(f"    • HW:   N₀ + σ²_γ/B = {N0 + HW_psd:.2e} W/Hz")
        print(f"    • 差异: ≈ {100 * HW_psd / (N0 + HW_psd):.1f}% (很小！)")
        print(f"\n  但当加入PN后:")
        print(f"    • HW+PN: N₀ + σ²_γ/B + PN = {N0 + HW_psd + PN_psd:.2e} W/Hz")
        print(f"    • 差异: ≈ {100 * PN_psd / (N0 + HW_psd + PN_psd):.1f}% (巨大！)")

        print(f"\n💡 解决方案 - 使HW可见的方法:")
        print(f"\n【方案A】关闭相位噪声（仅用于理解HW影响）")
        print(f"  在消融研究中，AWGN和HW配置都应该关闭PN:")
        print(f"  ```python")
        print(f"  cfg['pn_model']['S_phi_c_K2'] = 0.0")
        print(f"  cfg['pn_model']['S_phi_c_K0'] = 0.0")
        print(f"  ```")
        print(f"  这样HW vs AWGN的差异才能显现。")

        pn_to_hw_factor = PN_psd / HW_psd
        hw_boost_needed = np.sqrt(pn_to_hw_factor)  # 需要让HW接近PN量级

        print(f"\n【方案B】增强硬件失真到PN量级")
        print(f"  当前PN是HW的 {pn_to_hw_factor:.1f}× ")
        print(f"  要使HW与PN相当，需要:")
        print(f"  1. 发射功率 ↑ {hw_boost_needed:.1f}×")
        print(f"     P_tx_fixed = {P_tx * hw_boost_needed:.3e} W")
        print(f"  2. 或 硬件质量 ↓ {hw_boost_needed:.1f}×")
        print(f"     gamma_pa_floor = {0.15 * hw_boost_needed:.3f}")
        print(f"  3. 或两者平衡 (各 {np.sqrt(hw_boost_needed):.1f}×)")

        print(f"\n【方案C】理解现有结果（推荐）")
        print(f"  你的图已经是正确的！")
        print(f"  - AWGN ≈ HW: 因为两者都被PN淹没")
        print(f"  - HW+PN >> HW: PN的巨大影响")
        print(f"  这是物理正确的，不是bug！")
        print(f"\n  如果要分离HW的影响，需要:")
        print(f"  1. 在AWGN和HW配置中都关闭PN和DSE")
        print(f"  2. 或者将hardware_ablation_study.py中的create_config_variants")
        print(f"     修改为在所有配置中都保持PN=0")

    else:
        # HW不够显著的情况（相对于N0）
        target_db = 0  # 目标：硬件失真等于热噪声
        gain_needed_db = target_db - ratio_db
        gain_needed_linear = 10 ** (gain_needed_db / 10)

        print(f"\n为使硬件失真与热噪声相当 (0 dB):")
        print(f"  需要提升: {gain_needed_db:.1f} dB = {gain_needed_linear:.1f}×")

        print(f"\n方案1: 仅增大发射功率")
        P_tx_new = P_tx * gain_needed_linear
        print(f"  P_tx_fixed = {P_tx_new:.3e} W  (增加 {gain_needed_linear:.1f}×)")

        print(f"\n方案2: 仅增大硬件失真系数")
        Gamma_new = Gamma_eff * gain_needed_linear
        print(f"  这需要修改多个硬件参数:")
        print(f"  - gamma_pa_floor ↑ {np.sqrt(gain_needed_linear):.1f}× (约)")
        print(f"  - gamma_adc_bits ↓ 1-2 bits")
        print(f"  - gamma_iq_irr_dbc ↑ 10 dB")
        print(f"  - gamma_lo_jitter ↑ {np.sqrt(gain_needed_linear):.1f}×")

        print(f"\n方案3: 平衡方案（推荐）")
        power_factor = np.sqrt(gain_needed_linear)
        hw_factor = np.sqrt(gain_needed_linear)
        print(f"  发射功率 ↑ {power_factor:.1f}×")
        print(f"  硬件质量 ↓ {hw_factor:.1f}× (劣化)")
        print(f"  具体参数:")
        print(f"    P_tx_fixed      = {P_tx * power_factor:.3e} W")
        print(f"    gamma_pa_floor  = {0.15 * hw_factor:.4f} (示例)")
        print(f"    gamma_adc_bits  = 6  (降低精度)")
        print(f"    gamma_iq_irr_dbc = -20 dBc (降低质量)")
        print(f"    gamma_lo_jitter = {30e-15 * hw_factor:.1e} s")


def plot_noise_breakdown(noise_diag: Dict, output_dir: Path):
    """可视化噪声组成"""
    print_section("生成噪声组成图")

    components = noise_diag['hardware_components']
    N0 = noise_diag['N0_white']
    sigma2_gamma_psd = noise_diag['sigma2_gamma_psd']

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 子图1: 硬件失真组件
    labels = ['PA', 'ADC', 'IQ', 'LO']
    values = [components['Gamma_pa'], components['Gamma_adc'],
              components['Gamma_iq'], components['Gamma_lo']]

    ax1.bar(labels, values, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    ax1.set_ylabel('Distortion Coefficient')
    ax1.set_title('Hardware Distortion Breakdown')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 子图2: 总噪声PSD对比
    labels2 = ['Thermal\nNoise\n(N₀)', 'Hardware\nDistortion\n(σ²_γ/B)']
    values2 = [N0, sigma2_gamma_psd]
    colors = ['#3498db', '#e74c3c']

    bars = ax2.bar(labels2, values2, color=colors, alpha=0.7)
    ax2.set_ylabel('Noise PSD (W/Hz)')
    ax2.set_title('Noise Power Comparison')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, val in zip(bars, values2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height * 1.5,
                 f'{val:.2e}',
                 ha='center', va='bottom', fontsize=8)

    # 添加比例标注
    ratio_db = noise_diag['ratio_db']
    ax2.text(0.5, 0.95, f'Ratio: {ratio_db:.1f} dB',
             transform=ax2.transAxes,
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
             fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'noise_diagnosis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存图表: {output_file}")
    plt.close()


def main(config_path='config.yaml'):
    """主诊断流程"""
    print("=" * 80)
    print(" 硬件损伤影响诊断工具")
    print(" 目的：找出为什么修改硬件参数不影响RMSE")
    print("=" * 80)

    # 加载配置
    print(f"\n加载配置: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        sys.exit(1)

    # 输出目录
    output_dir = Path('./figures')
    output_dir.mkdir(exist_ok=True)

    # 运行诊断
    noise_diag = diagnose_noise_components(config)
    bcrlb_diag = diagnose_bcrlb_calculation(config, noise_diag)
    sensitivity = diagnose_parameter_sensitivity(config)
    recommend_parameters(noise_diag)

    # 可视化
    plot_noise_breakdown(noise_diag, output_dir)

    # 总结
    print_section("诊断总结")

    ratio_db = noise_diag['ratio_db']

    print(f"\n根本原因:")
    if ratio_db < -20:
        print(f"  ❌ 硬件失真功率远小于热噪声 ({ratio_db:.1f} dB)")
        print(f"     即使修改硬件参数，失真仍然可以忽略")
        print(f"     需要同时提升发射功率和/或降低硬件质量")
    elif ratio_db < -10:
        print(f"  ⚠️  硬件失真偏小 ({ratio_db:.1f} dB)")
        print(f"     HW 影响微弱但可能存在")
    else:
        print(f"  ✓ 硬件失真足够显著 ({ratio_db:.1f} dB)")

    print(f"\n参数敏感性测试结果:")
    print(f"  PA失真×10:     RMSE变化 {(sensitivity['RMSE_PA_10x'] / sensitivity['RMSE_base'] - 1) * 100:+.2f}%")
    print(f"  发射功率×1000: RMSE变化 {(sensitivity['RMSE_Ptx_1000x'] / sensitivity['RMSE_base'] - 1) * 100:+.2f}%")
    print(f"  两者结合:      RMSE变化 {(sensitivity['RMSE_both'] / sensitivity['RMSE_base'] - 1) * 100:+.2f}%")

    print(f"\n下一步行动:")
    print(f"  1. 修改 config.yaml 按照「方案3」的参数建议")
    print(f"  2. 重新运行 hardware_ablation_study.py")
    print(f"  3. 检查 HW 曲线是否高于 AWGN")
    print(f"  4. 查看 noise_diagnosis.png 确认噪声比例")

    print("\n" + "=" * 80)
    print("✓ 诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hardware Impact Diagnosis Tool')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Configuration file (default: config.yaml)')

    args = parser.parse_args()
    main(config_path=args.config)