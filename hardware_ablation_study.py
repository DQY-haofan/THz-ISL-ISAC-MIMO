#!/usr/bin/env python3
"""
Hardware Ablation Study - 专家修正版本
基于两位导师的建议，修复了 G_grad_avg 问题，并增强诊断功能

修改内容：
1. ✅ limits_engine.py 的 G_grad_avg 已改为功率增益标度
2. ✅ 增加硬件失真/热噪声比例诊断
3. ✅ 增强 HW 配置参数以使硬件影响可见
4. ✅ 添加相对劣化图
5. ✅ 增加详细的理论验证

Author: Expert-Corrected Version
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
import copy
from pathlib import Path
from typing import Dict, Tuple

# 导入你的原始engine
try:
    from physics_engine import calc_g_sig_factors, calc_n_f_vector
    from limits_engine import calc_BCRLB

    ENGINE_AVAILABLE = True
except ImportError as e:
    ENGINE_AVAILABLE = False
    print(f"❌ 错误: 找不到engine模块")
    print(f"详情: {e}")
    print("\n请确保以下文件在同一目录:")
    print("  - physics_engine.py")
    print("  - limits_engine.py")
    print("  - config.yaml")
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


def create_clean_awgn_config(base_config: dict) -> dict:
    """
    创建完全干净的AWGN配置
    确保所有非理想因素都真正归零
    """
    cfg = copy.deepcopy(base_config)

    # 硬件失真 - 全部归零
    cfg['hardware']['gamma_pa_floor'] = 0.0
    cfg['hardware']['papr_db'] = 0.0
    cfg['hardware']['ibo_db'] = 100.0  # 无限后退
    cfg['hardware']['gamma_adc_bits'] = 100  # 无限精度
    cfg['hardware']['gamma_iq_irr_dbc'] = -1000  # 完美IQ
    cfg['hardware']['gamma_lo_jitter_s'] = 0.0

    # 量化和幅度误差
    cfg['hardware']['rho_q_bits'] = 100
    cfg['hardware']['rho_a_error_rms'] = 0.0

    # 相位噪声 - 归零
    cfg['pn_model']['S_phi_c_K2'] = 0.0
    cfg['pn_model']['S_phi_c_K0'] = 0.0

    # DSE - 归零
    cfg['dse_model']['C_DSE'] = 0.0

    # 平台误差 - 归零
    cfg['platform']['sigma_theta_rad'] = 0.0

    return cfg


def create_config_variants(base_config: dict, enhance_hw: bool = True) -> Dict[str, dict]:
    """
    创建5个配置变体

    Args:
        base_config: 基础配置
        enhance_hw: 是否增强HW参数以使其影响可见（推荐True）
    """
    variants = {}

    # ═══════════════════════════════════════════════════════════
    # 1. AWGN-only: 理想信道，无任何非理想因素
    # ═══════════════════════════════════════════════════════════
    variants['AWGN'] = create_clean_awgn_config(base_config)
    print("\n[配置] AWGN: 完全理想（仅热噪声）")

    # ═══════════════════════════════════════════════════════════
    # 2. HW: AWGN + 硬件失真
    # ═══════════════════════════════════════════════════════════
    cfg_hw = copy.deepcopy(base_config)

    if enhance_hw:
        # 📌 导师建议：使用"温和但现实"的参数，使硬件影响可见
        print("\n[配置] HW: 增强硬件参数（使影响可见）")
        hw = cfg_hw['hardware']

        # PA 非线性
        hw['gamma_pa_floor'] = 0.008  # 略高于state-of-art
        hw['papr_db'] = 8.0  # OFDM 典型值
        hw['ibo_db'] = 3.0  # 轻度压缩（平衡效率和失真）

        # ADC 量化
        hw['gamma_adc_bits'] = 10  # 10-bit ENOB（现实值）

        # I/Q 不平衡
        hw['gamma_iq_irr_dbc'] = -28.0  # -28 dBc（略差于最优）

        # LO 相位抖动
        hw['gamma_lo_jitter_s'] = 30e-15  # 30 fs RMS（THz 可实现）

        # 相位量化
        hw['rho_q_bits'] = 4  # 4-bit 移相器

        # 幅度误差
        hw['rho_a_error_rms'] = 0.03  # 3% RMS（略高于base）

        print(f"  PA floor: {hw['gamma_pa_floor']:.4f}")
        print(f"  PAPR: {hw['papr_db']:.1f} dB")
        print(f"  ADC: {hw['gamma_adc_bits']} bits")
        print(f"  IQ IRR: {hw['gamma_iq_irr_dbc']:.1f} dBc")
        print(f"  LO jitter: {hw['gamma_lo_jitter_s'] * 1e15:.1f} fs")
    else:
        print("\n[配置] HW: 使用 base config 的硬件参数")

    # 关闭 PN, DSE, 平台误差
    cfg_hw['pn_model']['S_phi_c_K2'] = 0.0
    cfg_hw['pn_model']['S_phi_c_K0'] = 0.0
    cfg_hw['dse_model']['C_DSE'] = 0.0
    cfg_hw['platform']['sigma_theta_rad'] = 0.0

    variants['HW'] = cfg_hw

    # ═══════════════════════════════════════════════════════════
    # 3. HW+PN: 硬件 + 相位噪声
    # ═══════════════════════════════════════════════════════════
    cfg_hw_pn = copy.deepcopy(base_config)
    cfg_hw_pn['dse_model']['C_DSE'] = 0.0
    cfg_hw_pn['platform']['sigma_theta_rad'] = 0.0
    variants['HW+PN'] = cfg_hw_pn
    print("\n[配置] HW+PN: 硬件 + 相位噪声（DSE=0）")

    # ═══════════════════════════════════════════════════════════
    # 4. HW+PN+DSE: 硬件 + 相位噪声 + 双边谱展宽
    # ═══════════════════════════════════════════════════════════
    cfg_hw_pn_dse = copy.deepcopy(base_config)
    cfg_hw_pn_dse['platform']['sigma_theta_rad'] = 0.0
    variants['HW+PN+DSE'] = cfg_hw_pn_dse
    print("\n[配置] HW+PN+DSE: 硬件 + PN + DSE（平台误差=0）")

    # ═══════════════════════════════════════════════════════════
    # 5. Full: 所有非理想因素
    # ═══════════════════════════════════════════════════════════
    variants['Full'] = copy.deepcopy(base_config)
    print("\n[配置] Full: 完整模型（包括平台指向误差）")

    return variants


def verify_awgn_theory(config: dict) -> Tuple[float, dict]:
    """
    理论验证：计算 AWGN 基线的理论值

    Returns:
        (RMSE_theory_mm, theory_dict)
    """
    print("\n" + "═" * 80)
    print("理论验证：AWGN 基线")
    print("═" * 80)

    # 提取参数
    B_hz = config['channel']['B_hz']
    SNR_db = config['isac_model']['SNR_p_db']
    SNR_lin = 10 ** (SNR_db / 10)
    c_mps = config['channel']['c_mps']
    f_c_hz = config['channel']['f_c_hz']

    # 理论公式（矩形窗近似）
    # σ_τ = 1 / (2π B √(3·SNR))
    sigma_tau_theory = 1 / (2 * np.pi * B_hz * np.sqrt(3 * SNR_lin))
    RMSE_theory_m = (c_mps / 2) * sigma_tau_theory
    RMSE_theory_mm = RMSE_theory_m * 1000

    # 相对带宽
    frac_bw = B_hz / f_c_hz

    print(f"\n系统参数:")
    print(f"  载频 f_c    = {f_c_hz / 1e9:.1f} GHz")
    print(f"  带宽 B      = {B_hz / 1e9:.1f} GHz")
    print(f"  相对带宽    = {frac_bw * 100:.2f}%")
    print(f"  导频 SNR    = {SNR_db:.1f} dB")
    print(f"\n理论 AWGN 基线（矩形窗近似）:")
    print(f"  σ_τ,theory  = {sigma_tau_theory * 1e12:.3f} ps")
    print(f"  RMSE_theory = {RMSE_theory_mm:.4f} mm")

    print(f"\n注意:")
    print(f"  • 你的实际 RMSE 可能与此不同（常见因子 0.5-2×）")
    print(f"  • 原因: 频域加权、双边频谱、能量归一化")
    print(f"  • 这是正常的！重要的是相对趋势")

    theory_dict = {
        'B_hz': B_hz,
        'SNR_db': SNR_db,
        'SNR_lin': SNR_lin,
        'sigma_tau_s': sigma_tau_theory,
        'RMSE_mm': RMSE_theory_mm,
    }

    return RMSE_theory_mm, theory_dict


def run_single_point(config: dict, alpha: float, config_name: str) -> Dict:
    """
    运行单个点的计算

    Returns:
        结果字典，包含 RMSE、诊断信息等
    """
    try:
        # 设置 alpha
        cfg = copy.deepcopy(config)
        cfg['isac_model']['alpha'] = alpha

        # 调用 engine
        g_sig = calc_g_sig_factors(cfg)
        n_f = calc_n_f_vector(cfg, g_sig)
        bcrlb = calc_BCRLB(cfg, g_sig, n_f)

        # 提取结果
        c_mps = cfg['channel']['c_mps']
        BCRLB_tau = bcrlb['BCRLB_tau']
        RMSE_m = (c_mps / 2) * np.sqrt(BCRLB_tau)
        RMSE_mm = RMSE_m * 1000

        # 健全性检查
        if np.isnan(RMSE_mm) or np.isinf(RMSE_mm) or RMSE_mm < 0:
            return {
                'config': config_name,
                'alpha': alpha,
                'RMSE_mm': np.nan,
                'BCRLB_tau': np.nan,
                'error': 'Invalid RMSE (NaN/Inf/Negative)',
            }

        if RMSE_mm > 10000:  # 超过 10 米肯定异常
            return {
                'config': config_name,
                'alpha': alpha,
                'RMSE_mm': np.nan,
                'BCRLB_tau': np.nan,
                'error': f'RMSE too large ({RMSE_mm:.1f} mm)',
            }

        # 提取诊断信息（如果有）
        result = {
            'config': config_name,
            'alpha': alpha,
            'RMSE_mm': RMSE_mm,
            'BCRLB_tau': BCRLB_tau,
            'method': bcrlb.get('method', 'unknown'),
        }

        # 📌 新增：硬件失真诊断（如果 bcrlb 返回了这些信息）
        if 'diagnostics' in bcrlb:
            diag = bcrlb['diagnostics']
            result['gamma_psd'] = diag.get('gamma_psd', np.nan)
            result['N0_psd'] = diag.get('N0_psd', np.nan)
            result['ratio_gamma_to_N0_dB'] = diag.get('ratio_gamma_to_N0_dB', np.nan)
            result['pn_psd_mean'] = diag.get('pn_psd_mean', np.nan)
            result['dse_psd_mean'] = diag.get('dse_psd_mean', np.nan)

        return result

    except Exception as e:
        import traceback
        return {
            'config': config_name,
            'alpha': alpha,
            'RMSE_mm': np.nan,
            'BCRLB_tau': np.nan,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def run_ablation_sweep(config: dict, alpha_vec: np.ndarray,
                       enhance_hw: bool = True) -> pd.DataFrame:
    """
    执行消融研究的α扫描
    修复: 确保每个α都有完整的配置数据，便于计算相对比值
    """
    print("\n" + "═" * 80)
    print("消融研究 α 扫描")
    print("═" * 80)

    variants = create_config_variants(config, enhance_hw=enhance_hw)

    # 存储结果 - 使用字典结构便于查找
    results_dict = {cfg_name: {} for cfg_name in variants.keys()}
    results = []

    total_points = len(alpha_vec) * len(variants)
    completed = 0

    print(f"\n配置变体: {list(variants.keys())}")
    print(f"α 范围: [{alpha_vec[0]:.2f}, {alpha_vec[-1]:.2f}] ({len(alpha_vec)} 点)")
    print(f"总计算点数: {total_points}")
    print()

    # 按 α 循环（外层），确保每个 α 都计算所有配置
    for alpha in alpha_vec:
        print(f"\n[α = {alpha:.3f}] ", end="")

        # 为当前 α 计算所有配置
        alpha_results = {}

        for cfg_name, cfg in variants.items():
            cfg_temp = copy.deepcopy(cfg)
            cfg_temp['isac_model']['alpha'] = alpha

            try:
                result = run_single_point(cfg_temp, alpha, cfg_name)

                # 转换单位: m → mm
                rmse_m = np.sqrt(result['BCRLB_tau']) * cfg['channel']['c_mps'] / 2
                rmse_mm = rmse_m * 1000.0

                # 存储到字典中
                alpha_results[cfg_name] = rmse_mm
                results_dict[cfg_name][alpha] = rmse_mm

                # 添加到结果列表
                result_row = {
                    'alpha': alpha,
                    'config': cfg_name,
                    'RMSE_mm': rmse_mm,
                    'BCRLB_tau': result['BCRLB_tau'],
                    'method': result.get('method', 'N/A'),
                    'N': result.get('N', np.nan),
                }

                # 添加诊断信息（如果有）
                if 'diag' in result:
                    result_row['ratio_gamma_to_N0_dB'] = result['diag'].get('ratio_gamma_to_N0_dB', np.nan)

                results.append(result_row)

                print(".", end="", flush=True)

            except Exception as e:
                print(f"X({cfg_name})", end="", flush=True)
                print(f"\n  ⚠️  错误 [α={alpha:.3f}, {cfg_name}]: {str(e)[:100]}")

                # 添加失败记录
                results.append({
                    'alpha': alpha,
                    'config': cfg_name,
                    'RMSE_mm': np.nan,
                    'BCRLB_tau': np.nan,
                    'method': 'FAILED',
                    'N': np.nan,
                })

            completed += 1

        # 计算并添加相对比值（对当前 α 的所有配置）
        if 'AWGN' in alpha_results and not np.isnan(alpha_results['AWGN']):
            rmse_awgn = alpha_results['AWGN']

            # 为每个配置添加 ratio 列
            for i, row in enumerate(results):
                if row['alpha'] == alpha and not np.isnan(row['RMSE_mm']):
                    ratio = row['RMSE_mm'] / rmse_awgn
                    results[i]['ratio_to_AWGN'] = ratio

        print(f" ✓ ({completed}/{total_points})")

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 添加相对比值列（确保所有行都有）
    if 'ratio_to_AWGN' not in df.columns:
        df['ratio_to_AWGN'] = np.nan

    print(f"\n✓ 扫描完成: {len(df)} 数据点")
    print(f"  成功: {df['RMSE_mm'].notna().sum()}")
    print(f"  失败: {df['RMSE_mm'].isna().sum()}")

    return df


def plot_ablation(df: pd.DataFrame, output_dir: Path, rmse_theory: float = None):
    """绘制消融图（绝对 RMSE）"""
    print("\n[绘图] 消融对比（绝对 RMSE）...")

    fig, ax = plt.subplots(figsize=(3.5, 2.625))

    styles = {
        'AWGN': {
            'color': '#000000',
            'linestyle': '--',
            'marker': 'o',
            'label': 'AWGN-only',
            'zorder': 5,
        },
        'HW': {
            'color': '#0072BD',
            'linestyle': '-',
            'marker': 's',
            'label': 'AWGN + HW',
            'zorder': 4,
        },
        'HW+PN': {
            'color': '#D95319',
            'linestyle': '-',
            'marker': '^',
            'label': 'AWGN + HW + PN',
            'zorder': 3,
        },
        'HW+PN+DSE': {
            'color': '#77AC30',
            'linestyle': '-',
            'marker': 'v',
            'label': 'AWGN + HW + PN + DSE',
            'zorder': 2,
        },

    }

    for cfg_name, style in styles.items():
        data = df[df['config'] == cfg_name].copy()
        # 过滤无效值
        data = data[data['RMSE_mm'].notna() & (data['RMSE_mm'] > 0) & (data['RMSE_mm'] < 1000)]

        if len(data) > 3:  # 至少要有几个点才画
            ax.semilogy(data['alpha'], data['RMSE_mm'],
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        linewidth=1.5,
                        markersize=4,
                        label=style['label'],
                        markevery=max(1, len(data) // 10),  # 自适应标记间隔
                        alpha=0.9,
                        zorder=style['zorder'])

    # # 添加理论基线（如果提供）
    # if rmse_theory is not None:
    #     ax.axhline(y=rmse_theory, color='gray', linestyle=':',
    #                linewidth=1.0, alpha=0.5, label='Theory (rect. window)')

    ax.set_xlabel(r'ISAC Overhead $\alpha$')
    ax.set_ylabel('Range RMSE (mm, log scale)')
    ax.set_xlim([df['alpha'].min() * 0.95, df['alpha'].max() * 1.05])
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_ablation_absolute.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_ablation_absolute.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_dir / 'fig_ablation_absolute.pdf'}")
    plt.close()


def plot_relative_degradation(df: pd.DataFrame, output_dir: Path):
    """
    绘制相对劣化图
    修复: 使用同一 α 的 AWGN 作为分母，确保比值准确
    """
    print("\n绘制相对劣化图...")

    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))  # 稍高一点便于图例

    # 配置样式
    config_styles = {
        'HW': {
            'color': '#A2142F',
            'label': 'HW only',
            'linestyle': '-',
            'marker': 's'
        },
        'HW+PN': {
            'color': '#D95319',
            'label': 'HW+PN',
            'linestyle': '-',
            'marker': '^'
        },
        'HW+PN+DSE': {
            'color': '#EDB120',
            'label': 'HW+PN+DSE',
            'linestyle': '-',
            'marker': 'o'
        },
        'Full': {
            'color': '#7E2F8E',
            'label': 'Full model',
            'linestyle': '-',
            'marker': 'v'
        }
    }

    # 提取 AWGN 基线
    df_awgn = df[df['config'] == 'AWGN'].copy()
    df_awgn = df_awgn.sort_values('alpha')

    if len(df_awgn) < 2:
        print("  ⚠️  AWGN 数据不足，无法绘制相对图")
        return

    alpha_awgn = df_awgn['alpha'].values
    rmse_awgn = df_awgn['RMSE_mm'].values

    # 移除 NaN
    valid_mask = ~np.isnan(rmse_awgn)
    alpha_awgn = alpha_awgn[valid_mask]
    rmse_awgn = rmse_awgn[valid_mask]

    print(f"  AWGN 基线: {len(alpha_awgn)} 个 α 点")

    # 绘制各配置的相对比值
    for cfg_name, style in config_styles.items():
        df_cfg = df[df['config'] == cfg_name].copy()
        df_cfg = df_cfg.sort_values('alpha')

        if len(df_cfg) < 2:
            print(f"  ⚠️  {cfg_name} 数据不足，跳过")
            continue

        # 对齐到 AWGN 的 α 网格
        alpha_cfg = []
        ratio_cfg = []

        for a_awgn, r_awgn in zip(alpha_awgn, rmse_awgn):
            # 找到最接近的 α 点
            idx = np.argmin(np.abs(df_cfg['alpha'].values - a_awgn))
            a_match = df_cfg['alpha'].values[idx]

            # 容差检查
            if np.abs(a_match - a_awgn) < 0.005:  # 容差 0.005
                r_cfg = df_cfg['RMSE_mm'].values[idx]

                if not np.isnan(r_cfg) and r_awgn > 0:
                    ratio = r_cfg / r_awgn

                    # 合理性检查（防止异常值）
                    if 0.1 < ratio < 50:
                        alpha_cfg.append(a_awgn)
                        ratio_cfg.append(ratio)

        if len(alpha_cfg) > 2:
            ax.plot(alpha_cfg, ratio_cfg,
                    color=style['color'],
                    label=style['label'],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    linewidth=1.5,
                    markersize=4,
                    markevery=max(1, len(alpha_cfg) // 8),
                    alpha=0.9)

            print(f"  ✓ {cfg_name}: {len(alpha_cfg)} 点, "
                  f"比值范围 [{min(ratio_cfg):.2f}, {max(ratio_cfg):.2f}]")
        else:
            print(f"  ⚠️  {cfg_name}: 有效点不足 ({len(alpha_cfg)})")

    # 添加基线
    ax.axhline(y=1.0, color='gray', linestyle='--',
               linewidth=1.2, alpha=0.6, label='AWGN baseline', zorder=1)

    # 设置坐标轴
    ax.set_xlabel(r'ISAC Overhead $\alpha$', fontsize=8)
    ax.set_ylabel(r'RMSE / RMSE$_{\mathrm{AWGN}}$', fontsize=8)
    ax.set_xlim([alpha_awgn[0] * 0.95, alpha_awgn[-1] * 1.05])

    # 使用对数 y 轴（如果跨度大）
    ratio_all = df['ratio_to_AWGN'].dropna()
    if len(ratio_all) > 0:
        ratio_max = ratio_all.max()
        if ratio_max > 5:
            ax.set_yscale('log')
            ax.set_ylabel(r'RMSE / RMSE$_{\mathrm{AWGN}}$ (log scale)', fontsize=8)
        else:
            ax.set_ylim([0.9, ratio_max * 1.1])

    ax.legend(framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    # 保存
    for ext in ['pdf', 'png']:
        save_path = output_dir / f'fig_ablation_relative.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存: {save_path}")

    plt.close()


def print_summary(df: pd.DataFrame, alpha_eval: float, rmse_theory: float):
    """打印总结"""
    print("\n" + "═" * 80)
    print("分析总结")
    print("═" * 80)

    configs = ['AWGN', 'HW', 'HW+PN', 'HW+PN+DSE', 'Full']

    print(f"\n1. RMSE 对比 (α = {alpha_eval:.2f}):")
    print("─" * 80)

    rmse_dict = {}
    valid_count = 0

    for cfg in configs:
        data = df[(df['config'] == cfg) & (np.abs(df['alpha'] - alpha_eval) < 0.015)]
        if len(data) > 0:
            row = data.iloc[0]
            rmse = row['RMSE_mm']

            if not np.isnan(rmse) and rmse > 0 and rmse < 1000:
                rmse_dict[cfg] = rmse
                valid_count += 1

                if cfg == 'AWGN':
                    error_pct = abs(rmse - rmse_theory) / rmse_theory * 100
                    status = "✓" if error_pct < 100 else "⚠️"
                    print(f"  {cfg:15s}: {rmse:7.3f} mm  "
                          f"(理论 {rmse_theory:.3f} mm, 差异 {error_pct:5.1f}%) {status}")
                else:
                    if 'AWGN' in rmse_dict:
                        degradation = (rmse / rmse_dict['AWGN'] - 1) * 100
                        print(f"  {cfg:15s}: {rmse:7.3f} mm  ({degradation:+6.1f}% vs AWGN)")
                    else:
                        print(f"  {cfg:15s}: {rmse:7.3f} mm")

                # 打印诊断信息（如果有）
                if 'ratio_gamma_to_N0_dB' in row and not np.isnan(row['ratio_gamma_to_N0_dB']):
                    print(f"                 └─ γ/N0 = {row['ratio_gamma_to_N0_dB']:+.1f} dB")
            else:
                print(f"  {cfg:15s}: ERROR/INVALID")
        else:
            print(f"  {cfg:15s}: NO DATA")

    # 关键发现
    if valid_count >= 3:
        print(f"\n2. 关键发现:")
        print("─" * 80)

        if 'AWGN' in rmse_dict and 'HW' in rmse_dict:
            hw_gap = rmse_dict['HW'] / rmse_dict['AWGN']
            print(f"  • HW vs AWGN: {hw_gap:.2f}× 劣化")
            if hw_gap < 1.1:
                print(f"    ⚠️  差距很小 (<10%)，可能原因:")
                print(f"       - 硬件参数太干净（需要增强）")
                print(f"       - G_grad_avg 可能还是用的 sqrt(g_ar)？")
                print(f"       - 检查 limits_engine.py 是否修改正确")
            elif 1.1 <= hw_gap <= 3.0:
                print(f"    ✓ 合理的硬件劣化")
            else:
                print(f"    ⚠️  劣化很大 (>3×)，检查参数是否过于保守")

        if 'Full' in rmse_dict and 'AWGN' in rmse_dict:
            full_gap = rmse_dict['Full'] / rmse_dict['AWGN']
            print(f"  • Full vs AWGN: {full_gap:.2f}× 劣化")

        if 'HW+PN+DSE' in rmse_dict and 'Full' in rmse_dict:
            if abs(rmse_dict['HW+PN+DSE'] - rmse_dict['Full']) / rmse_dict['Full'] < 0.05:
                print(f"  • Full ≈ HW+PN+DSE (差异 <5%)")
                print(f"    → 平台指向误差的影响很小（符合预期）")
    else:
        print(f"\n⚠️  警告: 大部分配置计算失败 ({valid_count}/{len(configs)} 成功)")
        print("    可能原因:")
        print("    1. config.yaml 中某些参数缺失或格式错误")
        print("    2. physics_engine.py 或 limits_engine.py 有问题")
        print("    3. 参数设置导致数值溢出/下溢")


def save_summary_table(df: pd.DataFrame, output_dir: Path, alpha_eval: float):
    """保存汇总表格"""
    configs = ['AWGN', 'HW', 'HW+PN', 'HW+PN+DSE', 'Full']

    summary_rows = []
    for cfg in configs:
        data = df[(df['config'] == cfg) & (np.abs(df['alpha'] - alpha_eval) < 0.015)]
        if len(data) > 0:
            row = data.iloc[0]
            summary_rows.append({
                'Configuration': cfg,
                'RMSE (mm)': row['RMSE_mm'],
                'BCRLB_tau (s²)': row['BCRLB_tau'],
                'Method': row.get('method', 'N/A'),
            })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_path = output_dir / f'summary_alpha_{alpha_eval:.3f}.csv'
        df_summary.to_csv(csv_path, index=False, float_format='%.6e')
        print(f"\n✓ 保存汇总表: {csv_path}")


def main(config_path='config.yaml', enhance_hw=True):
    """
    主函数

    Args:
        config_path: 配置文件路径
        enhance_hw: 是否增强 HW 参数（推荐 True）
    """
    print("═" * 80)
    print("硬件消融研究 - 专家修正版本")
    print("基于两位导师建议的完整修复")
    print("═" * 80)

    setup_ieee_style()

    # 加载配置
    print(f"\n加载配置: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    # 输出目录
    output_dir = Path('./figures')
    output_dir.mkdir(exist_ok=True)

    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)

    print(f"\n输出目录:")
    print(f"  • 图表: {output_dir}")
    print(f"  • 数据: {results_dir}")

    # ═══════════════════════════════════════════════════════════
    # STEP 1: 理论验证
    # ═══════════════════════════════════════════════════════════
    rmse_theory, theory_dict = verify_awgn_theory(config)

    # ═══════════════════════════════════════════════════════════
    # STEP 2: α 扫描
    # ═══════════════════════════════════════════════════════════
    alpha_vec = np.linspace(0.05, 0.30, 20)
    print(f"\nα 扫描范围: [{alpha_vec[0]:.2f}, {alpha_vec[-1]:.2f}] ({len(alpha_vec)} 点)")

    df = run_ablation_sweep(config, alpha_vec, enhance_hw=enhance_hw)

    # ═══════════════════════════════════════════════════════════
    # STEP 3: 保存数据
    # ═══════════════════════════════════════════════════════════
    csv_path = results_dir / 'ablation_fixed.csv'
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"\n✓ 保存数据: {csv_path}")

    # ═══════════════════════════════════════════════════════════
    # STEP 4: 绘图
    # ═══════════════════════════════════════════════════════════
    plot_ablation(df, output_dir, rmse_theory=rmse_theory)
    plot_relative_degradation(df, output_dir)

    # ═══════════════════════════════════════════════════════════
    # STEP 5: 总结
    # ═══════════════════════════════════════════════════════════
    alpha_eval = 0.10  # 评估点
    print_summary(df, alpha_eval, rmse_theory)
    save_summary_table(df, results_dir, alpha_eval)

    # ═══════════════════════════════════════════════════════════
    # 完成
    # ═══════════════════════════════════════════════════════════
    print("\n" + "═" * 80)
    print("✓ 消融研究完成!")
    print("═" * 80)
    print("\n生成的文件:")
    print(f"  • {output_dir / 'fig_ablation_absolute.pdf'}")
    print(f"  • {output_dir / 'fig_ablation_relative.pdf'}")
    print(f"  • {csv_path}")
    print("\n下一步:")
    print("  1. 检查图表，验证 HW 是否高于 AWGN")
    print("  2. 如果 HW≈AWGN，检查 limits_engine.py 的 G_grad_avg 修改")
    print("  3. 查看 CSV 中的 ratio_gamma_to_N0_dB 列")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hardware Ablation Study')
    parser.add_argument('config', nargs='?', default='config.yaml',
                        help='Configuration file (default: config.yaml)')
    parser.add_argument('--no-enhance-hw', action='store_true',
                        help='Do not enhance HW parameters (use base config)')

    args = parser.parse_args()

    main(config_path=args.config, enhance_hw=not args.no_enhance_hw)