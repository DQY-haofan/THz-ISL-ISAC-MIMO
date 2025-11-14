#!/usr/bin/env python3
"""
===================================================================
综合诊断脚本：精确定位 HW ≈ AWGN 的根本原因
===================================================================

诊断层级：
1. 能量标度诊断（G_grad_avg 是否正确）
2. 硬件失真量级诊断（σ²_γ/N0 比值）
3. 参数敏感性诊断（哪些参数影响最大）

使用方法：
    python comprehensive_diagnosis.py config.yaml
"""

import numpy as np
import yaml
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt

# 导入你的引擎
try:
    from physics_engine import calc_g_sig_factors, calc_eta_bsq_factors
    from limits_engine import calc_BCRLB, calc_n_f_vector
except ImportError as e:
    print(f"❌ 无法导入引擎模块: {e}")
    print("请确保 physics_engine.py 和 limits_engine.py 在当前目录")
    sys.exit(1)


class HWDiagnostics:
    """硬件失真诊断工具类"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 固定 α = 0.1 用于诊断
        self.config['isac_model']['alpha'] = 0.1

        # 提取关键参数
        self.Nt = self.config['array']['Nt']
        self.Nr = self.config['array']['Nr']
        self.g_ar = self.Nt * self.Nr
        self.B_hz = self.config['channel']['B_hz']
        self.f_c_hz = self.config['channel']['f_c_hz']
        self.c_mps = self.config['channel']['c_mps']
        self.SNR_p_db = self.config['isac_model']['SNR_p_db']
        self.SNR_p_lin = 10 ** (self.SNR_p_db / 10)

    def diagnose_energy_scaling(self):
        """第一层诊断：能量标度是否正确"""
        print("\n" + "=" * 80)
        print("【诊断层1】能量标度检查：G_grad_avg 计算方式")
        print("=" * 80)

        g_sig = calc_g_sig_factors(self.config)

        eta_bsq_avg = g_sig['eta_bsq_avg']
        rho_Q = g_sig['rho_Q']
        rho_APE = g_sig['rho_APE']
        rho_A = g_sig['rho_A']

        # 两种计算方式
        G_grad_amplitude = np.sqrt(self.g_ar) * rho_Q * rho_APE * rho_A
        G_grad_power = self.g_ar * eta_bsq_avg * rho_Q * rho_APE * rho_A

        print(f"\n系统参数：")
        print(f"  Nt × Nr = {self.Nt} × {self.Nr} = {self.g_ar}")
        print(f"  η²_bsq_avg = {eta_bsq_avg:.6f}")
        print(f"  ρ_Q = {rho_Q:.6f}")
        print(f"  ρ_APE = {rho_APE:.6f}")
        print(f"  ρ_A = {rho_A:.6f}")

        print(f"\n方式A（幅度增益 - 可能有问题）：")
        print(f"  G_grad = √g_ar × ρ_Q × ρ_APE × ρ_A")
        print(f"         = √{self.g_ar} × {rho_Q:.4f} × {rho_APE:.4f} × {rho_A:.4f}")
        print(f"         = {G_grad_amplitude:.2f}")
        print(f"         = {10 * np.log10(G_grad_amplitude):.1f} dB")

        print(f"\n方式B（功率增益 - 推荐）：")
        print(f"  G_grad = g_ar × η²_bsq × ρ_Q × ρ_APE × ρ_A")
        print(f"         = {self.g_ar} × {eta_bsq_avg:.4f} × {rho_Q:.4f} × {rho_APE:.4f} × {rho_A:.4f}")
        print(f"         = {G_grad_power:.2f}")
        print(f"         = {10 * np.log10(G_grad_power):.1f} dB")

        ratio = G_grad_power / G_grad_amplitude
        print(f"\n差异倍数：{ratio:.1f}×")
        print(f"差异 (dB)：{10 * np.log10(ratio):.1f} dB")

        # 判断当前使用的是哪种
        print(f"\n" + "-" * 80)
        if ratio > 10:
            print("⚠️  检测到：功率增益应该比幅度增益大 {:.1f}× ({:.1f} dB)".format(ratio, 10 * np.log10(ratio)))
            print("   如果你的 limits_engine.py 使用了方式A，这会导致：")
            print("   • P_tx_eff 被低估 {:.1f}× ".format(ratio))
            print("   • σ²_γ 被低估 {:.1f}× ".format(ratio))
            print("   • HW失真'消失'，HW ≈ AWGN")
            verdict_scaling = "❌ 错误"
        else:
            print("✓ 两种方式差异不大，能量标度可能正确")
            verdict_scaling = "✓ 正确"

        return verdict_scaling, G_grad_amplitude, G_grad_power

    def diagnose_hardware_magnitude(self):
        """第二层诊断：硬件失真量级"""
        print("\n" + "=" * 80)
        print("【诊断层2】硬件失真量级：σ²_γ/N0 比值分析")
        print("=" * 80)

        g_sig = calc_g_sig_factors(self.config)
        n_f = calc_n_f_vector(self.config, g_sig)

        # 提取关键量
        N0_white = n_f['N0_white']
        sigma2_gamma_new = n_f.get('sigma2_gamma_new', n_f.get('sigma2_gamma', 0))

        # 计算 PSD
        gamma_psd = sigma2_gamma_new / self.B_hz
        ratio_gamma2white = gamma_psd / N0_white
        ratio_db = 10 * np.log10(ratio_gamma2white) if ratio_gamma2white > 0 else -np.inf

        print(f"\n热噪声基线：")
        print(f"  N0 = {N0_white:.3e} W/Hz")
        print(f"  N0 = {10 * np.log10(N0_white * 1e3):.1f} dBm/Hz")

        print(f"\n硬件失真：")
        print(f"  σ²_γ = {sigma2_gamma_new:.3e} W")
        print(f"  σ²_γ/B = {gamma_psd:.3e} W/Hz")

        print(f"\n关键比值：")
        print(f"  (σ²_γ/B) / N0 = {ratio_gamma2white:.6f}")
        print(f"                = {ratio_db:.1f} dB")

        # 判断
        print(f"\n" + "-" * 80)
        if ratio_db < -20:
            print(f"📊 硬件失真远小于热噪声 ({ratio_db:.1f} dB < -20 dB)")
            print(f"   结论：HW ≈ AWGN 是**合理的物理现象**")
            print(f"   建议：如需在图上看到差异，需增强硬件失真参数")
            verdict_magnitude = "物理合理"
        elif -20 <= ratio_db < -10:
            print(f"📊 硬件失真略小于热噪声 ({ratio_db:.1f} dB)")
            print(f"   结论：HW 与 AWGN 差异应该微弱可见（~1-5%）")
            print(f"   建议：使用相对劣化图（RMSE_hw/RMSE_awgn）放大差异")
            verdict_magnitude = "边界情况"
        elif -10 <= ratio_db < 0:
            print(f"📊 硬件失真接近热噪声 ({ratio_db:.1f} dB)")
            print(f"   结论：HW 与 AWGN 应有明显差异（~10-50%）")
            print(f"   如果图上看不出，请检查 BCRLB 计算逻辑")
            verdict_magnitude = "应该可见"
        else:
            print(f"📊 硬件失真大于热噪声 ({ratio_db:.1f} dB)")
            print(f"   结论：HW 应显著劣于 AWGN")
            print(f"   如果图上看不出，BCRLB 计算可能有严重错误")
            verdict_magnitude = "必须可见"

        return verdict_magnitude, ratio_db, sigma2_gamma_new, N0_white

    def diagnose_parameter_sensitivity(self):
        """第三层诊断：参数敏感性分析"""
        print("\n" + "=" * 80)
        print("【诊断层3】参数敏感性：哪些硬件参数影响最大")
        print("=" * 80)

        # 基准配置
        g_sig_base = calc_g_sig_factors(self.config)
        n_f_base = calc_n_f_vector(self.config, g_sig_base)
        sigma2_gamma_base = n_f_base.get('sigma2_gamma_new', n_f_base.get('sigma2_gamma', 0))

        # 测试各参数的影响
        params_to_test = [
            ('gamma_pa_floor', [0.001, 0.005, 0.01, 0.02]),
            ('papr_db', [0.1, 3.0, 6.0, 9.0]),
            ('ibo_db', [0.5, 3.0, 6.0, 10.0]),
            ('gamma_adc_bits', [6, 8, 10, 12]),
            ('gamma_iq_irr_dbc', [-40, -30, -20, -15]),
            ('gamma_lo_jitter_s', [1e-15, 10e-15, 50e-15, 100e-15]),
        ]

        sensitivity_results = []

        for param_name, param_values in params_to_test:
            config_test = yaml.safe_load(yaml.dump(self.config))  # 深拷贝

            for val in param_values:
                config_test['hardware'][param_name] = val
                g_sig_test = calc_g_sig_factors(config_test)
                n_f_test = calc_n_f_vector(config_test, g_sig_test)
                sigma2_gamma_test = n_f_test.get('sigma2_gamma_new', n_f_test.get('sigma2_gamma', 0))

                change_ratio = sigma2_gamma_test / sigma2_gamma_base if sigma2_gamma_base > 0 else 0
                change_db = 10 * np.log10(change_ratio) if change_ratio > 0 else -np.inf

                sensitivity_results.append({
                    'parameter': param_name,
                    'value': val,
                    'sigma2_gamma': sigma2_gamma_test,
                    'change_ratio': change_ratio,
                    'change_db': change_db
                })

        # 找出影响最大的参数
        print(f"\n当前配置的 σ²_γ 基准值：{sigma2_gamma_base:.3e} W\n")

        # 按参数分组显示
        for param_name, _ in params_to_test:
            param_results = [r for r in sensitivity_results if r['parameter'] == param_name]
            print(f"\n参数：{param_name}")
            print("-" * 60)

            table_data = []
            for r in param_results:
                table_data.append([
                    f"{r['value']:.2e}" if isinstance(r['value'], float) else r['value'],
                    f"{r['sigma2_gamma']:.3e}",
                    f"{r['change_db']:+.1f} dB"
                ])

            print(tabulate(table_data,
                           headers=['取值', 'σ²_γ', '变化'],
                           tablefmt='simple'))

        return sensitivity_results

    def diagnose_bcrlb_computation(self):
        """第四层诊断：BCRLB计算链路完整性"""
        print("\n" + "=" * 80)
        print("【诊断层4】BCRLB 计算链路检查")
        print("=" * 80)

        g_sig = calc_g_sig_factors(self.config)
        n_f = calc_n_f_vector(self.config, g_sig)
        bcrlb = calc_BCRLB(self.config, g_sig, n_f)

        # 提取中间量
        N0 = n_f['N0_white']
        sigma2_gamma = n_f.get('sigma2_gamma_new', n_f.get('sigma2_gamma', 0))

        # 理论RMSE（AWGN基线）
        sigma_tau_theory = 1 / (2 * np.pi * self.B_hz * np.sqrt(3 * self.SNR_p_lin))
        RMSE_theory = (self.c_mps / 2) * sigma_tau_theory * 1000

        # 实际RMSE
        RMSE_actual = (self.c_mps / 2) * np.sqrt(bcrlb['BCRLB_tau']) * 1000

        # 比较
        ratio = RMSE_actual / RMSE_theory

        print(f"\n理论 AWGN 基线：")
        print(f"  RMSE_theory = {RMSE_theory:.4f} mm")
        print(f"  (基于 B={self.B_hz / 1e9:.0f} GHz, SNR_p={self.SNR_p_db:.0f} dB)")

        print(f"\n实际 BCRLB 计算：")
        print(f"  RMSE_actual = {RMSE_actual:.4f} mm")

        print(f"\n比值：")
        print(f"  RMSE_actual / RMSE_theory = {ratio:.4f}")

        print(f"\n" + "-" * 80)
        if 0.5 <= ratio <= 2.0:
            print("✓ BCRLB 计算链路正常")
            print("  实际 RMSE 在理论值的合理范围内")
            verdict_bcrlb = "✓ 正常"
        elif ratio < 0.5:
            print("⚠️  实际 RMSE 显著小于理论值")
            print("   可能原因：")
            print("   • 频域加窗效应（η_bsq 权重）")
            print("   • G_grad_avg 过度放大")
            print("   • 能量归一化问题")
            verdict_bcrlb = "⚠️ 偏优"
        else:
            print("❌ 实际 RMSE 显著大于理论值")
            print("   可能原因：")
            print("   • G_grad_avg 过度缩小")
            print("   • 噪声项重复计入")
            print("   • FIM 计算错误")
            verdict_bcrlb = "❌ 偏差"

        return verdict_bcrlb, RMSE_theory, RMSE_actual

    def generate_report(self):
        """生成综合诊断报告"""
        print("\n")
        print("=" * 80)
        print(" " * 20 + "THz-ISL MIMO ISAC 系统诊断报告")
        print("=" * 80)

        # 运行所有诊断
        verdict1, G_amp, G_pow = self.diagnose_energy_scaling()
        verdict2, ratio_db, sigma2_gamma, N0 = self.diagnose_hardware_magnitude()
        _ = self.diagnose_parameter_sensitivity()
        verdict4, RMSE_th, RMSE_ac = self.diagnose_bcrlb_computation()

        # 生成总结
        print("\n" + "=" * 80)
        print("【总结】诊断结果汇总")
        print("=" * 80)

        summary_table = [
            ["能量标度", verdict1, "G_grad 功率增益 vs 幅度增益"],
            ["失真量级", verdict2, f"σ²_γ/N0 = {ratio_db:.1f} dB"],
            ["BCRLB链路", verdict4, f"RMSE 比值 = {RMSE_ac / RMSE_th:.2f}"],
        ]

        print("\n" + tabulate(summary_table,
                              headers=['诊断项', '状态', '详情'],
                              tablefmt='grid'))

        # 给出建议
        print("\n" + "=" * 80)
        print("【建议】修复优先级")
        print("=" * 80)

        recommendations = []

        if verdict1 == "❌ 错误":
            recommendations.append({
                'priority': 'P0',
                'action': '修改 limits_engine.py 中的 G_grad_avg',
                'detail': f'将 √g_ar 改为 g_ar × η²_bsq_avg (增益提升 {G_pow / G_amp:.1f}×)',
                'file': 'limits_engine.py',
                'line': '搜索 "G_grad_avg = " '
            })

        if ratio_db < -20:
            recommendations.append({
                'priority': 'P1',
                'action': '增强硬件失真参数（可选）',
                'detail': f'当前 σ²_γ/N0 = {ratio_db:.1f} dB，建议提升至 -10~-5 dB',
                'file': 'hardware_ablation_study.py 或 config.yaml',
                'line': 'hardware 部分'
            })

        recommendations.append({
            'priority': 'P1',
            'action': '添加诊断日志',
            'detail': '在 limits_engine.py 中打印 σ²_γ/N0 比值',
            'file': 'limits_engine.py',
            'line': 'calc_BCRLB 函数末尾'
        })

        recommendations.append({
            'priority': 'P2',
            'action': '实现相对劣化图',
            'detail': '绘制 RMSE_cfg / RMSE_awgn 随 α 的曲线',
            'file': '新建 plot_relative_degradation.py',
            'line': ''
        })

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority']}] {rec['action']}")
            print(f"   说明：{rec['detail']}")
            print(f"   位置：{rec['file']}")
            if rec['line']:
                print(f"   行号：{rec['line']}")


def main():
    if len(sys.argv) < 2:
        print("使用方法：python comprehensive_diagnosis.py config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        diagnostics = HWDiagnostics(config_path)
        diagnostics.generate_report()

        print("\n" + "=" * 80)
        print("诊断完成！请根据上述建议修改代码。")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 诊断过程中出错：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()