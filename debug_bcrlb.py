#!/usr/bin/env python3
"""
BCRLB Diagnostic Script - 精确定位问题 (FIXED VERSION)
包含所有专家修复：
1. 能量定标乘 G_grad_avg
2. Schur 补稳健求解
3. PSD 单位统一比较
"""

import numpy as np
import yaml
from physics_engine import calc_g_sig_factors, calc_n_f_vector


def diagnose_bcrlb(config_path='config.yaml', test_alpha=0.1):
    """诊断BCRLB计算的每个关键步骤"""

    print("=" * 80)
    print("BCRLB DIAGNOSTIC SCRIPT (FIXED VERSION)")
    print("=" * 80)

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['isac_model']['alpha'] = test_alpha

    print(f"\n[CONFIG] Testing at α = {test_alpha}")
    print(f"  SNR_p_db: {config['isac_model']['SNR_p_db']}")
    print(f"  B_hz: {config['channel']['B_hz']:.2e}")
    print(f"  N: {config['simulation']['N']}")
    print(f"  Nt×Nr: {config['array']['Nt']}×{config['array']['Nr']}")

    # STEP 1: 物理参数
    print("\n" + "=" * 80)
    print("STEP 1: Physics")
    print("=" * 80)

    g_sig = calc_g_sig_factors(config)
    n_f = calc_n_f_vector(config, g_sig)

    g_ar = g_sig['g_ar']
    print(f"  g_ar: {g_ar:.0f}")
    print(f"  G_sig_avg: {g_sig['G_sig_avg']:.2f}")

    # STEP 2-3: SNR
    N = int(config['simulation']['N'])
    B_hz = float(config['channel']['B_hz'])
    Delta_f = B_hz / N
    T_obs = N / B_hz

    alpha_model = config['isac_model'].get('alpha_model', 'CONST_POWER')
    base_SNR_db = float(config['isac_model']['SNR_p_db'])

    if alpha_model == 'CONST_POWER':
        SNR_p_db = base_SNR_db
    else:
        SNR_p_db = base_SNR_db - 10 * np.log10(max(test_alpha, 1e-10))

    SNR_p = 10.0 ** (SNR_p_db / 10.0)

    # STEP 4: G_grad_avg (✅ 不含 rho_PN)
    print("\n" + "=" * 80)
    print("STEP 4: G_grad_avg")
    print("=" * 80)

    G_grad_avg = (g_ar * g_sig['eta_bsq_avg'] * g_sig['rho_Q'] *
                  g_sig['rho_APE'] * g_sig['rho_A'])

    print(f"  G_grad_avg: {G_grad_avg:.2f}")
    print(f"  Ratio G/g_ar: {G_grad_avg / g_ar:.4f}")

    # STEP 5: 能量归一化 (✅ 乘 G_grad_avg)
    print("\n" + "=" * 80)
    print("STEP 5: Energy Normalization")
    print("=" * 80)

    kB = 1.380649e-23
    N0 = float(n_f['N0'])

    P_psd_target = SNR_p * N0 * G_grad_avg  # ✅ 关键修复
    E_target = P_psd_target * (B_hz * T_obs)

    sig_k = g_sig['sig_amp_k']
    E_current = float(np.sum(np.abs(sig_k) ** 2) * Delta_f)

    A = np.sqrt(max(E_target, 1e-300) / max(E_current, 1e-300))
    s_k = A * sig_k

    print(f"  P_psd_target: {P_psd_target:.3e} W/Hz")
    print(f"  Scaling A: {A:.3e}")

    # STEP 6: 失真重建
    Nt = int(config['array']['Nt'])
    Nr = int(config['array']['Nr'])

    E_actual = float(np.sum(np.abs(s_k) ** 2) * Delta_f)
    P_rx = E_actual / T_obs
    P_tx = P_rx / max(g_sig['G_sig_avg'], 1e-30)

    Gamma_tot = (n_f['Gamma_pa'] + n_f['Gamma_adc'] +
                 n_f['Gamma_iq'] + n_f['Gamma_lo'])

    sigma2_gamma_new = Gamma_tot * P_tx * (Nt + Nr)

    # STEP 7: 噪声PSD
    S_pn = np.asarray(n_f.get('S_phi_c_res_k', np.zeros(N)))
    S_dse = np.asarray(n_f.get('S_DSE_k', np.zeros(N)))
    S_rsm = np.asarray(n_f.get('S_RSM_k', np.zeros(N)))

    N_psd = N0 + sigma2_gamma_new / B_hz + S_pn + S_dse + S_rsm
    N_psd = np.maximum(N_psd, 1e-30)

    print("\n" + "=" * 80)
    print("STEP 7: Noise Composition")
    print("=" * 80)
    total = N_psd.mean()
    print(f"  Thermal: {100 * N0 / total:.1f}%")
    print(f"  HW dist: {100 * (sigma2_gamma_new / B_hz) / total:.1f}%")
    print(f"  PN: {100 * S_pn.mean() / total:.1f}%")
    print(f"  DSE: {100 * S_dse.mean() / total:.1f}%")

    # STEP 8: FIM
    print("\n" + "=" * 80)
    print("STEP 8: FIM")
    print("=" * 80)

    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    ds_tau = -1j * 2 * np.pi * f_vec * s_k
    ds_fD = 1j * 2 * np.pi * T_obs * s_k

    F_00 = 2 * np.sum(np.abs(ds_tau) ** 2 / N_psd) * Delta_f
    F_11 = 2 * np.sum(np.abs(ds_fD) ** 2 / N_psd) * Delta_f
    F_01 = 2 * np.sum(np.real(np.conj(ds_tau) * ds_fD) / N_psd) * Delta_f

    FIM = np.array([[F_00, F_01], [F_01, F_11]])
    cond = np.linalg.cond(FIM)

    print(f"  F_00: {F_00:.3e}")
    print(f"  F_11: {F_11:.3e}")
    print(f"  cond: {cond:.3e}")

    # STEP 9: CRLB (✅ Schur 补)
    print("\n" + "=" * 80)
    print("STEP 9: CRLB (Schur)")
    print("=" * 80)

    eps_reg = 1e-9 * np.median([F_00, F_11])
    F_11_reg = F_11 + eps_reg

    BCRLB_tau = 1.0 / (F_00 - F_01 ** 2 / F_11_reg)

    if not np.isfinite(BCRLB_tau) or BCRLB_tau <= 0:
        from numpy.linalg import pinv
        BCRLB_tau = max(pinv(FIM)[0, 0].real, 1e-30)
        method = "pinv"
    else:
        method = "Schur"

    c = config['channel']['c_mps']
    RMSE_m = (c / 2) * np.sqrt(BCRLB_tau)

    print(f"  Method: {method}")
    print(f"  BCRLB_τ: {BCRLB_tau:.3e} s²")
    print(f"  RMSE: {RMSE_m * 1000:.3f} mm")

    # 诊断
    print("\n" + "=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)

    checks = []

    if G_grad_avg / g_ar > 0.7:
        checks.append(f"✓ G_grad/g_ar = {G_grad_avg / g_ar:.3f}")
    else:
        checks.append(f"⚠️  G_grad/g_ar = {G_grad_avg / g_ar:.3f} (low)")

    if 1e-4 < A < 1e4:
        checks.append(f"✓ Scaling A = {A:.2e}")
    else:
        checks.append(f"✗ Scaling A = {A:.2e} (extreme)")

    ratio_hw = (sigma2_gamma_new / B_hz) / N0
    if 0.01 < ratio_hw < 100:
        checks.append(f"✓ HW/thermal = {ratio_hw:.2f}")
    else:
        checks.append(f"⚠️  HW/thermal = {ratio_hw:.2f}")

    if cond < 1e12:
        checks.append(f"✓ FIM cond = {cond:.2e}")
    else:
        checks.append(f"⚠️  FIM cond = {cond:.2e} (ill)")

    if 1e-6 < RMSE_m < 1.0:
        checks.append(f"✓ RMSE = {RMSE_m * 1000:.1f} mm")
    elif RMSE_m > 1.0:
        checks.append(f"⚠️  RMSE = {RMSE_m:.2f} m (large)")
    else:
        checks.append(f"⚠️  RMSE = {RMSE_m * 1e6:.2f} μm (small)")

    for c in checks:
        print(f"  {c}")

    print("\n" + "=" * 80)
    return {'RMSE_m': RMSE_m, 'cond': cond, 'method': method}


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    res = diagnose_bcrlb(cfg, alpha)
    print(f"\n>>> RMSE: {res['RMSE_m'] * 1000:.3f} mm (method={res['method']})")