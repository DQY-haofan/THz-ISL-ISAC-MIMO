#!/usr/bin/env python3
"""
BCRLB Diagnostic Script - 精确定位问题
诊断calc_BCRLB函数的每个步骤
"""

import numpy as np
import yaml
from physics_engine import calc_g_sig_factors, calc_n_f_vector


def diagnose_bcrlb(config_path='config.yaml', test_alpha=0.1):
    """
    诊断BCRLB计算的每个关键步骤
    """

    print("=" * 80)
    print("BCRLB DIAGNOSTIC SCRIPT")
    print("=" * 80)

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['isac_model']['alpha'] = test_alpha

    print(f"\n[CONFIG] Testing at α = {test_alpha}")
    print(f"  SNR_p_db: {config['isac_model']['SNR_p_db']}")
    print(f"  alpha_model: {config['isac_model'].get('alpha_model', 'CONST_POWER')}")
    print(f"  B_hz: {config['channel']['B_hz']:.2e}")
    print(f"  N: {config['simulation']['N']}")
    print(f"  Nt: {config['array']['Nt']}, Nr: {config['array']['Nr']}")

    # ===================================================================
    # STEP 1: 物理参数计算
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Physics Calculation")
    print("=" * 80)

    g_sig_factors = calc_g_sig_factors(config)
    n_f_outputs = calc_n_f_vector(config, g_sig_factors)

    g_ar = g_sig_factors['g_ar']
    G_sig_avg = g_sig_factors['G_sig_avg']

    print(f"  g_ar (Nt×Nr): {g_ar:.0f}")
    print(f"  G_sig_avg: {G_sig_avg:.2f}")
    print(f"  η_bsq_avg: {g_sig_factors['eta_bsq_avg']:.4f}")
    print(f"  ρ_Q: {g_sig_factors['rho_Q']:.4f}")
    print(f"  ρ_APE: {g_sig_factors['rho_APE']:.6f}")
    print(f"  ρ_A: {g_sig_factors['rho_A']:.6f}")
    print(f"  ρ_PN: {g_sig_factors['rho_PN']:.4f}")

    print(f"\n  Noise from physics_engine:")
    print(f"    N0 (thermal): {n_f_outputs['N0']:.3e} W/Hz")
    print(f"    σ²_γ (old): {n_f_outputs['sigma2_gamma']:.3e} W")
    print(f"    Γ_eff_total (old): {n_f_outputs['Gamma_eff_total']:.3e}")

    # ===================================================================
    # STEP 2: BCRLB参数提取
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 2: BCRLB Parameter Extraction")
    print("=" * 80)

    N = int(config['simulation']['N'])
    B_hz = float(config['channel']['B_hz'])
    Nt = int(config['array']['Nt'])
    Nr = int(config['array']['Nr'])

    Delta_f_hz = B_hz / N
    T_obs = N / B_hz

    print(f"  N: {N}")
    print(f"  B_hz: {B_hz:.2e} Hz")
    print(f"  Δf: {Delta_f_hz:.2e} Hz")
    print(f"  T_obs: {T_obs:.2e} s")

    # ===================================================================
    # STEP 3: SNR计算
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Target SNR Calculation")
    print("=" * 80)

    alpha = float(config['isac_model']['alpha'])
    alpha_model = config['isac_model'].get('alpha_model', 'CONST_POWER')
    base_SNRp_db = float(config['isac_model'].get('SNR_p_db', 30.0))

    if alpha_model == 'CONST_POWER':
        SNR_p_db = base_SNRp_db
    elif alpha_model == 'CONST_ENERGY':
        SNR_p_db = base_SNRp_db - 10 * np.log10(max(alpha, 1e-10))
    else:
        SNR_p_db = base_SNRp_db

    SNR_p_lin = 10.0 ** (SNR_p_db / 10.0)

    print(f"  Base SNR_p: {base_SNRp_db:.1f} dB")
    print(f"  Alpha model: {alpha_model}")
    print(f"  Actual SNR_p: {SNR_p_db:.1f} dB")
    print(f"  SNR_p (linear): {SNR_p_lin:.2e}")

    # ===================================================================
    # STEP 4: G_grad_avg计算
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Gradient Gain (G_grad_avg)")
    print("=" * 80)

    eta_bsq_avg = float(g_sig_factors['eta_bsq_avg'])
    rho_Q = float(g_sig_factors['rho_Q'])
    rho_APE = float(g_sig_factors['rho_APE'])
    rho_A = float(g_sig_factors['rho_A'])

    G_grad_avg = g_ar * eta_bsq_avg * rho_Q * rho_APE * rho_A

    print(f"  g_ar: {g_ar:.0f}")
    print(f"  × η_bsq_avg: {eta_bsq_avg:.4f}")
    print(f"  × ρ_Q: {rho_Q:.4f}")
    print(f"  × ρ_APE: {rho_APE:.6f}")
    print(f"  × ρ_A: {rho_A:.6f}")
    print(f"  ───────────────────────")
    print(f"  = G_grad_avg: {G_grad_avg:.2f}")
    print(f"\n  ⚠️  CRITICAL CHECK:")
    print(f"     G_grad_avg should be ≈ g_ar (if losses small)")
    print(f"     Ratio G_grad/g_ar: {G_grad_avg / g_ar:.4f}")

    # ===================================================================
    # STEP 5: 能量归一化
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Energy Normalization")
    print("=" * 80)

    kB = 1.380649e-23
    T0_K = float(config.get('channel', {}).get('T0_K', 290.0))
    N0_white = float(n_f_outputs.get('N0', kB * T0_K))

    # ⚠️ 正确：不乘quality_factor
    P_sig_psd_target = SNR_p_lin * N0_white
    E_sig_target = P_sig_psd_target * (B_hz * T_obs)

    sig_amp_k = g_sig_factors['sig_amp_k']
    E_sig_current = float(np.sum(np.abs(sig_amp_k) ** 2) * Delta_f_hz)

    A = np.sqrt(max(E_sig_target, 1e-300) / max(E_sig_current, 1e-300))
    s_k = A * sig_amp_k

    print(f"  N0_white: {N0_white:.3e} W/Hz")
    print(f"  P_sig_psd_target: {P_sig_psd_target:.3e} W/Hz")
    print(f"  E_sig_target: {E_sig_target:.3e} J")
    print(f"  E_sig_current: {E_sig_current:.3e} J")
    print(f"  Scaling factor A: {A:.3e}")

    E_actual = float(np.sum(np.abs(s_k) ** 2) * Delta_f_hz)
    print(f"\n  Signal power after scaling:")
    print(f"    E_actual: {E_actual:.3e} J")
    print(f"    |s_k|² (mean): {np.mean(np.abs(s_k) ** 2):.3e}")
    print(f"    |s_k|² (max): {np.max(np.abs(s_k) ** 2):.3e}")
    print(f"    |s_k|² (min): {np.min(np.abs(s_k) ** 2):.3e}")

    # Parseval检查
    rel_err = abs(E_actual - E_sig_target) / max(E_sig_target, 1e-12)
    print(f"\n  ✓ Parseval check: {rel_err:.2e} {'PASS' if rel_err < 1e-3 else 'FAIL'}")

    # ===================================================================
    # STEP 6: 失真功率重建
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Hardware Distortion Reconstruction")
    print("=" * 80)

    P_rx_target = E_actual / T_obs
    P_tx_eff = P_rx_target / max(G_sig_avg, 1e-30)

    Gamma_pa = float(n_f_outputs.get('Gamma_pa', 0.0))
    Gamma_adc = float(n_f_outputs.get('Gamma_adc', 0.0))
    Gamma_iq = float(n_f_outputs.get('Gamma_iq', 0.0))
    Gamma_lo = float(n_f_outputs.get('Gamma_lo', 0.0))
    Gamma_per_elem = Gamma_pa + Gamma_adc + Gamma_iq + Gamma_lo

    sigma2_gamma_new = Gamma_per_elem * P_tx_eff * (Nt + Nr)

    print(f"  P_rx_target: {P_rx_target:.3e} W")
    print(f"  P_tx_eff (per element): {P_tx_eff:.3e} W")
    print(f"\n  Hardware distortion coefficients:")
    print(f"    Γ_pa: {Gamma_pa:.3e}")
    print(f"    Γ_adc: {Gamma_adc:.3e}")
    print(f"    Γ_iq: {Gamma_iq:.3e}")
    print(f"    Γ_lo: {Gamma_lo:.3e}")
    print(f"    Γ_total (per elem): {Gamma_per_elem:.3e}")
    print(f"\n  σ²_γ (NEW): {sigma2_gamma_new:.3e} W")
    print(f"  σ²_γ (OLD from physics): {n_f_outputs['sigma2_gamma']:.3e} W")
    print(f"  Ratio (new/old): {sigma2_gamma_new / n_f_outputs['sigma2_gamma']:.3f}")

    # ===================================================================
    # STEP 7: 噪声PSD重建
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Noise PSD Reconstruction")
    print("=" * 80)

    S_phi_c_res_k = np.asarray(n_f_outputs.get('S_phi_c_res_k', np.zeros(N)))
    S_DSE_k = np.asarray(n_f_outputs.get('S_DSE_k', np.zeros(N)))
    S_RSM_k = np.asarray(n_f_outputs.get('S_RSM_k', np.zeros(N)))

    if S_RSM_k.size == 0:
        S_RSM_k = np.zeros(N, dtype=float)

    # 组件贡献
    term_thermal = N0_white
    term_gamma = sigma2_gamma_new / B_hz
    term_pn = np.mean(S_phi_c_res_k)
    term_dse = np.mean(S_DSE_k)
    term_rsm = np.mean(S_RSM_k)

    print(f"  Noise PSD components (mean):")
    print(f"    Thermal (N0): {term_thermal:.3e} W/Hz")
    print(f"    Distortion (σ²_γ/B): {term_gamma:.3e} W/Hz")
    print(f"    Phase noise: {term_pn:.3e} W/Hz")
    print(f"    DSE: {term_dse:.3e} W/Hz")
    print(f"    RSM: {term_rsm:.3e} W/Hz")

    total_mean = term_thermal + term_gamma + term_pn + term_dse + term_rsm
    print(f"\n  Total PSD (mean): {total_mean:.3e} W/Hz")

    print(f"\n  Composition:")
    print(f"    Thermal: {100 * term_thermal / total_mean:.2f}%")
    print(f"    Distortion: {100 * term_gamma / total_mean:.2f}%")
    print(f"    Phase noise: {100 * term_pn / total_mean:.2f}%")
    print(f"    DSE: {100 * term_dse / total_mean:.2f}%")
    print(f"    RSM: {100 * term_rsm / total_mean:.2f}%")

    N_k_psd = (N0_white + sigma2_gamma_new / B_hz +
               S_phi_c_res_k + S_DSE_k + S_RSM_k)
    N_k_psd = np.maximum(N_k_psd, 1e-30)

    print(f"\n  N_k_psd statistics:")
    print(f"    Min: {N_k_psd.min():.3e} W/Hz")
    print(f"    Mean: {N_k_psd.mean():.3e} W/Hz")
    print(f"    Max: {N_k_psd.max():.3e} W/Hz")

    # ===================================================================
    # STEP 8: FIM计算
    # ===================================================================
    print("\n" + "=" * 80)
    print("STEP 8: FIM Calculation")
    print("=" * 80)

    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    ds_dtau_k = -1j * 2 * np.pi * f_vec * s_k
    ds_dfD_k = 1j * 2 * np.pi * T_obs * s_k

    print(f"  Gradient magnitudes:")
    print(f"    |s_k|² (mean): {np.mean(np.abs(s_k) ** 2):.3e}")
    print(f"    |∂s/∂τ|² (mean): {np.mean(np.abs(ds_dtau_k) ** 2):.3e}")
    print(f"    |∂s/∂τ|² (max): {np.max(np.abs(ds_dtau_k) ** 2):.3e}")
    print(f"    |∂s/∂fD|² (mean): {np.mean(np.abs(ds_dfD_k) ** 2):.3e}")

    # FIM元素计算（Whittle）
    eps = np.finfo(float).eps
    N_k_psd_safe = np.maximum(N_k_psd, eps)

    integrand_tt = (1.0 / N_k_psd_safe) * np.abs(ds_dtau_k) ** 2
    F_00 = 2 * np.sum(integrand_tt) * Delta_f_hz

    integrand_ff = (1.0 / N_k_psd_safe) * np.abs(ds_dfD_k) ** 2
    F_11 = 2 * np.sum(integrand_ff) * Delta_f_hz

    integrand_tf = (1.0 / N_k_psd_safe) * 2 * np.real(np.conj(ds_dtau_k) * ds_dfD_k)
    F_01 = np.sum(integrand_tf) * Delta_f_hz

    FIM = np.array([[F_00, F_01],
                    [F_01, F_11]])

    print(f"\n  FIM matrix:")
    print(f"    F[0,0] (delay): {F_00:.3e}")
    print(f"    F[1,1] (Doppler): {F_11:.3e}")
    print(f"    F[0,1] (cross): {F_01:.3e}")

    # 条件数
    cond = np.linalg.cond(FIM)
    print(f"\n  FIM condition number: {cond:.3e}")
    print(f"    {'GOOD' if cond < 1e12 else 'BAD - ILL-CONDITIONED'}")

    # CRLB
    try:
        from scipy.linalg import inv
        CRLB_matrix = inv(FIM)

        BCRLB_tau = max(CRLB_matrix[0, 0].real, np.finfo(float).eps)
        BCRLB_fD = max(CRLB_matrix[1, 1].real, np.finfo(float).eps)

        print(f"\n  CRLB matrix:")
        print(f"    BCRLB_τ: {BCRLB_tau:.3e} s²")
        print(f"    BCRLB_fD: {BCRLB_fD:.3e} Hz²")

        # RMSE
        c_mps = config['channel']['c_mps']
        RMSE_tau_s = np.sqrt(BCRLB_tau)
        RMSE_range_m = (c_mps / 2.0) * RMSE_tau_s

        print(f"\n  RMSE:")
        print(f"    στ: {RMSE_tau_s:.3e} s")
        print(f"    Range: {RMSE_range_m:.6f} m = {RMSE_range_m * 1000:.3f} mm")
        print(f"    Doppler: {np.sqrt(BCRLB_fD):.3f} Hz")

    except Exception as e:
        print(f"\n  ✗ CRLB inversion failed: {e}")
        BCRLB_tau = np.inf
        BCRLB_fD = np.inf

    # ===================================================================
    # STEP 9: 诊断总结
    # ===================================================================
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    # 关键检查
    checks = []

    # 1. G_grad_avg检查
    if G_grad_avg / g_ar < 0.5:
        checks.append("⚠️  G_grad_avg too small (< 0.5×g_ar)")
    elif G_grad_avg / g_ar > 0.8:
        checks.append("✓ G_grad_avg reasonable")
    else:
        checks.append("⚠️  G_grad_avg moderate losses")

    # 2. 能量归一化检查
    if A > 1e6 or A < 1e-6:
        checks.append(f"✗ Scaling factor A extreme: {A:.2e}")
    else:
        checks.append(f"✓ Scaling factor A reasonable: {A:.2e}")

    # 3. 失真功率检查
    if sigma2_gamma_new / N0_white < 0.01:
        checks.append("⚠️  Hardware distortion << thermal noise")
    elif sigma2_gamma_new / N0_white > 100:
        checks.append("✗ Hardware distortion >> thermal noise")
    else:
        checks.append("✓ Hardware distortion comparable to thermal")

    # 4. FIM检查
    if cond > 1e12:
        checks.append("✗ FIM ill-conditioned")
    elif F_00 < 1e-10:
        checks.append("✗ F[0,0] too small (delay info lost)")
    else:
        checks.append("✓ FIM well-conditioned")

    # 5. BCRLB检查
    if BCRLB_tau < 1e-15:
        checks.append("✗ BCRLB_τ = machine epsilon (FAILED)")
    elif RMSE_range_m > 1.0:
        checks.append(f"⚠️  RMSE > 1 meter: {RMSE_range_m:.3f} m")
    elif RMSE_range_m < 1e-6:
        checks.append(f"⚠️  RMSE < 1 μm (unrealistic)")
    else:
        checks.append(f"✓ RMSE reasonable: {RMSE_range_m * 1000:.3f} mm")

    print("\n  Diagnostic Checks:")
    for check in checks:
        print(f"    {check}")

    print("\n" + "=" * 80)

    return {
        'G_grad_avg': G_grad_avg,
        'A': A,
        'sigma2_gamma_new': sigma2_gamma_new,
        'N_k_psd_mean': N_k_psd.mean(),
        'FIM': FIM,
        'BCRLB_tau': BCRLB_tau,
        'RMSE_m': RMSE_range_m,
        'cond': cond
    }


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    test_alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    results = diagnose_bcrlb(config_path, test_alpha)

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)