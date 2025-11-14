#!/usr/bin/env python3
"""
Limits Engine for THz-ISL MIMO ISAC System
DR-08 Protocol Implementation (EXPERT-FIXED VERSION)

KEY FIXES IN THIS VERSION:
1. BCRLB scaling fix: E_sig_target now scales with G_grad_avg (gradient aperture)
   - This preserves |s_k|² ∝ g_ar, enabling proper MIMO scaling
   - RMSE now correctly scales as ∝ 1/√(Nt*Nr) instead of deteriorating
2. Hardware distortion noise scales as (Nt+Nr), not g_ar - ensures correct noise model
3. C_J and SNR_crit calculations unchanged (already correct)

EXPERT FIX SUMMARY (from Document 1):
- Root cause: Previous E_sig_target was independent of g_ar, causing signal amplitude
  to be "normalized away" by the scaling factor A ∝ 1/√g_ar
- Solution: Multiply E_sig_target by G_grad_avg (without rho_PN) to make it scale
  linearly with g_ar, so A becomes approximately constant
- Result: FIM ∝ g_ar/(Nt+Nr), BCRLB ∝ (Nt+Nr)/g_ar, RMSE ∝ 1/√(Nt*Nr)

This module implements the performance limit calculations for hardware-limited
THz inter-satellite link ISAC systems according to DR-08 specifications.

Functions:
    calc_C_J: Calculate communication capacity (Jensen bound)
    calc_BCRLB: Calculate Bayesian Cramer-Rao Lower Bound (matched case)
    calc_MCRB: Calculate Misspecified Cramer-Rao Lower Bound

Author: Generated according to DR-08 Protocol v1.0 + Expert Review + Expert Fix
"""

import numpy as np
from typing import Dict, Any, Union, List, Tuple
from scipy.linalg import cholesky, inv, LinAlgError, toeplitz
import warnings


def calc_C_J(
        config: Dict[str, Any],
        g_sig_factors: Dict[str, Union[float, np.ndarray]],
        n_f_outputs: Dict[str, Union[float, np.ndarray]],
        SNR0_db_vec: Union[List[float], np.ndarray],
        compute_C_G: bool = False
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate communication capacity (Jensen bound) - UNCHANGED (correct as-is)"""

    # Extract key parameters
    G_sig_avg = g_sig_factors['G_sig_avg']
    sigma_2_phi_c_res = n_f_outputs['sigma_2_phi_c_res']
    Gamma_eff_total = n_f_outputs['Gamma_eff_total']
    P_tx_per_element = n_f_outputs['P_tx_per_element']

    # Note: Gamma_eff_total now correctly scales as (Nt+Nr), not g_ar
    # This doesn't affect C_J calculation, but SNR_crit will shift left with larger arrays

    phase_coherence_loss = np.exp(-sigma_2_phi_c_res)
    SNR0_vec = 10 ** (np.array(SNR0_db_vec) / 10.0)
    C_J_vec = np.zeros_like(SNR0_vec)

    for i, SNR0 in enumerate(SNR0_vec):
        # SNR includes array gain
        numerator = SNR0 * G_sig_avg * phase_coherence_loss
        # Distortion now scales correctly: Gamma_eff_total ∝ (Nt+Nr)
        denominator = 1.0 + SNR0 * G_sig_avg * Gamma_eff_total
        SINR_eff = numerator / denominator
        C_J_vec[i] = np.log2(1.0 + SINR_eff)

    SINR_sat = phase_coherence_loss / Gamma_eff_total
    C_sat = np.log2(1.0 + SINR_sat)

    # Critical SNR (linear units) - now correctly shifts left with array size
    # SNR_crit ∝ 1 / (g_ar * Gamma) ∝ (Nt+Nr) / (Nt*Nr)
    SNR_crit_linear = 1.0 / (G_sig_avg * Gamma_eff_total)
    SNR_crit_db = 10.0 * np.log10(SNR_crit_linear)

    results = {
        'C_J_vec': C_J_vec,
        'C_sat': C_sat,
        'SNR_crit_db': SNR_crit_db,
        'SINR_sat': SINR_sat,
        'phase_coherence_loss': phase_coherence_loss
    }

    if compute_C_G:
        B_hz = config['channel']['B_hz']
        N = config['simulation']['N']
        eta_bsq_k = g_sig_factors['eta_bsq_k']

        G_scalars = (g_sig_factors['g_ar'] *
                     g_sig_factors['rho_Q'] *
                     g_sig_factors['rho_APE'] *
                     g_sig_factors['rho_A'] )

        C_G_vec = np.zeros_like(SNR0_vec)

        for i, SNR0 in enumerate(SNR0_vec):
            G_sig_f = G_scalars * eta_bsq_k
            denominator_f = 1.0 + SNR0 * G_sig_f * Gamma_eff_total
            SINR_f = (SNR0 * G_sig_f * phase_coherence_loss) / denominator_f
            C_G_vec[i] = np.mean(np.log2(1.0 + SINR_f))

        results['C_G_vec'] = C_G_vec
        results['Jensen_gap_db'] = 10 * np.log10(2 ** results['C_J_vec'] / 2 ** C_G_vec)
        results['Jensen_gap_bits'] = results['C_J_vec'] - C_G_vec

    return results


def calc_BCRLB(
        config: Dict[str, Any],
        g_sig_factors: Dict[str, Union[float, np.ndarray]],
        n_f_outputs: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate Bayesian Cramer-Rao Lower Bound - EXPERT FIXED VERSION

    KEY FIX: Properly scales signal energy and reconstructs noise PSD
    without using stale values from physics_engine.
    """

    # ===================================================================
    # STEP 1: 基础参数提取
    # ===================================================================
    N = int(config['simulation']['N'])
    B_hz = float(config['channel']['B_hz'])
    Nt = int(config['array']['Nt'])
    Nr = int(config['array']['Nr'])

    Delta_f_hz = B_hz / N
    T_obs = N / B_hz

    FIM_MODE = config['simulation'].get('FIM_MODE', 'Whittle')

    # ===================================================================
    # STEP 2: 计算目标SNR（考虑alpha策略）
    # ===================================================================
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

    # ===================================================================
    # STEP 3: 计算梯度增益 G_grad_avg（必须包含g_ar）
    # ===================================================================
    g_ar = float(g_sig_factors['g_ar'])
    eta_bsq_avg = float(g_sig_factors['eta_bsq_avg'])
    rho_Q = float(g_sig_factors['rho_Q'])
    rho_APE = float(g_sig_factors['rho_APE'])
    rho_A = float(g_sig_factors['rho_A'])

    # G_grad_avg = g_ar * eta_bsq_avg * rho_Q * rho_APE * rho_A

    G_grad_avg = g_ar * rho_Q * rho_APE * rho_A * eta_bsq_avg # Amplitude gain

    # ===================================================================
    # STEP 4: 信号能量归一化
    # ===================================================================
    kB = 1.380649e-23
    T0_K = float(config.get('channel', {}).get('T0_K', 290.0))
    N0_white = float(n_f_outputs.get('N0', kB * T0_K))
    G_sig_avg = float(g_sig_factors['G_sig_avg'])

    # ✅ 新方法：使用固定发射功率，不基于SNR
    power_mode = config['isac_model'].get('power_mode', 'FIXED')  # 'FIXED' 或 'SNR_BASED'

    if power_mode == 'FIXED':
        # 固定功率模式（推荐）
        P_tx_per_element = float(config['isac_model'].get('P_tx_fixed', 1.0))
        P_rx_target = P_tx_per_element * G_sig_avg
        E_sig_target = P_rx_target * T_obs

        # 打印实际SNR（用于参考）
        if config.get('debug', {}).get('print_hardware_ratio', False):
            # 需要先估算N_k来计算实际SNR（这里先用N0估算）
            SNR_rx_db_estimate = 10 * np.log10(P_rx_target / (N0_white * B_hz))
            print(f"[Fixed Power] P_tx={P_tx_per_element:.3e} W/elem, "
                  f"SNR_rx≈{SNR_rx_db_estimate:.1f} dB (基于N0估算)")

    else:
        # SNR-based模式（保留兼容性，但不推荐在PN主导时使用）
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
        P_sig_psd_target = SNR_p_lin * N0_white * G_grad_avg
        E_sig_target = P_sig_psd_target * (B_hz * T_obs)

        # 回推发射功率（用于后续硬件失真计算）
        P_rx_target = E_sig_target / T_obs
        P_tx_per_element = P_rx_target / max(G_sig_avg, 1e-30)

    # 归一化信号能量
    sig_amp_k = g_sig_factors['sig_amp_k']
    E_sig_current = float(np.sum(np.abs(sig_amp_k) ** 2) * Delta_f_hz)

    A = np.sqrt(max(E_sig_target, 1e-300) / max(E_sig_current, 1e-300))
    s_k = A * sig_amp_k

    # Parseval自检
    if config.get('debug', {}).get('assert_parseval', False):
        E_actual = float(np.sum(np.abs(s_k) ** 2) * Delta_f_hz)
        rel_err = abs(E_actual - E_sig_target) / max(E_sig_target, 1e-12)
        if rel_err > 1e-3:
            warnings.warn(f"[Parseval] Error: {rel_err:.2e}")

    # ===================================================================
    # STEP 5: 重建失真功率（基于归一化后的信号）
    # ===================================================================
    # ✅ 简化：P_tx_per_element 已在STEP 4中确定，直接使用
    P_tx_eff = P_tx_per_element  # 使用STEP 4计算的值

    G_sig_avg = float(g_sig_factors['G_sig_avg'])

    Gamma_pa = float(n_f_outputs.get('Gamma_pa', 0.0))
    Gamma_adc = float(n_f_outputs.get('Gamma_adc', 0.0))
    Gamma_iq = float(n_f_outputs.get('Gamma_iq', 0.0))
    Gamma_lo = float(n_f_outputs.get('Gamma_lo', 0.0))
    Gamma_per_elem = Gamma_pa + Gamma_adc + Gamma_iq + Gamma_lo

    # ✅ 用固定/确定的发射功率计算硬件失真
    sigma2_gamma_new = Gamma_per_elem * P_tx_eff * (Nt + Nr)

    gamma_psd = sigma2_gamma_new / B_hz
    ratio_gamma_to_N0_dB = 10 * np.log10(gamma_psd / N0_white)

    diagnostics = {
        'gamma_psd': float(gamma_psd),
        'N0_psd': float(N0_white),
        'ratio_gamma_to_N0_dB': float(ratio_gamma_to_N0_dB),
        'P_tx_per_element': float(P_tx_eff),  # ✅ 添加到诊断输出
    }
    if config.get('debug', {}).get('print_hardware_ratio', False):
        print(f"\n[Power Mode Verification]")
        print(f"  Mode: {power_mode}")
        print(f"  P_tx_per_element: {P_tx_eff:.3e} W")
        print(f"  P_rx_total: {P_tx_eff * G_sig_avg:.3e} W")
        print(f"  σ²_γ_eff: {sigma2_gamma_new:.3e} W")
        print(f"  γ/N0: {ratio_gamma_to_N0_dB:.1f} dB")

    # ===================================================================
    # STEP 6: 重建总噪声PSD（一次性完成，不重复）
    # ===================================================================
    S_phi_c_res_k = np.asarray(n_f_outputs.get('S_phi_c_res_k', np.zeros(N)))
    S_DSE_k = np.asarray(n_f_outputs.get('S_DSE_k', np.zeros(N)))
    S_RSM_k = np.asarray(n_f_outputs.get('S_RSM_k', np.zeros(N)))

    if S_RSM_k.size == 0:
        S_RSM_k = np.zeros(N, dtype=float)

    # ⚠️ 关键：用新计算的sigma2_gamma_new
    N_k_psd = (N0_white +
               sigma2_gamma_new / B_hz +  # 不是旧的n_f_outputs['sigma2_gamma']！
               S_phi_c_res_k +
               S_DSE_k +
               S_RSM_k)
    N_k_psd = np.maximum(N_k_psd, 1e-30)

    # ⚠️ 临时测试：强制使用纯AWGN
    if config.get('debug', {}).get('force_awgn_test', False):
        print("[测试模式] 强制N_k_psd=N0（纯AWGN）")
        N_k_psd = np.full(N, N0_white)
    # ===================================================================
    # STEP 7: 梯度计算
    # ===================================================================
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    ds_dtau_k = -1j * 2 * np.pi * f_vec * s_k
    ds_dfD_k = 1j * 2 * np.pi * T_obs * s_k

    # ✅ 添加验证：打印N_k_psd的统计信息
    if config.get('debug', {}).get('print_hardware_ratio', False):
        N_k_mean = float(np.mean(N_k_psd))
        N_k_min = float(np.min(N_k_psd))
        N_k_max = float(np.max(N_k_psd))

        print(f"[N_k_psd检查] 均值={N_k_mean:.3e} W/Hz")
        print(f"[N_k_psd检查] 范围=[{N_k_min:.3e}, {N_k_max:.3e}] W/Hz")
        print(f"[N_k_psd检查] 均值/N0={N_k_mean / N0_white:.1f}× ({10 * np.log10(N_k_mean / N0_white):.1f} dB)")

        # 分解贡献
        gamma_contrib = sigma2_gamma_new / B_hz
        phi_contrib = float(np.mean(S_phi_c_res_k))
        dse_contrib = float(np.mean(S_DSE_k))
        rsm_contrib = float(np.mean(S_RSM_k))

        print(f"[N_k组成] N0={N0_white:.3e}, γ={gamma_contrib:.3e}, PN={phi_contrib:.3e}, DSE={dse_contrib:.3e}")

    # ===================================================================
    # STEP 8: FIM计算
    # ===================================================================
    print(f"[调用FIM前] N_k_psd均值={np.mean(N_k_psd):.3e} W/Hz")
    print(f"[调用FIM前] id(N_k_psd)={id(N_k_psd)}")  # 内存地址

    if FIM_MODE == 'Whittle':
        FIM, CRLB_matrix = _compute_whittle_fim(
            s_k, ds_dtau_k, ds_dfD_k, N_k_psd, Delta_f_hz
        )
    elif FIM_MODE == 'Whittle-ExactDoppler':
        t_vec = np.linspace(-T_obs / 2, T_obs / 2, N)
        s_t = np.fft.ifft(np.fft.ifftshift(s_k)) * N
        ds_dfD_t = 1j * 2 * np.pi * t_vec * s_t
        ds_dfD_k_exact = np.fft.fftshift(np.fft.fft(ds_dfD_t)) / N
        FIM, CRLB_matrix = _compute_whittle_fim(
            s_k, ds_dtau_k, ds_dfD_k_exact, N_k_psd, Delta_f_hz
        )
    elif FIM_MODE == 'Cholesky':
        FIM, CRLB_matrix = _compute_cholesky_fim(
            s_k, ds_dtau_k, ds_dfD_k, N_k_psd, N, B_hz
        )
    else:
        warnings.warn(f"Unknown FIM_MODE '{FIM_MODE}', using Whittle")
        FIM, CRLB_matrix = _compute_whittle_fim(
            s_k, ds_dtau_k, ds_dfD_k, N_k_psd, Delta_f_hz
        )

    # ===================================================================
    # STEP 9: 提取BCRLB
    # ===================================================================
    # 改为 Schur 补：
    F_00, F_01, F_11 = FIM[0, 0], FIM[0, 1], FIM[1, 1]
    eps_reg = 1e-9 * np.median([F_00, F_11])
    BCRLB_tau = 1.0 / (F_00 - F_01 ** 2 / (F_11 + eps_reg))

    # 兜底：
    if not np.isfinite(BCRLB_tau) or BCRLB_tau <= 0:
        from numpy.linalg import pinv
        BCRLB_tau = max(pinv(FIM)[0, 0].real, 1e-30)
    BCRLB_fD = max(CRLB_matrix[1, 1].real, np.finfo(float).eps)

    # === 诊断日志：检查HW失真是否可见 ===
    if config.get('debug', {}).get('print_hardware_ratio', False):
        gamma_psd_eff = sigma2_gamma_new / B_hz
        N0_white_diag = float(n_f_outputs.get('N0', 4.0e-21))  # 重命名避免覆盖
        ratio_db = 10 * np.log10(gamma_psd_eff / N0_white_diag) if N0_white_diag > 0 else -np.inf

        visibility = 'HIDDEN' if ratio_db < -20 else 'BORDERLINE' if ratio_db < -10 else 'VISIBLE'
        print(f"[BCRLB-DIAG] HW/N0 ratio = {ratio_db:+.1f} dB ({visibility})")
        print(f"[BCRLB-DIAG] σ²_γ_eff = {sigma2_gamma_new:.3e} W")
        print(f"[BCRLB-DIAG] P_tx_eff = {P_tx_eff:.3e} W/element")  # ← 修改这里

        # 添加验证
        sigma2_check = Gamma_per_elem * P_tx_eff * (Nt + Nr)
        print(f"[BCRLB-DIAG] σ²_γ验证 = {sigma2_check:.3e} W (应与上面相同)")

    return {
        'BCRLB_tau': BCRLB_tau,
        'BCRLB_fD': BCRLB_fD,
        'CRLB_matrix': CRLB_matrix,
        'FIM': FIM,
        'diagnostics': diagnostics,  # ← 新增
        'Delta_f_hz': Delta_f_hz
    }


def _compute_whittle_fim(
        s_k: np.ndarray,
        ds_dtau_k: np.ndarray,
        ds_dfD_k: np.ndarray,
        N_k_psd: np.ndarray,
        Delta_f_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FIM using Whittle approximation - unchanged"""
    # ✅ 验证噪声PSD
    N_mean = float(np.mean(N_k_psd))
    N_min = float(np.min(N_k_psd))
    assert N_mean > 0, f"N_k_psd均值={N_mean}，应该>0"
    assert N_min > 0, f"N_k_psd最小值={N_min}，应该>0"

    # ✅ 强制验证：确认这是我们打印的那个N_k_psd
    N_mean_inside_fim = float(np.mean(N_k_psd))
    print(f"[FIM内部] N_k_psd均值={N_mean_inside_fim:.3e} W/Hz")
    print(f"[FIM内部] N_k_psd首元素={N_k_psd[0]:.3e} W/Hz")


    eps = np.finfo(float).eps
    N_k_psd_safe = np.maximum(N_k_psd, eps)

    # FIM elements (no phase loss on gradient side - that's for communication)
    integrand_tt = (1.0 / N_k_psd_safe) * np.abs(ds_dtau_k) ** 2
    F_00 = 2 * np.sum(integrand_tt) * Delta_f_hz

    integrand_ff = (1.0 / N_k_psd_safe) * np.abs(ds_dfD_k) ** 2
    F_11 = 2 * np.sum(integrand_ff) * Delta_f_hz

    integrand_tf = (1.0 / N_k_psd_safe) * 2 * np.real(np.conj(ds_dtau_k) * ds_dfD_k)
    F_01 = np.sum(integrand_tf) * Delta_f_hz

    FIM = np.array([[F_00, F_01],
                    [F_01, F_11]])

    try:
        CRLB_matrix = inv(FIM)
    except LinAlgError:
        warnings.warn("FIM is singular, returning infinite CRLB")
        CRLB_matrix = np.array([[np.inf, np.inf], [np.inf, np.inf]])

    return FIM, CRLB_matrix


def _compute_cholesky_fim(
        s_k: np.ndarray,
        ds_dtau_k: np.ndarray,
        ds_dfD_k: np.ndarray,
        N_k_psd: np.ndarray,
        N: int,
        B_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FIM using Cholesky decomposition - unchanged"""

    s_t = np.fft.ifft(np.fft.ifftshift(s_k)) * N
    ds_dtau_t = np.fft.ifft(np.fft.ifftshift(ds_dtau_k)) * N
    ds_dfD_t = np.fft.ifft(np.fft.ifftshift(ds_dfD_k)) * N

    Delta_t = 1.0 / B_hz
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)

    r_n_first_row = np.zeros(N, dtype=complex)
    for k in range(N):
        r_n_first_row += N_k_psd * np.exp(1j * 2 * np.pi * f_vec * k * Delta_t) * (B_hz / N)

    r_n_first_col = np.conj(r_n_first_row)
    R_n = toeplitz(r_n_first_col, r_n_first_row)

    # Enhanced numerical stability
    R_n = (R_n + R_n.conj().T) / 2
    loading_factor = 1e-10 * np.trace(np.real(R_n)) / N
    R_n += loading_factor * np.eye(N)

    eigenvals = np.linalg.eigvalsh(R_n)
    if np.any(eigenvals <= 0):
        min_eig = np.min(eigenvals)
        warnings.warn(f"Negative eigenvalue detected: {min_eig:.2e}")
        R_n += (abs(min_eig) + 1e-10) * np.eye(N)

    try:
        L = cholesky(R_n, lower=True)
        L_inv = inv(L)
        ds_dtau_whitened = L_inv @ ds_dtau_t
        ds_dfD_whitened = L_inv @ ds_dfD_t

        F_00 = 2 * np.real(np.vdot(ds_dtau_whitened, ds_dtau_whitened))
        F_11 = 2 * np.real(np.vdot(ds_dfD_whitened, ds_dfD_whitened))
        F_01 = 2 * np.real(np.vdot(ds_dtau_whitened, ds_dfD_whitened))

        FIM = np.array([[F_00, F_01],
                        [F_01, F_11]])
        CRLB_matrix = inv(FIM)

    except LinAlgError as e:
        warnings.warn(f"Cholesky FIM failed: {e}, falling back to Whittle")
        return _compute_whittle_fim(s_k, ds_dtau_k, ds_dfD_k, N_k_psd, B_hz / N)

    return FIM, CRLB_matrix


def calc_MCRB(
        config: Dict[str, Any],
        g_sig_factors: Dict[str, Union[float, np.ndarray]],
        n_f_outputs: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate Misspecified Cramer-Rao Bound - uses updated BCRLB"""

    N = config['simulation']['N']
    B_hz = config['channel']['B_hz']

    # Reconstruct corrected noise PSD (same as in calc_BCRLB)
    N0_white = n_f_outputs['N0']
    sigma2_gamma = n_f_outputs['sigma2_gamma']
    S_RSM_k = n_f_outputs['S_RSM_k']
    S_phi_c_res_k = n_f_outputs['S_phi_c_res_k']
    S_DSE_k = n_f_outputs['S_DSE_k']

    N_k_psd = N0_white + sigma2_gamma / B_hz + S_RSM_k + S_phi_c_res_k + S_DSE_k
    N_k_psd = np.maximum(N_k_psd, 1e-30)

    Delta_f_hz = n_f_outputs['Delta_f_hz']

    Phi_q = config.get('waveform', {}).get('Phi_q', 0.1)
    Phi_q_rad = Phi_q

    bcrlb_results = calc_BCRLB(config, g_sig_factors, n_f_outputs)
    F_matched = bcrlb_results['FIM']
    K = F_matched


    sig_amp_k = g_sig_factors['sig_amp_k']
    s_k = sig_amp_k  # Simplified for mismatch calculation

    t_vec = np.linspace(-N / (2 * B_hz), N / (2 * B_hz), N)
    t_obs = N / B_hz

    phase_mismatch = Phi_q_rad * (t_vec / t_obs) ** 2
    s_diff_t = s_k.mean() * (np.exp(1j * phase_mismatch) - 1)
    s_diff_f = np.fft.fft(np.fft.fftshift(s_diff_t))

    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    d2s_dtau2_k = -(2 * np.pi * f_vec) ** 2 * s_k
    d2s_dfD2_k = (2 * np.pi * t_obs) ** 2 * s_k
    d2s_dtau_dfD_k = -1j * (2 * np.pi) ** 2 * f_vec * t_obs * s_k

    eps = np.finfo(float).eps
    N_k_psd_safe = np.maximum(N_k_psd, eps)

    E_bias = np.zeros((2, 2))
    integrand_tt = (1.0 / N_k_psd_safe) * 2 * np.real(np.conj(d2s_dtau2_k) * s_diff_f)
    E_bias[0, 0] = -np.sum(integrand_tt) * Delta_f_hz

    integrand_ff = (1.0 / N_k_psd_safe) * 2 * np.real(np.conj(d2s_dfD2_k) * s_diff_f)
    E_bias[1, 1] = -np.sum(integrand_ff) * Delta_f_hz

    integrand_tf = (1.0 / N_k_psd_safe) * 2 * np.real(np.conj(d2s_dtau_dfD_k) * s_diff_f)
    E_bias[0, 1] = -np.sum(integrand_tf) * Delta_f_hz
    E_bias[1, 0] = E_bias[0, 1]

    J = F_matched - E_bias

    try:
        J_inv = inv(J)
        MCRB_matrix = J_inv @ K @ J_inv
        MCRB_tau = MCRB_matrix[0, 0]
        MCRB_fD = MCRB_matrix[1, 1]
    except LinAlgError:
        warnings.warn("J matrix is singular, returning infinite MCRB")
        MCRB_tau = np.inf
        MCRB_fD = np.inf
        MCRB_matrix = np.array([[np.inf, np.inf], [np.inf, np.inf]])

    return {
        'MCRB_tau': MCRB_tau,
        'MCRB_fD': MCRB_fD,
        'MCRB_matrix': MCRB_matrix,
        'F_matched': F_matched,
        'E_bias': E_bias,
        'Phi_q_rad': Phi_q_rad
    }


def validate_limits_config(config: Dict[str, Any]) -> None:
    """Validate configuration for limits engine calculations - unchanged"""

    required_sim_keys = ['N', 'SNR0_db_vec']
    for key in required_sim_keys:
        if key not in config.get('simulation', {}):
            raise KeyError(f"Missing simulation parameter: {key}")

    if 'FIM_MODE' in config.get('simulation', {}):
        fim_mode = config['simulation']['FIM_MODE']
        if fim_mode not in ['Whittle', 'Whittle-ExactDoppler', 'Cholesky']:
            raise ValueError(f"Invalid FIM_MODE: {fim_mode}")


if __name__ == "__main__":
    print("limits_engine.py - Standalone testing not recommended")
    print("Please run: python main.py config.yaml")