#!/usr/bin/env python3
"""
Limits Engine for THz-ISL MIMO ISAC System
DR-08 Protocol Implementation (FIXED per Expert Review)

KEY FIXES IN THIS VERSION:
1. C_G denominator now uses frequency-dependent G_sig_f (was G_sig_avg) - CRITICAL FIX
2. Improved numerical stability
3. Enhanced comments for RMSE calculation

This module implements the performance limit calculations for hardware-limited
THz inter-satellite link ISAC systems according to DR-08 specifications.

Updates in this version:
- FIXED: C_G denominator now frequency-dependent (line ~129)
- Added 'Whittle-ExactDoppler' FIM mode for precise Doppler gradient calculation
- Enhanced numerical stability in Cholesky decomposition (symmetrization, diagonal loading)
- Added eps clamping in Whittle FIM integration

Functions:
    calc_C_J: Calculate communication capacity (Jensen bound)
    calc_BCRLB: Calculate Bayesian Cramer-Rao Lower Bound (matched case)
    calc_MCRB: Calculate Misspecified Cramer-Rao Lower Bound

Author: Generated according to DR-08 Protocol v1.0
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
    """
    Calculate communication capacity (Jensen bound) - DR-08, Sec 5.1
    FIXED: C_G denominator now uses frequency-dependent G_sig_f
    """

    # PN once-counting for COMMUNICATION
    G_sig_avg = g_sig_factors['G_sig_avg']
    sigma_2_phi_c_res = n_f_outputs['sigma_2_phi_c_res']
    Gamma_eff_total = n_f_outputs['Gamma_eff_total']

    phase_coherence_loss = np.exp(-sigma_2_phi_c_res)
    SNR0_vec = 10 ** (np.array(SNR0_db_vec) / 10.0)
    C_J_vec = np.zeros_like(SNR0_vec)

    for i, SNR0 in enumerate(SNR0_vec):
        numerator = SNR0 * G_sig_avg * phase_coherence_loss
        denominator = 1.0 + SNR0 * G_sig_avg * Gamma_eff_total
        SINR_eff = numerator / denominator
        C_J_vec[i] = np.log2(1.0 + SINR_eff)

    SINR_sat = phase_coherence_loss / Gamma_eff_total
    C_sat = np.log2(1.0 + SINR_sat)

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

        G_scalars = (g_sig_factors['G_sig_ideal'] *
                     g_sig_factors['rho_Q'] *
                     g_sig_factors['rho_APE'] *
                     g_sig_factors['rho_A'] *
                     g_sig_factors['rho_PN'])

        C_G_vec = np.zeros_like(SNR0_vec)

        for i, SNR0 in enumerate(SNR0_vec):
            G_sig_f = G_scalars * eta_bsq_k

            # CRITICAL FIX: Use frequency-dependent G_sig_f in denominator
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
    Calculate Bayesian Cramer-Rao Lower Bound (matched case) - DR-08, Sec 5.2

    关键修复：用 ESD/观测时长 标定信号幅度（而不是 PSD），避免把 FIM 推爆、CRLB 压扁。
    """

    import warnings
    N = config['simulation']['N']
    B_hz = config['channel']['B_hz']
    FIM_MODE = config['simulation'].get('FIM_MODE', 'Whittle')

    # —— 噪声 PSD（Whittle-一次计量：频域用 N_k_psd 向量）——
    N_k_psd = n_f_outputs['N_k_psd']
    Delta_f_hz = n_f_outputs['Delta_f_hz']

    # —— 频率相关的阵列/硬件形状 ——
    eta_bsq_k = g_sig_factors['eta_bsq_k']
    G_scalars = (g_sig_factors['G_sig_ideal'] *
                 g_sig_factors['rho_Q'] *
                 g_sig_factors['rho_APE'] *
                 g_sig_factors['rho_A'] *
                 g_sig_factors['rho_PN'])

    # =============  幅度归一化：按 “能量=功率×时间” 进行  =============
    # 形状：不含绝对幅度（方便统一缩放）
    s_k_shape = np.sqrt(G_scalars * eta_bsq_k)

    # 基础白噪 PSD（W/Hz）。你文件已返回 N0_psd；若缺省则用中位数兜底。
    N0_psd = n_f_outputs.get('N0_psd', None)
    if N0_psd is None:
        N0_psd = np.median(N_k_psd)
        warnings.warn(f"N0_psd not found, using median(N_k_psd) = {N0_psd:.2e}")

    # 观测 SNR_p（支持 alpha 的能量/功率模型）
    alpha = config['isac_model']['alpha']
    alpha_model = config['isac_model'].get('alpha_model', 'CONST_POWER')
    base_SNRp_db = config['isac_model'].get('SNR_p_db',
                                            config['simulation'].get('SNR0_db_fixed', 20.0))
    if alpha_model == 'CONST_POWER':
        SNR_p_db = base_SNRp_db
    elif alpha_model == 'CONST_ENERGY':
        SNR_p_db = base_SNRp_db + 10 * np.log10(max(alpha, np.finfo(float).eps))
    else:
        SNR_p_db = base_SNRp_db
    SNR_p = 10.0 ** (SNR_p_db / 10.0)

    # 目标功率谱密度（W/Hz）
    P_sig_psd_target = SNR_p * N0_psd

    # —— 关键修正：把 PSD→能量，保证 sum |S[k]|^2 Δf = P_sig * T_obs ——
    T_obs = N / B_hz
    mean_eta = float(np.mean(eta_bsq_k))
    denom = max(G_scalars * mean_eta, np.finfo(float).eps)
    # A^2 = (P_sig * T_obs) / (B * denom)   （其中 denom = G_scalars * <eta>）
    A = np.sqrt((P_sig_psd_target * T_obs) / (denom * B_hz))
    s_k = A * s_k_shape

    # —— Parseval 自检（可选）——
    if config.get('debug', {}).get('assert_parseval', False):
        E_freq = np.sum(np.abs(s_k) ** 2) * Delta_f_hz
        E_time = P_sig_psd_target * T_obs
        rel_err = abs(E_freq - E_time) / max(E_time, np.finfo(float).eps)
        assert rel_err < 1e-3, f"Parseval energy mismatch: {rel_err:.3e}"

    # —— 构造梯度并计算 FIM ——
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    t_obs = T_obs
    ds_dtau_k = -1j * 2 * np.pi * f_vec * s_k
    # 频移梯度（Whittle 近似）：∂s/∂fD ≈ j 2π t_obs s_k
    ds_dfD_k = 1j * 2 * np.pi * t_obs * s_k

    if FIM_MODE == 'Whittle':
        FIM, CRLB_matrix = _compute_whittle_fim(s_k, ds_dtau_k, ds_dfD_k, N_k_psd, Delta_f_hz)

    elif FIM_MODE == 'Whittle-ExactDoppler':
        # 先到时域做“确切”多普勒梯度，再回频域
        t_vec = np.linspace(-t_obs / 2, t_obs / 2, N)
        s_t = np.fft.ifft(np.fft.ifftshift(s_k)) * N
        ds_dfD_t = 1j * 2 * np.pi * t_vec * s_t
        ds_dfD_k_exact = np.fft.fftshift(np.fft.fft(ds_dfD_t)) / N
        FIM, CRLB_matrix = _compute_whittle_fim(s_k, ds_dtau_k, ds_dfD_k_exact, N_k_psd, Delta_f_hz)

    elif FIM_MODE == 'Cholesky':
        FIM, CRLB_matrix = _compute_cholesky_fim(s_k, ds_dtau_k, ds_dfD_k, N_k_psd, B_hz)

    else:
        warnings.warn(f"Unknown FIM_MODE='{FIM_MODE}', falling back to Whittle")
        FIM, CRLB_matrix = _compute_whittle_fim(s_k, ds_dtau_k, ds_dfD_k, N_k_psd, Delta_f_hz)

    # —— 取对角得到 BCRLB ——
    BCRLB_tau = max(CRLB_matrix[0, 0].real, np.finfo(float).eps)
    BCRLB_fD = max(CRLB_matrix[1, 1].real, np.finfo(float).eps)

    return {
        'BCRLB_tau': BCRLB_tau,
        'BCRLB_fD': BCRLB_fD,
        'CRLB_matrix': CRLB_matrix,
        'FIM': FIM,
        'Delta_f_hz': Delta_f_hz
    }



def _compute_whittle_fim(
        s_k: np.ndarray,
        ds_dtau_k: np.ndarray,
        ds_dfD_k: np.ndarray,
        N_k_psd: np.ndarray,
        Delta_f_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FIM using Whittle approximation"""

    eps = np.finfo(float).eps
    N_k_psd_safe = np.maximum(N_k_psd, eps)

    # FIM elements
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
    """Compute FIM using Cholesky decomposition with enhanced stability"""

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
        warnings.warn(f"Negative eigenvalue detected: {min_eig:.2e}, adding correction")
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
    """Calculate Misspecified Cramer-Rao Bound"""

    N = config['simulation']['N']
    B_hz = config['channel']['B_hz']
    N_k_psd = n_f_outputs['N_k_psd']
    Delta_f_hz = n_f_outputs['Delta_f_hz']

    Phi_q = config.get('waveform', {}).get('Phi_q', 0.1)
    Phi_q_rad = Phi_q

    bcrlb_results = calc_BCRLB(config, g_factors, n_f_outputs)
    F_matched = bcrlb_results['FIM']
    K = F_matched

    eta_bsq_k = g_sig_factors['eta_bsq_k']
    G_scalars = (g_sig_factors['G_sig_ideal'] *
                 g_sig_factors['rho_Q'] *
                 g_sig_factors['rho_APE'] *
                 g_sig_factors['rho_A'] *
                 g_sig_factors['rho_PN'])
    s_k = np.sqrt(G_scalars * eta_bsq_k)

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
    """Validate configuration for limits engine calculations"""

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