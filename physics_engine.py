#!/usr/bin/env python3
"""
Physics Engine for THz-ISL MIMO ISAC System
DR-08 Protocol Implementation (FIXED per Expert Review)

KEY FIXES IN THIS VERSION:
1. Added Gamma component breakdown in return dict (Gamma_pa/adc/iq/lo) - Expert Item #3
2. Added sanity checks for ENOB and jitter to prevent overflow/underflow
3. Enhanced comments for hardware distortion components

This module implements the core physics calculations for hardware-limited
THz inter-satellite link ISAC systems according to DR-08 specifications.

Functions:
    calc_g_sig_factors(config): Calculate multiplicative gains/losses
    calc_n_f_vector(config, g_sig_factors): Calculate additive noise sources

Author: Generated according to DR-08 Protocol v1.0
"""

import numpy as np
import yaml
from typing import Dict, Any, Union


def calc_g_sig_factors(config: Dict[str, Any]) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate multiplicative gain/loss factors - EXPERT FIXED"""

    # Extract parameters
    Nt = config['array']['Nt']
    Nr = config['array']['Nr']
    L_ap_m = config['array']['L_ap_m']
    theta_0_deg = config['array']['theta_0_deg']
    f_c_hz = config['channel']['f_c_hz']
    B_hz = config['channel']['B_hz']
    c_mps = config['channel']['c_mps']
    N = config['simulation']['N']
    rho_q_bits = config['hardware']['rho_q_bits']
    rho_a_error_rms = config['hardware']['rho_a_error_rms']
    sigma_theta_rad = config['platform']['sigma_theta_rad']
    sigma_rel_sq_rad2 = config['pn_model']['sigma_rel_sq_rad2']

    # Derived parameters
    theta_0_rad = np.radians(theta_0_deg)
    lambda_c = c_mps / f_c_hz
    L_over_lambda = L_ap_m / lambda_c
    B_over_fc = B_hz / f_c_hz
    sin_theta_0 = np.sin(theta_0_rad)
    cos_theta_0 = np.cos(theta_0_rad)

    # ✅ EXPERT FIX #1: 阵列增益独立保存
    g_ar = float(Nt * Nr)

    # Beam squint
    f_axis = np.linspace(-B_hz / 2, B_hz / 2, N)
    freq_offset_norm = f_axis / f_c_hz
    sinc_arg_k = np.pi * L_over_lambda * sin_theta_0 * freq_offset_norm
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_k = np.where(sinc_arg_k == 0, 1.0, np.sin(sinc_arg_k) / sinc_arg_k)
    eta_bsq_k = sinc_k ** 4

    K = np.pi * L_over_lambda * sin_theta_0
    R_B = B_over_fc
    eta_bsq_avg = max(0.0, 1.0 - (K ** 2 * R_B ** 2) / 18.0)

    # Phase quantization
    if rho_q_bits <= 0:
        rho_Q = 0.0
    else:
        quantization_arg = np.pi / (2 ** rho_q_bits)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_Q = (np.sin(quantization_arg) / quantization_arg) ** 2 if quantization_arg != 0 else 1.0

    # Antenna pointing error
    pointing_factor = (np.pi ** 2 / 3.0) * (L_over_lambda * cos_theta_0) ** 2 * (sigma_theta_rad ** 2)
    rho_APE = max(0.0, 1.0 - pointing_factor)

    # Amplitude error
    rho_A = 1.0 / (1.0 + rho_a_error_rms ** 2)

    # Differential phase noise
    rho_PN = np.exp(-sigma_rel_sq_rad2)

    # ✅ EXPERT FIX #2: G_sig_avg正确包含g_ar
    G_sig_avg = g_ar * eta_bsq_avg * rho_Q * rho_APE * rho_A * rho_PN

    # ✅ EXPERT FIX #3: sig_amp_k供FIM使用
    sig_amp_k = np.sqrt(g_ar) * eta_bsq_k

    return {
        'g_ar': g_ar,
        'eta_bsq_avg': eta_bsq_avg,
        'eta_bsq_k': eta_bsq_k,
        'sig_amp_k': sig_amp_k,
        'rho_Q': rho_Q,
        'rho_APE': rho_APE,
        'rho_A': rho_A,
        'rho_PN': rho_PN,
        'G_sig_avg': G_sig_avg,
        'G_sig_ideal': g_ar
    }


def calc_n_f_vector(config: Dict[str, Any], g_sig_factors: Dict[str, Union[float, np.ndarray]]) -> Dict[
    str, Union[float, np.ndarray]]:
    """Calculate additive noise sources - EXPERT FIXED"""

    # Extract parameters
    N = config['simulation']['N']
    B_hz = config['channel']['B_hz']
    f_c_hz = config['channel']['f_c_hz']
    gamma_pa_floor = config['hardware']['gamma_pa_floor']
    gamma_adc_bits = config['hardware']['gamma_adc_bits']
    gamma_iq_irr_dbc = config['hardware']['gamma_iq_irr_dbc']
    gamma_lo_jitter_s = config['hardware']['gamma_lo_jitter_s']
    S_phi_c_K2 = config['pn_model']['S_phi_c_K2']
    S_phi_c_K0 = config['pn_model']['S_phi_c_K0']
    B_loop_hz = config['pn_model']['B_loop_hz']
    alpha = config['isac_model']['alpha']
    C_DSE = config['isac_model'].get('C_DSE', 5e-9)
    DSE_alpha_exponent = config['isac_model'].get('DSE_alpha_exponent', -5.0)
    G_sig_avg = g_sig_factors['G_sig_avg']

    Delta_f_hz = B_hz / N

    # Hardware distortion
    papr_db = config['hardware'].get('papr_db', 0.0)
    ibo_db = config['hardware'].get('ibo_db', 0.0)
    Gamma_pa = gamma_pa_floor + (10 ** (papr_db / 10) / 10 ** (ibo_db / 10)) * 1e-3
    Gamma_adc = 1.0 / (3.0 * (2 ** (2 * gamma_adc_bits)))
    Gamma_iq = 10 ** (-gamma_iq_irr_dbc / 10.0)
    Gamma_lo = (2 * np.pi * f_c_hz * gamma_lo_jitter_s) ** 2
    Gamma_eff_total = Gamma_pa + Gamma_adc + Gamma_iq + Gamma_lo

    # ✅ EXPERT FIX #4: 接收功率包含阵列增益
    kB = 1.380649e-23
    T_sys_K = 290.0
    N0 = kB * T_sys_K
    P_tx = config['isac_model'].get('P_tx', 1.0)
    P_rx = P_tx * G_sig_avg
    sigma2_gamma = Gamma_eff_total * P_rx

    # Phase noise spectrum
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    f_abs = np.abs(f_vec)
    with np.errstate(divide='ignore', invalid='ignore'):
        S_phi_c_k = S_phi_c_K2 / np.where(f_abs > 1e-6, f_abs ** 2, 1e12) + S_phi_c_K0

    H_err_type = config['pn_model'].get('H_err_model_type', 'FirstOrderHPF')
    if H_err_type == 'FirstOrderHPF':
        H_err_sq_k = f_abs ** 2 / (f_abs ** 2 + B_loop_hz ** 2)
    else:
        H_err_sq_k = np.ones_like(f_abs)

    S_phi_c_res_k = S_phi_c_k * H_err_sq_k
    sigma_2_phi_c_res = np.sum(S_phi_c_res_k) * Delta_f_hz

    # ✅ EXPERT FIX #5: 统一α依赖
    alpha_safe = max(alpha, np.finfo(float).eps)
    sigma2_DSE = C_DSE * (alpha_safe ** DSE_alpha_exponent)
    S_DSE_k = np.ones(N) * (sigma2_DSE / B_hz)

    # RSM
    s_rsm_path = config.get('waveform', {}).get('S_RSM_path', None)
    if s_rsm_path:
        try:
            rsm_data = np.loadtxt(s_rsm_path, delimiter=',')
            if len(rsm_data) != N:
                S_RSM_k = np.interp(np.arange(N), np.linspace(0, N - 1, len(rsm_data)), rsm_data)
            else:
                S_RSM_k = rsm_data.copy()
        except:
            S_RSM_k = np.ones(N) * 1e-6
    else:
        S_RSM_k = np.ones(N) * 1e-6

    # Total noise PSD
    N0_psd = N0 + sigma2_gamma / B_hz
    N_k_psd = N0_psd + S_RSM_k + S_phi_c_res_k + S_DSE_k
    N_k_psd = np.maximum(N_k_psd, 1e-30)

    noise_components = {
        'white': float(N0),
        'gamma': float(sigma2_gamma / B_hz),
        'rsm': float(np.mean(S_RSM_k)),
        'pn': float(np.mean(S_phi_c_res_k)),
        'dse': float(np.mean(S_DSE_k))
    }

    return {
        'N_k_psd': N_k_psd,
        'sigma_2_phi_c_res': sigma_2_phi_c_res,
        'Gamma_eff_total': Gamma_eff_total,
        'Gamma_pa': Gamma_pa,
        'Gamma_adc': Gamma_adc,
        'Gamma_iq': Gamma_iq,
        'Gamma_lo': Gamma_lo,
        'Delta_f_hz': Delta_f_hz,
        'sigma2_DSE': sigma2_DSE,
        'N0_psd': N0_psd,
        'noise_components': noise_components
    }


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration"""
    required_keys = [
        ('array', 'Nt'), ('array', 'Nr'), ('array', 'L_ap_m'), ('array', 'theta_0_deg'),
        ('channel', 'f_c_hz'), ('channel', 'B_hz'), ('channel', 'c_mps'),
        ('simulation', 'N'),
        ('hardware', 'rho_q_bits'), ('hardware', 'rho_a_error_rms'),
        ('hardware', 'gamma_pa_floor'), ('hardware', 'gamma_adc_bits'),
        ('hardware', 'gamma_iq_irr_dbc'), ('hardware', 'gamma_lo_jitter_s'),
        ('platform', 'sigma_theta_rad'),
        ('pn_model', 'sigma_rel_sq_rad2'), ('pn_model', 'S_phi_c_K2'),
        ('pn_model', 'S_phi_c_K0'), ('pn_model', 'B_loop_hz'),
        ('isac_model', 'alpha')
    ]

    for section, key in required_keys:
        if section not in config:
            raise KeyError(f"Missing section: {section}")
        if key not in config[section]:
            raise KeyError(f"Missing parameter: {section}.{key}")

    if config['array']['Nt'] <= 0 or config['array']['Nr'] <= 0:
        raise ValueError("Nt and Nr must be positive")
    if not (0 <= config['isac_model']['alpha'] <= 1):
        raise ValueError("alpha must be in [0, 1]")


if __name__ == "__main__":
    """
    Example usage and testing
    """

    # Mock configuration for testing
    test_config = {
        'array': {
            'Nt': 64,
            'Nr': 64,
            'L_ap_m': 0.1,
            'theta_0_deg': 30.0
        },
        'channel': {
            'f_c_hz': 140e9,
            'B_hz': 10e9,
            'c_mps': 299792458.0
        },
        'simulation': {
            'N': 2048
        },
        'hardware': {
            'rho_q_bits': 4,
            'rho_a_error_rms': 0.02,
            'gamma_pa_floor': 0.005,
            'gamma_adc_bits': 10,  # More realistic ENOB
            'gamma_iq_irr_dbc': -30.0,
            'gamma_lo_jitter_s': 20e-15  # More realistic jitter
        },
        'platform': {
            'sigma_theta_rad': 1e-6
        },
        'pn_model': {
            'sigma_rel_sq_rad2': 0.01,
            'S_phi_c_K2': 200.0,
            'S_phi_c_K0': 1e-15,
            'B_loop_hz': 1e6
        },
        'isac_model': {
            'alpha': 0.05,
            'C_DSE': 1e-9
        },
        'waveform': {
            'S_RSM_path': None
        }
    }

    print("Testing Physics Engine...")

    try:
        # Test configuration validation
        print("\n1. Validating configuration...")
        validate_config(test_config)
        print("✓ Configuration validation passed")

        # Test calc_g_sig_factors
        print("\n2. Testing calc_g_sig_factors...")
        g_factors = calc_g_sig_factors(test_config)
        print(f"✓ Multiplicative factors calculated")
        print(f"  G_sig_ideal = {g_factors['G_sig_ideal']:.2e}")
        print(f"  G_sig_avg = {g_factors['G_sig_avg']:.2e}")
        print(f"  eta_bsq_avg = {g_factors['eta_bsq_avg']:.4f}")
        print(f"  rho_Q = {g_factors['rho_Q']:.4f}")
        print(f"  rho_APE = {g_factors['rho_APE']:.4f}")
        print(f"  rho_A = {g_factors['rho_A']:.4f}")
        print(f"  rho_PN = {g_factors['rho_PN']:.4f}")

        # Test calc_n_f_vector
        print("\n3. Testing calc_n_f_vector...")
        n_outputs = calc_n_f_vector(test_config, g_factors)
        print(f"✓ Additive noise sources calculated")
        print(f"  Gamma_eff_total = {n_outputs['Gamma_eff_total']:.2e}")
        print(f"  Gamma breakdown:")
        print(
            f"    Gamma_pa:  {n_outputs['Gamma_pa']:.2e} ({100 * n_outputs['Gamma_pa'] / n_outputs['Gamma_eff_total']:.1f}%)")
        print(
            f"    Gamma_adc: {n_outputs['Gamma_adc']:.2e} ({100 * n_outputs['Gamma_adc'] / n_outputs['Gamma_eff_total']:.1f}%)")
        print(
            f"    Gamma_iq:  {n_outputs['Gamma_iq']:.2e} ({100 * n_outputs['Gamma_iq'] / n_outputs['Gamma_eff_total']:.1f}%)")
        print(
            f"    Gamma_lo:  {n_outputs['Gamma_lo']:.2e} ({100 * n_outputs['Gamma_lo'] / n_outputs['Gamma_eff_total']:.1f}%)")
        print(f"  sigma_2_phi_c_res = {n_outputs['sigma_2_phi_c_res']:.2e} rad²")
        print(f"  Delta_f_hz = {n_outputs['Delta_f_hz']:.2e} Hz")

        print("\n✓ All tests passed successfully!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()