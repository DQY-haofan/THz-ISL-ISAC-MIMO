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
    """
    Calculate multiplicative gain/loss factors (DR-08, Sec 4.1)

    This function computes all multiplicative degradation factors affecting
    the signal path, following the unified multiplicative gain skeleton.

    Args:
        config: Configuration dictionary from YAML file

    Returns:
        Dictionary containing:
            - eta_bsq_k: np.ndarray, shape (N,), beam squint per frequency
            - eta_bsq_avg: float, average beam squint factor
            - rho_Q: float, phase quantization factor
            - rho_APE: float, antenna pointing error factor
            - rho_A: float, amplitude error factor
            - rho_PN: float, differential phase noise factor
            - G_sig_ideal: float, ideal MIMO gain (Nt * Nr)
            - G_sig_avg: float, average signal gain with all impairments
    """

    # 1. Extract basic parameters from config
    try:
        # Array parameters
        Nt = config['array']['Nt']
        Nr = config['array']['Nr']
        L_ap_m = config['array']['L_ap_m']
        theta_0_deg = config['array']['theta_0_deg']

        # Channel parameters
        f_c_hz = config['channel']['f_c_hz']
        B_hz = config['channel']['B_hz']
        c_mps = config['channel']['c_mps']

        # Simulation parameters
        N = config['simulation']['N']

        # Hardware parameters
        rho_q_bits = config['hardware']['rho_q_bits']
        rho_a_error_rms = config['hardware']['rho_a_error_rms']

        # Platform parameters
        sigma_theta_rad = config['platform']['sigma_theta_rad']

        # Phase noise parameters
        sigma_rel_sq_rad2 = config['pn_model']['sigma_rel_sq_rad2']

    except KeyError as e:
        raise KeyError(f"Missing required configuration parameter: {e}")

    # 2. Calculate derived parameters
    theta_0_rad = np.radians(theta_0_deg)
    lambda_c = c_mps / f_c_hz
    L_over_lambda = L_ap_m / lambda_c
    B_over_fc = B_hz / f_c_hz
    sin_theta_0 = np.sin(theta_0_rad)
    cos_theta_0 = np.cos(theta_0_rad)

    # 3. Ideal MIMO gain
    G_sig_ideal = float(Nt * Nr)

    # 4. Calculate beam squint factors (DR-01)

    # 4a. Frequency axis for beam squint calculation
    f_axis = np.linspace(-B_hz / 2, B_hz / 2, N)
    freq_offset_norm = f_axis / f_c_hz  # Normalized frequency offset

    # 4b. Beam squint per frequency bin (Tx/Rx combined = sinc^4)
    # Based on DR-01 equation for joint Tx/Rx beam squint
    sinc_arg_k = np.pi * L_over_lambda * sin_theta_0 * freq_offset_norm

    # Handle sinc(0) = 1 case properly
    with np.errstate(divide='ignore', invalid='ignore'):
        sinc_k = np.where(sinc_arg_k == 0, 1.0, np.sin(sinc_arg_k) / sinc_arg_k)

    eta_bsq_k = sinc_k ** 4  # Joint Tx/Rx beam squint factor

    # 4c. Average beam squint factor (analytical approximation)
    # From DR-01: Small angle approximation for eta_bsq_avg
    K = np.pi * L_over_lambda * sin_theta_0
    R_B = B_over_fc
    eta_bsq_avg = 1.0 - (K ** 2 * R_B ** 2) / 18.0

    # Ensure eta_bsq_avg is non-negative
    eta_bsq_avg = max(0.0, eta_bsq_avg)

    # 5. Calculate phase quantization factor (DR-01)
    b_phi = rho_q_bits
    if b_phi <= 0:
        rho_Q = 0.0  # No quantization bits = complete loss
    else:
        quantization_arg = np.pi / (2 ** b_phi)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_Q = (np.sin(quantization_arg) / quantization_arg) ** 2 if quantization_arg != 0 else 1.0

    # 6. Calculate antenna pointing error factor (DR-01)
    # Based on Gaussian beam pattern and pointing error statistics
    sigma_theta = sigma_theta_rad
    pointing_factor = (np.pi ** 2 / 3.0) * (L_over_lambda * cos_theta_0) ** 2 * (sigma_theta ** 2)
    rho_APE = 1.0 - pointing_factor

    # Ensure rho_APE is non-negative
    rho_APE = max(0.0, rho_APE)

    # 7. Calculate amplitude error factor (DR-01)
    # Ruze formula for amplitude errors
    sigma_a_rms = rho_a_error_rms
    rho_A = 1.0 / (1.0 + sigma_a_rms ** 2)

    # 8. Calculate differential phase noise factor (DR-02)
    # Exponential loss due to differential phase noise between array elements
    sigma_rel_sq = sigma_rel_sq_rad2
    rho_PN = np.exp(-sigma_rel_sq)

    # 9. Aggregate total average signal gain (unified multiplicative skeleton)
    # Following DR-08 Sec 3.2.1 equation
    G_sig_avg = G_sig_ideal * eta_bsq_avg * rho_Q * rho_APE * rho_A * rho_PN

    # 10. Return results dictionary
    return {
        'G_sig_ideal': G_sig_ideal,
        'eta_bsq_avg': eta_bsq_avg,
        'eta_bsq_k': eta_bsq_k,
        'rho_Q': rho_Q,
        'rho_APE': rho_APE,
        'rho_A': rho_A,
        'rho_PN': rho_PN,
        'G_sig_avg': G_sig_avg
    }


def calc_n_f_vector(config: Dict[str, Any], g_sig_factors: Dict[str, Union[float, np.ndarray]]) -> Dict[
    str, Union[float, np.ndarray]]:
    """
    Calculate additive noise sources (DR-08, Sec 4.2)

    This function computes all additive noise components and implements the
    "PN once-counting" dual aperture approach for phase noise handling.

    FIXED (Expert Review Item #3): Now returns breakdown of Gamma components
    for diagnostics and visualization.

    Args:
        config: Configuration dictionary from YAML file
        g_sig_factors: Output from calc_g_sig_factors function

    Returns:
        Dictionary containing:
            - N_k_psd: np.ndarray, shape (N,), noise PSD per frequency bin [power/Hz]
            - sigma_2_phi_c_res: float, residual phase noise variance [rad²]
            - Gamma_eff_total: float, total hardware distortion factor [dimensionless]
            - Gamma_pa: float, PA contribution to distortion
            - Gamma_adc: float, ADC contribution to distortion
            - Gamma_iq: float, I/Q imbalance contribution to distortion
            - Gamma_lo: float, LO jitter contribution to distortion
            - Delta_f_hz: float, frequency bin width [Hz]
    """

    # 1. Extract parameters from config
    try:
        # Simulation parameters
        N = config['simulation']['N']
        B_hz = config['channel']['B_hz']

        # Array parameters
        Nt = config['array']['Nt']

        # Hardware distortion parameters
        gamma_pa_floor = config['hardware']['gamma_pa_floor']
        gamma_adc_bits = config['hardware']['gamma_adc_bits']
        gamma_iq_irr_dbc = config['hardware']['gamma_iq_irr_dbc']
        gamma_lo_jitter_s = config['hardware']['gamma_lo_jitter_s']

        # Phase noise model parameters
        S_phi_c_K2 = config['pn_model']['S_phi_c_K2']
        S_phi_c_K0 = config['pn_model']['S_phi_c_K0']
        B_loop_hz = config['pn_model']['B_loop_hz']

        # ISAC model parameters
        alpha = config['isac_model']['alpha']
        C_DSE = config['isac_model']['C_DSE']

        # Extract from g_sig_factors (causality dependency)
        G_sig_avg = g_sig_factors['G_sig_avg']

    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e}")

    # 2. Calculate frequency bin width
    Delta_f_hz = B_hz / N

    # ========================================================================
    # 3. Calculate hardware distortion factor Gamma_eff_total
    # FIXED (Expert Review Item #3): Now returning component breakdown
    # ========================================================================

    # 3a. Power amplifier contribution (dominant)
    Gamma_pa = gamma_pa_floor

    # 3b. ADC contribution (ENOB-based)
    ENOB = gamma_adc_bits

    # SANITY CHECK: Clamp ENOB to reasonable range [4, 20]
    # Prevents numerical overflow/underflow (Expert Review Item #3)
    ENOB = np.clip(ENOB, 4, 20)

    Gamma_adc = 10 ** (-((6.02 * ENOB + 1.76) / 10))

    # 3c. I/Q imbalance contribution (IRR-based)
    IRR_dBc = gamma_iq_irr_dbc
    Gamma_iq = 10 ** (IRR_dBc / 10)

    # 3d. LO jitter contribution
    sigma_t_jitter = gamma_lo_jitter_s

    # SANITY CHECK: Clamp jitter to reasonable range [1fs, 1ns]
    # Prevents numerical explosion (Expert Review Item #3)
    sigma_t_jitter = np.clip(sigma_t_jitter, 1e-15, 1e-9)

    Gamma_lo = (np.pi * B_hz * sigma_t_jitter) ** 2

    # 3e. Total hardware distortion factor
    Gamma_eff_total = Gamma_pa + Gamma_adc + Gamma_iq + Gamma_lo

    # ========================================================================
    # DIAGNOSTIC: Print Gamma breakdown (Expert Review Item #3)
    # Uncomment for debugging or when running diagnostics
    # ========================================================================
    # print(f"  [Gamma Breakdown]")
    # print(f"    Gamma_pa:  {Gamma_pa:.2e} ({100*Gamma_pa/Gamma_eff_total:.1f}%)")
    # print(f"    Gamma_adc: {Gamma_adc:.2e} ({100*Gamma_adc/Gamma_eff_total:.1f}%)")
    # print(f"    Gamma_iq:  {Gamma_iq:.2e} ({100*Gamma_iq/Gamma_eff_total:.1f}%)")
    # print(f"    Gamma_lo:  {Gamma_lo:.2e} ({100*Gamma_lo/Gamma_eff_total:.1f}%)")
    # print(f"    Total:     {Gamma_eff_total:.2e}")

    # 4. Calculate phase noise components (PN once-counting implementation)

    # 4a. Frequency axis for noise PSD calculation
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    f_abs = np.abs(f_vec)

    # Avoid division by zero at DC
    f_abs = np.where(f_abs < 1e-6, 1e-6, f_abs)

    # 4b. Total common phase noise PSD (Wiener + white floor)
    S_phi_c_tot_f = S_phi_c_K2 / (f_abs ** 2) + S_phi_c_K0

    # 4c. Tracking loop error transfer function (first-order high-pass)
    # H_err(f) = f / (f + B_L) for f >> B_L approximation
    H_err_f = f_abs / (f_abs + B_loop_hz)

    # 4d. Residual phase noise PSD after tracking
    S_phi_c_res_f = S_phi_c_tot_f * (H_err_f ** 2)

    # 4e. Convert to discrete PSD and integrate for total variance
    S_phi_c_res_k = S_phi_c_res_f  # Discrete samples
    sigma_2_phi_c_res = np.sum(S_phi_c_res_k) * Delta_f_hz  # Integration for [rad²]

    # 5. Calculate white noise components

    # 5a. Thermal noise (normalized reference)
    # 5a. Thermal noise (normalized reference)
    N0 = 1.0  # Normalized thermal noise floor

    # 5b. Hardware distortion noise (power-dependent, Gamma_eff normalization)
    P_pa_normalized = 1.0  # Normalized PA power
    P_sig_avg_rx = Nt * P_pa_normalized * G_sig_avg
    sigma2_gamma = (P_sig_avg_rx * Gamma_eff_total) / B_hz

    # 5c. DSE noise
    if alpha > 0:
        sigma2_DSE = C_DSE / (alpha ** 5)
    else:
        sigma2_DSE = 1e6

    # ✅ ADD THIS LINE (base noise PSD for BCRLB calibration)
    N0_psd = N0 + sigma2_gamma  # Base white noise PSD

    # ====================================================================
    # CRITICAL FIX (Expert Review Item #6 - RMSE Amplitude Calibration):
    # Add N0_psd for BCRLB signal amplitude normalization
    # ====================================================================
    # Base noise PSD (white components only, no colored noise yet)
    N0_psd = N0 + sigma2_gamma  # W/Hz, linear scale
    # NOTE: We exclude sigma2_DSE from N0_psd because it scales with alpha
    # and will be added separately in N_k_psd

    # 6. Load residual spectral modulation (RSM) if available
    try:
        s_rsm_path = config.get('waveform', {}).get('S_RSM_path', None)
        if s_rsm_path:
            try:
                rsm_data = np.loadtxt(s_rsm_path, delimiter=',')
                if len(rsm_data) != N:
                    S_RSM_k = np.interp(np.arange(N),
                                        np.linspace(0, N - 1, len(rsm_data)),
                                        rsm_data)
                else:
                    S_RSM_k = rsm_data.copy()
            except:
                S_RSM_k = np.ones(N) * 1e-6
        else:
            S_RSM_k = np.ones(N) * 1e-6
    except:
        S_RSM_k = np.ones(N) * 1e-6

    # 7. Aggregate total noise PSD
    white_noise_components = N0 + sigma2_gamma + sigma2_DSE
    N_k_psd = white_noise_components + S_RSM_k + S_phi_c_res_k
    N_k_psd = np.maximum(N_k_psd, 1e-12)

    # ✅ MODIFY RETURN: Add 'N0_psd' to the return dictionary
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
        'N0_psd': N0_psd  # ✅ NEW: Add this line!
    }


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary for required parameters

    Args:
        config: Configuration dictionary to validate

    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameter values are invalid
    """

    required_keys = [
        # Array parameters
        ('array', 'Nt'),
        ('array', 'Nr'),
        ('array', 'L_ap_m'),
        ('array', 'theta_0_deg'),

        # Channel parameters
        ('channel', 'f_c_hz'),
        ('channel', 'B_hz'),
        ('channel', 'c_mps'),

        # Simulation parameters
        ('simulation', 'N'),

        # Hardware parameters
        ('hardware', 'rho_q_bits'),
        ('hardware', 'rho_a_error_rms'),
        ('hardware', 'gamma_pa_floor'),
        ('hardware', 'gamma_adc_bits'),
        ('hardware', 'gamma_iq_irr_dbc'),
        ('hardware', 'gamma_lo_jitter_s'),

        # Platform parameters
        ('platform', 'sigma_theta_rad'),

        # Phase noise parameters
        ('pn_model', 'sigma_rel_sq_rad2'),
        ('pn_model', 'S_phi_c_K2'),
        ('pn_model', 'S_phi_c_K0'),
        ('pn_model', 'B_loop_hz'),

        # ISAC parameters
        ('isac_model', 'alpha'),
        ('isac_model', 'C_DSE')
    ]

    for section, key in required_keys:
        if section not in config:
            raise KeyError(f"Missing configuration section: {section}")
        if key not in config[section]:
            raise KeyError(f"Missing required parameter: {section}.{key}")

    # Validate value ranges
    if config['array']['Nt'] <= 0 or config['array']['Nr'] <= 0:
        raise ValueError("Nt and Nr must be positive integers")

    if config['channel']['B_hz'] <= 0 or config['channel']['f_c_hz'] <= 0:
        raise ValueError("Bandwidth and carrier frequency must be positive")

    if config['simulation']['N'] <= 0:
        raise ValueError("Number of frequency bins N must be positive")

    if not (0 <= config['isac_model']['alpha'] <= 1):
        raise ValueError("ISAC overhead parameter alpha must be in [0, 1]")


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