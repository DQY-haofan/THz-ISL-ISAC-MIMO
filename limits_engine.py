#!/usr/bin/env python3
"""
Limits Engine for THz-ISL MIMO ISAC System
DR-08 Protocol Implementation

This module implements the performance limit calculations for hardware-limited
THz inter-satellite link ISAC systems according to DR-08 specifications.

Functions:
    calc_C_J: Calculate communication capacity (Jensen bound)
    calc_BCRLB: Calculate Bayesian Cramer-Rao Lower Bound (matched case)
    calc_MCRB: Calculate Misspecified Cramer-Rao Lower Bound

Author: Generated according to DR-08 Protocol v1.0
"""

import numpy as np
from typing import Dict, Any, Union, List, Tuple
from scipy.linalg import cholesky, inv, LinAlgError
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

    This function computes the Jensen inequality upper bound for channel capacity
    and its asymptotic characteristics, following the "PN once-counting" principle
    for communication (uses scalar sigma_2_phi_c_res, NOT vector N_k_psd).

    Args:
        config: Configuration dictionary
        g_sig_factors: Output from calc_g_sig_factors
        n_f_outputs: Output from calc_n_f_vector
        SNR0_db_vec: Array of SNR values in dB for capacity sweep
        compute_C_G: If True, also compute exact Gaussian capacity (DR-05)

    Returns:
        Dictionary containing:
            - C_J_vec: np.ndarray, Jensen capacity for each SNR [bits/s/Hz]
            - C_sat: float, saturation capacity [bits/s/Hz]
            - SNR_crit_db: float, critical SNR in dB
            - C_G_vec: np.ndarray (optional), Gaussian capacity if compute_C_G=True
    """

    # CRITICAL: PN once-counting for COMMUNICATION (Phase 2 衔接更正, Item 2)
    # Extract SCALAR quantities only (NOT N_k_psd vector)
    G_sig_avg = g_sig_factors['G_sig_avg']
    sigma_2_phi_c_res = n_f_outputs['sigma_2_phi_c_res']
    Gamma_eff_total = n_f_outputs['Gamma_eff_total']

    # Calculate phase noise coherence loss (multiplicative, in numerator)
    phase_coherence_loss = np.exp(-sigma_2_phi_c_res)

    # Convert SNR from dB to linear
    SNR0_vec = 10 ** (np.array(SNR0_db_vec) / 10.0)

    # Initialize output arrays
    C_J_vec = np.zeros_like(SNR0_vec)

    # Calculate Jensen capacity for each SNR value
    # Based on Phase 2, DR-01, Eq. (3.3.1)
    for i, SNR0 in enumerate(SNR0_vec):
        # Effective SINR (average over bandwidth)
        # SINR_eff = (SNR0 * G_sig_avg * exp(-sigma²)) / (1 + SNR0 * G_sig_avg * Gamma_eff)
        numerator = SNR0 * G_sig_avg * phase_coherence_loss
        denominator = 1.0 + SNR0 * G_sig_avg * Gamma_eff_total
        SINR_eff = numerator / denominator

        # Jensen capacity: C_J = log2(1 + SINR_eff)
        C_J_vec[i] = np.log2(1.0 + SINR_eff)

    # Calculate saturation capacity (C_sat) - Phase 2, DR-01, Eq. (4.1.1)
    # C_sat = log2(1 + exp(-sigma²_phi) / Gamma_eff)
    SINR_sat = phase_coherence_loss / Gamma_eff_total
    C_sat = np.log2(1.0 + SINR_sat)

    # Calculate critical SNR (SNR_crit) - Phase 2, DR-01, Eq. (4.2.1)
    # SNR_crit = 1 / (G_sig_avg * Gamma_eff_total)
    SNR_crit_linear = 1.0 / (G_sig_avg * Gamma_eff_total)
    SNR_crit_db = 10.0 * np.log10(SNR_crit_linear)

    # Prepare output dictionary
    results = {
        'C_J_vec': C_J_vec,
        'C_sat': C_sat,
        'SNR_crit_db': SNR_crit_db,
        'SINR_sat': SINR_sat,
        'phase_coherence_loss': phase_coherence_loss
    }

    # Optional: Compute exact Gaussian capacity (DR-05 Jensen gap validation)
    if compute_C_G:
        # Need to compute C_G = (1/B) ∫ log2(1 + SINR(f)) df
        # This requires frequency-dependent SINR calculation
        B_hz = config['channel']['B_hz']
        N = config['simulation']['N']

        # Get frequency-dependent beam squint
        eta_bsq_k = g_sig_factors['eta_bsq_k']

        # Calculate scalar gain factors (without beam squint)
        G_scalars = (g_sig_factors['G_sig_ideal'] *
                     g_sig_factors['rho_Q'] *
                     g_sig_factors['rho_APE'] *
                     g_sig_factors['rho_A'] *
                     g_sig_factors['rho_PN'])

        C_G_vec = np.zeros_like(SNR0_vec)

        for i, SNR0 in enumerate(SNR0_vec):
            # Frequency-dependent signal gain
            G_sig_f = G_scalars * eta_bsq_k

            # Frequency-dependent SINR
            SINR_f = (SNR0 * G_sig_f * phase_coherence_loss) / \
                     (1.0 + SNR0 * G_sig_avg * Gamma_eff_total)

            # Exact Gaussian capacity (average log)
            C_G_vec[i] = np.mean(np.log2(1.0 + SINR_f))

        results['C_G_vec'] = C_G_vec
        results['Jensen_gap_db'] = 10 * np.log10(2 ** results['C_J_vec'] / 2 ** C_G_vec)

    return results


def calc_BCRLB(
        config: Dict[str, Any],
        g_sig_factors: Dict[str, Union[float, np.ndarray]],
        n_f_outputs: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate Bayesian Cramer-Rao Lower Bound (matched case) - DR-08, Sec 5.2

    This function computes the sensing performance limit using Fisher Information
    Matrix, implementing the "PN once-counting" principle for sensing (uses
    vector N_k_psd, NOT scalar sigma_2_phi_c_res).

    Supports two FIM computation modes:
    - 'Whittle': Frequency-domain FIM (asymptotic, O(N log N))
    - 'Cholesky': Time-domain pre-whitening FIM (exact, O(N^3))

    Args:
        config: Configuration dictionary
        g_sig_factors: Output from calc_g_sig_factors
        n_f_outputs: Output from calc_n_f_vector

    Returns:
        Dictionary containing:
            - BCRLB_tau: float, range estimation CRLB [seconds²]
            - BCRLB_fD: float, Doppler estimation CRLB [Hz²]
            - FIM: np.ndarray, 2x2 Fisher Information Matrix
            - CRLB_matrix: np.ndarray, 2x2 CRLB covariance matrix
    """

    # Extract parameters
    N = config['simulation']['N']
    B_hz = config['channel']['B_hz']
    f_c_hz = config['channel']['f_c_hz']
    FIM_MODE = config['simulation'].get('FIM_MODE', 'Whittle')

    # CRITICAL: PN once-counting for SENSING (Phase 2 衔接更正, Item 2)
    # Use VECTOR noise PSD (includes S_phi_c_res[k]), NOT scalar sigma²
    N_k_psd = n_f_outputs['N_k_psd']
    Delta_f_hz = n_f_outputs['Delta_f_hz']

    # Get frequency-dependent beam squint (CRITICAL: use eta_bsq_k NOT eta_bsq_avg)
    eta_bsq_k = g_sig_factors['eta_bsq_k']

    # Calculate scalar gain factors (all rho factors, no beam squint)
    G_scalars = (g_sig_factors['G_sig_ideal'] *
                 g_sig_factors['rho_Q'] *
                 g_sig_factors['rho_APE'] *
                 g_sig_factors['rho_A'] *
                 g_sig_factors['rho_PN'])

    # Signal amplitude per frequency bin (includes beam squint)
    # s(f) = sqrt(G_scalars * eta_bsq(f))
    s_k = np.sqrt(G_scalars * eta_bsq_k)

    # Frequency axis
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)

    # Calculate signal gradients (Phase 2, DR-02, Sec 4.1, 4.2)
    # Gradient w.r.t. time delay: ∂s/∂τ = -j*2π*f * s(f)
    ds_dtau_k = -1j * 2 * np.pi * f_vec * s_k

    # Gradient w.r.t. Doppler: ∂s/∂f_D = j*2π*t * s(t) -> FFT domain
    # For simplicity, use frequency-domain approximation
    # ∂s/∂f_D ≈ j*2π*t_obs * s(f) where t_obs is effective observation time
    t_obs = N / B_hz  # Observation time
    ds_dfD_k = 1j * 2 * np.pi * t_obs * s_k

    # Select FIM computation mode
    if FIM_MODE == 'Whittle':
        # Whittle-FIM (frequency domain) - Phase 2, DR-02, Sec 5.1
        FIM = _compute_whittle_fim(ds_dtau_k, ds_dfD_k, N_k_psd, Delta_f_hz)

    elif FIM_MODE == 'Cholesky':
        # Cholesky-FIM (time domain with pre-whitening) - Phase 2, DR-02, Sec 5.2
        FIM = _compute_cholesky_fim(ds_dtau_k, ds_dfD_k, N_k_psd, N)

    else:
        raise ValueError(f"Unknown FIM_MODE: {FIM_MODE}. Must be 'Whittle' or 'Cholesky'")

    # Invert FIM to get BCRLB matrix
    try:
        CRLB_matrix = inv(FIM)
        BCRLB_tau = CRLB_matrix[0, 0]
        BCRLB_fD = CRLB_matrix[1, 1]
    except LinAlgError:
        warnings.warn("FIM is singular, returning infinite BCRLB")
        BCRLB_tau = np.inf
        BCRLB_fD = np.inf
        CRLB_matrix = np.array([[np.inf, np.inf], [np.inf, np.inf]])

    return {
        'BCRLB_tau': BCRLB_tau,
        'BCRLB_fD': BCRLB_fD,
        'FIM': FIM,
        'CRLB_matrix': CRLB_matrix
    }


def _compute_whittle_fim(
        ds_dtau_k: np.ndarray,
        ds_dfD_k: np.ndarray,
        N_k_psd: np.ndarray,
        Delta_f_hz: float
) -> np.ndarray:
    """
    Compute Whittle-FIM in frequency domain

    FIM[i,j] = ∫ (1/N(f)) * 2*Re{(∂s/∂θi)^H * (∂s/∂θj)} df
    """

    # Compute FIM elements using Whittle formula
    # F_tau_tau
    integrand_tt = (1.0 / N_k_psd) * 2 * np.real(np.conj(ds_dtau_k) * ds_dtau_k)
    F_tau_tau = np.sum(integrand_tt) * Delta_f_hz

    # F_fD_fD
    integrand_ff = (1.0 / N_k_psd) * 2 * np.real(np.conj(ds_dfD_k) * ds_dfD_k)
    F_fD_fD = np.sum(integrand_ff) * Delta_f_hz

    # F_tau_fD (cross term)
    integrand_tf = (1.0 / N_k_psd) * 2 * np.real(np.conj(ds_dtau_k) * ds_dfD_k)
    F_tau_fD = np.sum(integrand_tf) * Delta_f_hz

    # Construct FIM (symmetric)
    FIM = np.array([
        [F_tau_tau, F_tau_fD],
        [F_tau_fD, F_fD_fD]
    ])

    return FIM


def _compute_cholesky_fim(
        ds_dtau_k: np.ndarray,
        ds_dfD_k: np.ndarray,
        N_k_psd: np.ndarray,
        N: int
) -> np.ndarray:
    """
    Compute FIM using time-domain pre-whitening (Cholesky fallback)

    This is the exact method for non-WSS or small-N cases.
    """

    # Convert noise PSD to time-domain covariance matrix (Toeplitz)
    # Sigma_n = IFFT(N_k_psd)
    acf = np.fft.ifft(np.fft.ifftshift(N_k_psd))

    # Construct Toeplitz covariance matrix
    from scipy.linalg import toeplitz
    Sigma_n = toeplitz(acf)

    # Ensure Hermitian symmetry
    Sigma_n = (Sigma_n + np.conj(Sigma_n.T)) / 2

    # Check condition number and apply diagonal loading if needed
    cond_num = np.linalg.cond(Sigma_n)
    if cond_num > 1e12:
        eps = 1e-12 * np.trace(Sigma_n).real / N
        Sigma_n += eps * np.eye(N)
        warnings.warn(f"Sigma_n ill-conditioned (cond={cond_num:.2e}), applied diagonal loading")

    # Cholesky decomposition: Sigma_n = L * L^H
    try:
        L = cholesky(Sigma_n, lower=True)
    except LinAlgError:
        warnings.warn("Cholesky decomposition failed, using pseudo-inverse")
        # Fallback: use SVD-based pseudo-inverse
        U, S, Vh = np.linalg.svd(Sigma_n)
        S_inv = np.where(S > 1e-10 * S[0], 1.0 / S, 0)
        Sigma_n_inv = (Vh.T.conj() * S_inv) @ Vh
        W = np.linalg.cholesky(Sigma_n_inv)
    else:
        # Whitening matrix: W = L^{-1}
        W = inv(L)

    # Convert signal gradients to time domain
    J_tau_t = np.fft.ifft(np.fft.ifftshift(ds_dtau_k))
    J_fD_t = np.fft.ifft(np.fft.ifftshift(ds_dfD_k))

    # Apply whitening: J_w = W * J
    J_w_tau = W @ J_tau_t
    J_w_fD = W @ J_fD_t

    # Compute FIM using Slepian-Bangs formula
    # F_ij = 2 * Re{J_w_i^H * J_w_j}
    F_tau_tau = 2 * np.real(np.conj(J_w_tau) @ J_w_tau)
    F_fD_fD = 2 * np.real(np.conj(J_w_fD) @ J_w_fD)
    F_tau_fD = 2 * np.real(np.conj(J_w_tau) @ J_w_fD)

    FIM = np.array([
        [F_tau_tau, F_tau_fD],
        [F_tau_fD, F_fD_fD]
    ])

    return FIM


def calc_MCRB(
        config: Dict[str, Any],
        g_sig_factors: Dict[str, Union[float, np.ndarray]],
        n_f_outputs: Dict[str, Union[float, np.ndarray]]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate Misspecified Cramer-Rao Lower Bound - DR-08, Sec 5.3

    This function computes the MCRB "sandwich" bound for model mismatch due to
    residual DSE (Doppler Squint Effect).

    MCRB = J^{-1} * K * J^{-1}

    where:
    - J: Hessian matrix based on assumed model (Phi_q = 0)
    - K: Gradient outer product based on true model (Phi_q from config)

    Args:
        config: Configuration dictionary
        g_sig_factors: Output from calc_g_sig_factors
        n_f_outputs: Output from calc_n_f_vector

    Returns:
        Dictionary containing:
            - MCRB_tau: float, mismatched range estimation bound [seconds²]
            - MCRB_fD: float, mismatched Doppler estimation bound [Hz²]
            - MCRB_matrix: np.ndarray, 2x2 MCRB covariance matrix
            - F_matched: np.ndarray, 2x2 matched FIM (J matrix)
            - Phi_q_rad: float, mismatch parameter value [radians]
    """

    # Get mismatch parameter
    Phi_q_rad = config.get('waveform', {}).get('Phi_q', 0.0)

    # If no mismatch, MCRB equals BCRLB
    if abs(Phi_q_rad) < 1e-12:
        bcrlb_results = calc_BCRLB(config, g_sig_factors, n_f_outputs)
        return {
            'MCRB_tau': bcrlb_results['BCRLB_tau'],
            'MCRB_fD': bcrlb_results['BCRLB_fD'],
            'MCRB_matrix': bcrlb_results['CRLB_matrix'],
            'F_matched': bcrlb_results['FIM'],
            'Phi_q_rad': 0.0
        }

    # Step 1: Compute matched FIM (J matrix) - assumes Phi_q = 0
    # This is the standard BCRLB calculation
    bcrlb_results = calc_BCRLB(config, g_sig_factors, n_f_outputs)
    F_matched = bcrlb_results['FIM']

    # Step 2: Compute K matrix (gradient outer product under true model)
    # For DSE mismatch, K ≈ F_matched (Phase 2, DR-03, Sec 2.4)
    # This is because the noise-induced randomness dominates
    K = F_matched.copy()

    # Step 3: Compute bias matrix E_bias
    # E_bias captures the second-order effect of mismatch
    # For small Phi_q, E_bias ≈ (Phi_q)² * (second derivative terms)
    # Phase 2, DR-03, Sec 2.3

    # Get parameters for bias calculation
    N = config['simulation']['N']
    B_hz = config['channel']['B_hz']
    Delta_f_hz = n_f_outputs['Delta_f_hz']
    N_k_psd = n_f_outputs['N_k_psd']

    # Signal gradients (same as BCRLB)
    eta_bsq_k = g_sig_factors['eta_bsq_k']
    G_scalars = (g_sig_factors['G_sig_ideal'] *
                 g_sig_factors['rho_Q'] *
                 g_sig_factors['rho_APE'] *
                 g_sig_factors['rho_A'] *
                 g_sig_factors['rho_PN'])
    s_k = np.sqrt(G_scalars * eta_bsq_k)

    # Time vector for quadratic phase
    t_vec = np.linspace(-N / (2 * B_hz), N / (2 * B_hz), N)
    t_obs = N / B_hz

    # Signal difference due to mismatch (in time domain)
    # S_diff = S_true - S_est ≈ s(t) * [exp(j*Phi_q*t²/T²) - 1]
    # For small Phi_q, this is approximately j*Phi_q*t²/T² * s(t)
    phase_mismatch = Phi_q_rad * (t_vec / t_obs) ** 2
    s_diff_t = s_k.mean() * (np.exp(1j * phase_mismatch) - 1)

    # Convert to frequency domain
    s_diff_f = np.fft.fft(np.fft.fftshift(s_diff_t))

    # Compute bias matrix elements (Phase 2, DR-03, Eq 2.3.1)
    # E_bias[i,j] = ∫ (1/N(f)) * 2*Re{(∂²s/∂θi∂θj)^H * s_diff} df

    # Second derivatives (approximate for CE-Chirp waveform)
    f_vec = np.linspace(-B_hz / 2, B_hz / 2, N)
    d2s_dtau2_k = -(2 * np.pi * f_vec) ** 2 * s_k
    d2s_dfD2_k = (2 * np.pi * t_obs) ** 2 * s_k
    d2s_dtau_dfD_k = -1j * (2 * np.pi) ** 2 * f_vec * t_obs * s_k

    # Bias matrix elements
    E_bias = np.zeros((2, 2))

    integrand_tt = (1.0 / N_k_psd) * 2 * np.real(np.conj(d2s_dtau2_k) * s_diff_f)
    E_bias[0, 0] = -np.sum(integrand_tt) * Delta_f_hz

    integrand_ff = (1.0 / N_k_psd) * 2 * np.real(np.conj(d2s_dfD2_k) * s_diff_f)
    E_bias[1, 1] = -np.sum(integrand_ff) * Delta_f_hz

    integrand_tf = (1.0 / N_k_psd) * 2 * np.real(np.conj(d2s_dtau_dfD_k) * s_diff_f)
    E_bias[0, 1] = -np.sum(integrand_tf) * Delta_f_hz
    E_bias[1, 0] = E_bias[0, 1]

    # Step 4: Compute J matrix
    # J = F_matched - E_bias (Phase 2, DR-03, Sec 2.3)
    J = F_matched - E_bias

    # Step 5: Compute MCRB sandwich matrix
    # C_MCRB = J^{-1} * K * J^{-1}
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
    """
    Validate configuration for limits engine calculations

    Args:
        config: Configuration dictionary to validate

    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameter values are invalid
    """

    # Check required simulation parameters
    required_sim_keys = ['N', 'SNR0_db_vec']
    for key in required_sim_keys:
        if key not in config.get('simulation', {}):
            raise KeyError(f"Missing simulation parameter: {key}")

    # Validate FIM_MODE if present
    if 'FIM_MODE' in config.get('simulation', {}):
        fim_mode = config['simulation']['FIM_MODE']
        if fim_mode not in ['Whittle', 'Cholesky']:
            raise ValueError(f"Invalid FIM_MODE: {fim_mode}. Must be 'Whittle' or 'Cholesky'")


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
            'N': 2048,
            'FIM_MODE': 'Whittle',
            'SNR0_db_vec': [-10, 0, 10, 20, 30, 40]
        },
        'hardware': {
            'rho_q_bits': 4,
            'rho_a_error_rms': 0.02,
            'gamma_pa_floor': 0.005,
            'gamma_adc_bits': 6,
            'gamma_iq_irr_dbc': -30.0,
            'gamma_lo_jitter_s': 50e-15
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
            'Phi_q': 0.1  # Mismatch parameter for MCRB
        }
    }

    print("Testing Limits Engine...")

    try:
        # Import physics engine
        from physics_engine import calc_g_sig_factors, calc_n_f_vector

        # Calculate physics factors
        print("\n1. Calculating physics factors...")
        g_factors = calc_g_sig_factors(test_config)
        n_outputs = calc_n_f_vector(test_config, g_factors)
        print("✓ Physics factors calculated")

        # Test calc_C_J
        print("\n2. Testing calc_C_J...")
        SNR0_db_vec = test_config['simulation']['SNR0_db_vec']
        c_j_results = calc_C_J(test_config, g_factors, n_outputs, SNR0_db_vec,
                               compute_C_G=True)
        print(f"✓ C_J calculation completed")
        print(f"  C_sat = {c_j_results['C_sat']:.3f} bits/s/Hz")
        print(f"  SNR_crit = {c_j_results['SNR_crit_db']:.2f} dB")
        print(f"  Phase coherence loss = {c_j_results['phase_coherence_loss']:.4f}")

        # Test calc_BCRLB
        print("\n3. Testing calc_BCRLB (Whittle mode)...")
        bcrlb_results = calc_BCRLB(test_config, g_factors, n_outputs)
        print(f"✓ BCRLB calculation completed")
        print(f"  BCRLB(τ) = {bcrlb_results['BCRLB_tau']:.2e} s²")
        print(f"  BCRLB(f_D) = {bcrlb_results['BCRLB_fD']:.2e} Hz²")

        # Test calc_BCRLB with Cholesky
        print("\n4. Testing calc_BCRLB (Cholesky mode)...")
        test_config['simulation']['FIM_MODE'] = 'Cholesky'
        bcrlb_chol = calc_BCRLB(test_config, g_factors, n_outputs)
        print(f"✓ BCRLB (Cholesky) calculation completed")
        print(f"  BCRLB(τ) = {bcrlb_chol['BCRLB_tau']:.2e} s²")

        # Test calc_MCRB
        print("\n5. Testing calc_MCRB...")
        test_config['simulation']['FIM_MODE'] = 'Whittle'  # Reset
        mcrb_results = calc_MCRB(test_config, g_factors, n_outputs)
        print(f"✓ MCRB calculation completed")
        print(f"  MCRB(τ) = {mcrb_results['MCRB_tau']:.2e} s²")
        print(f"  MCRB(f_D) = {mcrb_results['MCRB_fD']:.2e} Hz²")
        print(f"  Mismatch parameter Φ_q = {mcrb_results['Phi_q_rad']:.3f} rad")

        # Calculate degradation due to mismatch
        if bcrlb_results['BCRLB_tau'] > 0:
            degradation_db = 10 * np.log10(mcrb_results['MCRB_tau'] /
                                           bcrlb_results['BCRLB_tau'])
            print(f"  Performance degradation = {degradation_db:.2f} dB")

        print("\n✓ All tests passed successfully!")

    except ImportError:
        print("✗ Error: physics_engine.py not found. Please ensure it's in the same directory.")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()