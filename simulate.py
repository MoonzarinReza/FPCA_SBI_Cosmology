import numpy as np
import pandas as pd
from astropy.cosmology import LambdaCDM
from scipy import integrate
from scipy.stats import truncnorm

from config import SimGenConfig

C_LIGHT_KM_S = 300000 


def sample_truncnorm(mu, sigma, lower, upper, size=None):
    """Draws from the GLOBAL numpy random state (matches legacy script)."""
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)

def draw_cosmo_param(prior, bounds, gauss):
    """One draw from a uniform (bounds) or truncated-normal (gauss) prior."""
    if prior == "uniform":
        return np.random.uniform(*bounds)
    if prior == "gaussian":
        return sample_truncnorm(*gauss)
    raise ValueError(f"Unknown cosmo_prior: {prior!r} (use 'uniform' or 'gaussian')")


def _E_z(z, Om, Ol):
    Ok = 1.0 - Om - Ol
    return np.sqrt(Om * (1 + z) ** 3 + Ok * (1 + z) ** 2 + Ol)


def _chi_of_z(z, Om, Ol, H0):
    integrand = lambda zp: 1.0 / _E_z(zp, Om, Ol)
    return (C_LIGHT_KM_S / H0) * integrate.quad(integrand, 0, z)[0]


def _D_M(z, Om, Ol, H0):
    Ok = 1.0 - Om - Ol
    chi = _chi_of_z(z, Om, Ol, H0)
    if np.isclose(Ok, 0.0):
        return chi
    sqrtOk = np.sqrt(abs(Ok))
    factor = (C_LIGHT_KM_S / H0) / sqrtOk
    x = sqrtOk * H0 * chi / C_LIGHT_KM_S
    return factor * (np.sinh(x) if Ok > 0 else np.sin(x))


def _w_shape(z, Om, Ol, H0):
    Ez = _E_z(z, Om, Ol)
    DM = _D_M(z, Om, Ol, H0)
    return (1 + z) ** 0.5 * DM ** 2 / Ez


def sample_redshifts(n_samples, z_edges, Om, Ol, H0):
    """np.random.choice / np.random.random on global state -- matches original."""
    n_bins = len(z_edges) - 1
    counts = np.empty(n_bins)
    for i in range(n_bins):
        z1, z2 = z_edges[i], z_edges[i + 1]
        counts[i] = integrate.quad(lambda z: _w_shape(z, Om, Ol, H0), z1, z2)[0]

    frac = counts / counts.sum()
    bin_idx = np.random.choice(n_bins, size=n_samples, p=frac)
    u = np.random.random(size=n_samples)
    return z_edges[bin_idx] + u * (z_edges[bin_idx + 1] - z_edges[bin_idx])


def simulate_one_cosmology(cosmo_id: int, cfg: SimGenConfig) -> pd.DataFrame:
    # --- per-cosmology draws: match original exactly (global state, scalar calls) ---
    omega_m = draw_cosmo_param(cfg.cosmo_prior, cfg.omega_m_bounds, cfg.omega_m_gauss)
    omega_l = draw_cosmo_param(cfg.cosmo_prior, cfg.omega_l_bounds, cfg.omega_l_gauss)

    alpha = sample_truncnorm(*cfg.alpha_prior)
    beta = sample_truncnorm(*cfg.beta_prior)

    MB_mu = np.random.normal(loc=cfg.MB_mu_loc, scale=cfg.MB_mu_scale)
    MB_sigma = np.random.uniform(*cfg.MB_sigma_bounds)

    z_edges = np.arange(cfg.z_min, cfg.z_max, cfg.z_bin_width)
    n_sn = cfg.n_sn_per_cosmo

    redshifts = sample_redshifts(n_sn, z_edges, omega_m, omega_l, cfg.H0)

    # --- per-SN draws: kept vectorized -- values won't match original 1:1,
    # only their distribution does, since batching consumes the RNG
    # stream differently than one-at-a-time calls ---
    c = sample_truncnorm(*cfg.c_prior, size=n_sn)
    x1 = sample_truncnorm(*cfg.x1_prior, size=n_sn)
    
    MB_abs = sample_truncnorm(MB_mu, MB_sigma, *cfg.MB_abs_bounds, size=n_sn)

    cosmo = LambdaCDM(H0=cfg.H0, Om0=omega_m, Ode0=omega_l)
    mu = cosmo.distmod(redshifts).value

    m_b_theory = MB_abs - alpha * x1 + beta * c + mu

    return pd.DataFrame({
        "cosmo_id": cosmo_id,
        "omega_m": omega_m,
        "omega_lambda": omega_l,
        "MB_mu": MB_mu,
        "MB_sigma": MB_sigma,
        "alpha": alpha,
        "beta": beta,
        "sn_id": np.arange(n_sn),
        "redshift": redshifts,
        "x1": x1,
        "c": c,
        "mb_theory": m_b_theory,
    })


def simulate(cfg: SimGenConfig, show_progress: bool = True) -> pd.DataFrame:
    
    np.random.seed(cfg.seed)   

    iterator = range(cfg.n_cosmologies)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="simulating cosmologies")

    frames = [simulate_one_cosmology(i, cfg) for i in iterator]
    return pd.concat(frames, ignore_index=True)