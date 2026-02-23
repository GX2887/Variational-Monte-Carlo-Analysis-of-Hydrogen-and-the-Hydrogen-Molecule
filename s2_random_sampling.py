import numpy as np

# --- wavefunction and target pdf ------------------------------------------

def psi_0(x):
    """Ground-state HO wavefunction use H(0)"""
    return np.exp(-x**2 / 2)

def rho_unnormalised(x):
    """Target PDF from wave function"""
    return np.abs(psi_0(x))**2 

# --- rejection sampler ----------------------------------------------------

def sample_from_rho_rejection(Ns, x_min=-5.0, x_max=5.0):
    """
    Generate Ns samples x^(i) from ρ(x) using rejection sampling.

    Parameters
    ----------
    Ns : int
        Number of samples to return (after burn-in).

    Returns
    -------
    samples : ndarray shape (Ns,)
        Correlated samples distributed as ρ(x).

    Comparison function: f(x) = C (constant) with C ≥ max_x ρ(x).
    For the HO ground state, ρ(x)=exp(-x^2) has max 1 at x=0, so C=1.

    """
    C = 1.0 # upper bound on ρ(x) in [x_min, x_max]
    samples = []

    while len(samples) < Ns:
        # propose a batch of trial points uniformly in [x_min, x_max]
        x_trial = np.random.uniform(x_min, x_max, size=Ns)
        # uniform random numbers for acceptance test in [0, C]
        u = np.random.uniform(0.0, C, size=Ns)

        # accept where u < ρ(x)/C  (here C=1, so just u < ρ)
        accept_mask = u < rho_unnormalised(x_trial)
        accepted = x_trial[accept_mask]

        samples.extend(accepted.tolist())

    # return exactly Ns samples
    return np.array(samples[:Ns])

def metropolis_sample_rho(Ns, x0=0.0, delta=1.0, burn_in=0):
    """
    Metropolis algorithm to sample from ρ(x) ∝ |ψ(x)|^2.

    Parameters
    ----------
    Ns : int
        Number of samples to return (after burn-in).
    x0 : float
        Initial position of the Markov chain.
    delta : float
        Proposal step size: x' = x + U(-delta, delta).
    burn_in : int
        Number of initial steps to discard.

    Returns
    -------
    samples : ndarray shape (Ns,)
        Correlated samples distributed as ρ(x).
    """
    samples = np.zeros(Ns)
    x = x0

    # total steps = burn_in + Ns
    for i in range(Ns + burn_in):
        # propose a move: symmetric proposal distribution
        # x_prop = x + np.random.uniform(-delta, delta) # unifrom move
        x_prop = x + np.random.normal(0, delta) # gaussian move
        
        # Metropolis acceptance probability
        w_current = rho_unnormalised(x)
        w_prop    = rho_unnormalised(x_prop)
        A = min(1.0, w_prop / w_current)

        # accept or reject
        if np.random.rand() < A:
            x = x_prop

        # after burn-in, store samples
        if i >= burn_in:
            samples[i - burn_in] = x

    return samples


def rho_unnormalised_3d(R,theta):
    r = np.linalg.norm(R)
    return np.exp(-2 * theta * r)   # or whatever θ you’re using

def metropolis_sample_rho_3d(Ns, theta, R0=None, delta=1.05, burn_in=2000):
    """
    Metropolis algorithm to sample from ρ(R) ∝ |ψ(R)|^2 in 3D.

    Parameters
    ----------
    Ns : int
        Number of samples to return (after burn-in).
    R0 : array-like shape (3,), optional
        Initial position of the Markov chain. If None -> (0,0,0).
    delta : float
        Proposal step size: R' = R + N(0, delta^2 I_3).
    burn_in : int
        Number of initial steps to discard.

    Returns
    -------
    samples : ndarray shape (Ns, 3)
        Correlated samples distributed as ρ(R).
        Each row is a 3D coordinate [x, y, z].
    """
    if R0 is None:
        R = np.zeros(3, dtype=float)
    else:
        R = np.array(R0, dtype=float)

    samples = np.zeros((Ns, 3), dtype=float)

    # total steps = burn_in + Ns
    for i in range(Ns + burn_in):
        # propose a move: 3D Gaussian step
        R_prop = R + np.random.normal(0.0, delta, size=3)

        # Metropolis acceptance probability (unnormalised pdf)
        w_current = rho_unnormalised_3d(R, theta)
        w_prop    = rho_unnormalised_3d(R_prop, theta)
        A = min(1.0, w_prop / w_current)

        # accept or reject
        if np.random.rand() < A:
            R = R_prop

        # after burn-in, store samples
        if i >= burn_in:
            samples[i - burn_in] = R

    return samples

def psi_H2(R, theta, q1, q2):
    """
    H2 trial wavefunction ψ(r1,r2;θ).

    Parameters
    ----------
    R : array-like, shape (6,)
        [x1, y1, z1, x2, y2, z2]
    theta : array-like, length 3
        (theta1, theta2, theta3)
    q1, q2 : array-like, shape (3,)
        Positions of nuclei 1 and 2.

    Returns
    -------
    float
        ψ_H2(R; θ)
    """
    R = np.asarray(R, dtype=float)
    r1 = R[:3]
    r2 = R[3:]

    theta1, theta2, theta3 = theta

    # distances to nuclei
    r1q1 = np.linalg.norm(r1 - q1)
    r1q2 = np.linalg.norm(r1 - q2)
    r2q1 = np.linalg.norm(r2 - q1)
    r2q2 = np.linalg.norm(r2 - q2)

    # electron–electron distance
    r12 = np.linalg.norm(r1 - r2)

    # symmetric part
    sym_part = np.exp(-theta1 * (r1q1 + r2q2)) + np.exp(-theta1 * (r1q2 + r2q1))

    # Jastrow-type correlation factor
    corr = np.exp(-theta2 / (1.0 + theta3 * r12))

    return sym_part * corr

def rho_unnormalised_H2(R, theta, q1, q2):
    """Unnormalised PDF ρ(R) ∝ |ψ_H2(R;θ)|² for 6D configuration R."""
    psi_val = psi_H2(R, theta, q1, q2)
    return psi_val * psi_val   # square modulus

def metropolis_sample_H2(Ns, theta, q1, q2, delta=1.0, burn_in=200):
    """
    Metropolis sampling for H2 in 6D:
        R = [x1,y1,z1,x2,y2,z2]

    Parameters
    ----------
    Ns : int
        Number of returned samples.
    theta : array-like length 3
        (theta1, theta2, theta3)
    q1, q2 : array-like shape (3,)
        Nuclear positions.
    R0 : array-like shape (6,), optional
        Initial position. If None → zeros.
    delta : float
        Gaussian proposal step size.
    burn_in : int
        Discarded initial steps.

    Returns
    -------
    samples : ndarray (Ns,6)
    """
    R = np.concatenate([q1, q2], axis=0)

    samples = np.zeros((Ns, 6), dtype=float)

    for i in range(Ns + burn_in):
        # propose a 6D Gaussian move
        R_prop = R + delta * np.random.randn(6)

        # compute unnormalised weight
        w_current = rho_unnormalised_H2(R, theta, q1, q2)
        w_prop    = rho_unnormalised_H2(R_prop, theta, q1, q2)

        # acceptance probability
        A = min(1.0, w_prop / w_current)

        # accept or reject
        if np.random.rand() < A:
            R = R_prop

        # save after burn-in
        if i >= burn_in:
            samples[i - burn_in] = R

    return samples

def rho_unnormalised_H2_batch(R_samples, theta, q1, q2, eps=1e-3):
    """
    Unnormalised PDF ρ(R) ∝ |ψ_H2(R;θ)|² for a batch of configurations.

    Parameters
    ----------
    R_samples : array-like, shape (Ns, 6)
        Each row is [x1, y1, z1, x2, y2, z2].
    theta : array-like, length 3
        (theta1, theta2, theta3)
    q1, q2 : array-like, shape (3,)
        Nuclear positions.
    eps : float
        Small cutoff passed to psi_H2 to avoid singularities.

    Returns
    -------
    rho_vals : ndarray, shape (Ns,)
        Unnormalised densities ρ(R_i; θ) for each sample.
    """
    R_samples = np.asarray(R_samples, dtype=float)
    Ns = R_samples.shape[0]
    rho_vals = np.empty(Ns, dtype=float)

    for i in range(Ns):
        psi_val = psi_H2(R_samples[i], theta, q1, q2)
        rho_vals[i] = psi_val * psi_val

    return rho_vals


# for 1e6 sampling, use multi core processing to speed up
from multiprocessing import Pool, cpu_count

def metropolis_sample_H2_single_chain(Ns, theta, q1, q2, delta=1.0, burn_in=200, seed=1234):
    np.random.seed(seed)
    R = np.concatenate([q1, q2], axis=0)
    samples = np.zeros((Ns, 6), dtype=float)

    for i in range(Ns + burn_in):
        R_prop = R + delta * np.random.randn(6)
        w_current = rho_unnormalised_H2(R, theta, q1, q2)
        w_prop    = rho_unnormalised_H2(R_prop, theta, q1, q2)
        A = min(1.0, w_prop / w_current)
        if np.random.rand() < A:
            R = R_prop
        if i >= burn_in:
            samples[i - burn_in] = R

    return samples

def metropolis_sample_H2_multi(Ns_total, theta, q1, q2, delta=1.0, burn_in=200, n_chains=None, base_seed=1234):
    if n_chains is None:
        n_chains = cpu_count()

    Ns_per_chain = Ns_total // n_chains
    remainder = Ns_total % n_chains

    args_list = []
    for k in range(n_chains):
        Ns_chain = Ns_per_chain + (1 if k < remainder else 0)
        seed = base_seed + 1000 * k
        args_list.append((Ns_chain, theta, q1, q2, delta, burn_in, seed))

    with Pool(processes=n_chains) as pool:
        results = pool.starmap(metropolis_sample_H2_single_chain, args_list)

    samples = np.vstack(results)
    return samples[:Ns_total]