import numpy as np
import s2_random_sampling as rs
import s2_MS_accp_rate_1d as ms


def neff_ratio_for_delta_3d(delta, theta,
                            Ns=50000, burn_in=5000,
                            n_repeats=3, R0=None):
    """
    Estimate Neff/N for a given proposal step size delta in 3D.
    Uses the radial distance r = ||R|| as a 1D time series.
    Averages over several independent runs to reduce noise.
    """
    if delta <= 0:
        return 0.0, 0.0  # ratio, std

    ratios = []
    for _ in range(n_repeats):
        samples = rs.metropolis_sample_rho_3d(Ns, theta, R0=R0,
                                           delta=delta, burn_in=burn_in)
        r = np.linalg.norm(samples, axis=1)  # 1D series
        Neff, tau_int = ms.effective_sample_size(r)
        ratios.append(Neff / Ns)

    ratios = np.array(ratios)
    return float(ratios.mean())

def neff_ratio_for_delta_3d_s(delta, theta,
                            Ns=50000, burn_in=5000,
                            n_repeats=3, R0=None):
    """
    Estimate Neff/N for a given proposal step size delta in 3D.
    Uses the radial distance r = ||R|| as a 1D time series.
    Averages over several independent runs to reduce noise.
    """
    if delta <= 0:
        return 0.0, 0.0, np.array([])  # ratio_mean, ratio_std, ratios

    ratios = []
    for _ in range(n_repeats):
        samples = rs.metropolis_sample_rho_3d(Ns, theta, R0=R0,
                                              delta=delta, burn_in=burn_in)
        r = np.linalg.norm(samples, axis=1)  # 1D series
        Neff, tau_int = ms.effective_sample_size(r)
        ratios.append(Neff / Ns)

    ratios = np.array(ratios, dtype=float)
    ratio_mean = float(np.mean(ratios))
    ratio_std  = float(np.std(ratios, ddof=1)) if n_repeats > 1 else 0.0

    return ratio_mean, ratio_std, ratios
# ---------- main search: find optimal delta ----------

def optimise_delta_3d_gradient(
    theta,
    delta_init=1.0,
    Ns=50000,
    burn_in=5000,
    n_repeats=3,
    lr=0.5,
    max_iter=15,
    eps=0.1,
    delta_min=1e-3,
    delta_max=5.0,
    grad_tol=5e-4,
    verbose=True,
    # NEW: parameters for std(delta) estimation
    n_std_samples=20,
    std_perturb_scale=0.05,
    ratio_threshold=0.8,
):
    """
    Use gradient ascent in log(delta) to find the delta that maximizes Neff/N,
    and estimate the statistical uncertainty (std) of the optimal delta.

    Parameters
    ----------
    theta : float
        Wavefunction parameter in rho_unnormalised_3d.
    delta_init : float
        Initial guess for proposal step size.
    Ns, burn_in, n_repeats : int
        MCMC and averaging parameters.
    lr : float
        Learning rate for gradient ascent in log(delta).
    max_iter : int
        Maximum number of gradient iterations.
    eps : float
        Finite-difference step in log(delta) for gradient estimate.
    delta_min, delta_max : float
        Bounds on delta.
    grad_tol : float
        Stop if |grad| < grad_tol.
    verbose : bool
        Print progress if True.
    n_std_samples : int
        Number of perturbation samples used to estimate std(delta).
    std_perturb_scale : float
        Standard deviation of Gaussian perturbations in log(delta).
    ratio_threshold : float
        Keep only perturbations with Neff/N >= ratio_threshold * best_ratio.

    Returns
    -------
    best_delta : float
        Delta with largest Neff/N encountered.
    delta_std : float
        Estimated standard deviation of the optimal delta.
    history : list of dict
        Each entry: {'iter', 'delta', 'ratio', 'grad'}.
    """

    # work in theta_d = log(delta) to keep delta > 0
    theta_d = np.log(delta_init)
    history = []

    # evaluate initial point
    delta = np.clip(np.exp(theta_d), delta_min, delta_max)
    f_curr = neff_ratio_for_delta_3d(
        delta, theta, Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
    )
    best_delta = delta
    best_ratio = f_curr

    if verbose:
        print(f"init: delta = {delta:.5f}, Neff/N ≈ {f_curr:.5f}")

    # ---------- gradient ascent loop ----------
    for it in range(1, max_iter + 1):

        # central finite-difference gradient in log(delta) space
        theta_plus  = theta_d + eps
        theta_minus = theta_d - eps

        delta_plus  = np.clip(np.exp(theta_plus),  delta_min, delta_max)
        delta_minus = np.clip(np.exp(theta_minus), delta_min, delta_max)

        f_plus = neff_ratio_for_delta_3d(
            delta_plus, theta, Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
        )
        f_minus = neff_ratio_for_delta_3d(
            delta_minus, theta, Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
        )

        grad = (f_plus - f_minus) / (2.0 * eps)   # d f / d log(delta)

        # gradient *ascent* step (we want to maximize Neff/N)
        theta_d = theta_d + lr * grad
        delta   = np.clip(np.exp(theta_d), delta_min, delta_max)

        f_curr = neff_ratio_for_delta_3d(
            delta, theta, Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
        )

        if f_curr > best_ratio:
            best_ratio = f_curr
            best_delta = delta

        history.append({
            "iter": it,
            "delta": delta,
            "ratio": f_curr,
            "grad": grad,
        })

        if verbose:
            print(f"iter {it:2d}: delta = {delta:.5f}, "
                  f"Neff/N ≈ {f_curr:.5f}, grad = {grad:.5f}")

        # simple stopping condition
        if abs(grad) < grad_tol:
            if verbose:
                print("Gradient small; stopping.")
            break

    # ---------- estimate std of optimal delta ----------
    theta_best = np.log(best_delta)
    delta_samples = []

    for _ in range(n_std_samples):
        # perturb around optimal log(delta)
        theta_pert = theta_best + std_perturb_scale * np.random.randn()
        delta_pert = np.clip(np.exp(theta_pert), delta_min, delta_max)

        r_pert = neff_ratio_for_delta_3d(
            delta_pert, theta, Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
        )

        # keep only near-optimal deltas
        if r_pert >= ratio_threshold * best_ratio:
            delta_samples.append(delta_pert)

    if len(delta_samples) > 1:
        delta_std = np.std(delta_samples, ddof=1)
    else:
        delta_std = 0.0

    if verbose:
        print(
            f"\nBest delta ≈ {best_delta:.5f} ± {delta_std:.5f} "
            f"with Neff/N ≈ {best_ratio:.5f}"
        )

    return best_delta, delta_std, history


# theta = 1
# best_delta, std, history = optimise_delta_3d_gradient(
#     theta,
#     delta_init=1.05,
#     Ns=100000,
#     burn_in=10,
#     n_repeats=4,
#     lr=0.1,
#     grad_tol=5e-4,
#     max_iter=200,
#     eps=0.2
# )

# test the acceptance rate

def metropolis_acceptance_rate_3d(Ns, theta, delta=1.00853,
                                  burn_in=5000, R0=None,
                                  n_repeats=5):
    """
    Compute the Metropolis acceptance rate for a 3D sampler,
    averaged over several independent runs.
    """
    acc_rates = []

    for j in range(n_repeats):
        # reset chain for each repeat
        if R0 is None:
            R = np.zeros(3)
        else:
            R = np.array(R0, dtype=float)

        accepted = 0
        total = 0

        # we don't really need to keep samples here, but keep the structure
        samples = np.zeros((Ns, 3))

        for i in range(Ns + burn_in):
            # propose a Gaussian 3D move
            R_prop = R + np.random.normal(0.0, delta, size=3)

            # unnormalised Metropolis weights
            w_current = rs.rho_unnormalised_3d(R, theta)
            w_prop    = rs.rho_unnormalised_3d(R_prop, theta)

            A = min(1.0, w_prop / w_current)

            # accept / reject
            if np.random.rand() < A:
                R = R_prop
                accepted += 1

            total += 1

            # store after burn-in (optional)
            if i >= burn_in and (i - burn_in) < Ns:
                samples[i - burn_in] = R

        acc_rate = accepted / total
        acc_rates.append(acc_rate)

    acc_rates = np.array(acc_rates)
    acc_mean = float(np.mean(acc_rates))
    acc_std  = float(np.std(acc_rates, ddof=1)) if n_repeats > 1 else 0.0

    return acc_mean, acc_std, acc_rates


def summarize_delta_3d(delta_best, theta,
                       Ns=100000, burn_in=2000,
                       n_repeats_acc=5, n_repeats_neff=3,
                       R0=None):
    """
    Run both acceptance-rate and Neff/N diagnostics and print a summary.
    """
    acc_mean, acc_std, acc_rates = metropolis_acceptance_rate_3d(
        Ns, theta, delta=delta_best, burn_in=burn_in,
        R0=R0, n_repeats=n_repeats_acc
    )

    neff_mean, neff_std, neff_ratios = neff_ratio_for_delta_3d_s(
        delta_best, theta, Ns=Ns, burn_in=burn_in,
        n_repeats=n_repeats_neff, R0=R0
    )

    print("\nSummary over runs:")
    print(f"delta_best      = {delta_best:.5f}")
    print(f"mean acc_rate   = {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"mean Neff/N     = {neff_mean:.4f} ± {neff_std:.4f}")

    return {
        "delta_best": delta_best,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_rates": acc_rates,
        "neff_mean": neff_mean,
        "neff_std": neff_std,
        "neff_ratios": neff_ratios,
    }


# -------------------------------
# Example usage:
# -------------------------------

theta = 1.0                 
delta_best = 1.05
summary = summarize_delta_3d(delta_best, theta)