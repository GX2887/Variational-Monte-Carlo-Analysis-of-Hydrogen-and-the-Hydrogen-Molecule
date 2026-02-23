import numpy as np
import s2_random_sampling as rs
# --- assume these already exist in your file ---
# def rho_unnormalised(x): ...
# def metropolis_sample_rho(...): ...
# def effective_sample_size(x, max_lag=None): ...
# -----------------------------------------------

def autocorr_1d(x):
    """
    Normalized autocorrelation function for a 1D array.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)
    var = np.var(x)
    if var == 0:
        return np.array([1.0])
    # FFT-based ACF (fast, but simple)
    fx = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(fx * np.conjugate(fx))[:N].real
    acf /= acf[0]
    return acf

def integrated_autocorr_time(x, max_lag=None):
    """
    Estimate integrated autocorrelation time tau_int.
    """
    acf = autocorr_1d(x)
    if max_lag is None:
        max_lag = len(acf) // 10  # simple window
    max_lag = min(max_lag, len(acf) - 1)

    # standard estimator: 1/2 + sum_{k>=1} rho(k)
    tau_int = 0.5
    for k in range(1, max_lag + 1):
        if acf[k] <= 0:
            break
        tau_int += acf[k]
    return tau_int

def effective_sample_size(x, max_lag=None):
    N = len(x)
    tau_int = integrated_autocorr_time(x, max_lag=max_lag)
    Neff = N / (2.0 * tau_int)
    return Neff, tau_int


def neff_over_n_for_delta(delta, Ns=50000, burn_in=5000, n_repeats=3, x0=0.0):
    """
    Estimate Neff/N for a given proposal step size delta by averaging over
    several independent Metropolis runs (to reduce noise).
    """
    if delta <= 0:
        return 0.0

    ratios = []
    for _ in range(n_repeats):
        samples = rs.metropolis_sample_rho(Ns, x0=x0, delta=delta, burn_in=burn_in)
        Neff, tau_int = effective_sample_size(samples)
        ratios.append(Neff / Ns)
    return np.mean(ratios)

'''
def optimise_delta_gradient_descent(
    delta_init=0.5,
    Ns=50000,
    burn_in=5000,
    n_repeats=3,
    lr=0.5,
    max_iter=200,
    eps=0.1,
    delta_min=1e-3,
    delta_max=10.0,
    verbose=True,
    tol = 1e-3
):
    """
    Gradient-descent search (in log(delta)) for the delta that maximizes Neff/N.

    Returns
    -------
    best_delta : float
        Step size with highest Neff/N encountered.
    history : list of dict
        Each entry contains {'iter', 'delta', 'ratio', 'grad'}.
    """
    # work in theta = log(delta) to enforce delta > 0
    theta = np.log(delta_init)
    history = []

    # evaluate initial point
    delta = np.clip(np.exp(theta), delta_min, delta_max)
    f_curr = neff_over_n_for_delta(delta, Ns=Ns, burn_in=burn_in,
                                   n_repeats=n_repeats)
    best_delta = delta
    best_ratio = f_curr

    if verbose:
        print(f"init: delta = {delta:.5f}, Neff/N ≈ {f_curr:.5f}")

    for it in range(1, max_iter + 1):
        # central finite-difference gradient in theta space
        theta_plus  = theta + eps
        theta_minus = theta - eps

        delta_plus  = np.clip(np.exp(theta_plus),  delta_min, delta_max)
        delta_minus = np.clip(np.exp(theta_minus), delta_min, delta_max)

        f_plus  = neff_over_n_for_delta(delta_plus,  Ns=Ns, burn_in=burn_in,
                                        n_repeats=n_repeats)
        f_minus = neff_over_n_for_delta(delta_minus, Ns=Ns, burn_in=burn_in,
                                        n_repeats=n_repeats)

        # gradient wrt theta (log delta)
        grad = (f_plus - f_minus) / (2.0 * eps)

        # gradient ascent step (maximize f)
        theta = theta + lr * grad
        delta = np.clip(np.exp(theta), delta_min, delta_max)

        f_curr = neff_over_n_for_delta(delta, Ns=Ns, burn_in=burn_in,
                                       n_repeats=n_repeats)
        
        # set stopping condition    
        if np.abs(grad) <tol:
            break
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
        

    if verbose:
        print(f"\nBest delta ≈ {best_delta:.5f} with Neff/N ≈ {best_ratio:.5f}")

    return best_delta, history
''' 

def optimise_delta_gradient_descent(
    delta_init=0.5,
    Ns=50000,
    burn_in=5000,
    n_repeats=3,
    lr=0.5,
    max_iter=200,
    eps=0.1,
    delta_min=1e-3,
    delta_max=10.0,
    verbose=True,
    tol=1e-3,
    n_std_samples=20,          # number of perturbations to estimate std(delta)
    std_perturb_scale=0.05     # width of perturbations in log(delta)
):
    """
    Gradient-descent search (in log(delta)) for the delta that maximizes Neff/N.
    Also estimates the statistical uncertainty (std) of the optimal delta.
    """

    theta = np.log(delta_init)
    history = []

    # evaluate initial
    delta = np.clip(np.exp(theta), delta_min, delta_max)
    f_curr = neff_over_n_for_delta(delta, Ns=Ns, burn_in=burn_in,
                                   n_repeats=n_repeats)
    best_delta = delta
    best_ratio = f_curr

    if verbose:
        print(f"init: delta = {delta:.5f}, Neff/N ≈ {f_curr:.5f}")

    # -------- GRADIENT ASCENT LOOP --------
    for it in range(1, max_iter + 1):

        # finite-difference gradient in theta
        theta_plus  = theta + eps
        theta_minus = theta - eps

        delta_plus  = np.clip(np.exp(theta_plus),  delta_min, delta_max)
        delta_minus = np.clip(np.exp(theta_minus), delta_min, delta_max)

        f_plus  = neff_over_n_for_delta(delta_plus,  Ns=Ns, burn_in=burn_in,
                                        n_repeats=n_repeats)
        f_minus = neff_over_n_for_delta(delta_minus, Ns=Ns, burn_in=burn_in,
                                        n_repeats=n_repeats)

        grad = (f_plus - f_minus) / (2.0 * eps)

        # gradient ascent
        theta = theta + lr * grad
        delta = np.clip(np.exp(theta), delta_min, delta_max)

        f_curr = neff_over_n_for_delta(delta, Ns=Ns, burn_in=burn_in,
                                       n_repeats=n_repeats)

        # stopping condition
        if np.abs(grad) < tol:
            break

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

    # -------- ESTIMATE STD OF OPTIMAL DELTA --------
    theta_best = np.log(best_delta)
    delta_samples = []

    for _ in range(n_std_samples):
        # random perturbation around optimum in log(delta)
        theta_pert = theta_best + std_perturb_scale * np.random.randn()
        delta_pert = np.clip(np.exp(theta_pert), delta_min, delta_max)

        # re-evaluate ratio at perturbed delta
        r_pert = neff_over_n_for_delta(delta_pert, Ns=Ns, burn_in=burn_in,
                                       n_repeats=n_repeats)

        # accept perturbation if ratio is close to max (i.e., plausible optimum)
        if r_pert > 0.8 * best_ratio:
            delta_samples.append(delta_pert)

    # compute std
    if len(delta_samples) > 1:
        delta_std = np.std(delta_samples, ddof=1)
    else:
        delta_std = 0.0

    if verbose:
        print(f"\nBest delta ≈ {best_delta:.5f} ± {delta_std:.5f} "
              f"with Neff/N ≈ {best_ratio:.5f}")

    return best_delta, delta_std, history


# gradient descent to search the optimal delta value
# Best delta ≈ 1.71632 ± 0.07225 with Neff/N ≈ 0.22950

# best_delta, std, hist = optimise_delta_gradient_descent(
#     delta_init=1.7, Ns=100000, burn_in=200, n_repeats=4,lr=0.1
# )

def metropolis_sample_rho_with_stats(Ns, x0=0.0, delta=1.0, burn_in=0):
    """
    Metropolis algorithm to sample from ρ(x) ∝ |ψ(x)|^2 and track acceptance.

    Parameters
    ----------
    Ns : int
        Number of samples to return (after burn-in).
    x0 : float
        Initial position of the Markov chain.
    delta : float
        Proposal step size: x' = x + N(0, delta^2).
    burn_in : int
        Number of initial steps to discard.

    Returns
    -------
    samples : ndarray, shape (Ns,)
        Correlated samples distributed as ρ(x).
    acc_rate : float
        Acceptance rate (accepted moves / total proposals).
    """
    samples = np.zeros(Ns)
    x = x0

    n_accept = 0
    n_total  = 0

    # total steps = burn_in + Ns
    for i in range(Ns + burn_in):
        # propose a move: Gaussian move
        x_prop = x + np.random.normal(0.0, delta)

        # Metropolis acceptance probability
        w_current = rs.rho_unnormalised(x)
        w_prop    = rs.rho_unnormalised(x_prop)
        A = min(1.0, w_prop / w_current)

        # accept or reject
        if np.random.rand() < A:
            x = x_prop
            n_accept += 1
        n_total += 1

        # after burn-in, store samples
        if i >= burn_in:
            samples[i - burn_in] = x

    acc_rate = n_accept / n_total
    return samples, acc_rate

# ---- Verify acceptance rate at your best delta ----

def verify_acceptance_rate_at_best_delta(
    delta_best=1.71632,
    Ns=int(1e6),
    burn_in=2000,
    n_repeats=5,
    x0=0.0
):
    acc_rates = []
    neff_ratios = []

    for r in range(n_repeats):
        samples, acc = metropolis_sample_rho_with_stats(
            Ns=Ns, x0=x0, delta=delta_best, burn_in=burn_in
        )
        Neff, tau_int = effective_sample_size(samples)
        acc_rates.append(acc)
        neff_ratios.append(Neff / Ns)
        print(f"run {r+1}: acc_rate = {acc:.4f}, Neff/N ≈ {Neff/Ns:.4f}, tau_int ≈ {tau_int:.3f}")

    std_acc = np.std(acc_rates)/np.sqrt(n_repeats)
    std_neff = np.std(neff_ratios)/np.sqrt(n_repeats)
    print("\nSummary over runs:")
    print(f"delta_best      = {delta_best:.5f}")
    print(f"mean acc_rate   = {np.mean(acc_rates):.4f} ± {std_acc:.4f}")
    print(f"mean Neff/N     = {np.mean(neff_ratios):.4f} ± {std_neff:.4f}")

verify_acceptance_rate_at_best_delta()