import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Target distribution and Metropolis sampler
# ----------------------------------------

def log_rho(x):
    """
    Log of target PDF rho(x) ∝ exp(-x) on [0, ∞).
    Note: rho(x) = exp(-x) is already normalized for x >= 0.
    """
    if x < 0.0:
        return -np.inf
    return -x


def metropolis_exp_1d(Ns, delta=1.0, burn_in=1000, x0=0.0, seed=None):
    """
    Metropolis sampler for p(x) ∝ exp(-x) on [0, ∞).
    
    Parameters
    ----------
    Ns : int
        Number of samples (after burn-in).
    delta : float
        Proposal step size (Gaussian).
    burn_in : int
        Number of burn-in steps (discarded).
    x0 : float
        Initial position.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    samples : ndarray, shape (Ns,)
        Samples approximately distributed as exp(-x) on [0, ∞).
    """
    rng = np.random.default_rng(seed)
    x = float(x0)
    samples = np.zeros(Ns)

    # Precompute current log density
    logw_current = log_rho(x)

    i_sample = 0
    n_steps = Ns + burn_in

    for step in range(n_steps):
        # Gaussian proposal
        x_prop = x + delta * rng.normal()

        # Enforce domain [0, ∞): if x_prop < 0, auto-reject
        if x_prop < 0:
            accept = False
        else:
            logw_prop = log_rho(x_prop)
            # Metropolis acceptance probability
            logA = logw_prop - logw_current
            if np.log(rng.random()) < logA:
                accept = True
            else:
                accept = False

        if accept:
            x = x_prop
            logw_current = logw_prop

        # Store after burn-in
        if step >= burn_in:
            samples[i_sample] = x
            i_sample += 1

    return samples


# ----------------------------------------
# Monte Carlo integration setup
# ----------------------------------------

def mc_integral_estimate(Ns, n_repeats=10, delta=1.0, burn_in=1000, seed_base=1234):
    """
    Estimate integral I = ∫_0^∞ x^2 e^{-x} dx using Metropolis-importance sampling.

    Since p(x) = e^{-x} is normalized, we have:
        I = E_p[x^2].
    So we just sample x ~ p(x) and average x^2.

    Parameters
    ----------
    Ns : int
        Number of samples per run.
    n_repeats : int
        Number of independent runs (for error bar).
    delta, burn_in : Metropolis parameters.
    seed_base : int
        Base seed for reproducibility.

    Returns
    -------
    I_mean : float
        Mean of the integral estimate across repeats.
    I_stderr : float
        Standard error of the mean across repeats.
    """
    estimates = []
    for k in range(n_repeats):
        samples = metropolis_exp_1d(
            Ns=Ns,
            delta=delta,
            burn_in=burn_in,
            x0=0.0,
            seed=seed_base + k
        )
        # Importance estimator: I ~ <x^2>_p
        I_hat = np.mean(samples**2)
        estimates.append(I_hat)

    estimates = np.array(estimates)
    I_mean = estimates.mean()
    # Standard error of the mean (CLT)
    I_stderr = estimates.std(ddof=1) / np.sqrt(n_repeats)
    return I_mean, I_stderr


# ----------------------------------------
# Main experiment: Ns from 1e2 to 1e6
# ----------------------------------------

if __name__ == "__main__":
    # Exact value of the test integral ∫_0^∞ x^2 e^{-x} dx = 2
    I_exact = 2.0

    # Sampling sizes (log-spaced from 1e2 to 1e6)
    Ns_values = np.logspace(2, 6.5, num=10, dtype=int)  # e.g. 1e2, 2e2, ..., 1e6

    I_means = []
    I_errors = []

    for Ns in Ns_values:
        print(f"Running Ns = {Ns}")
        I_mean, I_stderr = mc_integral_estimate(
            Ns,
            n_repeats=10,
            delta=1.0,
            burn_in=1000
        )
        I_means.append(I_mean)
        I_errors.append(I_stderr)

    I_means = np.array(I_means)
    I_errors = np.array(I_errors)

    # ----------------------------------------
    # Plot: Integral estimate vs N_s with error bars
    # ----------------------------------------
    # plt.figure(figsize=(7, 5))
    # plt.errorbar(
    #     Ns_values,
    #     I_means,
    #     yerr=I_errors,
    #     fmt='o-',
    #     capsize=4,
    #     label='MC estimate'
    # )
    # plt.axhline(I_exact, linestyle='--', label='Exact value = 2')
    # plt.xscale('log')
    # plt.xlabel(r'Number of samples $N_s$')
    # plt.ylabel(r'Integral estimate $I$')
    # plt.title(r'1D Metropolis Monte Carlo: $\int_0^\infty x^2 e^{-x} dx$')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ----------------------------------------
    # (Optional) Plot: absolute error vs N_s on log-log
    # to see ~ 1/sqrt(N_s) scaling from CLT
    # ----------------------------------------
    abs_error = np.abs(I_means - I_exact)

    plt.figure(figsize=(7, 3))

    plt.errorbar(
        Ns_values,
        abs_error,
        yerr=I_errors,
        fmt='o-',
        capsize=4,
        label='|error|'
    )

    # Reference line ~ 1/sqrt(N_s)
    c_ref = abs_error[0] * np.sqrt(Ns_values[0])
    ref_line = c_ref / np.sqrt(Ns_values)
    plt.plot(Ns_values, ref_line, '--', label=r'$\propto 1/\sqrt{N_s}$ (CLT)')

    plt.xscale('log')
    plt.yscale('log')

    # ---- font size control ----
    plt.xlabel(r'Number of samples $N_s$', fontsize=12)
    plt.ylabel(r'|Estimate - Exact|', fontsize=12)

    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    plt.legend(fontsize=11)
    # ---------------------------

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_plot\1d_Montecarlo_convergence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
