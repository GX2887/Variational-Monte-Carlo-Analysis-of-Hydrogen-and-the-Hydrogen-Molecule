import numpy as np
import s2_random_sampling as rs
import s1_differentiation as drt
from s1_differentiation import psi_0
import matplotlib.pyplot as plt

def local_energy_num(x):
    # analytic psi and d2psi for HO ground state)
    d2psi = drt.d2_fd_4th_order(psi_0,x)    # second derivative of psi
    return -0.5 * (d2psi / psi_0(x)) + 0.5 * x**2

def mean_local_energy_num():
    Ns = 200 # define Sampling point

    # use random sampling method to generate x_i
    x_samples = rs.metropolis_sample_rho(Ns)

    # calculate the local energy
    El_values = local_energy_num(x_samples)

    # Monte-Carlo estimator
    E_estimate = np.mean(El_values)

    return E_estimate

def test_burn_in_effect(
        Ns=200,
        N_estimates=100,
        burn_in_values=np.arange(0, 1001, 50),
        seed=123,
        plot=True
    ):
    """
    Investigate how burn-in length affects the Monte-Carlo estimate of <E_l>.

    Parameters
    ----------
    Ns : int
        Number of MCMC samples per estimate.
    N_estimates : int
        Number of independent energy estimates per burn-in length.
    burn_in_values : array-like
        List of burn-in steps to test.
    seed : int
        RNG seed for reproducibility.
    plot : bool
        Whether to produce a matplotlib plot.

    Returns
    -------
    burn_in_values : array
    means : array
        Mean E_estimate for each burn-in value.
    stds : array
        Standard deviation of the estimates for each burn-in value.
    """

    np.random.seed(seed)

    means = []
    stds = []

    # --- helper inside function ---
    def E_estimate_for_burnin(Ns, burn_in):
        x_samples = rs.metropolis_sample_rho(Ns, burn_in=burn_in)
        El_values = local_energy_num(x_samples)
        return np.mean(El_values)

    # --- main loop ---
    for burn_in in burn_in_values:
        estimates = []
        for _ in range(N_estimates):
            estimates.append(E_estimate_for_burnin(Ns, burn_in))

        estimates = np.array(estimates)
        means.append(estimates.mean())
        stds.append(estimates.std(ddof=1))

    means = np.array(means)
    stds = np.array(stds)

    # --- plotting ---
    if plot:
        plt.figure(figsize=(7,5))
        plt.errorbar(burn_in_values, means, yerr=stds, fmt='o-')
        plt.axhline(0.5, linestyle='--', label='Exact ground state E=0.5')
        plt.xlabel("Burn-in steps")
        plt.ylabel(r"$\langle E_l \rangle$ estimate")
        plt.title("Effect of burn-in length on energy estimate")
        plt.legend()
        plt.show()

    return burn_in_values, means, stds

def test_Ns_effect(
        Ns_values=None,
        N_estimates=10,
        burn_in=500,
        seed=121,
        plot=True
    ):
    """
    Investigate how the number of samples Ns affects the Monte-Carlo
    estimate of the local-energy expectation value <E_l>.

    Parameters
    ----------
    Ns_values : array-like
        List of Ns values to test. If None, uses log-spaced values.
    N_estimates : int
        Number of independent E_estimates per Ns.
    burn_in : int
        Burn-in steps for Metropolis sampler.
    seed : int
        Random seed.
    plot : bool
        Whether to plot the results.

    Returns
    -------
    Ns_values : array
    means : array
    stds : array
    """

    if Ns_values is None:
        # Default: log-spaced samples from 50 → 20000
        Ns_values = np.unique(
            np.logspace(np.log10(50), np.log10(200000), 15).astype(int)
        )

    np.random.seed(seed)

    means = []
    stds = []

    # --- helper: one E estimate for a given Ns ---
    def one_E_estimate(Ns):
        x_samples = rs.metropolis_sample_rho(Ns, burn_in=burn_in)
        El = local_energy_num(x_samples)
        return np.mean(El)

    # --- main loop ---
    for Ns in Ns_values:
        E_list = []
        for _ in range(N_estimates):
            E_list.append(one_E_estimate(Ns))
        E_arr = np.array(E_list)

        means.append(E_arr.mean())
        stds.append(E_arr.std(ddof=1))

    means = np.array(means)
    stds = np.array(stds)

    # --- plotting ---
    if plot:
        plt.figure(figsize=(10,7))
        plt.errorbar(Ns_values, means, yerr=stds, fmt='o-', capsize=4)
        plt.axhline(0.5, ls='--', label='Exact ground state E = 0.5')
        plt.xscale('log')
        plt.xlabel("Ns (number of Monte-Carlo samples)")
        plt.ylabel(r"$\langle E_l \rangle$ estimate")
        plt.title("Energy estimate convergence vs number of samples")
        plt.legend()
        plt.grid(True)
        # plt.ticklabel_format(useOffset=False)
        plt.show()

    return Ns_values, means, stds

# Ns_values, means, stds = test_Ns_effect(plot=True)

def compare_local_energy(Ns=int(1e6), delta=1.74491, burn_in=200):
    """
    Samples x from ρ(x), computes numerical local energy,
    and plots x vs [E_L(x) - 1/2].
    """
    # Step 1: sample x
    samples = rs.metropolis_sample_rho(Ns, delta=delta, burn_in=burn_in)

    # Step 2: compute local energies
    E_loc = np.array([local_energy_num(x) for x in samples])

    # Step 3: analytical ground-state energy
    E_exact = 0.5
    dE = E_loc - E_exact

    # Step 4: plot
    plt.figure(figsize=(7,5))
    plt.scatter(samples, dE, s=8, alpha=0.5, label=r"$E_L(x)-E_0$")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel(r"$E_L(x)$ - $E_0$")
    plt.title("Local Energy Error for Harmonic Oscillator Ground State")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print average energy as a correctness check
    print(f"Mean sampled local energy = {np.mean(E_loc):.6f} (expected 0.5)")

    return samples, E_loc

samples, E_vals = compare_local_energy(Ns=int(1e6), delta=1.7, burn_in=200)

E_std = np.std(E_vals)/(1e3*np.sqrt(0.2261))

E_mean = np.mean(E_vals)
print(E_mean,E_std)
print(0.5-E_mean)
