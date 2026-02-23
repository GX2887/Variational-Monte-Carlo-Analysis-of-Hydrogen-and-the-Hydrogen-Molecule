import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Define 1D Gaussian probability density
# ----------------------------------------

def rho_gaussian_1d(x, alpha):
    """Unnormalised 1D Gaussian PDF: exp(-alpha * x^2)."""
    return np.exp(-alpha * x**2)

# ----------------------------------------
# 2. Metropolis sampler (1D)
# ----------------------------------------

def metropolis_sample_gaussian_1d(Ns, alpha, x0=0.0, delta=1.0, burn_in=2000):
    """
    Metropolis sampler for 1D Gaussian PDF: ρ(x) ∝ exp(-α x^2).

    Parameters
    ----------
    Ns : int
        Number of samples AFTER burn-in.
    alpha : float
        Gaussian parameter.
    x0 : float
        Initial position.
    delta : float
        Proposal step size (Gaussian).
    burn_in : int
        Number of burn-in steps.

    Returns
    -------
    samples : ndarray, shape (Ns,)
        Generated samples.
    acc_rate : float
        Acceptance rate.
    """
    x = float(x0)
    samples = np.zeros(Ns)
    accepted = 0
    total = 0

    for i in range(Ns + burn_in):

        # Propose Gaussian step
        x_prop = x + np.random.normal(0, delta)

        # Unnormalised PDF ratio
        w_current = rho_gaussian_1d(x, alpha)
        w_prop    = rho_gaussian_1d(x_prop, alpha)

        A = min(1.0, w_prop / w_current)

        if np.random.rand() < A:
            x = x_prop
            accepted += 1

        total += 1

        # Store after burn-in
        if i >= burn_in:
            samples[i - burn_in] = x

    acc_rate = accepted / total
    return samples, acc_rate

# ----------------------------------------
# 3. Exact 1D Gaussian PDF (normalised)
# ----------------------------------------

def gaussian_pdf_1d(x, alpha):
    """
    Normalised 1D Gaussian PDF:
    p(x) = sqrt(alpha/pi) * exp(-alpha x^2)
    """
    norm = np.sqrt(alpha / np.pi)
    return norm * np.exp(-alpha * x**2)

# ----------------------------------------
# 4. Run the sampler + compare with exact
# ----------------------------------------

def compare_metropolis_gaussian_1d(alpha=1.0, delta=0.5, Ns=50000, bins=100):
    """
    Run 1D Metropolis sampler and compare histogram to exact Gaussian.
    """
    samples, acc_rate = metropolis_sample_gaussian_1d(
        Ns=Ns, alpha=alpha, delta=delta, burn_in=2000
    )

    print(f"1D Gaussian: acceptance rate = {acc_rate:.4f}")

    # Histogram of sampled x
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Exact Gaussian PDF over a suitable range
    x_min, x_max = np.min(samples), np.max(samples)
    x_plot = np.linspace(x_min, x_max, 400)
    p_exact = gaussian_pdf_1d(x_plot, alpha)

    # Plot comparison
    plt.figure(figsize=(7, 3))

    # Histogram for Metropolis samples
    plt.hist(
        samples,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Metropolis samples (histogram)",
    )

    # Exact curve
    plt.plot(x_plot, p_exact, lw=2, label="Exact 1D Gaussian PDF")

    plt.xlabel("x",fontsize=13)
    plt.ylabel("Probability density",fontsize=13)
    plt.xlim(-4,4)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    # plt.title("1D Gaussian Sampling: Exact vs. Metropolis Histogram")
    plt.legend(fontsize=11, loc = 'upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
    r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_plot\ms_1d_verify_2.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

    return samples, centers, hist, acc_rate


# ----------------------------------------
# Example usage
# ----------------------------------------

if __name__ == "__main__":
    samples_1d, centers_1d, hist_1d, acc_rate_1d = compare_metropolis_gaussian_1d(
        alpha=1.0, delta=1.0, Ns=2000000
    )
