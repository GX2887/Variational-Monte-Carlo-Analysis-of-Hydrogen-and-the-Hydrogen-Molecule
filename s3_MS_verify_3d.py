import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Define 3D Gaussian probability density
# ----------------------------------------

def rho_gaussian_3d(R, alpha):
    """Unnormalised 3D Gaussian PDF: exp(-alpha * r^2)."""
    r2 = np.dot(R, R)
    return np.exp(-alpha * r2)

# ----------------------------------------
# 2. Metropolis sampler (3D)
# ----------------------------------------

def metropolis_sample_gaussian_3d(Ns, alpha, R0=None, delta=1, burn_in=2000):
    """
    Metropolis sampler for 3D Gaussian PDF: ρ(R) ∝ exp(-α r^2).

    Returns:
        samples : (Ns, 3)
        acc_rate : acceptance ratio
    """
    if R0 is None:
        R = np.zeros(3)
    else:
        R = np.array(R0, dtype=float)

    samples = np.zeros((Ns, 3))
    accepted = 0
    total = 0

    for i in range(Ns + burn_in):

        # Propose Gaussian step
        R_prop = R + np.random.normal(0, delta, size=3)

        # Unnormalised PDF ratio
        w_current = rho_gaussian_3d(R, alpha)
        w_prop    = rho_gaussian_3d(R_prop, alpha)

        A = min(1.0, w_prop / w_current)

        if np.random.rand() < A:
            R = R_prop
            accepted += 1

        total += 1

        # Store after burn-in
        if i >= burn_in:
            samples[i - burn_in] = R

    acc_rate = accepted / total
    return samples, acc_rate

# ----------------------------------------
# 3. Exact radial distribution of 3D Gaussian
# ----------------------------------------

def gaussian_radial_pdf(r, alpha):
    """
    Radial probability density for 3D Gaussian.
    Normalised analytical expression:
    P(r) = 4π r^2 (α/π)^(3/2) exp(-α r^2)
    """
    norm = (alpha / np.pi)**(3/2)
    return 4 * np.pi * r**2 * norm * np.exp(-alpha * r**2)

# ----------------------------------------
# 4. Run the sampler + compare with exact
# ----------------------------------------

def compare_metropolis_gaussian(alpha=1.0, delta=0.5, Ns=50000, bins=100):
    samples, acc_rate = metropolis_sample_gaussian_3d(
        Ns=Ns, alpha=alpha, delta=delta, burn_in=2000
    )

    print(f"Acceptance rate = {acc_rate:.4f}")

    # compute radii from samples
    r = np.linalg.norm(samples, axis=1)

    # histogram of sampled radii (for returning values if needed)
    hist, edges = np.histogram(r, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # exact Gaussian radial distribution
    r_plot = np.linspace(0, np.max(r), 400)
    p_exact = gaussian_radial_pdf(r_plot, alpha)

    # plot comparison
    plt.figure(figsize=(7, 3))

    # Histogram for Metropolis samples
    plt.hist(
        r,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Metropolis samples (histogram)",
    )

    # Exact curve
    plt.plot(r_plot, p_exact, lw=2, label="Exact Gaussian radial PDF")

    plt.xlabel("r",fontsize=13)
    plt.ylabel("Probability density",fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    # plt.title("3D Gaussian Sampling: Exact vs. Metropolis Histogram")
    plt.legend(fontsize=11, loc = 'upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
    r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_plot\ms_3d_verify_2.png",
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

    return samples, r, centers, hist, acc_rate


# ----------------------------------------
# Example usage
# ----------------------------------------

samples, r, centers, hist, acc_rate = compare_metropolis_gaussian(
    alpha=1.0, delta=1, Ns=5000000
)
