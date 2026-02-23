import numpy as np
import matplotlib.pyplot as plt
import s1_differentiation as drt



# ----------------------------
# psi(R) = exp(r) and analytic Laplacian
# ----------------------------

def psi_exp_r(R, theta):
    """
    Test wavefunction psi(R) = exp(r), r = |R|.
    theta is unused but kept for interface compatibility.
    """
    r = np.linalg.norm(R)
    return np.exp(r)

def laplacian_exp_r_analytic(R):
    """
    Analytic 3D Laplacian of psi(R) = exp(r), r = |R|.
    For a purely radial f(r), in 3D:
        ∇^2 f = f''(r) + (2/r) f'(r)
    For f(r) = e^r: f' = e^r, f'' = e^r ⇒ ∇^2 e^r = e^r (1 + 2/r).
    """
    r = np.linalg.norm(R)
    if r == 0.0:
        # singular at r=0; you can skip or define a limit if needed
        return np.nan
    return np.exp(r) * (1.0 + 2.0 / r)


def psi_r2_exp_r(R, theta):
    """
    Test wavefunction psi(R) = r^2 e^{r}, r = |R|.
    theta is unused but kept for interface compatibility.
    """
    r = np.linalg.norm(R)
    return (r**2) * np.exp(r)

def laplacian_r2_exp_r_analytic(R):
    """
    Analytic 3D Laplacian of psi(R) = r^2 e^{r}, r = |R|.
    For f(r) = r^2 e^r in 3D:
        ∇^2 f = e^r (r^2 + 6r + 6).
    """
    r = np.linalg.norm(R)
    return np.exp(r) * (r**2 + 6.0*r + 6.0)

# ----------------------------
# Verification + heatmap
# ----------------------------

def verify_laplacian_exp_r_2d_slice(theta=0.0,
                                    h=3.078e-3,
                                    L=0.5,
                                    n_points=101,
                                    r_min_factor=5.0):
    """
    Compare numerical vs analytic Laplacian of exp(r) on the z=0 plane
    and plot an error heatmap in (x, y).

    Parameters
    ----------
    theta : float
        Dummy parameter for psi_func (not used for exp(r)).
    h : float
        Step size for finite-difference Laplacian.
    L : float
        Half-size of the square domain [-L, L] in x and y.
    n_points : int
        Number of grid points in each direction.
    r_min_factor : float
        Exclude points with r < r_min_factor * h to avoid the r=0 singularity.
    """
    xs = np.linspace(-L, L, n_points)
    ys = np.linspace(-L, L, n_points)

    err = np.zeros((n_points, n_points), dtype=float)

    r_min = r_min_factor * h  # avoid region too close to r=0

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            R = np.array([x, y, 0.0], dtype=float)
            r = np.linalg.norm(R)

            if r < r_min:
                err[iy, ix] = np.nan  # mark center as invalid
                continue

            lap_num = drt.laplacian_3d_fd(psi_exp_r, R, theta, h=h)
            lap_ana = laplacian_exp_r_analytic(R)
            err[iy, ix] = lap_num - lap_ana  # signed error; use abs(...) if you like

    # Plot heatmap of error in x-y plane
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        err,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin='lower',
        aspect='equal'
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r'$\nabla^2_{\mathrm{num}} \psi - \nabla^2_{\mathrm{exact}} \psi$')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Error in $\nabla^2 \mathrm{e}^{r}$ (z=0 slice)')
    plt.tight_layout()
    plt.show()

    return xs, ys, err

def verify_laplacian_r2_exp_r_2d_slice(theta=0.0,
                                       h=3.078e-3,
                                       L=0.5,
                                       n_points=1001):
    """
    Compare numerical vs analytic Laplacian of r^2 e^{r} on the z=0 plane
    and plot an error heatmap in (x, y).

    Parameters
    ----------
    theta : float
        Dummy parameter for psi_func (not used).
    h : float
        Step size for finite-difference Laplacian.
    L : float
        Half-size of the square domain [-L, L] in x and y.
    n_points : int
        Number of grid points in each direction.
    """
    xs = np.linspace(L/2, L, n_points)
    ys = np.linspace(L/2, L, n_points)

    err = np.zeros((n_points, n_points), dtype=float)

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            R = np.array([x, y, 0.0], dtype=float)

            lap_num = drt.laplacian_3d_fd(psi_r2_exp_r, R, theta, h=h)
            lap_ana = laplacian_r2_exp_r_analytic(R)
            err[iy, ix] = lap_num - lap_ana  # signed error

    # Plot heatmap of error in x-y plane
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        err,
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        origin='lower',
        aspect='equal'
    )
    cbar = plt.colorbar(im)
    cbar.set_label('Deviation')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Error in $\nabla^2 (r^2 e^{r})$ (z=0 slice)')
    plt.show()

    return xs, ys, err

# xs, ys, err = verify_laplacian_exp_r_2d_slice()
# xs, ys, err = verify_laplacian_r2_exp_r_2d_slice()

def compare_laplacian_at_point():
    """
    Compare analytic vs numerical Laplacian at R = (1,1,1)
    for the two test wavefunctions:
      - psi = exp(r)
      - psi = r^2 * exp(r)
    """

    R = np.array([1.0, 1.0, 1.0])
    theta = None   # unused, kept for interface compatibility

    print("Comparing at R = (1,1,1)\n")

    # ----------------------------------------------------------
    # 1. psi(R) = exp(r)
    # ----------------------------------------------------------
    lap_num_exp = drt.laplacian_3d_fd(psi_exp_r, R, theta)
    lap_ana_exp = laplacian_exp_r_analytic(R)

    print("=== psi = exp(r) ===")
    print(f"Numerical Laplacian: {lap_num_exp:.8f}")
    print(f"Analytic  Laplacian: {lap_ana_exp:.8f}")
    print(f"Relative error      : {abs(lap_num_exp - lap_ana_exp) / abs(lap_ana_exp):.3e}\n")

    # ----------------------------------------------------------
    # 2. psi(R) = r^2 exp(r)
    # ----------------------------------------------------------
    lap_num_r2exp = drt.laplacian_3d_fd(psi_r2_exp_r, R, theta)
    lap_ana_r2exp = laplacian_r2_exp_r_analytic(R)

    print("=== psi = r^2 * exp(r) ===")
    print(f"Numerical Laplacian: {lap_num_r2exp:.8f}")
    print(f"Analytic  Laplacian: {lap_ana_r2exp:.8f}")
    print(f"Relative error      : {abs(lap_num_r2exp - lap_ana_r2exp) / abs(lap_ana_r2exp):.3e}\n")

    return {
        "exp_r": (lap_num_exp, lap_ana_exp),
        "r2_exp_r": (lap_num_r2exp, lap_ana_r2exp),
    }

a = compare_laplacian_at_point()