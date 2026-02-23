import os
import time

import numpy as np
from numba import njit  # noqa: F401  # if unused but kept intentionally
from openpyxl import Workbook
from scipy.optimize import curve_fit

from s3_H2_copy_numba_quasi_newton import quasi_newton_min_energy_analytic


def save_r_e_to_excel(filepath, r_values, e_values, e_std_values):
    """
    Save r, E, and E_std into an Excel (.xlsx) file.

    Columns:
        r   |   E(r)   |   E_std(r)
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "H2_EnergyCurve"

    # Header row
    ws.append(["r", "E(r)", "E_std(r)"])

    # Data rows
    for r_val, e_val, e_std in zip(r_values, e_values, e_std_values):
        ws.append([float(r_val), float(e_val), float(e_std)])

    # Ensure directory exists (only if a directory is specified)
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    wb.save(filepath)
    print(f"\nSaved Excel file to:\n{filepath}\n")


def collect_e_vs_r_for_fit(
    r_min=0.5,
    r_max=3.0,
    n_points=25,
    theta_init=None,
    ns=int(2e5),
    base_seed=123,
):
    """
    Sweep over internuclear distances r from r_min to r_max.

    Returns
    -------
    tuple of np.ndarray
        r_values, e_values, e_std_values
    """
    if theta_init is None:
        theta_init = np.array([1.5, 0.355, 0.55])

    r_values = np.linspace(r_min, r_max, n_points)
    e_values = []
    e_std_values = []

    for i, r_val in enumerate(r_values):
        print(f"\n==== Optimising at r = {r_val:.3f} ====")

        q1 = np.array([-r_val / 2.0, 0.0, 0.0])
        q2 = np.array([r_val / 2.0, 0.0, 0.0])

        theta = theta_init.copy()
        theta_mini, history = quasi_newton_min_energy_analytic(
            theta,
            q1,
            q2,
            ns=ns,
            n_steps=50,
            base_seed=base_seed + 10 * i,
            tol_grad=1e-3,
            tol_step=1e-4,
            max_step_norm=0.2,
        )

        # 'theta_mini' is unused here, but we keep it for completeness.
        _ = theta_mini

        last = history[-1]
        energy = last["E"]
        energy_std = last["E_std"]

        e_values.append(energy)
        e_std_values.append(energy_std)

        print(
            "Converged E(r={:.3f}) = {:.6f} ± {:.6f}".format(
                r_val,
                energy,
                energy_std,
            )
        )

    return (
        np.array(r_values),
        np.array(e_values),
        np.array(e_std_values),
    )


def morse_potential(r_val, d_param, a_param, r0_param, e_single):
    """
    Morse potential function.

    Parameters
    ----------
    r_val : array_like
        Internuclear distance(s).
    d_param : float
        Dissociation energy parameter.
    a_param : float
        Stiffness parameter.
    r0_param : float
        Equilibrium bond length.
    e_single : float
        Single atom energy.

    Returns
    -------
    array_like
        Potential energy at r_val.
    """
    return (
        d_param * (1.0 - np.exp(-a_param * (r_val - r0_param))) ** 2
        - d_param
        + 2.0 * e_single
    )


def main():
    """Main execution block."""
    t0 = time.perf_counter()

    r_vals, e_vals, e_stds = collect_e_vs_r_for_fit(
        r_min=2.0,
        r_max=2.1,
        n_points=2,
        theta_init=np.array([1.5, 0.355, 0.55]),
        ns=int(1e6),
    )

    # Save to Excel
    save_path = "S3_H2/fitting_data/H2_E_vs_r.xlsx"
    save_r_e_to_excel(save_path, r_vals, e_vals, e_stds)

    t1 = time.perf_counter()
    print(f"\nTotal sweep runtime: {t1 - t0:.2f} s\n")

    # Example for finishing the fit (commented out, as in original):

    # e_single = -0.5  # Hartree

    # def morse_fixed_e_single(r_val, d_param, a_param, r0_param):
    #     return morse_potential(r_val, d_param, a_param, r0_param, e_single)
    #
    # popt, pcov = curve_fit(
    #     morse_fixed_e_single,
    #     r_vals,
    #     e_vals,
    #     sigma=e_stds,
    #     absolute_sigma=True,
    # )
    #
    # d_fit, a_fit, r0_fit = popt
    # d_std, a_std, r0_std = np.sqrt(np.diag(pcov))
    #
    # print(f"Equilibrium bond length r0  = {r0_fit:.6f} ± {r0_std:.6f} bohr")
    # print(f"Dissociation energy D       = {d_fit:.6f} ± {d_std:.6f} Hartree")
    # print(f"Stiffness parameter a       = {a_fit:.6f} ± {a_std:.6f} bohr^-1")


if __name__ == "__main__":
    main()