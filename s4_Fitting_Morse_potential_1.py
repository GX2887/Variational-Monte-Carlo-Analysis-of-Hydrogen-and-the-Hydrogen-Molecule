import numpy as np
from numba import njit
from openpyxl import Workbook
import os
from s3_H2_copy_numba_quasi_newton import quasi_newton_min_energy_analytic
from scipy.optimize import curve_fit

def save_r_E_to_excel(filepath, r_values, E_values, E_std_values):
    """
    Save r, E, and E_std into an Excel (.xlsx) file.
    Columns:
    r   |   E(r)   |   E_std(r)
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "H2_EnergyCurve"

    # header row
    ws.append(["r", "E(r)", "E_std(r)"])

    # data rows
    for r, E, Es in zip(r_values, E_values, E_std_values):
        ws.append([float(r), float(E), float(Es)])

    # ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    wb.save(filepath)
    print(f"\nSaved Excel file to:\n{filepath}\n")

def collect_E_vs_r_for_fit(r_min=0.5, r_max=3.0, n_points=25,
                           theta_init=np.array([1.5, 0.355, 0.55]),
                           Ns=int(2e5),
                           base_seed=123):
    """
    Sweep over internuclear distances r from r_min to r_max.
    Returns arrays r_values, E_values, E_std_values.
    """

    r_values = np.linspace(r_min, r_max, n_points)
    E_values = []
    E_std_values = []

    for i, r in enumerate(r_values):
        print(f"\n==== Optimising at r = {r:.3f} ====")

        q1 = np.array([-r/2, 0, 0])
        q2 = np.array([ r/2, 0, 0])

        theta = theta_init.copy()
        theta_mini, history = quasi_newton_min_energy_analytic(
            theta, q1, q2,
            Ns=Ns,
            n_steps=50,
            base_seed=base_seed + 10*i,
            tol_grad=1e-3,
            tol_step=1e-4,
            max_step_norm=0.2
        )

        last = history[-1]
        E = last["E"]
        E_std = last["E_std"]

        E_values.append(E)
        E_std_values.append(E_std)

        print(f"Converged E(r={r:.3f}) = {E:.6f} ± {E_std:.6f}")

    return (
        np.array(r_values),
        np.array(E_values),
        np.array(E_std_values),
    )

#fitting fucntion
def morse_potential(r, D, a, r0, E_single):
    return D * (1 - np.exp(-a * (r - r0)))**2 - D + 2*E_single


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()

    r_vals, E_vals, E_stds = collect_E_vs_r_for_fit(
        r_min=1.025 , r_max=2.025, n_points=11,
        theta_init=np.array([1.5, 0.355, 0.55]),
        Ns=int(1e6)
    )

    # save to excel
    save_path = "S3_H2/fitting_data/H2_E_vs_r_1.xlsx"
    save_r_E_to_excel(save_path, r_vals, E_vals, E_stds)

    t1 = time.perf_counter()
    print(f"\nTotal sweep runtime: {t1 - t0:.2f} s\n")

    E_single  = -0.5 # Hartree

    popt, pcov = curve_fit(
    lambda r, D, a, r0: morse_potential(r, D, a, r0, E_single),
    r_vals, E_vals, sigma=E_stds
    )
    D_fit, a_fit, r0_fit = popt
    D_std, a_std, r0_std = np.sqrt(np.diag(pcov))
    print(f"Equilibrium bond length r0  = {r0_fit:.6f} ± {r0_std:.6f} bohr")
    print(f"Dissociation energy D       = {D_fit:.6f} ± {D_std:.6f} Hartree")
    print(f"Stiffness parameter a       = {a_fit:.6f} ± {a_std:.6f} bohr^-1")