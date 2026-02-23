import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def morse_potential(r, D, a, r0, E_single):
    return D * (1 - np.exp(-a * (r - r0)))**2 - D + 2*E_single

def fit_and_plot_morse(filepath, E_single = -0.4997):
    """
    Read r, E(r), E_std(r) from Excel, fit a Morse potential and
    make an error-bar plot with the fitted curve.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    E_single : float
        Single-atom energy used in the Morse potential.
    savefig : str or None
        If not None, path to save the figure (e.g. 'morse_fit.png').
    """

    # --- load data from Excel (assumes columns: r, E(r), E_std(r)) ---
    df = pd.read_excel(filepath)
    r_vals   = df.iloc[:, 0].to_numpy(dtype=float)
    E_vals   = df.iloc[:, 1].to_numpy(dtype=float)
    E_stds   = df.iloc[:, 2].to_numpy(dtype=float)

    # --- initial guesses for the fit ---
    D0   = E_vals.max() - E_vals.min()
    a0   = 1.0
    r0_0 = r_vals[np.argmin(E_vals)]
    p0   = [D0, a0, r0_0]

    # --- fit with weights from E_stds ---
    popt, pcov = curve_fit(
        lambda r, D, a, r0: morse_potential(r, D, a, r0, E_single),
        r_vals, E_vals,
        sigma=E_stds,
        absolute_sigma=True,
        p0=p0
    )

    D_fit, a_fit, r0_fit = popt
    perr = np.sqrt(np.diag(pcov))   # standard deviations of fitted params



    D_fit =  0.1425
    a_fit = 1.136
    r0_fit = 1.404


    # --- smooth curve for plotting the fit ---
    r_fine = np.linspace(r_vals.min(), r_vals.max(), 400)
    E_fit  = morse_potential(r_fine, D_fit, a_fit, r0_fit, E_single)

    # --- locate r = 1.4 data point ---
    r_target = 1.4
    idx = np.argmin(np.abs(r_vals - r_target))     # nearest entry
    r_hl     = r_vals[idx]
    E_hl     = E_vals[idx]
    E_hl_std = E_stds[idx]

    # --- make error-bar plot + fit curve ---
    plt.figure(figsize=(7,4))
    plt.errorbar(
        r_vals, E_vals, yerr=E_stds,
        fmt='o', capsize=3, label='Data with error bars'
    )
    plt.plot(r_fine, E_fit, label='Morse fit')

    # --- highlight: vertical dashed line at r = 1.4 ---
    # plt.axvline(r_target, color='red', linestyle='--', linewidth=1.3, alpha = 0.4,
    #             label='r = 1.4 highlight')

    # --- highlight the actual data point at r = 1.4 ---
    plt.errorbar(
        r_hl, E_hl, yerr=E_hl_std,
        fmt='o', color='red',
        ecolor='red', elinewidth=2, capsize=4,
        label=f'E(1.4) = {E_hl:.3f} ± {E_hl_std:.3f} Hartree'
    )

    # --- annotate numeric value ---
    # plt.text(
    #     r_hl + 0.02, E_hl-0.04,
    #     f"{E_hl:.4f} ± {E_hl_std:.4f}",
    #     color='red', fontsize=10, weight='bold'
    # )

    plt.xlabel('r / bohr',fontsize=13)
    plt.ylabel('E(r) / Hartree',fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    plt.legend(fontsize=11, loc = 'lower right')
    plt.grid(True)
    plt.tight_layout()

    
    plt.savefig(
    r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_plot\Morse_potential_2.png",
    dpi=300,
    bbox_inches="tight"
    )

    plt.show()

    # --- print fitted parameters with std dev ---
    print(f"Equilibrium bond length r0  = {r0_fit:.5f} ± {perr[2]:.5f} bohr")
    print(f"Dissociation energy D       = {D_fit:.5f} ± {perr[0]:.5f} Hartree")
    print(f"Stiffness parameter a       = {a_fit:.5f} ± {perr[1]:.5f} bohr^-1")

    return popt, pcov

fp = str(r'S3_H2\fitting_data\H2_E_vs_r_delta_0.75.xlsx')
popt, pcov = fit_and_plot_morse(fp)


