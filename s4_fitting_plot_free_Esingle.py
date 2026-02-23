import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def morse_potential_4p(r, D, a, r0, E_single):
    """
    4-parameter Morse:
    E(r) = D(1 - exp(-a (r-r0)))^2 - D + 2 E_single
    """
    return D * (1 - np.exp(-a * (r - r0)))**2 - D + 2 * E_single

def fit_and_plot_morse_free_Esingle(filepath, savefig=None):
    # --- load data ---
    df = pd.read_excel(filepath)
    r_vals = df.iloc[:, 0].to_numpy(dtype=float)
    E_vals = df.iloc[:, 1].to_numpy(dtype=float)
    E_stds = df.iloc[:, 2].to_numpy(dtype=float)

    # avoid zero/negative sigmas
    eps = 1e-12
    E_stds = np.where(E_stds <= 0, eps, E_stds)

    # --- initial guesses ---
    D0   = E_vals.max() - E_vals.min()
    a0   = 1.0
    r0_0 = r_vals[np.argmin(E_vals)]
    Es0  = -0.5  # starting guess, but now free
    p0   = [D0, a0, r0_0, Es0]

    # --- weighted fit ---
    popt, pcov = curve_fit(
        morse_potential_4p,
        r_vals, E_vals,
        sigma=E_stds,
        absolute_sigma=True,
        p0=p0
    )

    D_fit, a_fit, r0_fit, Es_fit = popt
    perr = np.sqrt(np.diag(pcov))

    # --- smooth curve for plotting ---
    r_fine = np.linspace(r_vals.min(), r_vals.max(), 400)
    E_fit  = morse_potential_4p(r_fine, D_fit, a_fit, r0_fit, Es_fit)

    # --- plot ---
    plt.figure()
    plt.errorbar(r_vals, E_vals, yerr=E_stds,
                 fmt='o', capsize=3, label='Data with error bars')
    plt.plot(r_fine, E_fit, label='Morse fit (4-param)')
    plt.xlabel('r / bohr')
    plt.ylabel('E(r) / Hartree')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    plt.show()

    print(f"r0        = {r0_fit:.5f} ± {perr[2]:.5f} bohr")
    print(f"D         = {D_fit:.5f} ± {perr[0]:.5f} Hartree")
    print(f"a         = {a_fit:.5f} ± {perr[1]:.5f} bohr^-1")
    print(f"E_single  = {Es_fit:.5f} ± {perr[3]:.5f} Hartree")

    return popt, pcov

fp = str(r'S3_H2\fitting_data\H2_E_vs_r_delta_0.75.xlsx')
popt, pcov = fit_and_plot_morse_free_Esingle(fp)
