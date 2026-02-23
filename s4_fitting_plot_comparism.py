import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from scipy.optimize import curve_fit


def morse_potential(r, D, a, r0, E_single):
    return D * (1 - np.exp(-a * (r - r0)))**2 - D + 2 * E_single


def fit_morse_from_excel(filepath, E_single=-0.5):
    """
    Read r, E(r), E_std(r) from Excel and fit a Morse potential.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    E_single : float
        Single-atom energy used in the Morse potential.

    Returns
    -------
    result : dict
        {
          'r': r_vals,
          'E': E_vals,
          'E_std': E_stds,
          'popt': (D, a, r0),
          'pcov': covariance_matrix
        }
    """
    df = pd.read_excel(filepath)

    # Assumes first 3 columns are: r, E, E_std
    r_vals = df.iloc[:, 0].to_numpy(dtype=float)
    E_vals = df.iloc[:, 1].to_numpy(dtype=float)
    E_stds = df.iloc[:, 2].to_numpy(dtype=float)

    # Initial guesses
    D0 = E_vals.max() - E_vals.min()
    a0 = 1.0
    r0_0 = r_vals[np.argmin(E_vals)]
    p0 = [D0, a0, r0_0]

    popt, pcov = curve_fit(
        lambda r, D, a, r0: morse_potential(r, D, a, r0, E_single),
        r_vals,
        E_vals,
        sigma=E_stds,
        absolute_sigma=True,
        p0=p0
    )

    return {
        "r": r_vals,
        "E": E_vals,
        "E_std": E_stds,
        "popt": popt,
        "pcov": pcov,
    }


def compare_two_morse_files(filepath1,
                            filepath2,
                            E_single=-0.5,
                            label1="Data set 1",
                            label2="Data set 2",
                            savefig=None):
    """
    Fit Morse potential to two Excel data sets and make an interactive plot
    where you can choose what to show.

    Parameters
    ----------
    filepath1, filepath2 : str
        Paths to the Excel files.
    E_single : float
        Single-atom energy used in the Morse potential.
    label1, label2 : str
        Labels for the two data sets.
    savefig : str or None
        If not None, save figure to this path.

    Returns
    -------
    res1, res2 : dict
        Fit results for file 1 and file 2 (see fit_morse_from_excel).
    """
    # --- Fit both data sets ---
    res1 = fit_morse_from_excel(filepath1, E_single=E_single)
    res2 = fit_morse_from_excel(filepath2, E_single=E_single)

    D1, a1, r01 = res1["popt"]
    D2, a2, r02 = res2["popt"]

    perr1 = np.sqrt(np.diag(res1["pcov"]))
    perr2 = np.sqrt(np.diag(res2["pcov"]))

    # r-range for smooth curves (combine both ranges)
    r_min = min(res1["r"].min(), res2["r"].min())
    r_max = max(res1["r"].max(), res2["r"].max())
    r_fine = np.linspace(r_min, r_max, 400)

    E_fit1 = morse_potential(r_fine, D1, a1, r01, E_single)
    E_fit2 = morse_potential(r_fine, D2, a2, r02, E_single)

    # --- Plot with interactive checkboxes ---
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, right=0.95)  # space on left for CheckButtons

    # Data 1
    data1_line = ax.errorbar(
        res1["r"], res1["E"], yerr=res1["E_std"],
        fmt='o', capsize=3, label=f'{label1} (data)'
    )[0]  # errorbar returns (line, caplines, barlinecols), we take main line

    # Fit 1
    fit1_line, = ax.plot(
        r_fine, E_fit1,
        label=f'{label1} (Morse fit)'
    )

    # Data 2
    data2_line = ax.errorbar(
        res2["r"], res2["E"], yerr=res2["E_std"],
        fmt='s', capsize=3, label=f'{label2} (data)'
    )[0]

    # Fit 2
    fit2_line, = ax.plot(
        r_fine, E_fit2,
        linestyle='--',
        label=f'{label2} (Morse fit)'
    )

    ax.set_xlabel('r / bohr')
    ax.set_ylabel('E(r) / Hartree')
    ax.grid(True)
    ax.legend(loc='upper right')

    # --- CheckButtons for toggling visibility ---
    # Define checkbox labels and initial visibility
    labels = [
        f'{label1} data',
        f'{label1} fit',
        f'{label2} data',
        f'{label2} fit'
    ]
    visibility = [
        data1_line.get_visible(),
        fit1_line.get_visible(),
        data2_line.get_visible(),
        fit2_line.get_visible()
    ]

    # Add axes for the CheckButtons
    check_ax = plt.axes([0.03, 0.4, 0.18, 0.2])  # [left, bottom, width, height]
    check = CheckButtons(check_ax, labels, visibility)

    # Map labels to the actual artists
    lines_dict = {
        labels[0]: data1_line,
        labels[1]: fit1_line,
        labels[2]: data2_line,
        labels[3]: fit2_line,
    }

    def toggle_visibility(label):
        line = lines_dict[label]
        line.set_visible(not line.get_visible())
        plt.draw()

    check.on_clicked(toggle_visibility)

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=300)

    plt.show()

    # --- Print fit parameters for both data sets ---
    print("\n=== Fit results for", label1, "===")
    print(f"r0  = {r01:.5f} ± {perr1[2]:.5f} bohr")
    print(f"D   = {D1:.5f} ± {perr1[0]:.5f} Hartree")
    print(f"a   = {a1:.5f} ± {perr1[1]:.5f} bohr^-1")

    print("\n=== Fit results for", label2, "===")
    print(f"r0  = {r02:.5f} ± {perr2[2]:.5f} bohr")
    print(f"D   = {D2:.5f} ± {perr2[0]:.5f} Hartree")
    print(f"a   = {a2:.5f} ± {perr2[1]:.5f} bohr^-1")

    return res1, res2


if __name__ == "__main__":
    # Example usage
    fp1 = r"S3_H2\fitting_data\H2_E_vs_r_delta_0.5.xlsx"
    fp2 = r"S3_H2\fitting_data\H2_E_vs_r_delta_0.75.xlsx"

    compare_two_morse_files(fp1, fp2,
                            E_single=-0.5,
                            label1="Run 1",
                            label2="Run 2",
                            savefig=None)
