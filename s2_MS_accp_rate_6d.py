import numpy as np
import matplotlib.pyplot as plt
from s2_random_sampling import psi_H2
import pandas as pd
import os
from scipy.optimize import curve_fit, brentq
from numpy.random import multivariate_normal
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed

# file saving
def save_delta_scan_to_excel(results,
                             out_dir=r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_data",
                             filename="delta_scan_results.xlsx",
                             sheet_name="delta_scan"):
    """
    Save delta-scan results (delta, acc, tau_int, Neff, Neff_frac) to an Excel file.

    Parameters
    ----------
    results : dict
        Dictionary returned by optimize_delta_by_neff with keys:
        'deltas', 'acc', 'Neff', 'Neff_frac', 'tau_int'.
    out_dir : str
        Directory where the Excel file will be stored.
    filename : str
        Name of the Excel file.
    sheet_name : str
        Excel sheet name.
    """
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)

    df = pd.DataFrame({
        "delta":      results["deltas"],
        "acc_rate":   results["acc"],
        "tau_int":    results["tau_int"],
        "Neff":       results["Neff"],
        "Neff_frac":  results["Neff_frac"],
    })

    df.to_excel(full_path, index=False, sheet_name=sheet_name)
    print(f"Saved delta scan results to:\n  {full_path}")

def log_prob_H2(R, theta, q1, q2):
    """
    Return log probability log(|Psi(R)|^2) for H2.

    Parameters
    ----------
    R : array-like length 6  -> [x1,y1,z1,x2,y2,z2]
    theta : variational parameters
    q1, q2 : nuclear positions (arrays of shape (3,))
    """
    psi_val = psi_H2(R, theta, q1, q2)   # You must define this separately

    # Avoid log(0)
    if psi_val <= 0:
        return -1e300

    return 2.0 * np.log(psi_val)

# two different sampling methods 3d and 6d

def metropolis_sample_H2_6D(Ns, theta, q1, q2, delta=1.0, burn_in=200):

    R = np.concatenate([q1, q2], axis=0)
    samples = np.zeros((Ns, 6))

    accept_count = 0     # ← counter for accepted moves
    total_steps = Ns + burn_in

    for step in range(total_steps):

        # propose move
        R_prop = R + delta * np.random.randn(6)

        # compute acceptance probability
        logp_old = log_prob_H2(R, theta, q1, q2)
        logp_new = log_prob_H2(R_prop, theta, q1, q2)
        acc_prob = np.exp(logp_new - logp_old)

        if np.random.rand() < acc_prob:  
            R = R_prop
            accept_count += 1            # ← count accepted move
        
        # store sample after burn-in
        if step >= burn_in:
            samples[step - burn_in] = R

    acceptance_rate = accept_count / total_steps
    return samples, acceptance_rate

import numpy as np

def metropolis_sample_H2_2x3D(Ns, theta, q1, q2, delta=1.0, burn_in=200):
    """
    Metropolis sampling for H2 in 6D using two 3D single-electron moves per sweep.

    Parameters
    ----------
    Ns : int
        Number of stored configurations (after burn-in).
    theta : array-like
        Variational parameters for psi_H2.
    q1, q2 : array-like shape (3,)
        Nuclear positions.
    delta : float
        3D Gaussian proposal step size for each electron.
    burn_in : int
        Number of initial sweeps discarded.

    Returns
    -------
    samples : ndarray, shape (Ns, 6)
        Stored configurations [x1,y1,z1,x2,y2,z2].
    acceptance_rate : float
        Fraction of accepted single-electron moves.
    """

    # Initial configuration: place electrons at nuclei midpoints or wherever you like
    R = np.concatenate([q1, q2], axis=0).astype(float)

    samples = np.zeros((Ns, 6))
    total_sweeps = Ns + burn_in

    accept_count = 0
    move_count = 0

    for sweep in range(total_sweeps):

        # --- electron 1 move (indices 0:3) ---
        for e in [0, 1]:
            R_old = R.copy()
            R_prop = R.copy()

            start = 3 * e
            end = start + 3

            # 3D Gaussian move for electron e
            R_prop[start:end] += delta * np.random.randn(3)

            # log probabilities
            logp_old = log_prob_H2(R_old, theta, q1, q2)
            logp_new = log_prob_H2(R_prop, theta, q1, q2)

            acc_prob = np.exp(logp_new - logp_old)

            move_count += 1
            if np.random.rand() < acc_prob:
                R = R_prop
                accept_count += 1
            else:
                R = R_old  # explicitly keep old (for clarity)

        # after both electrons have been updated, store the configuration
        if sweep >= burn_in:
            samples[sweep - burn_in] = R

    acceptance_rate = accept_count / move_count
    return samples, acceptance_rate

# R_samples, acc_rate = metropolis_sample_H2(Ns, theta, q1, q2, delta=0.5)

def sweep_delta_and_save(deltas, Ns, theta, q1, q2,
                         burn_in=200,
                         save_dir=r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_data"):
    """
    Sweep over delta values, plot acceptance rate, save plot and Excel file.

    Parameters
    ----------
    deltas : list or array
        Proposal step sizes to test.
    Ns : int
        Number of samples (after burn-in) per delta.
    theta : array
        Variational parameters.
    q1, q2 : nuclear positions.
    """

    acc_rates = []

    for delta in deltas:
        print(f"Running delta = {delta} ...")
        _, acc_rate = metropolis_sample_H2_2x3D(Ns, theta, q1, q2,
                                           delta=delta, burn_in=burn_in)
        acc_rates.append(acc_rate)

    # --- Plot acceptance rate vs delta ---
    plt.figure(figsize=(6,4))
    plt.plot(deltas, acc_rates, marker='o')
    plt.xlabel("Proposal step size δ")
    plt.ylabel("Acceptance rate")
    # plt.title("Metropolis Acceptance Rate vs δ")
    plt.grid(True)

    # Make sure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save high-resolution figure
    fig_path = os.path.join(save_dir, "acceptance_rate_vs_delta.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"Plot saved to: {fig_path}")

    # --- Save data to Excel ---
    df = pd.DataFrame({"delta": deltas, "acceptance_rate": acc_rates})

    excel_path = os.path.join(save_dir, "acceptance_rate_vs_delta.xlsx")
    df.to_excel(excel_path, index=False)

    print(f"Excel file saved to: {excel_path}")

    return np.array(acc_rates)

# acc_rate = sweep_delta_and_save(deltas, Ns, theta, q1, q2, burn_in=200)

# ===== Autocorrelation + effective N =======================
# this code is used to compare 3D and 6D sampling effect of corelation

def autocorr_1d(x):
    """
    Normalized autocorrelation function for a 1D array.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)
    var = np.var(x)
    if var == 0:
        return np.array([1.0])
    # FFT-based ACF (fast, but simple)
    fx = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(fx * np.conjugate(fx))[:N].real
    acf /= acf[0]
    return acf

def integrated_autocorr_time(x, max_lag=None):
    """
    Estimate integrated autocorrelation time tau_int.
    """
    acf = autocorr_1d(x)
    if max_lag is None:
        max_lag = len(acf) // 10  # simple window
    max_lag = min(max_lag, len(acf) - 1)

    # standard estimator: 1/2 + sum_{k>=1} rho(k)
    tau_int = 0.5
    for k in range(1, max_lag + 1):
        if acf[k] <= 0:
            break
        tau_int += acf[k]
    return tau_int

def effective_sample_size(x, max_lag=None):
    N = len(x)
    tau_int = integrated_autocorr_time(x, max_lag=max_lag)
    Neff = N / (2.0 * tau_int)
    return Neff, tau_int

# ===== Simple tuner to hit a target acceptance rate ========

def tune_delta(target_acc, sampler, Ns_test, theta, q1, q2,
               delta_init=1.0, burn_in=200, tol=0.01, max_iter=15):
    """
    Very simple multiplicative tuner to reach ~target acceptance.
    """
    delta = delta_init
    for _ in range(max_iter):
        _, acc = sampler(Ns_test, theta, q1, q2, delta=delta, burn_in=burn_in)
        if abs(acc - target_acc) < tol:
            break
        # if acceptance too high -> increase delta (bigger steps)
        if acc > target_acc:
            delta *= 1.2
        else:
            delta *= 0.8
    return delta, acc

# use plotting to fit the data
def fit_for_delta():
    excel_path = r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_data\acceptance_rate_vs_delta_2x3D.xlsx"

    df = pd.read_excel(excel_path)

    # Adjust these if your column names are different
    delta_data = df["delta"].to_numpy(dtype=float)
    acc_data   = df["acceptance_rate"].to_numpy(dtype=float)

    # If you have measurement errors for acceptance, put them here.
    # Otherwise, assume equal errors (sigma = 1 → chi2 is just RSS).
    sigma_data = np.ones_like(acc_data)


    # ============================================================
    # 2. Define models
    # ============================================================

    # Parabolic model: a * delta^2 + b * delta + c
    def model_parabola(delta, a, b, c):
        return a * delta**2 + b * delta + c

    # Exponential model: A * exp(B * delta) + C
    def model_exponential(delta, A, B, C):
        return A * np.exp(B * delta) + C


    # ============================================================
    # 3. Fit both models with curve_fit
    # ============================================================

    # Initial guesses (you can tweak if needed)
    p0_para = [0.0, 0.0, 0.5]   # roughly centered around acc ~ 0.5
    p0_exp  = [0.5, -1.0, 0.0]

    popt_para, pcov_para = curve_fit(
        model_parabola,
        delta_data, acc_data,
        p0=p0_para,
        sigma=sigma_data,
        absolute_sigma=True
    )

    popt_exp, pcov_exp = curve_fit(
        model_exponential,
        delta_data, acc_data,
        p0=p0_exp,
        sigma=sigma_data,
        absolute_sigma=True,
        maxfev=10000
    )


    # ============================================================
    # 4. Compute chi2 and reduced chi2 for each model
    # ============================================================

    def chi2_and_reduced(y, y_fit, sigma, n_params):
        residuals = (y - y_fit) / sigma
        chi2 = np.sum(residuals**2)
        dof  = len(y) - n_params
        chi2_red = chi2 / dof if dof > 0 else np.nan
        return chi2, chi2_red

    # Parabola
    acc_fit_para = model_parabola(delta_data, *popt_para)
    chi2_para, chi2_red_para = chi2_and_reduced(acc_data, acc_fit_para, sigma_data, n_params=3)

    # Exponential
    acc_fit_exp = model_exponential(delta_data, *popt_exp)
    chi2_exp, chi2_red_exp = chi2_and_reduced(acc_data, acc_fit_exp, sigma_data, n_params=3)

    print("=== Fit quality (chi^2) ===")
    print(f"Parabola:    chi2 = {chi2_para:.3f},  chi2_red = {chi2_red_para:.3f}")
    print(f"Exponential: chi2 = {chi2_exp:.3f},  chi2_red = {chi2_red_exp:.3f}")

    if chi2_red_para < chi2_red_exp:
        print("→ Parabolic model provides the better fit (lower reduced chi^2).")
    else:
        print("→ Exponential model provides the better fit (lower reduced chi^2).")


    # ============================================================
    # 5. Solve for delta where acceptance = 0.5
    # ============================================================

    TARGET_ACC = 0.5

    def find_delta_for_target(model, params, delta_min, delta_max, target=TARGET_ACC):
        """
        Find delta such that model(delta, *params) = target using brentq.
        Assumes the function crosses the target between delta_min and delta_max.
        """
        def f(delta):
            return model(delta, *params) - target

        # Expand the bracket a bit if needed
        lo, hi = delta_min, delta_max
        # Ensure we actually bracket a root
        for _ in range(10):
            if f(lo) * f(hi) < 0:
                break
            # Expand range
            lo *= 0.8
            hi *= 1.2
        else:
            raise RuntimeError("Could not bracket a root for target acceptance.")

        return brentq(f, lo, hi)

    delta_min = np.min(delta_data)
    delta_max = np.max(delta_data)

    delta_star_para = find_delta_for_target(model_parabola, popt_para, delta_min, delta_max)
    delta_star_exp  = find_delta_for_target(model_exponential, popt_exp, delta_min, delta_max)

    print("\n=== Delta solving for acceptance = 0.5 ===")
    print(f"Parabola:    delta* ≈ {delta_star_para:.5f}")
    print(f"Exponential: delta* ≈ {delta_star_exp:.5f}")


    # ============================================================
    # 6. Estimate uncertainty (std) on delta* via parameter sampling
    # ============================================================

    def delta_uncertainty_via_MC(model, popt, pcov, delta_min, delta_max,
                                target=TARGET_ACC, n_samples=2000):
        """
        Draw parameters from a multivariate normal with covariance pcov,
        solve for delta* for each draw, and return mean and std of delta*.
        """
        # If covariance is singular or ill-conditioned, this may fail.
        params_samples = multivariate_normal(mean=popt, cov=pcov, size=n_samples)

        delta_samples = []

        for params in params_samples:
            try:
                delta_s = find_delta_for_target(model, params, delta_min, delta_max, target)
                delta_samples.append(delta_s)
            except Exception:
                # If root finding fails for a sample (bad parameters), skip it
                continue

        delta_samples = np.array(delta_samples)
        return np.mean(delta_samples), np.std(delta_samples), delta_samples.size

    mean_para, std_para, n_ok_para = delta_uncertainty_via_MC(
        model_parabola, popt_para, pcov_para, delta_min, delta_max
    )

    mean_exp, std_exp, n_ok_exp = delta_uncertainty_via_MC(
        model_exponential, popt_exp, pcov_exp, delta_min, delta_max
    )

    print("\n=== Delta* with uncertainty for acceptance = 0.5 ===")
    print(f"Parabola:    delta* = {mean_para:.5f} ± {std_para:.5f}  (from {n_ok_para} MC samples)")
    print(f"Exponential: delta* = {mean_exp:.5f} ± {std_exp:.5f}  (from {n_ok_exp} MC samples)")

#fit plot
# fit_for_delta()

# ===== Comparison driver ===================================

def compare_effective_N(target_acc=0.5, Ns=int(1e6), Ns_test=5000,
                        theta=None, q1=None, q2=None,
                        observable=None):
    """
    Compare effective sample size for 6D vs 2x3D samplers
    at (approximately) the same target acceptance rate.
    """

    if theta is None:
        theta = np.array([1.0, 1.0, 1.0])
    if q1 is None or q2 is None:
        r = 1.0
        q1 = np.array([ r/2, 0.0, 0.0 ])
        q2 = np.array([-r/2, 0.0, 0.0 ])

    # Default observable: x-coordinate of electron 1
    if observable is None:
        def observable(R_samples):
            return R_samples[:, 0]
        
    # # --- Tune delta for each sampler to hit target_acc ------
    # delta_6D, acc_6D = tune_delta(target_acc,
    #                               metropolis_sample_H2_6D,
    #                               Ns_test, theta, q1, q2)
    # delta_3D, acc_3D = tune_delta(target_acc,
    #                               metropolis_sample_H2_2x3D,
    #                               Ns_test, theta, q1, q2)

    # print(f"Tuned 6D: delta = {delta_6D:.4f}, acceptance ≈ {acc_6D:.3f}")
    # print(f"Tuned 2x3D: delta = {delta_3D:.4f}, acceptance ≈ {acc_3D:.3f}")

    #read out the value from the plot:
    delta_6D = 0.55968 #from fitting result
    delta_3D = 0.84710 #from fitting result

    # --- Run long chains with tuned deltas ------------------
    samples_6D, acc_6D_long = metropolis_sample_H2_6D(
        Ns, theta, q1, q2, delta=delta_6D, burn_in=200
    )
    samples_3D, acc_3D_long = metropolis_sample_H2_2x3D(
        Ns, theta, q1, q2, delta=delta_3D, burn_in=200
    )

    obs_6D = observable(samples_6D)
    obs_3D = observable(samples_3D)

    Neff_6D, tau_6D = effective_sample_size(obs_6D)
    Neff_3D, tau_3D = effective_sample_size(obs_3D)

    print("\n=== Results at roughly the same acceptance rate ===")
    print(f"6D sampler:  acc ≈ {acc_6D_long:.3f}, "
          f"tau_int ≈ {tau_6D:.2f},  Neff ≈ {Neff_6D:.0f}")
    print(f"2x3D sampler: acc ≈ {acc_3D_long:.3f}, "
          f"tau_int ≈ {tau_3D:.2f},  Neff ≈ {Neff_3D:.0f}")

    return {
        "delta_6D": delta_6D,
        "delta_3D": delta_3D,
        "acc_6D": acc_6D_long,
        "acc_3D": acc_3D_long,
        "tau_6D": tau_6D,
        "tau_3D": tau_3D,
        "Neff_6D": Neff_6D,
        "Neff_3D": Neff_3D,
    }

# ---------- Scan on delta for tau and Neff/N ----------

def _eval_one_delta(i, delta, sampler, Ns_test,
                    theta, q1, q2,
                    burn_in, observable, max_lag):
    """
    Worker for a single delta.
    Runs in a separate process; must be at module top level to be picklable.
    """
    # Run sampler
    samples, acc = sampler(Ns_test, theta, q1, q2,
                           delta=delta, burn_in=burn_in)

    # Observable
    if observable is None:
        obs = samples[:, 0]   # x of electron 1 as default
    else:
        obs = observable(samples)

    # Use your existing effective_sample_size function
    Neff, tau = effective_sample_size(obs, max_lag=max_lag)
    Neff_frac = Neff / Ns_test

    # return index so we can reassemble in correct order
    return i, float(delta), acc, Neff, Neff_frac, tau

def optimize_delta_by_neff(sampler, deltas, Ns_test,
                           theta, q1, q2,
                           burn_in=200,
                           observable=None,
                           max_lag=None,
                           n_workers=6):
    """
    Search over delta values and find the one that maximizes Neff / Ns_test.

    Parameters
    ----------
    sampler : callable
        Function with signature sampler(Ns, theta, q1, q2, delta, burn_in)
        that returns (samples, acceptance_rate).
        IMPORTANT: must be defined at module top-level to be picklable.
    deltas : array-like
        List/array of proposal step sizes to test.
    Ns_test : int
        Number of samples (after burn-in) per delta.
    theta, q1, q2 :
        Parameters and nuclear positions for H2.
    burn_in : int
        Burn-in length for each run.
    observable : callable or None
        Function mapping samples array (Ns_test, 6) -> 1D observable array.
        Must be picklable if provided. If None, uses x-coordinate of electron 1.
    max_lag : int or None
        Max lag for autocorrelation integration.
    n_workers : int, optional
        Number of processes (cores) to use. Default 6.

    Returns
    -------
    best_delta : float
        Delta giving the largest Neff / Ns_test.
    results : dict
        Dictionary with arrays:
        'deltas', 'acc', 'Neff', 'Neff_frac', 'tau_int'
    """

    deltas = np.asarray(deltas, dtype=float)
    n_d = len(deltas)

    # Preallocate arrays
    Neff_vals = np.zeros(n_d, dtype=float)
    Neff_frac_vals = np.zeros(n_d, dtype=float)
    tau_vals = np.zeros(n_d, dtype=float)
    acc_vals = np.zeros(n_d, dtype=float)

    # If only one worker or one delta, use original sequential behaviour
    if n_workers <= 1 or n_d == 1:
        for i, delta in enumerate(deltas):
            print(f"Testing delta = {delta:.4f} ...")
            samples, acc = sampler(Ns_test, theta, q1, q2,
                                   delta=delta, burn_in=burn_in)
            acc_vals[i] = acc

            if observable is None:
                obs = samples[:, 0]
            else:
                obs = observable(samples)

            Neff, tau = effective_sample_size(obs, max_lag=max_lag)
            Neff_vals[i] = Neff
            Neff_frac_vals[i] = Neff / Ns_test
            tau_vals[i] = tau

            print(f"  acc ≈ {acc:.3f}, tau_int ≈ {tau:.2f}, "
                  f"Neff/N ≈ {Neff_vals[i] / Ns_test:.3f}")

    else:
        print(f"Running {n_d} deltas in parallel on up to {n_workers} cores ...")

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for i, delta in enumerate(deltas):
                print(f"Submitting delta = {delta:.4f} ...")
                fut = ex.submit(
                    _eval_one_delta,
                    i, float(delta), sampler, Ns_test,
                    theta, q1, q2,
                    burn_in, observable, max_lag
                )
                futures.append(fut)

            for fut in as_completed(futures):
                i_out, delta_out, acc, Neff, Neff_frac, tau = fut.result()

                acc_vals[i_out] = acc
                Neff_vals[i_out] = Neff
                Neff_frac_vals[i_out] = Neff_frac
                tau_vals[i_out] = tau

                print(f"Finished delta = {delta_out:.4f}")
                print(f"  acc ≈ {acc:.3f}, tau_int ≈ {tau:.2f}, "
                      f"Neff/N ≈ {Neff_frac:.3f}")

    # Maximise Neff / N
    idx_best = np.argmax(Neff_frac_vals)
    best_delta = deltas[idx_best]

    print("\n=== Optimisation result ===")
    print(f"Best delta ≈ {best_delta:.4f}")
    print(f"  acc ≈ {acc_vals[idx_best]:.3f}")
    print(f"  tau_int ≈ {tau_vals[idx_best]:.2f}")
    print(f"  Neff/N ≈ {Neff_frac_vals[idx_best]:.3f}")

    results = {
        "deltas": deltas,
        "acc": acc_vals,
        "Neff": Neff_vals,
        "Neff_frac": Neff_frac_vals,
        "tau_int": tau_vals,
    }

    return best_delta, results

def plot_tau_and_neff(results, title_prefix="Metropolis Sampling"):
    deltas = results["deltas"]
    tau = results["tau_int"]
    neff_frac = results["Neff_frac"]

    # -----------------------------
    # Plot 1: Autocorrelation time
    # -----------------------------
    plt.figure(figsize=(6,4))
    plt.plot(deltas, tau, marker='o')
    plt.xlabel("Proposal step size  δ")
    plt.ylabel(r"Integrated autocorrelation time  $\tau_{\mathrm{int}}$")
    plt.title(f"{title_prefix}: Autocorrelation Time vs δ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot 2: Neff / Ntest
    # -----------------------------
    plt.figure(figsize=(6,4))
    plt.plot(deltas, neff_frac, marker='o')
    plt.xlabel("Proposal step size  δ")
    plt.ylabel(r"Efficiency  $N_{\mathrm{eff}} / N_{\mathrm{test}}$")
    plt.title(f"{title_prefix}: Sampling Efficiency vs δ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_neff_frac_comparison(
    file_3D=r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_data\delta_scan_results2x3D.xlsx",
    file_6D=r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_data\delta_scan_results_6D.xlsx"
):
    """
    Read delta-scan results from two Excel files and plot Neff_frac vs delta.

    Parameters
    ----------
    file_3D : str
        Path to the 2x3D Metropolis scan results Excel file.
    file_6D : str
        Path to the 6D Metropolis scan results Excel file.
    """

    # --- Load both tables ---
    df3 = pd.read_excel(file_3D)
    df6 = pd.read_excel(file_6D)

    # --- Extract columns ---
    delta3 = df3["delta"]
    neff3 = df3["Neff_frac"]

    delta6 = df6["delta"]
    neff6 = df6["Neff_frac"]

    # --- Plot ---
    plt.figure(figsize=(7,5))
    plt.plot(delta3, neff3, marker="o", label="2×3D sampling")
    plt.plot(delta6, neff6, marker="s", label="6D sampling")

    plt.xlabel(r"$\delta$")
    plt.ylabel(r"$N_{\mathrm{eff}}/N$")
    plt.title("Comparison of Neff Fraction vs Step Size δ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Loaded and plotted both datasets successfully.")

# Compare the effective N
# if __name__ == "__main__":
    # Example: compare at target acceptance ~0.5
    # results = compare_effective_N(target_acc=0.5, Ns=int(1e6), Ns_test=5000)


# scan with different sampling method
if __name__ == "__main__":

    deltas = np.linspace(0.1, 3, 5)
    Ns_test = int(1e6)

    theta = np.array([1.0, 1.0, 1.0])
    r = 1.0
    q1 = np.array([ r/2, 0.0, 0.0 ])
    q2 = np.array([-r/2, 0.0, 0.0 ])

    # best_delta_6D, res_6D = optimize_delta_by_neff(
    #     sampler=metropolis_sample_H2_6D,
    #     deltas=deltas,
    #     Ns_test=Ns_test,
    #     theta=theta, q1=q1, q2=q2,
    #     burn_in=200
    # )
    
    
    # best_delta_2x3D, res_2x3D = optimize_delta_by_neff(
    #     sampler=metropolis_sample_H2_2x3D,
    #     deltas=np.linspace(0.1, 3.0, 50),
    #     Ns_test=int(1e6),
    #     theta=theta,
    #     q1=q1,
    #     q2=q2,
    #     burn_in=200, 
    #     max_lag=200,
    #     n_workers=6,  # 6-core run
    # )
    # save_delta_scan_to_excel(res_2x3D,sheet_name="delta_scan_2x3D")

    # best_delta_6D, res_6D = optimize_delta_by_neff(
    #     sampler=metropolis_sample_H2_6D,
    #     deltas=np.linspace(0.1, 3, 50),
    #     Ns_test=int(1e6),
    #     theta=theta,
    #     q1=q1,
    #     q2=q2,
    #     burn_in=200, 
    #     max_lag=200,
    #     n_workers=6,  # 6-core run
    # )
    # save_delta_scan_to_excel(res_6D, sheet_name="delta_scan_6D")

    # plot_neff_frac_comparison()

# curvefit
def minimize_chi2(cost_fn, x0, step=0.1, max_iter=5000, tol=1e-10):
    """
    Pure-Python random-search + coordinate descent minimizer.
    No SciPy required.

    Parameters
    ----------
    cost_fn : callable(params) -> chi2
    x0      : initial guess (list or array)
    step    : initial search step size
    max_iter: maximum iterations
    tol     : stopping tolerance

    Returns
    -------
    best_params, best_chi2
    """
    x = np.array(x0, dtype=float)
    best = cost_fn(x)
    
    for it in range(max_iter):
        improved = False
        
        # Try perturbing each parameter independently
        for i in range(len(x)):
            for direction in [+1, -1]:
                x_try = x.copy()
                x_try[i] += direction * step
                
                chi2_try = cost_fn(x_try)
                if chi2_try < best:
                    best = chi2_try
                    x = x_try
                    improved = True

        # If no improvement → reduce step size
        if not improved:
            step *= 0.5
            if step < tol:
                break

    return x, best

# Efficiency model
def efficiency_model(delta, A, p, delta0, q):
    return A * delta**p * np.exp(-(delta / delta0)**q)

def fit_delta_scan(file_path, label, color, x0=None):
    """
    Load an Excel delta-scan file, fit Neff_frac(delta) with the efficiency
    model using chi^2 minimization, and return data + fit params.

    Parameters
    ----------
    file_path : str
        Path to Excel file.
    label : str
        Label for legend.
    color : str
        Matplotlib color string.
    x0 : list or None
        Initial guess [A, p, delta0, q].

    Returns
    -------
    delta, neff_frac, params, chi2
    """
    df = pd.read_excel(file_path)
    delta = df["delta"].values
    neff_frac = df["Neff_frac"].values

    # Simple choice: constant sigma (used only as weights in chi^2)
    sigma = np.full_like(neff_frac, 5e-3)

    def chi2_fn(params):
        A, p, delta0, q = params
        model = efficiency_model(delta, A, p, delta0, q)
        return np.sum(((neff_frac - model) / sigma) ** 2)

    if x0 is None:
        x0 = [0.07, 2.0, 1.5, 2.0]

    params, chi2_val = minimize_chi2(chi2_fn, x0, step=0.1)

    print(f"=== Fit results for {label} ===")
    print("A      =", params[0])
    print("p      =", params[1])
    print("delta0 =", params[2], "(peak approx here)")
    print("q      =", params[3])
    print("chi^2  =", chi2_val)
    print()

    return delta, neff_frac, params, chi2_val

def plot_efficiency_fits():
    file_3D = (r"D:\OneDrive - Imperial College London\Y3 Computing\project"
               r"\report_writing_data\delta_scan_results2x3D.xlsx")
    file_6D = (r"D:\OneDrive - Imperial College London\Y3 Computing\project"
               r"\report_writing_data\delta_scan_results_6D.xlsx")

    # Fit 2×3D
    d3, neff3, p3, chi2_3 = fit_delta_scan(file_3D, "2×3D Metropolis", "C0")

    # Use 2×3D params as starting point for 6D fit (often helps)
    d6, neff6, p6, chi2_6 = fit_delta_scan(file_6D, "6D Metropolis", "C1",
                                           x0=p3)

    # Prepare smooth curves for plotting
    delta_min = min(d3.min(), d6.min())
    delta_max = max(d3.max(), d6.max())
    delta_plot = np.linspace(delta_min, delta_max, 400)

    fit3 = efficiency_model(delta_plot, *p3)
    fit6 = efficiency_model(delta_plot, *p6)

    # Plot
    plt.figure(figsize=(7, 5))

    # Data points
    plt.plot(d3, neff3, "o", ms=4, label="2×3D data")
    plt.plot(d6, neff6, "s", ms=4, label="6D data")

    # Fit curves
    plt.plot(delta_plot, fit3, "-", label="2×3D χ² fit")
    plt.plot(delta_plot, fit6, "-", label="6D χ² fit")

    plt.xlabel(r"Proposal step size $\delta$")
    plt.ylabel(r"$N_{\mathrm{eff}}/N_{\mathrm{test}}$")
    plt.title("Sampling Efficiency vs Step Size: 2×3D vs 6D Metropolis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# plot_efficiency_fits()

def optimum_from_fit_params(A, p, delta0, q):
        """
        Given fitted parameters (A, p, delta0, q), compute the delta that
        maximizes Neff/N and the corresponding maximum Neff/N.

        Parameters
        ----------
        A, p, delta0, q : float
            Fit parameters from the efficiency model.

        Returns
        -------
        delta_star : float
            Step size delta that maximizes Neff/N.
        neff_frac_star : float
            Maximum value of Neff/N at delta_star.
        """
        # Analytic maximum for f(delta) = A * delta^p * exp(-(delta/delta0)^q)
        delta_star = delta0 * (p / q)**(1.0 / q)
        neff_frac_star = efficiency_model(delta_star, A, p, delta0, q)
        return delta_star, neff_frac_star

def solving_for_maximun():
    #from fitting for 2x3D reuslt
    A_fit      = 0.09705078124999927
    p_fit      = 1.6313476562498814
    delta0_fit = 1.5645019531252964
    q_fit      = 1.5222656250001776

    delta_star, neff_frac_star = optimum_from_fit_params(
        A_fit, p_fit, delta0_fit, q_fit
    )

    print("delta*    =", delta_star)
    print("(Neff/N)* =", neff_frac_star)
