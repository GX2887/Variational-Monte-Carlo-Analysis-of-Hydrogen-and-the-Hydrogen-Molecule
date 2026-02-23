import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import s1_differentiation as drt
import s2_random_sampling as rs
import pandas as pd

# --- your psi and drt, rs must be defined/imported elsewhere ---

def psi(R, theta):
    r = np.linalg.norm(R)
    return np.exp(-theta * r)

def local_energy_single(R, theta):
    """
    Local energy E_L(R; theta) for a single configuration R.
    """
    r = np.linalg.norm(R)
    psi_val = psi(R, theta)
    lap = drt.laplacian_3d_fd(psi, R, theta)  # your finite-difference Laplacian
    return -0.5 * lap / psi_val - 1.0 / r

def _mc_worker(args):
    """
    Worker that generates Ns_chunk Metropolis samples and returns partial sums
    needed to compute E_avg, E_std, and d<E>/dθ.

    Returns
    -------
    Ns_chunk : int
    S_E      : sum E_i
    S_E2     : sum E_i^2
    S_r      : sum r_i
    S_Er     : sum E_i * r_i
    """
    theta, Ns_chunk, seed = args

    np.random.seed(seed)
    R_samples = rs.metropolis_sample_rho_3d(Ns_chunk, theta)

    S_E = 0.0
    S_E2 = 0.0
    S_r = 0.0
    S_Er = 0.0

    for i in range(Ns_chunk):
        R = R_samples[i]
        r = np.linalg.norm(R)
        E = local_energy_single(R, theta)

        S_E  += E
        S_E2 += E * E
        S_r  += r
        S_Er += E * r

    return Ns_chunk, S_E, S_E2, S_r, S_Er

def pd_H_theta_parallel(theta, Ns=int(1e6), r_eff=0.051, 
                        n_workers=None, seed0=123):
    """
    Parallel Monte Carlo estimate of d<E>/dθ, <H>, and its MC error.

    Parameters
    ----------
    theta : float
        Variational parameter.
    Ns : int
        Total number of Metropolis samples.
    r_eff : float
        Efficiency ratio (N_eff / N).
    n_workers : int or None
        Number of worker processes (defaults to cpu_count()).
    seed0 : int
        Base random seed.

    Returns
    -------
    grad : float
        Estimate of ∂⟨H⟩/∂θ.
    E_avg : float
        Estimate of ⟨H⟩.
    E_std : float
        MC standard error for ⟨H⟩.
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Split Ns into roughly equal chunks
    Ns_chunk = Ns // n_workers
    remainder = Ns % n_workers
    Ns_list = [Ns_chunk + (1 if i < remainder else 0) for i in range(n_workers)]
    Ns_list = [n for n in Ns_list if n > 0]

    # Prepare worker arguments with different seeds
    args_list = []
    cur_seed = seed0
    for n in Ns_list:
        args_list.append((theta, n, cur_seed))
        cur_seed += 1

    # Run workers in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(_mc_worker, args_list)

    # Combine partial sums
    N_total = 0
    S_E_total = 0.0
    S_E2_total = 0.0
    S_r_total = 0.0
    S_Er_total = 0.0

    for (N_i, S_E_i, S_E2_i, S_r_i, S_Er_i) in results:
        N_total   += N_i
        S_E_total += S_E_i
        S_E2_total += S_E2_i
        S_r_total += S_r_i
        S_Er_total += S_Er_i

    # Averages and variances
    E_avg = S_E_total / N_total
    var_E = S_E2_total / N_total - E_avg**2
    E_std = np.sqrt(var_E / (r_eff * N_total))

    # Gradient via algebra (no need for full Els, R)
    # grad = 2/N * [ -sum(E r) + E_avg * sum(r) ]
    grad = (2.0 / N_total) * (-S_Er_total + E_avg * S_r_total)

    return grad, E_avg, E_std

def gradient_descent_theta_parallel(theta0, lr=0.01, n_steps=200, 
                                    grad_tol=1e-6, seed0=123,
                                    Ns=int(1e6), r_eff=0.053,
                                    n_workers=None,
                                    excel_filename=r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_data\H_atom_gd_history.xlsx"):
    """
    Gradient-descent optimization of theta using parallel MC estimates.
    Saves the history of (step, theta, E, E_std) to an Excel file.
    """
    theta = theta0

    theta_hist = []   # theta values at which we measured E
    E_list = []
    std_list = []

    for k in range(n_steps):
        # store current theta for this measurement
        theta_hist.append(theta)

        # change seed each iteration to decorrelate MC runs
        seed_k = seed0 + 1000 * k  

        g, E_avg, E_std = pd_H_theta_parallel(
            theta, Ns=Ns, r_eff=r_eff, n_workers=n_workers, seed0=seed_k
        )

        E_list.append(E_avg)
        std_list.append(E_std)

        print(f"step {k:3d}: lr = {lr:.4f}, theta = {theta:.6f}, "
              f"grad = {g:.6f}, E = {E_avg:.6f} ± {E_std:.6f}")

        # Stopping condition
        if abs(g) < grad_tol:
            print(f"Converged at step {k}: |grad| = {abs(g):.4e} < {grad_tol}")
            break

        # Gradient descent update
        theta = theta - lr * g
        if theta <= 0:
            theta = 1e-6

    # Convert lists to arrays for plotting
    theta_hist = np.array(theta_hist)
    E_arr = np.array(E_list)
    std_arr = np.array(std_list)

    # --- Save history to Excel ---
    df_history = pd.DataFrame({
        "step": np.arange(len(theta_hist)),
        "theta": theta_hist,
        "E": E_arr,
        "E_std": std_arr,
    })
    df_history.to_excel(excel_filename, index=False)
    print(f"History saved to {excel_filename}")

    # Now these have exactly the same length
    plt.figure(figsize=(7, 5))
    plt.errorbar(theta_hist, E_arr, yerr=std_arr, capsize=4)
    plt.plot(theta_hist, E_arr, 'o')
    plt.xlabel('theta')
    plt.ylabel(r'$\langle H \rangle$')
    plt.title('Theta vs. ⟨H⟩ (parallel MC)')
    plt.grid(True)
    plt.show()

    # g is still the gradient from the last iteration
    return theta, abs(lr * g), std_arr[-1]

#calculate error of theta
def second_derivative(theta, h=3.078e-3):
        g_plus,  _, _ = pd_H_theta_parallel(theta + h)
        g_minus, _, _ = pd_H_theta_parallel(theta - h)
        return (g_plus - g_minus) / (2*h)


if __name__ == "__main__":
    # theta_opt, last_step, E_std = gradient_descent_theta_parallel(
    #     theta0=0.9999,
    #     lr=0.01,
    #     n_steps=100,
    #     grad_tol=1e-4,
    #     Ns=int(1e6),
    #     r_eff=0.053,
    #     n_workers=6  # or None to use all cores
    # )
    # print("Optimal theta:", theta_opt)
    
    # Hpp = second_derivative(theta_opt)
    # sigma_theta = E_std / abs(Hpp)

    # print(sigma_theta)
    grad, E_avg, E_std = pd_H_theta_parallel(0.9999946)
    print(E_avg,E_std)
    
