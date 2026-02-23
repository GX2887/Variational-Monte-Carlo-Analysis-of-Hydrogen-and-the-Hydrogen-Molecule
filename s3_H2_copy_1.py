import s2_random_sampling as rs
import s1_differentiation as drt
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from multiprocessing import Pool

def get_project_dir():
    """
    Returns the directory where the current script is located.
    This is your project root in VS Code.
    """
    return os.path.dirname(os.path.abspath(__file__))

def save_history(history, subfolder="S3_H2/history_gradient_search"):
    """
    Save history inside the project directory using a relative path.

    Example save path:
        <project_dir>/S3_H2/history_YYYY-MM-DD_HH-MM-SS.pkl
    """

    # Base project directory = folder where this script lives
    project_dir = get_project_dir()

    # Target directory: project/S3_H2/history_gradient_search
    save_dir = os.path.join(project_dir, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    # Time-based filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"history_{timestamp}.pkl"
    full_path = os.path.join(save_dir, filename)

    # Save
    with open(full_path, "wb") as f:
        pickle.dump(history, f)

    print(f"History saved to: {full_path}")
    return full_path

def load_history(filepath):
    """
    Load history back from a pkl file.
    """
    with open(filepath, "rb") as f:
        history = pickle.load(f)

    print(f"Loaded history from: {filepath}")
    return history

def psi_H2(R, theta, q1, q2 , eps = 1e-3):
    """
    H2 trial wavefunction ψ(r1,r2;θ).

    Parameters
    ----------
    R : array-like, shape (6,)
        [x1, y1, z1, x2, y2, z2]
    theta : array-like, length 3
        (theta1, theta2, theta3)
    q1, q2 : array-like, shape (3,)
        Positions of nuclei 1 and 2.

    Returns
    -------
    float
        ψ_H2(R; θ)
    """
    R = np.asarray(R, dtype=float)
    r1 = R[:3]
    r2 = R[3:]

    theta1, theta2, theta3 = theta

    # distances to nuclei
    eps = eps # add tiny cut off to avoid invalid sampling points
    # eps = 1e-3 is verified valid
    r1q1 = max(np.linalg.norm(r1 - q1), eps)
    r1q2 = max(np.linalg.norm(r1 - q2), eps)
    r2q1 = max(np.linalg.norm(r2 - q1), eps)
    r2q2 = max(np.linalg.norm(r2 - q2), eps)

    # electron–electron distance
    r12 = max(np.linalg.norm(r1 - r2) , eps)

    # symmetric part
    sym_part = np.exp(-theta1 * (r1q1 + r2q2)) + np.exp(-theta1 * (r1q2 + r2q1))

    # Jastrow-type correlation factor
    corr = np.exp(-theta2 / (1.0 + theta3 * r12))

    return sym_part * corr

def dlogpsi_dtheta(R, theta, q1, q2, eps = 1e-3):
    """
    Analytic derivatives ∂ lnψ / ∂θ_k, k=1,2,3.

    Returns
    -------
    dlogpsi : ndarray shape (3,)
        [∂ lnψ/∂θ1, ∂ lnψ/∂θ2, ∂ lnψ/∂θ3]
    """
    R = np.asarray(R, dtype=float)
    r1 = R[:3]
    r2 = R[3:]

    theta1, theta2, theta3 = theta

    # distances to nuclei
    eps = eps # add tiny cut off to avoid invalid sampling points
    # eps = 1e-3 is verified valid
    r1q1 = max(np.linalg.norm(r1 - q1), eps)
    r1q2 = max(np.linalg.norm(r1 - q2), eps)
    r2q1 = max(np.linalg.norm(r2 - q1), eps)
    r2q2 = max(np.linalg.norm(r2 - q2), eps)

    # electron–electron distance
    r12 = max(np.linalg.norm(r1 - r2) , eps)

    a = r1q1 + r2q2
    b = r1q2 + r2q1

    exp1 = np.exp(-theta1 * a)
    exp2 = np.exp(-theta1 * b)
    A = exp1 + exp2

    # ∂ lnψ / ∂θ1
    dlog_dtheta1 = (-a * exp1 - b * exp2) / A

    # ∂ lnψ / ∂θ2
    denom = 1.0 + theta3 * r12
    dlog_dtheta2 = -1.0 / denom

    # ∂ lnψ / ∂θ3
    dlog_dtheta3 = theta2 * r12 / (denom ** 2)

    return np.array([dlog_dtheta1, dlog_dtheta2, dlog_dtheta3])

def local_energy_H2_batch(R_samples, theta, q1, q2, h=2e-3, eps = 1e-3):
    """
    Local energy E_L(R) for the H2 molecule, evaluated for a batch of samples.

    Parameters
    ----------
    R_samples : array-like, shape (Ns, 6)
        Each row is [x1, y1, z1, x2, y2, z2].
    theta : array-like, length 3
        (theta1, theta2, theta3).
    q1, q2 : array-like, shape (3,)
        Nuclear positions.
    h : float
        Finite-difference step for the Laplacian.

    Returns
    -------
    Els : ndarray, shape (Ns,)
        Local energy values for each configuration.
    """
    R_samples = np.asarray(R_samples, dtype=float)
    Ns = R_samples.shape[0]
    Els = np.empty(Ns)

    # constant nuclear–nuclear repulsion
    R_nuc = np.linalg.norm(q1 - q2)
    V_nuc = 1.0 / R_nuc

    for i in range(Ns):
        R = R_samples[i]
        r1 = R[:3]
        r2 = R[3:]

        psi_val = psi_H2(R, theta, q1, q2, eps)

        # Laplacian with respect to r1 (holding r2 fixed)
        def psi_r1(r1_new, theta_local):
            R_full = np.concatenate([r1_new, r2])
            return psi_H2(R_full, theta_local, q1, q2, eps)

        lap1 = drt.laplacian_3d_fd(psi_r1, r1, theta, h)

        # Laplacian with respect to r2 (holding r1 fixed)
        def psi_r2(r2_new, theta_local):
            R_full = np.concatenate([r1, r2_new])
            return psi_H2(R_full, theta_local, q1, q2, eps)

        lap2 = drt.laplacian_3d_fd(psi_r2, r2, theta, h)

        lap_tot = lap1 + lap2

        # --- potential energy terms ---
        # distances to nuclei
        eps = eps # add tiny cut off to avoid invalid sampling points
        r1q1 = max(np.linalg.norm(r1 - q1), eps)
        r1q2 = max(np.linalg.norm(r1 - q2), eps)
        r2q1 = max(np.linalg.norm(r2 - q1), eps)
        r2q2 = max(np.linalg.norm(r2 - q2), eps)

        # electron–electron
        r12 = max(np.linalg.norm(r1 - r2),  eps)

        V_en = -(1.0 / r1q1 + 1.0 / r1q2 + 1.0 / r2q1 + 1.0 / r2q2)
        V_ee = 1.0 / r12

        V = V_en + V_ee + V_nuc

        Els[i] = -0.5 * lap_tot / psi_val + V

    return Els

def local_energy_chunk(args):
    R_chunk, theta, q1, q2, h, eps = args
    return local_energy_H2_batch(R_chunk, theta, q1, q2, h=h, eps=eps)

def estimate_mean_local_energy_parallel(Ns, theta, q1, q2,
                                        n_procs=14, h=2e-3, eps=1e-3):
    # one big sample set
    R_all = rs.metropolis_sample_H2(Ns, theta, q1, q2, delta=0.5, burn_in=200)

    # split into chunks
    R_chunks = np.array_split(R_all, n_procs)

    args = [(chunk, theta, q1, q2, h, eps) for chunk in R_chunks]

    with Pool(processes=n_procs) as pool:
        Els_list = pool.map(local_energy_chunk, args)

    Els = np.concatenate(Els_list)
    E_mean = np.mean(Els)
    E_std  = np.std(Els) / np.sqrt(len(Els))
    return E_mean, E_std



# gradient minimization for the local enenrgy
def estimate_mean_local_energy(theta, Ns, q1, q2,
                               delta=0.5, burn_in=200, h=2e-3,
                               seed=123,
                               eps = 1e-3):
    """
    One Monte Carlo estimate of <E_L> for given theta.
    """
    if seed is not None:
        np.random.seed(seed)

    # sample configurations from |ψ_H2(R;theta)|^2
    R_samples = rs.metropolis_sample_H2(Ns, theta, q1, q2,
                                     delta=delta, burn_in=burn_in)

    # compute local energies
    Els = local_energy_H2_batch(R_samples, theta, q1, q2, h=h, eps=eps)

    E_mean = np.mean(Els)
    E_std = np.std(Els) / np.sqrt(Ns)   # standard error of the mean
    return E_mean, E_std

def _chunk_local_energy_and_O(args):
    """
    Helper for multiprocessing: compute local energies and O = d log psi / d theta
    for a chunk of configurations.
    """
    R_chunk, theta, q1, q2, h, eps = args

    # local energies for this chunk
    Els_chunk = local_energy_H2_batch(R_chunk, theta, q1, q2, h=h, eps=eps)

    # O_k = ∂ lnψ / ∂θ_k for each sample in chunk
    Ns_chunk = R_chunk.shape[0]
    O_chunk = np.zeros((Ns_chunk, 3))
    for i in range(Ns_chunk):
        O_chunk[i, :] = dlogpsi_dtheta(R_chunk[i], theta, q1, q2)

    return Els_chunk, O_chunk

def energy_and_grad_analytic(theta, Ns, q1, q2,
                             delta=0.5, burn_in=200, h=2e-3,
                             seed=123, eps=1e-3,
                             n_procs=14):
    """
    Compute mean local energy E(θ) and its gradient ∇E(θ) using
    analytic d log ψ / dθ, parallelised over Monte Carlo samples.

    n_procs : number of worker processes (default: cpu_count()).
    """
    if seed is not None:
        np.random.seed(seed)

    # --- 1) Sample configurations from |ψ(θ)|^2 (still serial) ---
    R_samples = rs.metropolis_sample_H2(Ns, theta, q1, q2,
                                        delta=delta, burn_in=burn_in)

    # --- 2) Split samples into chunks for parallel processing ---
    R_chunks = np.array_split(R_samples, n_procs)
    # filter out possible empty chunks if Ns < n_procs
    R_chunks = [c for c in R_chunks if len(c) > 0]

    args_list = [(chunk, theta, q1, q2, h, eps) for chunk in R_chunks]

    # --- 3) Parallel map over chunks ---
    with Pool(processes=len(R_chunks)) as pool:
        results = pool.map(_chunk_local_energy_and_O, args_list)

    # --- 4) Reassemble Els and O from chunks ---
    Els_list, O_list = zip(*results)       # tuples of arrays
    Els = np.concatenate(Els_list)         # shape (Ns,)
    O   = np.vstack(O_list)                # shape (Ns, 3)

    # --- 5) Energy and gradient estimation ---
    E_mean = np.mean(Els)
    E_mean_std = np.std(Els) / np.sqrt(Ns)  # (this is std of local energies, not error on mean)

    diff = (Els - E_mean)[:, None]          # shape (Ns, 1)
    grad = 2.0 * np.mean(diff * O, axis=0)  # shape (3,)

    return E_mean, grad, E_mean_std


def gradient_search_min_energy_analytic(theta_init, q1, q2,
                                        Ns=int(1e4),
                                        n_steps=10,
                                        lr_vec = np.array([0.01, 1e-3, 1e-4]),
                                        delta=0.5,
                                        burn_in=200,
                                        h=2e-3,
                                        base_seed=123,
                                        tolerance = 1e-5,
                                        critical_step = 150):

    theta = np.asarray(theta_init, dtype=float)
    history = []

    #set stopping condition
    E_prev = -1000 # set particular starting condition(meaningless)
    for step in range(n_steps):

        seed = base_seed + 1000 * step

        E, grad, E_mean_std = energy_and_grad_analytic(theta, Ns, q1, q2,
                             delta=0.5, burn_in=burn_in, h=h,
                             seed=seed,
                             eps = 1e-3)
        
        # clip the gradient to avoid explosion
        max_norm = 2.0
        g_norm = np.linalg.norm(grad)
        if g_norm > max_norm:
            grad = grad * (max_norm / g_norm)

        # if step > 200:
        #     E_prev = history[-1]['E']
        #     if np.abs(E - E_prev) < E_std:
        #         print(f"Converged minimum found at step {step:3d}: E = {E: .6f}, θ = {theta}, |∇E| = {np.linalg.norm(grad):.4e}, std = {E_std:.3f} ")
        #         save_history(history)
        #         return theta, history
        # gradient descent update
        # if step > critical_step:
        #     lr = lr/(1 + 0.5*(step-critical_step))

        theta = theta - lr_vec * grad
        theta[1] = np.clip(theta[1], 0.0, 0.8)  # θ2
        theta[2] = np.clip(theta[2], 0.0, 1.0)  # θ3

        if step > 10 and np.linalg.norm(lr_vec * grad) < tolerance:
            print(f"Converged minimum found at step {step:3d}: E = {E: .6f}, θ = {theta}, |∇E| = {np.linalg.norm(grad):.4e}, std = {E_mean_std:.3f} ")
            save_history(history)
            return theta, history

        # (optional) enforce positivity for θ1, θ3 (physically sensible)
        history.append({"step": step,
                        "theta": theta.copy(),
                        "E": E,
                        "grad": grad.copy()})

        print(f"step {step:3d}: E = {E: .6f}, θ = {theta}, |∇E| = {np.linalg.norm(grad):.4e}")
    print(f'{n_steps} finished')
    save_history(history)
    return theta, history

def histogram_plot(history):
    theta1_vals = np.array([h["theta"][0] for h in history])
    E_vals      = np.array([h["E"] for h in history])

    plt.figure()
    plt.scatter(theta1_vals, E_vals, marker='o')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel('Energy E')
    plt.title(r'E vs $\theta_1$ (gradient search)')
    plt.grid(True)
    plt.show()

    # mask = (E_vals >= -10) & (E_vals <= -1)
    # E_filtered = E_vals[mask]

    plt.figure()
    plt.hist(E_vals, bins=25) # E_filtered/E_vals
    plt.xlabel('Energy E')
    plt.ylabel('Count')
    plt.title('Histogram of Energy Values')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_theta_history(history):
    """
    Plot theta1, theta2, theta3 against step number using subplots.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract values from history
    steps  = np.array([h["step"]  for h in history])
    theta1 = np.array([h["theta"][0] for h in history])
    theta2 = np.array([h["theta"][1] for h in history])
    theta3 = np.array([h["theta"][2] for h in history])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Plot theta1
    axes[0].plot(steps, theta1, marker='o')
    axes[0].set_ylabel("theta1")
    axes[0].set_title("Theta Parameters vs Step Number")
    axes[0].grid(True)

    # Plot theta2
    axes[1].plot(steps, theta2, marker='o')
    axes[1].set_ylabel("theta2")
    axes[1].grid(True)

    # Plot theta3
    axes[2].plot(steps, theta3, marker='o')
    axes[2].set_xlabel("Step Number")
    axes[2].set_ylabel("theta3")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return fig

def plot_energy_history(history):
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.array([h["step"] for h in history])
    Es    = np.array([h["E"]    for h in history])

    plt.figure(figsize=(7,4))
    plt.plot(steps, Es, marker='o')
    plt.xlabel("Step")
    plt.ylabel("Energy <H>")
    plt.title("Energy vs Step")
    plt.grid(True)
    plt.show()

def plot_grad_norm(history):
    import matplotlib.pyplot as plt
    import numpy as np

    steps = np.array([h["step"] for h in history])
    grad_norm = np.array([np.linalg.norm(h["grad"]) for h in history])

    plt.figure(figsize=(7,4))
    plt.semilogy(steps, grad_norm, marker='o')
    plt.xlabel("Step")
    plt.ylabel("||grad E||")
    plt.title("Gradient Norm vs Step")
    plt.grid(True)
    plt.show()


def check_cut_off_effect():
    # check the influence of cutoff in ets (r1r2, r1q1...)
    cutoffs = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
    E_vals = []
    err_vals = []

    for eps in cutoffs:
        E, std = estimate_mean_local_energy(theta=[1,1,1], 
                                            Ns=40000, 
                                            q1=q1, q2=q2,
                                            eps=eps,   # <-- modify your code to use this
                                            seed=123)
        E_vals.append(E)
        err_vals.append(std)
        print(f"eps={eps}:  E={E:.6f} ± {std:.6f}")
    
    delta_E = max(E_vals) - min(E_vals)
    print(f'delta_E_cutoff = {delta_E}, sigma_MC = {err_vals[0]}')
    plt.figure()
    plt.errorbar(cutoffs, E_vals, yerr=err_vals, fmt='o-', capsize=4)
    plt.xscale('log')
    plt.xlabel("cutoff eps")
    plt.ylabel("Energy E")
    plt.title("Convergence of E with cutoff")
    plt.grid(True)
    plt.show()

    # grad_list = []
    # for eps in cutoffs:
    #     E, grad = energy_and_grad_analytic(theta=[1,1,1], 
    #                                    Ns=20000,
    #                                    q1=q1, q2=q2,
    #                                    eps=eps,
    #                                    seed=123)
    #     grad_list.append(grad)
    #     print(f"eps={eps}, grad={grad}")
    
    # grad_list = np.array(grad_list)
    # cutoffs = np.array(cutoffs)
    # abs2_gradient = []
    # for i in range(5):
    #     abs2_gradient.append(grad_list[i,:]@grad_list[i,:])
    # abs2_gradient = np.array(abs2_gradient)

    # plt.figure(figsize=(6,4))
    # plt.plot(cutoffs, abs2_gradient, marker="o")
    # plt.xlabel("Cutoff ε")
    # plt.ylabel("Gradient component")
    # plt.title("Gradient vs cutoff ε")
    # plt.grid(True)
    # plt.show()

def El_theta1_scan(theta, q1, q2, Ns=int(1e4), seed=123, step = 101, l= -3.0 , r = 3.0):
    '''
    To determine the dependenc of H on theta 1 for given
    q1 = np.array([-dist/2, 0, 0])   # R0 = internuclear distance
    q2 = np.array([ dist/2, 0, 0])
    '''
    theta1 = np.linspace(l, r, int(step))
    theta2_fixed = theta[1]
    theta3_fixed = theta[2]
    theta_ref = np.array([2.0, theta2_fixed, theta3_fixed])
    R_ref = rs.metropolis_sample_H2(Ns, theta_ref, q1, q2,
                             delta=0.5, burn_in=200)
    E = []
    std = []
    j = 1
    for i in theta1:
        theta_curr = np.array([i, theta2_fixed, theta3_fixed])
        # seed += 10*j
        E_mean, E_std = estimate_mean_local_energy(
        theta_curr, theta_ref, R_ref, q1, q2, h=2e-3, eps=1e-3,
        seed=seed)

        E.append(E_mean)
        std.append(E_std)
        print(f'Step {j} Theta 1 = {i:.3f}, <H> = {E_mean:.6f}')
        j+=1
    
    E = np.array(E)
    std = np.array(std)
    fig = plt.figure(figsize=(7,5))
    plt.plot(theta1, E, 'o', color = 'royalblue')
    plt.errorbar(theta1, E, yerr=std, capsize=4)
    plt.xlabel('theta1')
    plt.ylabel('<H>')
    plt.title('<H> vs Theta1 (theta2,3 = 1)')
    plt.grid(True)
    plt.show()
    return fig

def El_theta2_scan(theta, q1, q2, Ns=int(1e5), seed=123, step = 101, l= -3.0 , r = 0):
    """
    Scan dependence of <H> on theta2 in range [-1, 1],
    keeping theta1 and theta3 fixed from the input theta.

    theta: initial guess array-like [theta1, theta2, theta3]
    """
    theta2_vals = np.linspace(l, r, int(step))
    theta1_fixed = theta[0]         # use current best guess
    theta3_fixed = theta[2]          # use current best guess

    # # Reference theta: use theta2 initial guess = 0
    # theta_ref = np.array([theta1_fixed, 0.0, theta3_fixed])

    # # Reference samples
    # R_ref = rs.metropolis_sample_H2(
    #     Ns, theta_ref, q1, q2, delta=0.5, burn_in=200
    # )

    E = []
    std = []
    j = 1

    for t2 in theta2_vals:
        seed = seed +100*j
        theta_curr = np.array([theta1_fixed, t2, theta3_fixed])
        E_mean, E_std = estimate_mean_local_energy_parallel(
            Ns, theta_curr, q1, q2,
            n_procs=14, h=2e-3, eps=1e-3
            )
        E.append(E_mean)
        std.append(E_std)
        print(f"Step {j:3d} Theta2 = {t2:.3f}, <H> = {E_mean:.6f}")
        j += 1

    E = np.array(E)
    std = np.array(std)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(theta2_vals, E, 'o', color='royalblue')
    plt.errorbar(theta2_vals, E, yerr=std, capsize=4)
    plt.xlabel('theta2')
    plt.ylabel('<H>')
    plt.title('<H> vs theta2 (theta1,3 fixed)')
    plt.grid(True)
    plt.show()

    return fig

def El_theta3_scan(theta, q1, q2, Ns=int(1e5), seed=123, step = 101, l= 1.0 , r = 2.0):
    """
    Scan dependence of <H> on theta3 in range [1, 2],
    keeping theta1 and theta2 fixed from the input theta.

    theta: initial guess array-like [theta1, theta2, theta3]
    """
    theta3_vals = np.linspace(l, r, int(step))
    theta1_fixed = theta[0]          # use current best guess
    theta2_fixed = theta[1]          # use current best guess

    # Reference theta: use theta3 initial guess = 1.5
    # theta_ref = np.array([theta1_fixed, theta2_fixed, 1.5])

    # Reference samples
    # R_ref = rs.metropolis_sample_H2(
    #     Ns, theta_ref, q1, q2, delta=0.5, burn_in=200
    # )

    E = []
    std = []
    j = 1
    for t3 in theta3_vals:
        theta_curr = np.array([theta1_fixed, theta2_fixed, t3])

        E_mean, E_std = estimate_mean_local_energy_parallel(Ns, theta_curr, q1, q2,
                                        n_procs=14, h=2e-3, eps=1e-3)

        # E_mean, E_std = estimate_mean_local_energy(
        #     theta_curr, Ns, q1, q2,
        #     h=2e-3, eps=1e-3, seed=seed
        # )
        E.append(E_mean)
        std.append(E_std)
        print(f"Step {j} Theta3 = {t3:.3f}, <H> = {E_mean:.6f}")
        j += 1

    E = np.array(E)
    std = np.array(std)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(theta3_vals, E, 'o', color='royalblue')
    plt.errorbar(theta3_vals, E, yerr=std, capsize=4)
    plt.xlabel('theta3')
    plt.ylabel('<H>')
    plt.title('<H> vs theta3 (theta1,2 fixed)')
    plt.grid(True)
    plt.show()

    return fig

if __name__ == "__main__":
    # H2 system parameter
    dist = 2
    theta1 = 1.5
    theta2 = 0.355
    theta3 = 0.55 #6



    # nuclei positions (example: aligned on x-axis)
    q1 = np.array([-dist/2, 0, 0])   # R0 = internuclear distance
    q2 = np.array([ dist/2, 0, 0])

    theta = np.array([theta1, theta2, theta3])

    # get_project_dir()
    theta_mini, history = gradient_search_min_energy_analytic(theta, q1, q2, Ns=int(1e5), n_steps=20, lr_vec = np.array([0.01, 1e-3, 1e-3]))
    # history = load_history('D:/OneDrive - Imperial College London/Y3 Computing/project/S3_H2/history_gradient_search/history_2025-11-27_07-37-56.pkl')
    histogram_plot(history)
    plot_theta_history(history)
    plot_energy_history(history)
    plot_grad_norm(history)



    # El_theta1_scan(theta, q1, q2, step = 20)
    # El_theta2_scan(theta, q1, q2, step = 20 , l = -0.2, r=0.2)
    # El_theta3_scan(theta, q1, q2, Ns=int(1e4), step = 100, l = 0, r = 4)






'''
# # random sample generation
# Ns = int(1e6) # numebr of sample
# samples = rs.metropolis_sample_H2(Ns, theta, q1, q2, delta=0.5)

# # local energy
# Els = local_energy_H2_batch(samples, theta, q1, q2)
# E_avg = np.mean(Els) # H_avg
# E_std = np.std(Els)
# print(E_avg)

# histogram plot
# x1 = samples[:,0]
# x2 = samples[:,3]
# y1 = samples[:,1]
# y2 = samples[:,4]

# x = np.concatenate([x1, x2], axis=0)
# y = np.concatenate([y1, y2], axis=0)

# plt.hist2d(x, y, bins=200, density=True, cmap='inferno')
# plt.colorbar(label='Probability density')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Projected density onto x-y plane')
# plt.show()
'''