import s2_random_sampling as rs
import s1_differentiation as drt
import numpy as np
import matplotlib.pyplot as plt

# define 3d ground state wave fucntion with parameter theta
def psi(R,theta):
    r = np.linalg.norm(R)
    return np.exp(- theta * r)   # or whatever θ you’re using


def local_energy_batch(R_samples, theta):
    R_samples = np.asarray(R_samples)
    Ns = R_samples.shape[0]
    Els = np.empty(Ns)

    for i in range(Ns):
        R = R_samples[i]
        r = np.linalg.norm(R)
        psi_val = psi(R, theta)
        lap = drt.laplacian_3d_fd(psi, R, theta)
        Els[i] = -0.5 * lap / psi_val - 1.0 / r

    return Els

# Ns = int(1e8) # optimal step 1e4??
# theta = 1
# sample = rs.metropolis_sample_rho_3d(Ns,theta)
# r = 0.053 # efficiency ratio
# Els = local_energy_batch(sample,theta)
# E_avg = np.mean(Els)
# E_std = np.std(Els)/(np.sqrt(Ns*r))

# # reuslt of Monte Carlo intergation
# print(E_avg,E_std)

# # histogram plot
# xy = sample[:, :2]  # take x and y
# x = xy[:, 0]
# y = xy[:, 1]

# plt.hist2d(x, y, bins=700, density=True, cmap='inferno')
# plt.colorbar(label='Probability density')
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Projected density onto x-y plane')
# plt.show()

#find the partial derivative of theta
'''
The partial derivative result is -1/r. Hence, the finite differenciation
of theta can be simplified to -1/r
'''
def partial_derivative_theta(H_avg, Els, R):
    Ns = len(Els)
    pd = 0.0
    for i in range(Ns):
        r_i = np.linalg.norm(R[i])
        pd += (Els[i] - H_avg) * (-r_i) # analytical ∂ψ/ψ
    return 2 * pd / Ns

def pd_H_theta(theta, seed = 123, r = 0.053):
    np.random.seed(seed)
    Ns = int(1e6)
    sample = rs.metropolis_sample_rho_3d(Ns,theta)
    Els = local_energy_batch(sample,theta)
    E_avg = np.mean(Els)
    E_std = np.std(Els)/(np.sqrt(r*Ns))
    return partial_derivative_theta(E_avg, Els, sample), E_avg, E_std

# find the upb and lowb initial guess
# upb_test = pd_H_theta(0.9)
# lowb_test = pd_H_theta(1.1)

def gradient_descent_theta(theta0, lr=0.01, n_steps=200, 
                           grad_tol=1e-6, seed0=123):
    """
    Gradient-descent optimization of theta with fixed learning rate and
    gradient tolerance stopping condition.

    Parameters
    ----------
    theta0 : float
        Initial value of theta.
    lr : float
        Fixed learning rate.
    n_steps : int
        Maximum number of GD iterations.
    grad_tol : float
        Stopping tolerance on |grad|.
    seed0 : int
        Base random seed for MC sampling.

    Returns
    -------
    theta : float
        Final optimized theta.
    last_update : float
        Last parameter update magnitude |lr * grad|.
    """

    thetas = [theta0]
    theta = theta0
    E = []
    std = []

    for k in range(n_steps):
        seed_k = seed0 + k
        g, E_avg, E_std = pd_H_theta(theta, seed=seed_k)

        E.append(E_avg)
        std.append(E_std)

        # ---- Stopping condition: gradient small enough ----
        if abs(g) < grad_tol:
            print(f"Converged at step {k}: |grad| = {abs(g):.4e} < {grad_tol}")
            break

        # ---- Gradient descent update ----
        theta = theta - lr * g

        # keep theta positive
        if theta <= 0:
            theta = 1e-6

        thetas.append(theta)
        print(f"step {k:3d}: lr = {lr:.4f}, theta = {theta:.6f}, grad = {g:.6f}")

    # Convert lists to arrays for plotting
    E = np.array(E)
    std = np.array(std)

    plt.figure(figsize=(7, 5))
    plt.errorbar(thetas[1:], E, yerr=std, capsize=4, color='cornflowerblue')
    plt.plot(thetas[1:], E, 'o', color='cornflowerblue')
    plt.xlabel('theta')
    plt.ylabel(r'$\langle H \rangle$')
    plt.title('Theta vs. ⟨H⟩')
    plt.grid(True)
    plt.show()

    return theta, abs(lr * g)

theta, lr = gradient_descent_theta(0.8, n_steps=50)
# 0.2 0.9 (0.9999999946733874, 5.108660128388243e-10)
# 0.4 0.9 (0.9999999920468501, 2.79644223291062e-10)
# 0.4 0.9 (0.9999999947189114, 5.213973971935457e-10)


# use paralel calculation




