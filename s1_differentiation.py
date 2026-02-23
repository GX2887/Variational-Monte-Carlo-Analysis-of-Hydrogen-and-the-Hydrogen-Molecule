import numpy as np
import matplotlib.pyplot as plt

#%%
#definition of the function

def H_0(x):
    return 1

def H_1(x):
    return 2*x

def H_2(x):
    return 4*x**2-2

def H_3(x):
    return 8*x**3-12*x

def psi_0(x):
    return H_0(x)*np.exp(-x**2/2)



def diff_4th_order(f, x):
    """
    Forth-order central difference for second derivative
    Uses 5-point stencil with O(h^3) accuracy
    """
    diff = []
    for i in range(len(x)):
        if i <= 1 or i >= len(x)-2:
            # Skip first two and last two terms
            continue
        else:
            h = x[i+1] - x[i]
            print(h)
            # Third-order central difference for second derivative
            diff_i = (-f(x[i-2]) + 16*f(x[i-1]) - 30*f(x[i]) + 16*f(x[i+1]) - f(x[i+2])) / (12*h**2)
            diff.append(diff_i)
    diff = np.array(diff)
    return diff

def d2_fd_4th_order(f, x, h=3.078e-3):
    """
    Forth-order central difference for second derivative

    Parameters
    ----------
    x : flot
        Sample point from PDF related random generation.
    h : float
        Optimize (roughly) step leghth of smallest error
    
    """
    return (-f(x+2*h) + 16*f(x+h) - 30*f(x) +
            16*f(x-h) - f(x-2*h)) / (12*h**2)

def laplacian_3d_fd(psi_func, R, theta, h=3.078e-3):
    
    """
    Forth-order central difference for second derivative in 3d

    Parameters
    ----------
    R : arrary
        Sample point from PDF related random generation.
    h : float
        Optimize (roughly) step leghth of smallest error
    thta : float
        Parameter for optimization
    """
    
    x, y, z = R

    def fx(x_new):
        return psi_func(np.array([x_new, y, z]), theta)

    def fy(y_new):
        return psi_func(np.array([x, y_new, z]), theta)

    def fz(z_new):
        return psi_func(np.array([x, y, z_new]), theta)

    d2x = d2_fd_4th_order(fx, x, h)
    d2y = d2_fd_4th_order(fy, y, h)
    d2z = d2_fd_4th_order(fz, z, h)

    return d2x + d2y + d2z

def laplacian_3d_fd_batch(psi_func, R_samples, theta, h=3.078e-3):
    """
    Compute ∇²ψ for many 3D positions.

    Parameters
    ----------
    psi_func   : callable psi_func(R, theta)
    R_samples  : array shape (Ns, 3)
    theta      : float
    h          : float  (FD step)

    Returns
    -------
    laps : array shape (Ns,)
        Laplacian at each sample point.
    """
    R_samples = np.asarray(R_samples)
    Ns = R_samples.shape[0]
    laps = np.empty(Ns)

    for i in range(Ns):
        laps[i] = laplacian_3d_fd(psi_func, R_samples[i], theta, h)

    return laps

def d1_fd_4th_order(f, x, h=3.078e-3):
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h)) / (12*h)


def d1_fd_partial_theta(psi, R, theta, h=3.078e-3):
    x, y, z = R
    def f_theta(theta_new):
        return psi(np.array[x,y,z],theta_new)
    
    return d1_fd_4th_order(f_theta,theta,h)

def diff_6th_order(f, x):
    """
    6th-order central difference for second derivative
    Uses 7-point stencil with O(h^6) accuracy
    Coefficients from standard 7-point central stencil.
    """
    diff = []
    for i in range(len(x)):
        if i <= 2 or i >= len(x)-3:
            # need points i-3 ... i+3
            continue
        else:
            h = x[i+1] - x[i]
            diff_i = ( (1/90)*f(x[i-3])
                       - (3/20)*f(x[i-2])
                       + (3/2)*f(x[i-1])
                       - (49/18)*f(x[i])
                       + (3/2)*f(x[i+1])
                       - (3/20)*f(x[i+2])
                       + (1/90)*f(x[i+3]) ) / h**2
            diff.append(diff_i)
    return np.array(diff)

def true_2nd_diff_psi_H_0(x):
    """
    analytic second derivative of psi_0(x)
    """
    return -np.exp(-x**2/2) + x**2*np.exp(-x**2/2)

#optimization for the step size

#2nd order
def diff_2nd_order(f, h, x):
    """
    Second-order central difference for second derivative:
        f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    O(h^2) truncation error.
    """
    x = np.asarray(x)
    return (f(x + h) - 2.0*f(x) + f(x - h)) / (h**2)

def error_2nd_2(h,x):
    test_diff_psi_H_0 = diff_2nd_order(psi_0,h,x)
    test_true_diff_2nd_order_psi_H_0 = true_2nd_diff_psi_H_0(x)
    diff = np.abs(test_diff_psi_H_0 - test_true_diff_2nd_order_psi_H_0)
    diff_avg = np.average(diff)
    return diff_avg

#4th order
def diff_2nd_order_4(f, h, x):
    """
    Fourth-order central difference for f''(x)
    5-point stencil, O(h^4) truncation error.
    """
    x = np.asarray(x)
    return (
        -f(x + 2*h) + 16*f(x + h) - 30*f(x)
        + 16*f(x - h) - f(x - 2*h)
    ) / (12*h**2)

def error_2nd_4(h, x):
    """
    Computes average absolute error of the 4th-order finite-difference
    approximation for the second derivative of psi_0.
    """
    test_diff = diff_2nd_order_4(psi_0, h, x)
    true_diff = true_2nd_diff_psi_H_0(x)

    diff = np.abs(test_diff - true_diff)
    diff_avg = np.mean(diff)
    return diff_avg

#6th order
def diff_2nd_order_6(f, h, x):
    """
    Sixth-order central difference for f''(x)
    7-point stencil, O(h^6) truncation error.
    """
    x = np.asarray(x)
    return (
        2*f(x - 3*h)
        - 27*f(x - 2*h)
        + 270*f(x - h)
        - 490*f(x)
        + 270*f(x + h)
        - 27*f(x + 2*h)
        + 2*f(x + 3*h)
    ) / (180*h**2)

def error_2nd_6(h, x):
    """
    Computes average absolute error of the 6th-order finite-difference
    approximation for the second derivative of psi_0.
    """
    test_diff = diff_2nd_order_6(psi_0, h, x)
    true_diff = true_2nd_diff_psi_H_0(x)

    diff = np.abs(test_diff - true_diff)
    diff_avg = np.mean(diff)
    return diff_avg


#1d first order derivative
def grad_1d(f, h, x, rel_step=1e-2):
    # choose a step proportional to h
    dh = rel_step * abs(h)
    if dh == 0.0:
        dh = rel_step  # fallback if h is exactly 0

    # ensure we don't cross h <= 0
    if h - dh <= 0:
        dh = 0.5 * h   # keep both points positive

    f_p = f(h + dh, x)
    f_m = f(h - dh, x)
    return (f_p - f_m) / (2.0 * dh)

def gradient_descent_1d(f, x,
                        h_init=1e-1,
                        lr=1.0,
                        max_iter=20_000,
                        h_min=1e-8,
                        h_max=1.0,
                        osc_window=100,
                        osc_grad_plateau=1e-5,  # ~ 5× typical grad noise
                        osc_rel_err_band=5e-2, # e.g. 5% band is "flat enough"
                        tol_grad=1e-6          # simple small-gradient stop
                        ):
    """
    Minimise f(h, x) over scalar h using gradient descent.

    Stopping criteria:
      - |grad| < tol_grad  (local small-gradient condition), OR
      - Over the last `osc_window` steps:
          * max |grad| < osc_grad_plateau, AND
          * relative error band (max - min)/mean < osc_rel_err_band,
        i.e. plateau / pseudo-oscillation around a flat minimum; OR
      - max_iter reached.

    Parameters
    ----------
    f : callable
        Objective f(h, x) -> scalar error.
    x : array-like
        Grid points for evaluating the error.
    h_init : float
        Initial step size.
    lr : float
        Learning rate in h-space.
    max_iter : int
        Maximum number of iterations.
    h_min, h_max : float
        Bounds for h (clipped after each update).
    osc_window : int
        Number of last steps used for plateau detection.
    osc_grad_plateau : float
        Threshold for max |grad| in window to be considered "flat".
    osc_rel_err_band : float
        Threshold for relative error band = (max-min)/mean in window.
    tol_grad : float
        Immediate small-gradient stopping threshold (per step).

    Returns
    -------
    h_opt : float
        Step size at stopping.
    f_opt : float
        Corresponding error f(h_opt, x).
    """

    h = float(h_init)

    grad_hist = []
    err_hist  = []
    h_hist    = []

    for i in range(max_iter):

        h_hist.append(h)
        
        g = grad_1d(f, h, x)   # df/dh
        f_curr = f(h, x)

        grad_hist.append(g)
        err_hist.append(f_curr)

        # # 1) simple local small-gradient stop
        # if abs(g) < tol_grad and i > 10:
        #     print(f"[converged: small grad] step {i}, h = {h:.3e}, "
        #           f"grad = {g:.3e}, error = {f_curr:.3e}")
        #     return h, f_curr

        # 2) plateau detection over last osc_window steps
        if i > 1000 and len(grad_hist) >= osc_window:
            g_win = np.array(grad_hist[-osc_window:])
            f_win = np.array(err_hist[-osc_window:])
            h_win = np.array(h_hist[-osc_window:])

            # use *mean* gradient magnitude over window
            mean_grad = np.mean(np.abs(g_win))

            f_max = np.max(f_win)
            f_min = np.min(f_win)
            f_mean = max(np.mean(np.abs(f_win)), 1e-16)
            rel_band = (f_max - f_min) / f_mean

            if mean_grad < osc_grad_plateau and rel_band < osc_rel_err_band:
                h_opt = np.mean(h_win)
                h_std = np.std(h_win)
                err_mean = np.mean(err_hist)
                err_std = np.std(err_hist)
                print(f"[converged: plateau] step {i}, "
                      f"h_opt = {h_opt:.3e} ± {h_std:.3e}, "
                      f"mean_grad = {mean_grad:.3e}, rel_band = {rel_band:.3e}, "
                      f"error = {f_curr:.3e}")
                return h_opt, h_std, err_mean, err_std

        # 3) gradient step in h
        h_new = h - lr * g
        h_new = float(np.clip(h_new, h_min, h_max))

        if i % 50 == 0:
            print(f"step {i}: h = {h:.3e}, grad = {g:.3e}, error = {f_curr:.3e}")

        h = h_new

    # 4) max iterations reached
    f_curr = f(h, x)
    if len(h_hist) >= osc_window:
        h_win = np.array(h_hist[-osc_window:])
        h_opt = np.mean(h_win)
        h_std = np.std(h_win)
        err_mean = np.mean(err_hist)
        err_std = np.std(err_hist)
    else:
        h_opt = h
        h_std = 0.0
    print(f"[max_iter reached] h ≈ {h_opt:.3e} ± {h_std:.3e}, error = {f_curr:.3e}")
    return h_opt, h_std, err_mean, err_std

def gradient_descent_logh(f, x,
                          h_init=1e-1,
                          lr_s=0.5,          # learning rate in log-space
                          tol_grad=1e-5,     # tolerance on |dE/dh|
                          tol_rel_h=1e-8,    # relative change in h
                          tol_rel_f=1e-10,   # relative change in f
                          max_iter=20_000,
                          h_min=1e-8,
                          h_max=1.0,
                          osc_window=50,
                          osc_min_sign_flips=5,
                          osc_rel_err_band=1e-3):
    """
    Minimise f(h, x) over h by gradient descent in s = log(h).

    Uses:
      - dE/dh from grad_1d
      - dE/ds = h * dE/dh
    """

    # initialise in log-space
    h_init = float(np.clip(h_init, h_min, h_max))
    s = np.log(h_init)

    h = np.exp(s)
    f_prev = f(h, x)
    h_prev = h

    grad_hist = []
    err_hist  = []

    for i in range(max_iter):
        h = float(np.clip(np.exp(s), h_min, h_max))

        g_h = grad_1d(f, h, x)     # dE/dh
        g_s = g_h * h              # dE/ds = h * dE/dh
        f_curr = f(h, x)

        grad_hist.append(g_h)
        err_hist.append(f_curr)

        # # 1) original "small grad" condition (on dE/dh)
        # if abs(g_h) < tol_grad:
        #     print(f"[converged: small grad] step {i}, h = {h:.3e}, grad = {g_h:.3e}, error = {f_curr:.3e}")
        #     return h, f_curr

        if i > 10:
            # # 2) flat region: small rel change in h and f
            # rel_h = abs(h - h_prev) / max(abs(h), abs(h_prev), 1.0)
            # rel_f = abs(f_curr - f_prev) / max(abs(f_prev), 1e-16)
            # if rel_h < tol_rel_h and rel_f < tol_rel_f:
            #     print(f"[converged: flat] step {i}, h = {h:.3e}, grad = {g_h:.3e}, error = {f_curr:.3e}")
            #     return h, f_curr

            # 3) oscillation detection (same logic as before, on grad_hist / err_hist)
            if len(grad_hist) >= osc_window:
                g_win = np.array(grad_hist[-osc_window:])
                f_win = np.array(err_hist[-osc_window:])

                signs = np.sign(g_win)
                nonzero = signs != 0
                if np.any(nonzero):
                    signs_nz = signs[nonzero]
                    if len(signs_nz) > 1:
                        sign_flips = np.sum(signs_nz[1:] * signs_nz[:-1] < 0)
                    else:
                        sign_flips = 0
                else:
                    sign_flips = 0

                f_max = np.max(f_win)
                f_min = np.min(f_win)
                f_mean = max(np.mean(np.abs(f_win)), 1e-16)
                rel_band = (f_max - f_min) / f_mean

                if sign_flips >= osc_min_sign_flips and rel_band < osc_rel_err_band:
                    print(f"[converged: oscillation] step {i}, h = {h:.3e}, grad = {g_h:.3e}, "
                          f"error = {f_curr:.3e}, sign_flips = {sign_flips}, rel_band = {rel_band:.3e}")
                    return h, f_curr

        # gradient step in s = log(h)
        s_new = s - lr_s * g_s    # note: g_s, not g_h
        # enforce bounds in h via s
        h_new = float(np.clip(np.exp(s_new), h_min, h_max))
        s_new = np.log(h_new)

        if i % 50 == 0:
            print(f"step {i}: h = {h:.3e}, grad = {g_h:.3e}, error = {f_curr:.3e}")

        s = s_new

    # max iterations
    h = float(np.clip(np.exp(s), h_min, h_max))
    f_curr = f(h, x)
    print(f"[max_iter reached] h = {h:.3e}, error = {f_curr:.3e}")
    return h, f_curr

def second_order():
    x = np.linspace(0, 2, int(1e4))
    h_opt, h_std, err_mean, err_std = gradient_descent_1d(error_2nd_2, x,
                                        h_init=1e-4,
                                        lr=0.1,
                                        osc_grad_plateau=1e-5,
                                        )
    print("2nd order: h_opt =", h_opt, "±", h_std)
    print("avg error =",err_mean, "±",err_std )

def forth_order():
    x = np.linspace(0, 2, int(1e4))
    h_opt, h_std, err_mean, err_std = gradient_descent_1d(error_2nd_4, x,
                                        h_init=3e-3,
                                        osc_grad_plateau=1e-9,
                                        lr=100)
    print("4th order: h_opt =", h_opt, "±", h_std)
    print("avg error =",err_mean, "±",err_std )

def sixth_order():
    x = np.linspace(0, 2, int(1e4))
    h_opt, h_std, err_mean, err_std = gradient_descent_1d(error_2nd_6, x,
                                        h_init=0.01155,
                                        osc_grad_plateau=1e-6,
                                        lr=100)
    print("6th order: h_opt =", h_opt, "±", h_std)
    print("avg error =",err_mean, "±",err_std )

# second_order()
# forth_order()
# sixth_order()

def scan_step_error(x, num_points=200):
    """
    Scan error vs step size for h in [1, 1e-8] on log scale.
    Computes 2nd, 4th, and 6th order truncation errors and plots them.
    """

    # Log-spaced step sizes
    h_vals = np.logspace(0, -6, num_points)  # 1 → 1e-8

    err2 = np.array([error_2nd_2(h, x)   for h in h_vals])
    err4 = np.array([error_2nd_4(h, x) for h in h_vals])
    err6 = np.array([error_2nd_6(h, x) for h in h_vals])

    # Plot
    plt.figure(figsize=(6, 4)) #fig size
    plt.loglog(h_vals, err2,'-o',markersize=3, label="2nd order (O(h²))")
    plt.loglog(h_vals, err4,'-o',markersize=3, label="4th order (O(h⁴))")
    plt.loglog(h_vals, err6,'-o',markersize=3, label="6th order (O(h⁶))")

    # Mark minima
    plt.scatter([h_vals[np.argmin(err2)]], [np.min(err2)], color='black', s=30)
    plt.scatter([h_vals[np.argmin(err4)]], [np.min(err4)], color='black', s=30)
    plt.scatter([h_vals[np.argmin(err6)]], [np.min(err6)], color='black', s=30)

    plt.xlabel("Step size h")
    plt.ylabel("Error")
    # plt.title("Truncation + Roundoff Error vs Step Size")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
    r"D:\OneDrive - Imperial College London\Y3 Computing\project\report_writing_plot\error_vs_h.png",
    dpi=600,
    bbox_inches='tight'
    )
    plt.show()

    # Return minima for convenience
    return {
        "h_min_2": h_vals[np.argmin(err2)], "err_min_2": np.min(err2),
        "h_min_4": h_vals[np.argmin(err4)], "err_min_4": np.min(err4),
        "h_min_6": h_vals[np.argmin(err6)], "err_min_6": np.min(err6),
    }

# plotting
# x = np.linspace(0, 2, int(1e4))
# results = scan_step_error(x)
# print(results)
