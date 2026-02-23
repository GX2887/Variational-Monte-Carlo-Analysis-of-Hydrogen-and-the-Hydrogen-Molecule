import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fp = str(r'S3_H2\fitting_data\H2_E_vs_r_delta_0.75.xlsx')


def numerical_grad(fun, x, args=(), eps=3e-3):
    """
    Compute numerical gradient of fun at x using central differences.
    fun: function(x, *args) -> scalar
    x  : 1D numpy array
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_p2 = x.copy(); x_p2[i] += 2*eps
        x_p1 = x.copy(); x_p1[i] += eps
        x_m1 = x.copy(); x_m1[i] -= eps
        x_m2 = x.copy(); x_m2[i] -= 2*eps

        f_p2 = fun(x_p2, *args)
        f_p1 = fun(x_p1, *args)
        f_m1 = fun(x_m1, *args)
        f_m2 = fun(x_m2, *args)

        grad[i] = (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12 * eps)
    return grad

def chi2_minimize(fun, x0, args=(), lr=1, maxiter=100000, tol=1e-6, verbose=True):
    """
    Very simple gradient-descent minimizer.

    Parameters
    ----------
    fun      : function(x, *args) -> scalar
    x0       : initial guess (1D array-like)
    args     : extra arguments passed to fun
    lr       : learning rate (step size)
    maxiter  : maximum iterations
    tol      : stopping tolerance on gradient norm
    verbose  : print progress if True

    Returns
    -------
    result : dict with keys
             'x'      : best parameters found
             'fun'    : function value at x
             'niter'  : number of iterations
             'grad'   : gradient at x
             'success': True/False
             'message': str
    """
    x = np.asarray(x0, dtype=float)
    
    for it in range(maxiter):
        fval = fun(x, *args)
        grad = numerical_grad(fun, x, args=args)
        grad_norm = np.linalg.norm(grad)

        if verbose and (it % 10 == 0 or it == 0):
            print(f"iter {it:4d}: f = {fval:.8f}, |grad| = {grad_norm:.3e}, x = {x}")

        # stopping condition: gradient small
        if grad_norm < tol:
            return {
                'x': x,
                'fun': fval,
                'niter': it,
                'grad': grad,
                'success': True,
                'message': 'Converged: gradient norm below tol'
            }

        # gradient descent update
        x = x - lr * grad

    # if we exit loop without converging
    return {
        'x': x,
        'fun': fun(x, *args),
        'niter': maxiter,
        'grad': numerical_grad(fun, x, args=args),
        'success': False,
        'message': 'Maximum iterations reached'
    }

def morse_potential(r, D, a, r0, E_single):
    return D * (1 - np.exp(-a * (r - r0)))**2 - D + 2 * E_single

def chi2(params, r_data, V_data, E_single, sigma=None):
    """
    params = [D, a, r0]
    """
    D, a, r0 = params
    V_model = morse_potential(r_data, D, a, r0, E_single)

    if sigma is None:
        residuals = V_model - V_data
    else:
        residuals = (V_model - V_data) / sigma

    return np.sum(residuals**2)

def residuals(params, r_data, V_data, E_single, sigma=None):
    D, a, r0 = params
    V_model = morse_potential(r_data, D, a, r0, E_single)

    if sigma is None:
        return V_model - V_data
    else:
        return (V_model - V_data) / sigma

def estimate_param_std(x_opt, residual_func, r_data, V_data, E_single, sigma=None):
    """
    Compute covariance and standard deviations of fitted parameters.
    """
    # residuals at optimum
    res = residual_func(x_opt, r_data, V_data, E_single, sigma)
    N = len(res)
    p = len(x_opt)

    # variance of residuals
    sigma2 = np.sum(res**2) / (N - p)

    # build Jacobian J (N × p)
    eps = 1e-6
    J = []
    for i in range(p):
        step = np.zeros_like(x_opt)
        step[i] = eps

        r_plus  = residual_func(x_opt + step, r_data, V_data, E_single, sigma)
        r_minus = residual_func(x_opt - step, r_data, V_data, E_single, sigma)

        deriv = (r_plus - r_minus) / (2 * eps)  # N-vector
        J.append(deriv)

    J = np.array(J).T   # shape (N, p)

    # covariance matrix
    JTJ_inv = np.linalg.inv(J.T @ J)
    Cov = sigma2 * JTJ_inv

    # standard deviations
    std = np.sqrt(np.diag(Cov))

    return Cov, std

def fit_from_excel(fp):
    df = pd.read_excel(fp)

    r_vals   = df.iloc[:, 0].to_numpy(dtype=float)
    E_vals   = df.iloc[:, 1].to_numpy(dtype=float)
    E_stds   = df.iloc[:, 2].to_numpy(dtype=float)
    E_single = -0.4997
    sigma_data = None 
    x0 = np.array([0.15, 1.2, 1.4])  # initial guess [D, a, r0]

    result = chi2_minimize(
    chi2,
    x0,
    args=(r_vals, E_vals, E_single, sigma_data),
    lr=0.01,          # you may need to tune step size
    maxiter=200000,
    tol=1e-5,
    verbose=True
    )

    x_opt = result["x"]

    Cov, std = estimate_param_std(
    x_opt,
    residuals,        # NOT chi2 — we need residuals
    r_vals,
    E_vals,
    E_single,
    sigma_data
    )


    print("\n--- Result ---")
    print("success:", result['success'])
    print("message:", result['message'])
    print(f"D_fit = {x_opt[0]:.6f} ± {std[0]:.6f}")
    print(f"a_fit = {x_opt[1]:.6f} ± {std[1]:.6f}")
    print(f"r0_fit = {x_opt[2]:.6f} ± {std[2]:.6f}")
    print("chi2_min =", result['fun'])
    return None

fit_from_excel(fp)
