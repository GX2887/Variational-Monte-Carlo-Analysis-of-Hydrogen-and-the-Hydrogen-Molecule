import numpy as np

# ----------------------------------------------------
# Objective and gradient for the test function
# f(x,y,z) = (x-1)^2 + 2(y-1)^2 + 3(z-1)^2
# ----------------------------------------------------
def f(x):
    """
    x : array-like, shape (3,)
        x = [x, y, z]
    """
    x1, y1, z1 = x
    return (x1 - 1.0)**2 + 2.0 * (y1 - 1.0)**2 + 3.0 * (z1 - 1.0)**2

def grad_f(x):
    """
    Gradient of f:
    ∂f/∂x = 2(x-1)
    ∂f/∂y = 4(y-1)
    ∂f/∂z = 6(z-1)
    """
    x1, y1, z1 = x
    return np.array([
        2.0 * (x1 - 1.0),
        4.0 * (y1 - 1.0),
        6.0 * (z1 - 1.0)
    ])


# ----------------------------------------------------
# Simple BFGS quasi-Newton implementation
# ----------------------------------------------------
def quasi_newton_bfgs(
        x0,
        max_iter=100,
        grad_tol=1e-6,
        verbose=True
    ):
    """
    Basic BFGS quasi-Newton optimizer with backtracking line search.
    """
    x = x0.astype(float)
    n = x.size

    # Initial inverse Hessian approximation = Identity
    H = np.eye(n)

    for k in range(max_iter):
        g = grad_f(x)

        # Check convergence
        if np.linalg.norm(g, ord=2) < grad_tol:
            if verbose:
                print(f"Converged at iter {k}")
                print(f"x = {x}")
                print(f"f(x) = {f(x):.8e}")
                print(f"||grad|| = {np.linalg.norm(g):.3e}")
            return x

        # BFGS search direction
        p = -H @ g

        # Backtracking line search (Armijo)
        alpha = 1.0
        c = 1e-4
        rho = 0.5
        f_curr = f(x)
        while f(x + alpha * p) > f_curr + c * alpha * np.dot(g, p):
            alpha *= rho
            if alpha < 1e-10:
                # step size too small, abort
                break

        # Update x
        s = alpha * p
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g

        # BFGS update: H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
        ys = np.dot(y, s)
        if ys > 1e-12:  # safeguard to avoid division by very small number
            rho_k = 1.0 / ys
            I = np.eye(n)
            Hy = H @ y
            H = (I - rho_k * np.outer(s, y)) @ H @ (I - rho_k * np.outer(y, s)) \
                + rho_k * np.outer(s, s)

        x = x_new

        if verbose:
            print(f"Iter {k:3d}: x = {x}, f(x) = {f(x):.6e}, ||grad|| = {np.linalg.norm(g):.3e}")

    print("Warning: did not converge within max_iter.")
    return x


# ----------------------------------------------------
# Run the verification
# ----------------------------------------------------
if __name__ == "__main__":
    # Start from a point away from the minimum
    x0 = np.array([0.0, 0.0, 0.0])
    x_opt = quasi_newton_bfgs(x0, max_iter=50, grad_tol=1e-6, verbose=True)

    print("\nFinal result:")
    print(f"x*  = {x_opt}")
    print(f"f(x*) = {f(x_opt):.10e}")
    print("Expected minimum at (1, 1, 1) with f = 0.")
