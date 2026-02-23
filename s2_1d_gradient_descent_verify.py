import numpy as np

def f(x):
    """Parabola function f(x) = (x-1)^2."""
    return (x - 1.0)**2

def grad_f(x):
    """Analytical gradient df/dx = 2(x-1)."""
    return 2.0 * (x - 1.0)

def gradient_descent(
        x0,
        lr=0.01,
        grad_tol=1e-5,
        max_iter=5000,
        verbose=True,
    ):
    """
    Simple 1D gradient descent verification.
    """
    x = x0
    for i in range(max_iter):
        g = grad_f(x)

        if abs(g) < grad_tol:
            if verbose:
                print(f"Converged at iteration {i}, x = {x:.8f}, gradient = {g:.2e}")
            return x

        x = x - lr * g

        if verbose and i % 100 == 0:
            print(f"Iter {i:4d}: x = {x:.8f}, f(x) = {f(x):.8f}, grad = {g:.2e}")

    print("Warning: did not converge within max_iter.")
    return x


# ----------------------------------------------------
# Run the test
# ----------------------------------------------------
x0 = 5.0  # starting far from minimum
x_opt = gradient_descent(x0, lr=0.01, grad_tol=1e-6, verbose=True)

print(f"\nFinal result: x ≈ {x_opt:.8f}")
print(f"Expected minimum at x = 1")
