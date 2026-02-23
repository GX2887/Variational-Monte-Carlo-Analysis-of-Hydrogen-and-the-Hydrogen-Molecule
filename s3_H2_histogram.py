import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import os
from multiprocessing import Pool, cpu_count

import s2_random_sampling as rs
import s1_differentiation as drt
import s3_H2_copy_numba_quasi_newton as hm

# ------------------ Numba 2D HISTOGRAM ------------------ #

@njit(parallel=True)
def hist2d_parallel(x, y, xmin, xmax, ymin, ymax, nbins):
    """
    Parallel 2D histogram (counts only).
    """
    H = np.zeros((nbins, nbins), dtype=np.int64)

    dx = (xmax - xmin) / nbins
    dy = (ymax - ymin) / nbins

    N = x.shape[0]

    for i in prange(N):
        xi = x[i]
        yi = y[i]

        if (xi >= xmin) and (xi < xmax) and (yi >= ymin) and (yi < ymax):
            ix = int((xi - xmin) / dx)
            iy = int((yi - ymin) / dy)

            if ix < 0:
                ix = 0
            elif ix >= nbins:
                ix = nbins - 1

            if iy < 0:
                iy = 0
            elif iy >= nbins:
                iy = nbins - 1

            H[ix, iy] += 1

    return H

# ------------------ PHYSICS PARAMETERS ------------------ #

dist = 2.0
theta1 = 1.5
theta2 = 0.355
theta3 = 0.55

q1 = np.array([-dist/2, 0.0, 0.0])
q2 = np.array([ dist/2, 0.0, 0.0])
theta = np.array([theta1, theta2, theta3])

# ------------------ HISTOGRAM PARAMETERS ------------------ #

nbins = 700
xmin, xmax = -3.0, 3.0
ymin, ymax = -3.0, 3.0

# ------------------ WORKER FUNCTION (RUNS ON EACH CORE) ------------------ #

def worker_chunk(args):
    """
    One worker:
    - generates a chunk of Metropolis samples
    - computes local energies
    - builds a 2D histogram for (x, y)
    Returns: (H_chunk, E_sum, E_sumsq, Ns_chunk)
    """
    (Ns_chunk, theta, q1, q2,
     xmin, xmax, ymin, ymax, nbins, delta, seed) = args

    # Make RNG independent between workers
    np.random.seed(seed)

    # Generate Metropolis samples for this chunk
    samples = rs.metropolis_sample_H2(Ns_chunk, theta, q1, q2, delta=delta)

    # Local energies for this chunk
    Els = hm.local_energy_H2_batch(samples, theta, q1, q2)
    E_sum = np.sum(Els)
    E_sumsq = np.sum(Els**2)

    # Projected coordinates (two electrons)
    x1 = samples[:, 0]
    x2 = samples[:, 3]
    y1 = samples[:, 1]
    y2 = samples[:, 4]

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    # Build histogram for this chunk (Numba parallel)
    H_chunk = hist2d_parallel(x.astype(np.float64),
                              y.astype(np.float64),
                              xmin, xmax, ymin, ymax,
                              nbins)

    return H_chunk, E_sum, E_sumsq, Ns_chunk

# ------------------ MAIN PARALLEL DRIVER ------------------ #

if __name__ == "__main__":
    # Target total number of *configurations* (not electrons)
    Ns_total = int(1e8)    # WARNING: physically huge → consider smaller for testing
    delta = 1.0

    # Choose chunk size per process call
    # e.g. 1e6 or 1e7 depending on RAM & speed
    chunk_size = int(1e6)

    n_chunks = Ns_total // chunk_size
    remainder = Ns_total % chunk_size
    if remainder > 0:
        n_chunks += 1  # last chunk will be smaller

    n_procs = cpu_count()  # number of processes to use

    print(f"Using {n_procs} processes for ~{Ns_total} samples in {n_chunks} chunks.")

    # Prepare argument list for each chunk
    args_list = []
    for k in range(n_chunks):
        Ns_chunk = chunk_size if k < n_chunks - 1 else (
            Ns_total - chunk_size * (n_chunks - 1)
        )
        seed = np.random.randint(0, 2**31 - 1)
        args = (Ns_chunk, theta, q1, q2,
                xmin, xmax, ymin, ymax, nbins, delta, seed)
        args_list.append(args)

    # Global accumulators
    H_global = np.zeros((nbins, nbins), dtype=np.int64)
    E_sum_total = 0.0
    E_sumsq_total = 0.0
    Ns_accum = 0

    # Run chunks in parallel
    with Pool(processes=n_procs) as pool:
        for H_chunk, E_sum, E_sumsq, Ns_chunk in pool.imap_unordered(worker_chunk, args_list):
            H_global += H_chunk
            E_sum_total += E_sum
            E_sumsq_total += E_sumsq
            Ns_accum += Ns_chunk

    # Compute energy mean and std from sums
    E_avg = E_sum_total / Ns_accum
    E_var = E_sumsq_total / Ns_accum - E_avg**2
    E_std = np.sqrt(max(E_var, 0.0))

    print(f"E_avg = {E_avg:.6f}, E_std = {E_std:.6f}, Ns = {Ns_accum}")

    # ------------------ DENSITY NORMALISATION & PLOT ------------------ #

    dx = (xmax - xmin) / nbins
    dy = (ymax - ymin) / nbins
    area = dx * dy

    total_counts = H_global.sum()
    if total_counts > 0:
        H_density = H_global / (total_counts * area)
    else:
        H_density = H_global.astype(np.float64)

    extent = [xmin, xmax, ymin, ymax]

    plt.imshow(H_density.T, origin='lower', extent=extent,
               aspect='auto', cmap='inferno')
    plt.colorbar(label='Probability density')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Projected density onto x–y plane')
    plt.show()
