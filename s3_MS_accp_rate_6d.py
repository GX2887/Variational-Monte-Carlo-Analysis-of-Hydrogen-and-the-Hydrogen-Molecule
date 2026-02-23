import numpy as np
import s2_MS_accp_rate_1d as ms
from s2_random_sampling import psi_H2
from multiprocessing import Pool, cpu_count

# -------------------------------------------------
# Helper: build a 1D physical observable from 6D samples
# -------------------------------------------------

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

# sampling from H2 minimization
def metropolis_sample_H2_single_chain(
    Ns, theta, q1, q2, delta=1.0, burn_in=200, seed=1234
):
    """
    Single-chain Metropolis sampler for H2 in 6D.
    Uses log_prob_H2 so it is consistent with the rest of this file.
    Returns (samples, acceptance_rate).
    """
    np.random.seed(seed)

    R = np.concatenate([q1, q2], axis=0).astype(float)
    samples = np.zeros((Ns, 6), dtype=float)

    accepted = 0
    total_steps = Ns + burn_in

    # cache old logp
    logp_old = log_prob_H2(R, theta, q1, q2)

    for i in range(total_steps):
        R_prop = R + delta * np.random.randn(6)

        logp_new = log_prob_H2(R_prop, theta, q1, q2)
        # acceptance probability in log form
        logA = logp_new - logp_old

        if np.log(np.random.rand()) < min(0.0, logA):
            R = R_prop
            logp_old = logp_new
            accepted += 1

        if i >= burn_in:
            samples[i - burn_in] = R

    acc_rate = accepted / total_steps
    return samples, acc_rate


# helper so Pool.starmap can call it
def _single_chain_worker(args):
    return metropolis_sample_H2_single_chain(*args)

def metropolis_sample_H2_multi(
    Ns_total, theta, q1, q2, delta=1.0, burn_in=200,
    n_chains=None, base_seed=1234
):
    """
    Run multiple independent chains in parallel and concatenate samples.
    Returns (samples, mean_acceptance_rate).
    """
    if n_chains is None:
        n_chains = cpu_count()

    Ns_per_chain = Ns_total // n_chains
    remainder = Ns_total % n_chains

    args_list = []
    for k in range(n_chains):
        Ns_chain = Ns_per_chain + (1 if k < remainder else 0)
        if Ns_chain == 0:
            continue  # in case Ns_total < n_chains
        seed = base_seed + 1000 * k
        args_list.append((Ns_chain, theta, q1, q2, delta, burn_in, seed))

    with Pool(processes=len(args_list)) as pool:
        results = pool.map(_single_chain_worker, args_list)

    samples_list = [res[0] for res in results]
    acc_rates    = [res[1] for res in results]

    samples = np.vstack(samples_list)
    acc_mean = float(np.mean(acc_rates))

    # keep exactly Ns_total samples
    return samples[:Ns_total], acc_mean

# test sampling
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

def scalar_obs_H2_total_enuc(R_samples, q1, q2):
    """
    Map 6D H2 samples to a 1D time series:
    s = |r1-q1| + |r1-q2| + |r2-q1| + |r2-q2|
    """
    R_samples = np.asarray(R_samples)
    # reshape to (Ns, 2, 3): electrons 1 and 2
    r = R_samples.reshape(-1, 2, 3)   # r[...,0,:] = r1, r[...,1,:] = r2

    r1 = r[:, 0, :]
    r2 = r[:, 1, :]

    q1 = np.asarray(q1).reshape(1, 3)
    q2 = np.asarray(q2).reshape(1, 3)

    d1q1 = np.linalg.norm(r1 - q1, axis=1)
    d1q2 = np.linalg.norm(r1 - q2, axis=1)
    d2q1 = np.linalg.norm(r2 - q1, axis=1)
    d2q2 = np.linalg.norm(r2 - q2, axis=1)

    s = d1q1 + d1q2 + d2q1 + d2q2
    return s

# -------------------------------------------------
# 6D Neff/N estimation for H2 Metropolis sampler
# -------------------------------------------------

def neff_ratio_for_delta_H2_6d(delta, theta, q1, q2,
                               Ns=50000, burn_in=5000,
                               n_repeats=3):
    """
    Estimate Neff/N for a given proposal step size delta in 6D (H2).
    Uses a 1D *physical* time series derived from the 6D coordinates:
        s = |r1-q1| + |r1-q2| + |r2-q1| + |r2-q2|.
    Averages over several runs.
    """
    if delta <= 0:
        return 0.0

    ratios = []
    for _ in range(n_repeats):
        samples, _ = metropolis_sample_H2_multi(
            Ns, theta, q1, q2, delta=delta, burn_in=burn_in
        )
        # 1D series from 6D coordinates (physical observable)
        s = scalar_obs_H2_total_enuc(samples, q1, q2)
        Neff, tau_int = ms.effective_sample_size(s)
        ratios.append(Neff / Ns)

    ratios = np.array(ratios, dtype=float)
    return float(ratios.mean())

def neff_ratio_for_delta_H2_6d_s(delta, theta, q1, q2,
                                 Ns=50000, burn_in=5000,
                                 n_repeats=3):
    """
    Same as neff_ratio_for_delta_H2_6d but also returns the std
    over the n_repeats and the individual Neff/N ratios.
    """
    if delta <= 0:
        return 0.0, 0.0, np.array([])

    ratios = []
    for _ in range(n_repeats):
        samples, _ = metropolis_sample_H2_6D(
            Ns, theta, q1, q2, delta=delta, burn_in=burn_in
        )
        s = scalar_obs_H2_total_enuc(samples, q1, q2)
        Neff, tau_int = ms.effective_sample_size(s)
        ratios.append(Neff / Ns)

    ratios = np.array(ratios, dtype=float)
    ratio_mean = float(np.mean(ratios))
    ratio_std  = float(np.std(ratios, ddof=1)) if n_repeats > 1 else 0.0

    return ratio_mean, ratio_std, ratios

# -------------------------------------------------
# Gradient-ascent optimisation of delta in 6D H2
# (unchanged except for calling the updated neff function)
# -------------------------------------------------

def optimise_delta_H2_6d_gradient(
    theta,
    q1, q2,
    delta_init=1.0,
    Ns=50000,
    burn_in=5000,
    n_repeats=3,
    lr=0.5,
    max_iter=15,
    eps=0.1,
    delta_min=1e-3,
    delta_max=5.0,
    grad_tol=5e-4,
    verbose=True,
    # parameters for std(delta) estimation
    n_std_samples=20,
    std_perturb_scale=0.05,
    ratio_threshold=0.8,
):
    """
    Use gradient ascent in log(delta) to find the delta that maximizes Neff/N
    for the 6D H2 Metropolis sampler, using a physical 1D observable.
    """

    theta_d = np.log(delta_init)
    history = []

    delta = np.clip(np.exp(theta_d), delta_min, delta_max)
    f_curr = neff_ratio_for_delta_H2_6d(
        delta, theta, q1, q2, Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
    )
    best_delta = delta
    best_ratio = f_curr

    if verbose:
        print(f"init: delta = {delta:.5f}, Neff/N ≈ {f_curr:.5f}")

    for it in range(1, max_iter + 1):

        theta_plus  = theta_d + eps
        theta_minus = theta_d - eps

        delta_plus  = np.clip(np.exp(theta_plus),  delta_min, delta_max)
        delta_minus = np.clip(np.exp(theta_minus), delta_min, delta_max)

        f_plus = neff_ratio_for_delta_H2_6d(
            delta_plus, theta, q1, q2, Ns=Ns, burn_in=burn_in,
            n_repeats=n_repeats
        )
        f_minus = neff_ratio_for_delta_H2_6d(
            delta_minus, theta, q1, q2, Ns=Ns, burn_in=burn_in,
            n_repeats=n_repeats
        )

        grad = (f_plus - f_minus) / (2.0 * eps)

        theta_d = theta_d + lr * grad
        delta   = np.clip(np.exp(theta_d), delta_min, delta_max)

        f_curr = neff_ratio_for_delta_H2_6d(
            delta, theta, q1, q2, Ns=Ns, burn_in=burn_in,
            n_repeats=n_repeats
        )

        if f_curr > best_ratio:
            best_ratio = f_curr
            best_delta = delta

        history.append({
            "iter": it,
            "delta": delta,
            "ratio": f_curr,
            "grad": grad,
        })

        if verbose:
            print(f"iter {it:2d}: delta = {delta:.5f}, "
                  f"Neff/N ≈ {f_curr:.5f}, grad = {grad:.5f}")

        if abs(grad) < grad_tol:
            if verbose:
                print("Gradient small; stopping.")
            break

    theta_best = np.log(best_delta)
    delta_samples = []

    for _ in range(n_std_samples):
        theta_pert = theta_best + std_perturb_scale * np.random.randn()
        delta_pert = np.clip(np.exp(theta_pert), delta_min, delta_max)

        r_pert = neff_ratio_for_delta_H2_6d(
            delta_pert, theta, q1, q2,
            Ns=Ns, burn_in=burn_in, n_repeats=n_repeats
        )

        if r_pert >= ratio_threshold * best_ratio:
            delta_samples.append(delta_pert)

    if len(delta_samples) > 1:
        delta_std = float(np.std(delta_samples, ddof=1))
    else:
        delta_std = 0.0

    if verbose:
        print(
            f"\nBest delta ≈ {best_delta:.5f} ± {delta_std:.5f} "
            f"with Neff/N ≈ {best_ratio:.5f}"
        )

    return best_delta, delta_std, history


# finding the optimal delta
deltas = np.linspace(0.1, 3, 5)
Ns_test = int(1e6)

theta = np.array([1.0, 1.0, 1.0])
r = 1.4
q1 = np.array([ r/2, 0.0, 0.0 ])
q2 = np.array([-r/2, 0.0, 0.0 ])

# if __name__ == "__main__":
#     best_delta, delta_std, history = optimise_delta_H2_6d_gradient(
#         theta,
#         q1, q2,
#         delta_init=0.7,
#         Ns=int(1e6),
#         burn_in=5000,
#         n_repeats=3,
#         lr=0.5,
#         max_iter=100,
#         eps=0.1,
#         delta_min=1e-3,
#         delta_max=5.0,
#         grad_tol=5e-4,
#         verbose=True,
#         # parameters for std(delta) estimation
#         n_std_samples=20,
#         std_perturb_scale=0.05,
#         ratio_threshold=0.8,
#     )



# -------------------------------------------------
# Acceptance rate and summary (unchanged logic)
# -------------------------------------------------

def metropolis_acceptance_rate_H2_6d(
    Ns, theta, q1, q2,
    delta=1.0,
    burn_in=5000,
    n_repeats=5,
    R0=None,
):
    acc_rates = []

    for j in range(n_repeats):
        if R0 is None:
            R = np.concatenate([q1, q2], axis=0).astype(float)
        else:
            R = np.array(R0, dtype=float)

        accepted = 0
        total = 0
        total_steps = Ns + burn_in

        for step in range(total_steps):
            R_prop = R + delta * np.random.randn(6)

            logp_old = log_prob_H2(R,      theta, q1, q2)
            logp_new = log_prob_H2(R_prop, theta, q1, q2)
            acc_prob = np.exp(logp_new - logp_old)

            if np.random.rand() < acc_prob:
                R = R_prop
                accepted += 1

            total += 1

        acc_rate = accepted / total
        acc_rates.append(acc_rate)

    acc_rates = np.array(acc_rates, dtype=float)
    acc_mean = float(np.mean(acc_rates))
    acc_std  = float(np.std(acc_rates, ddof=1)) if n_repeats > 1 else 0.0

    return acc_mean, acc_std, acc_rates


def summarize_delta_H2_6d(
    delta_best, theta, q1, q2,
    Ns=100000, burn_in=2000,
    n_repeats_acc=5, n_repeats_neff=3,
):
    acc_mean, acc_std, acc_rates = metropolis_acceptance_rate_H2_6d(
        Ns, theta, q1, q2, delta=delta_best, burn_in=burn_in,
        n_repeats=n_repeats_acc
    )

    neff_mean, neff_std, neff_ratios = neff_ratio_for_delta_H2_6d_s(
        delta_best, theta, q1, q2, Ns=Ns, burn_in=burn_in,
        n_repeats=n_repeats_neff
    )

    print("\nSummary over runs (H2 6D):")
    print(f"delta_best      = {delta_best:.5f}")
    print(f"mean acc_rate   = {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"mean Neff/N     = {neff_mean:.4f} ± {neff_std:.4f}")

    return {
        "delta_best": delta_best,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_rates": acc_rates,
        "neff_mean": neff_mean,
        "neff_std": neff_std,
        "neff_ratios": neff_ratios,
    }

# Best delta ≈ 0.74971 ± 0.04326 with Neff/N ≈ 0.02994
delta_best = 0.74971
a = summarize_delta_H2_6d(
    delta_best, theta, q1, q2,
    Ns=int(1e6), burn_in=2000,
    n_repeats_acc=5, n_repeats_neff=3,
)