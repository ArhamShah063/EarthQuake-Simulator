import numpy as np
from scipy.stats import poisson, expon


def poisson_probability(lam, t, k):
    """P(N(t) = k) for Poisson process with rate lam over time t."""
    return poisson.pmf(k, lam * t)


def poisson_cdf(lam, t, k):
    """P(N(t) <= k)"""
    return poisson.cdf(k, lam * t)


def probability_at_least_k(lam, t, k):
    """P(N(t) >= k) = 1 - P(N(t) <= k-1)"""
    if k == 0:
        return 1.0
    return 1.0 - poisson.cdf(k - 1, lam * t)


def probability_at_least_one(lam, t):
    """P(N(t) >= 1) = 1 - e^(-lambda*t)"""
    return 1.0 - np.exp(-lam * t)


def expected_earthquakes(lam, t):
    """E[N(t)] = lambda * t"""
    return lam * t


def mean_interarrival_time(lam):
    """E[T] = 1/lambda"""
    return 1.0 / lam


def variance_interarrival_time(lam):
    """Var[T] = 1/lambda^2"""
    return 1.0 / (lam ** 2)


def poisson_distribution_range(lam, t, k_max=None):
    """Return arrays of k values and their probabilities for a Poisson distribution."""
    mu = lam * t
    if k_max is None:
        k_max = int(mu + 4 * np.sqrt(mu)) + 5
    k_values = np.arange(0, k_max + 1)
    probs = poisson.pmf(k_values, mu)
    return k_values, probs


def confidence_interval_monte_carlo(counts, confidence=0.95):
    """
    Compute confidence interval for probability estimate from Monte Carlo.
    """
    alpha = 1 - confidence
    lower = np.percentile(counts, 100 * alpha / 2)
    upper = np.percentile(counts, 100 * (1 - alpha / 2))
    mean = np.mean(counts)
    return mean, lower, upper


def exceedance_probability(lam, t, k):
    """Probability of exceeding k earthquakes in time t."""
    return 1.0 - poisson.cdf(k, lam * t)


def return_period(lam, k=1):
    """
    Average time (in years) between events where at least k earthquakes occur.
    For k=1: return period = 1/lambda
    """
    return 1.0 / lam if k == 1 else None
