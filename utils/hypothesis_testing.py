import numpy as np
import matplotlib.pyplot as plt
from utils.distribution_analysis import create_pmf_from_kde
from utils.thinkstats import underride
from scipy.stats import gaussian_kde
from empiricaldist import Pmf

def simulate_groups(data):
    """
    Shuffle and split combined data into two groups of original sizes.

    Parameters:
    data (tuple): Tuple containing two arrays/lists (group1, group2).

    Returns:
    tuple: Two arrays representing the shuffled groups with the same sizes as the originals.
    """

    group1, group2 = data
    n, m = len(group1), len(group2)

    pool = np.hstack(data)
    np.random.shuffle(pool)
    return pool[:n], pool[n:]

def permutation_test(group1, group2, num_simulations=10000, figsize=(12,8)):
    """
    Perform a permutation test to compare means of two groups.

    Parameters:
    group1 (array-like): First sample group.
    group2 (array-like): Second sample group.
    num_simulations (int): Number of permutations to perform (default 10000).

    Returns:
    float: p-value representing the probability of observing a difference
           at least as extreme as the observed difference under the null hypothesis.
    """
        
    observed_diff = abs(np.mean(group1) - np.mean(group2))
    simulated_diffs = []

    for _ in range(num_simulations):
        new_g1, new_g2 = simulate_groups((group1, group2))
        diff = abs(np.mean(new_g1) - np.mean(new_g2))
        simulated_diffs.append(diff)

    pmf = create_pmf_from_kde(simulated_diffs, 0, max(simulated_diffs)*1.05)

    plt.figure(figsize=figsize)
    pmf.plot()
    fill_tail(pmf, observed_diff, "right")
    plt.title("Permutation Test")
    plt.xlabel("Absolute difference in means")
    plt.ylabel("Density")
    plt.grid()

    return compute_p_value(simulated_diffs, observed_diff)

def fill_tail(pmf, observed, side, **options):
    """Fill the area under a PMF, right or left of an observed value."""
    options = underride(options, alpha=0.3)

    if side == "right":
        condition = pmf.qs >= observed
    elif side == "left":
        condition = pmf.qs <= observed

    series = pmf[condition]
    plt.fill_between(series.index, 0, series, **options)

def compute_p_value(simulated, observed):
    """Fraction of simulated values as big or bigger than the observed value."""
    return (np.asarray(simulated) >= observed).mean()