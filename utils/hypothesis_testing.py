import numpy as np
import matplotlib.pyplot as plt
from utils.distribution_analysis import create_pmf_from_kde
from utils.thinkstats import underride
from scipy.stats import gaussian_kde
from empiricaldist import Pmf

def fill_tail(pmf, observed, side, **options):
    """Fill the area under a PMF, right or left of an observed value."""
    options = underride(options, alpha=0.3)

    if side == "right":
        condition = pmf.qs >= observed
    elif side == "left":
        condition = pmf.qs <= observed

    series = pmf[condition]
    plt.fill_between(series.index, 0, series, **options)

def plot_permutation_test(simulated, statistic, figsize=(12,8), xlabel="Statistic", ylabel="Density"):
    pmf = create_pmf_from_kde(simulated, 0, max(simulated)*1.05)

    plt.figure(figsize=figsize)
    pmf.plot()
    fill_tail(pmf, statistic, "right")
    plt.title("Permutation Test")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()