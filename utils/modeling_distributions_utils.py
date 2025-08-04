import numpy as np
import matplotlib.pyplot as plt
from empiricaldist import Pmf
from utils import thinkstats, pmf_utils
from scipy.special import gammaln

def estimate_r(mu, var):
    """
    Estimate the parameter 'r' of the negative binomial distribution 
    given the mean (mu) and variance (var).

    Parameters:
    mu (float): Mean of the distribution.
    var (float): Variance of the distribution.

    Returns:
    float: Estimated value of 'r'.

    Raises:
    ValueError: If the variance is less than or equal to the mean.
    """

    if var <= mu:
        raise ValueError("Variance must be larger than mean.")
    return mu**2 / (var - mu)

def negbinom_pmf(k, mu, r):
    """
    Compute the probability mass function (PMF) of the negative binomial distribution.

    Parameters:
    k (int or np.array): Value(s) at which to compute the PMF.
    mu (float): Mean of the distribution.
    r (float): Number of successes until the experiment is stopped (dispersion parameter).

    Returns:
    float or np.array: PMF value(s) at the given k.
    """

    log_coeff = gammaln(k + r) - gammaln(r) - gammaln(k + 1)
    log_p = r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu))
    return np.exp(log_coeff + log_p)

def plot_negative_binomial(df, feature, values, xlabel="Feature", ylabel="PMF"):
    """
    Plot the empirical PMF of a feature alongside the fitted negative binomial model.

    Parameters:
    df (DataFrame): Input pandas DataFrame containing the data.
    feature (str): Name of the column in the DataFrame to analyze.
    values (list or np.array): Discrete values at which to evaluate the negative binomial PMF.
    xlabel (str): Label for the x-axis of the plot. Default is "Feature".
    ylabel (str): Label for the y-axis of the plot. Default is "PMF".
    """
    
    pmf = pmf_utils.create_pmf(df, feature, name="Results")
    mean = pmf.mean()
    var = pmf.var()
    r = estimate_r(mean, var)

    vals = np.array(values)
    neg = negbinom_pmf(vals, mean, r)
    pmf_negbin = Pmf(neg, vals, name="Negative Binomial Model")

    plt.figure(figsize=(12,8))
    thinkstats.two_bar_plots(pmf, pmf_negbin)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
