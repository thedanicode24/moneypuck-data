import numpy as np
import matplotlib.pyplot as plt
from empiricaldist import Pmf
from utils.thinkstats import two_bar_plots
from utils.distribution_analysis import create_pmf
from scipy.special import gammaln
from scipy.stats import nbinom, chisquare

def mae(estimates, true_value):
    """
    Compute the Mean Absolute Error (MAE) between a list of estimates and the true value.

    MAE is defined as the average of the absolute differences between 
    each estimate and the true parameter value. It provides a more interpretable
    measure of average error than MSE, without squaring the differences.

    Parameters:
    ----------
    estimates : list or np.array
        A list or array of estimated values.

    true_value : float
        The true value of the parameter being estimated.

    Returns:
    -------
    float
        The mean absolute error.
    """
    errors = np.asarray(estimates) - true_value
    return np.mean(np.abs(errors))

def mse(estimates, true_value):
    """
    Compute the Mean Squared Error (MSE) between a list of estimates and the true value.

    MSE is defined as the average of the squared differences between 
    each estimate and the true parameter value. It measures the accuracy
    of an estimator by penalizing larger deviations more heavily.

    Parameters:
    ----------
    estimates : list or np.array
        A list or array of estimated values.

    true_value : float
        The true value of the parameter being estimated.

    Returns:
    -------
    float
        The mean squared error.
    """
    errors = np.array(estimates) - true_value
    return np.mean(errors ** 2)

####################################
# Negative Binomial Distribution
####################################

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

def negative_binomial_pmf(k, mu, r):
    log_coeff = gammaln(k + r) - gammaln(r) - gammaln(k + 1)
    log_p = r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu))
    return np.exp(log_coeff + log_p)

def plot_empirical_vs_negative_binomial_pmf(data, xlabel="Value", ylabel="PMF", figsize=(12,8)):
    data = np.array(data)
    data = data[data >= 0].astype(int)

    mu = np.mean(data)
    var = np.var(data, ddof=1)
    r = estimate_r(mu, var)

    values, counts = np.unique(data, return_counts=True)
    pmf_emp = counts / counts.sum()

    pmf_theor = negative_binomial_pmf(values, mu, r)

    plt.figure(figsize=figsize)
    plt.bar(values, pmf_emp, width=0.6, alpha=0.6, label='Empirical PMF', color='blue')
    plt.plot(values, pmf_theor, 'o-', label='Negative Binomial PMF', color='red')
    plt.title("Empirical vs Negative Binomial PMF")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Estimated parameters: r={r:.4f}, mu={mu:.4f}, var={var:.4f}")

    return values, pmf_theor, mu, r


def simulate_mean_estimation_nb(mean, r, figsize=(12,8), num_reps=1000):
    """
    Simulate the estimation of the mean from a negative binomial distribution
    across different sample sizes. Computes standard error and confidence intervals.

    Parameters:
    ----------
    mean : float
        Theoretical mean of the negative binomial distribution.

    r : float
        Dispersion parameter of the negative binomial distribution.

    num_reps : int, optional
        Number of repetitions/simulations per sample size (default is 1000).

    figsize : tuple, optional
        Figure size for the plot (default is (12, 8)).

    Returns:
    -------
    None
        The function prints MSE and MAE for the mean and median estimates,
        and displays a plot of the estimation performance across sample sizes.
    """

    p = r / (r + mean)
    ns = np.logspace(1, 5, num=10).astype(int)

    means = []
    medians = []
    stderr_means = []
    ci_low = []
    ci_high = []

    for n in ns:
        samples = [np.mean(nbinom.rvs(r, p, size=n)) for _ in range(num_reps)]
        sample_mean = np.mean(samples)
        sample_median = np.median(samples)
        std_err = np.std(samples, ddof=1)

        # 95% confidence interval from percentiles
        lower = np.percentile(samples, 2.5)
        upper = np.percentile(samples, 97.5)

        means.append(sample_mean)
        medians.append(sample_median)
        stderr_means.append(std_err)
        ci_low.append(lower)
        ci_high.append(upper)

    means = np.array(means)
    medians = np.array(medians)
    stderr_means = np.array(stderr_means)
    ci_low = np.array(ci_low)
    ci_high = np.array(ci_high)

    plt.figure(figsize=figsize)

    plt.axhline(mean, color="red", lw=1, alpha=0.5, linestyle="--", label="True Mean")
    plt.errorbar(ns, means, yerr=stderr_means, fmt='o', label="Sample Mean ± StdErr", capsize=4)
    plt.fill_between(ns, ci_low, ci_high, alpha=0.2, label="95% CI (Mean)")

    plt.plot(ns, medians, label="Sample Median")

    plt.title("Estimation of Negative Binomial Mean")
    plt.xlabel("Sample size (log scale)")
    plt.xscale("log")
    plt.ylabel("Estimate")
    plt.grid(True)
    plt.legend()

    print(f"Mean Squared Error (Mean): {mse(means, mean):.4f}")
    print(f"Mean Absolute Error (Mean): {mae(means, mean):.4f}")
    print(f"Mean Squared Error (Median): {mse(medians, mean):.4f}")
    print(f"Mean Absolute Error (Median): {mae(medians, mean):.4f}")


###############################
# Test
###############################

def chi_square_test(data, pmf_theor, values):
    """
    Perform a Chi-square goodness-of-fit test for discrete data.

    Parameters:
    -----------
    data : array-like
        Observed discrete data (integers).
    pmf_theor : array-like
        Theoretical probabilities (PMF values) corresponding to `values`.
    values : array-like
        Discrete values at which the theoretical PMF is evaluated.

    Returns:
    --------
    chi2_stat : float
        The Chi-square test statistic.
    p_value : float
        The p-value for the test.

    Notes:
    ------
    - The length of `pmf_theor` and `values` must be the same.
    - Observed data frequencies are calculated from `data`.
    - Expected frequencies are computed as pmf_theor * total_observations.
    - For valid results, expected frequencies should ideally be >= 5. 
      If needed, group rare categories before applying the test.
    """

    data = np.array(data)
    values = np.array(values)
    pmf_theor = np.array(pmf_theor)

    observed_counts = np.array([np.sum(data == v) for v in values])
    total = observed_counts.sum()

    expected_counts = pmf_theor * total
    expected_counts = expected_counts * total / expected_counts.sum()

    if np.any(expected_counts < 5):
        print("Warning: Some expected frequencies are less than 5; consider grouping categories.")

    chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

    print(f"Chi-square statistic: {chi2_stat:.4f}, p-value: {p_value:.4f}")

    #return chi2_stat, p_value
