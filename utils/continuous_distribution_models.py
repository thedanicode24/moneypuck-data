import matplotlib.pyplot as plt
import numpy as np
from empiricaldist import Pmf, Cdf
from utils.distribution_analysis import create_cdf
from scipy.stats import norm, shapiro, kstest, probplot

#############################
# Normal Distribution
#############################

def plot_empirical_vs_normal_cdf(data, figsize=(12,8)):
    mean, std = np.mean(data), np.std(data)
    sorted_data = np.sort(data)
    cdf_empirical = np.arange(1, len(data)+1) / len(data)
    
    qs = np.linspace(mean - 3.5*std, mean + 3.5*std, 1000)
    cdf_normal = norm.cdf(qs, mean, std)
    
    plt.figure(figsize=figsize)
    plt.step(sorted_data, cdf_empirical, where='post', label='Empirical CDF')
    plt.plot(qs, cdf_normal, 'r--', label='Theoretical Normal CDF')
    plt.title('Empirical CDF vs Theoretical Normal CDF')
    plt.xlabel('Data values')
    plt.ylabel('CDF')
    plt.grid()
    plt.legend()
    plt.show()

def plot_histogram_with_normal_pdf(data, figsize=(12,8)):
    mean, std = np.mean(data), np.std(data)
    plt.figure(figsize=figsize)  # Create figure first
    count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data')
    pdf = norm.pdf(bins, mean, std)
    plt.plot(bins, pdf, 'r-', lw=2, label='Theoretical Normal')
    plt.title('Histogram and Normal PDF')
    plt.legend()
    plt.show()

def qq_plot_normal(data, figsize=(12,8)):
    plt.figure(figsize=figsize)
    probplot(data, dist="norm", plot=plt)
    plt.title('Quantile-quantitle plot vs Normal')
    plt.grid()
    plt.show()


#################################
# Test
#################################

def shapiro_wilk_test(data):
    stat, p_value = shapiro(data)
    print(f"Shapiro-Wilk Test: stat={stat:.4f}, p-value={p_value:.4f}")
    if p_value > 0.05:
        print("Fail to reject the null hypothesis of normality")
    else:
        print("Reject the null hypothesis of normality")
    return stat, p_value

def kolmogorov_smirnov_test(data, distribution, params=None):
    if distribution == 'norm' and params is None:
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        params = (mean, std)
    ks_stat, ks_p = kstest(data, distribution, args=params)
    print(f"KS test {distribution}: stat={ks_stat:.4f}, p-value={ks_p:.4f}")
    return ks_stat, ks_p