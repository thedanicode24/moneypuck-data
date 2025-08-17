import matplotlib.pyplot as plt
import numpy as np
from empiricaldist import Pmf, Cdf
from utils.distribution_analysis import create_cdf
from scipy.stats import norm, shapiro, kstest, probplot, lognorm, cumfreq, gamma, weibull_min, beta, fisk, cauchy, laplace, logistic

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

######################################
# Lognormal
###################################

def plot_empirical_vs_lognorm_cdf(data, bins=100, figsize=(12,8)):

    positive_data = data[data > 0] 
    log_data = np.log(positive_data)

    mu, sigma = np.mean(log_data), np.std(log_data)

    shape = sigma
    scale = np.exp(mu)

    res = cumfreq(positive_data, numbins=bins)
    x_emp = np.linspace(min(positive_data), max(positive_data), bins)
    cdf_emp = res.cumcount / len(positive_data)

    cdf_lognorm = lognorm.cdf(x_emp, s=shape, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_lognorm, label='Theoretical Lognormal CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Lognormal CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()



######################################
# Gamma
#####################################


def plot_empirical_vs_gamma_cdf(data, bins=100, figsize=(12,8)):
    positive_data = data[data > 0]

    shape, loc, scale = gamma.fit(positive_data, floc=0)

    res = cumfreq(positive_data, numbins=bins)
    x_emp = np.linspace(min(positive_data), max(positive_data), bins)
    cdf_emp = res.cumcount / len(positive_data)

    cdf_gamma = gamma.cdf(x_emp, a=shape, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_gamma, label='Theoretical Gamma CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Gamma CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()


#####################################
# Weibull
##################################

def plot_empirical_vs_weibull_cdf(data, bins=100, figsize=(12,8)):
    positive_data = data[data > 0]

    shape, loc, scale = weibull_min.fit(positive_data, floc=0)

    res = cumfreq(positive_data, numbins=bins)
    x_emp = np.linspace(min(positive_data), max(positive_data), bins)
    cdf_emp = res.cumcount / len(positive_data)

    cdf_weibull = weibull_min.cdf(x_emp, c=shape, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_weibull, label='Theoretical Weibull CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Weibull CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()


#################################
# Beta
#################################


def plot_empirical_vs_beta_cdf(data, bins=100, figsize=(12,8), epsilon=1e-6):

    data_min, data_max = np.min(data), np.max(data)
    norm_data = (data - data_min) / (data_max - data_min)
    norm_data = np.clip(norm_data, epsilon, 1 - epsilon)

    a, b, loc, scale = beta.fit(norm_data, floc=0, fscale=1)

    res = cumfreq(norm_data, numbins=bins)
    x_emp = np.linspace(0, 1, bins)
    cdf_emp = res.cumcount / len(norm_data)

    cdf_beta = beta.cdf(x_emp, a=a, b=b, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_beta, label='Theoretical Beta CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Beta CDF (data normalized)")
    plt.xlabel("Normalized Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

#################################
# Log-logistic
##############################


def plot_empirical_vs_loglogistic_cdf(data, bins=100, figsize=(12,8)):
    positive_data = data[data > 0]

    shape, loc, scale = fisk.fit(positive_data, floc=0)

    res = cumfreq(positive_data, numbins=bins)
    x_emp = np.linspace(min(positive_data), max(positive_data), bins)
    cdf_emp = res.cumcount / len(positive_data)


    cdf_fisk = fisk.cdf(x_emp, c=shape, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_fisk, label='Theoretical Log-logistic CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Log-logistic (Fisk) CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()

#################################
# Cauchy
#################################

def plot_empirical_vs_cauchy_cdf(data, bins=100, figsize=(12,8)):

    loc = np.median(data)
    scale = (np.percentile(data, 75) - np.percentile(data, 25)) / 2

    res = cumfreq(data, numbins=bins)
    x_emp = np.linspace(min(data), max(data), bins)
    cdf_emp = res.cumcount / len(data)

    cdf_cauchy = cauchy.cdf(x_emp, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_cauchy, label='Theoretical Cauchy CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Cauchy CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid()
    plt.show()

###################################
# Laplace
#################################

def plot_empirical_vs_laplace_cdf(data, bins=100, figsize=(12,8)):
    loc = np.median(data)
    scale = np.mean(np.abs(data - loc))

    res = cumfreq(data, numbins=bins)
    x_emp = np.linspace(min(data), max(data), bins)
    cdf_emp = res.cumcount / len(data)

    cdf_laplace = laplace.cdf(x_emp, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_laplace, label='Theoretical Laplace CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Laplace CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

###################################
# Logistic
##################################

def plot_empirical_vs_logistic_cdf(data, bins=100, figsize=(12,8)):

    # Stima dei parametri
    loc = np.median(data)
    scale = np.std(data) * np.sqrt(3) / np.pi  # parametro di scala

    # CDF empirica
    res = cumfreq(data, numbins=bins)
    x_emp = np.linspace(min(data), max(data), bins)
    cdf_emp = res.cumcount / len(data)

    # CDF teorica logistica
    cdf_logistic = logistic.cdf(x_emp, loc=loc, scale=scale)

    plt.figure(figsize=figsize)
    plt.plot(x_emp, cdf_emp, label='Empirical CDF', color='blue')
    plt.plot(x_emp, cdf_logistic, label='Theoretical Logistic CDF', color='red', linestyle='--')
    plt.title("Empirical CDF vs Theoretical Logistic CDF")
    plt.xlabel("Data values")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
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


def kolmogorov_smirnov_test(data, distribution, params=None, epsilon=1e-6):
    if params is None:
        if distribution == 'norm':
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            params = (mean, std)
        elif distribution == 'lognorm':
            positive_data = data[data > 0]
            log_data = np.log(positive_data)
            mu, sigma = np.mean(log_data), np.std(log_data, ddof=1)
            params = (sigma, 0, np.exp(mu))
            data = positive_data
        elif distribution == 'gamma':
            positive_data = data[data > 0]
            shape, loc, scale = gamma.fit(positive_data, floc=0)
            params = (shape, 0, scale)
            data = positive_data
        elif distribution == 'weibull_min':
            positive_data = data[data > 0]
            shape, loc, scale = weibull_min.fit(positive_data, floc=0)
            params = (shape, 0, scale)
            data = positive_data
        elif distribution == 'beta':
            data_min, data_max = np.min(data), np.max(data)
            norm_data = (data - data_min) / (data_max - data_min)
            norm_data = np.clip(norm_data, epsilon, 1 - epsilon)
            a, b, loc, scale = beta.fit(norm_data, floc=0, fscale=1)
            params = (a, b, 0, 1)
            data = norm_data
        elif distribution == 'fisk':
            positive_data = data[data > 0]
            shape, loc, scale = fisk.fit(positive_data, floc=0)
            params = (shape, 0, scale)
            data = positive_data
        elif distribution == 'logistic':
            loc = np.median(data)
            scale = np.std(data, ddof=1) * np.sqrt(3) / np.pi
            params = (loc, scale)
        elif distribution == 'cauchy':
            loc = np.median(data)
            scale = (np.percentile(data, 75) - np.percentile(data, 25)) / 2
            params = (loc, scale)
        elif distribution == 'laplace':
            loc = np.median(data)
            scale = np.mean(np.abs(data - loc))
            params = (loc, scale)
        else:
            raise ValueError("No distribution")

    ks_stat, ks_p = kstest(data, distribution, args=params)
    print(f"KS test {distribution}: stat={ks_stat:.4f}, p-value={ks_p}")
    return ks_stat, ks_p
