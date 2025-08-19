import numpy as np
from empiricaldist import FreqTab, Pmf, Cdf
from utils.thinkstats import plot_kde, cohen_effect_size, two_bar_plots, bias, percentile_rank, median, iqr, quartile_skewness, Pdf, NormalPdf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis
###############################
# Frequency Table
###############################

def plot_ftab(data, xlabel="Feature", figsize=(12,8)):
    """
    Plot a frequency bar chart of a specific feature in a DataFrame.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data.
    xlabel (str, optional): Label for the x-axis. Default is "Feature".
    figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    """
    
    print_stats_discrete(data)
    ftab = FreqTab.from_seq(data, name=xlabel)
    plt.figure(figsize=figsize)
    ftab.bar()
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title("Frequency Table")
    plt.grid(True)

def plot_two_ftabs(df1, df2, label1="Name1", label2="Name2", figsize=(12,8), xlabel="Feature"):
    """
    Plot two frequency bar charts for the same feature from two different DataFrames
    and print the Cohen's effect size between the distributions.

    Parameters:
    df1 (DataFrame): The first pandas DataFrame.
    df2 (DataFrame): The second pandas DataFrame.
    name1 (str, optional): Label for the first dataset in the plot. Default is "Name1".
    name2 (str, optional): Label for the second dataset in the plot. Default is "Name2".
    figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    xlabel (str): Label for the x-axis. Default is "Feature".
    """

    print(f"Cohen's effect size: {cohen_effect_size(df1, df2):.2f}")
    ftab1 = FreqTab.from_seq(df1, name=label1)
    ftab2 = FreqTab.from_seq(df2, name=label2)
    plt.figure(figsize=figsize)
    two_bar_plots(ftab1, ftab2)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title("Frequency Table")

def print_stats_discrete(data):
    """
    Print basic statistics (mean, variance, standard deviation, and mode) for a given feature.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data.
    """

    print(f"Mean: {data.mean():.3f}")
    print(f"Variance: {data.var():.3f}")
    print(f"Standard deviation: {data.std(ddof=0):.3f}")
    mode_values = data.mode()
    if len(mode_values) == 1:
        print(f"Mode: {mode_values.iloc[0]}")
    else:
        print(f"Mode: {mode_values.values}")

def plot_grouped_ftab(data, bin_method='sturges', figsize=(12,8), xlabel="Intervals", ylabel="Frequency"):
    bins = np.histogram_bin_edges(data, bins=bin_method)
    frequencies, edges = np.histogram(data, bins=bins)
    
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)]
    
    plt.figure(figsize=figsize)
    plt.bar(labels, frequencies, width=0.6)
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Grouped Frequency Table")
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def print_stats_continuous(data):
    print(f"Mean: {data.mean():.3f}")
    print(f"Variance: {data.var():.3f}")
    print(f"Standard deviation: {data.std(ddof=0):.3f}")
    print(f"Skewness: {skew(data):.3f}")
    print(f"Kurtosis: {kurtosis(data)+3:.3f}")

def plot_two_grouped_ftabs(data1, data2, 
                           bin_size=0.5, 
                           figsize=(12,8), 
                           xlabel="Intervals", 
                           ylabel="Frequency",
                           label1="Dataset 1",
                           label2="Dataset 2"):
    min_val = min(min(data1), min(data2))
    max_val = max(max(data1), max(data2))

    bins = []
    start = min_val
    while start < max_val:
        bins.append((start, start + bin_size))
        start += bin_size

    def get_freq(data):
        return [sum(b[0] <= x < b[1] for x in data) for b in bins]

    freq1 = get_freq(data1)
    freq2 = get_freq(data2)

    labels = [f"{b[0]:.2f}-{b[1]:.2f}" for b in bins]
    x = np.arange(len(bins))

    width = 0.4

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, freq1, width, label=label1)
    ax.bar(x + width/2, freq2, width, label=label2)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Comparison of Grouped Frequency Tables')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()




###################################
# Probability Mass Function
####################################

def create_pmf(data, name="pmf"):
    """
    Creates a Probability Mass Function (PMF) from a given DataFrame column.

    Parameters:
        data (DataFrame): The input DataFrame.
        name (str): The name to assign to the PMF.

    Returns:
        Pmf: The resulting PMF.
    """
        
    return Pmf.from_seq(data, name=name)

def plot_pmf(data, width=2, xlabel="Feature", ylabel="PMF", figsize=(12,8)):
    """
    Plots the actual and observed (biased) PMFs of a given feature from a DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.
        width (int): Bar width for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    """

    actual_pmf = create_pmf(data, name="Actual")
    observed_pmf = bias(actual_pmf, name="Observed")

    plt.figure(figsize=figsize)
    two_bar_plots(actual_pmf, observed_pmf, width=width)
    plt.title("Probability Mass Function")
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    print("Actual PMF: ")
    print_pmf_stats(actual_pmf)
    print()
    print(f"Observed PMF: ")
    print_pmf_stats(observed_pmf)

def print_pmf_stats(pmf):
    """
    Prints statistical measures for a given PMF.

    Parameters:
        pmf (Pmf): The PMF to analyze.
    """
        
    print(f"Mean: {pmf.mean():.3f}")
    print(f"Variance: {pmf.var():.2f}")
    print(f"Standard deviation: {pmf.std():.3f}")
    print(f"Mode: {pmf.mode()}")
    print(f"Skewness: {pmf_skewness(pmf):.3f}")

def plot_two_pmfs(values1, values2, label1="Name1", label2="Name2", xlabel="Feature", ylabel="Probability", figsize=(12,8)):
    """
    Plots the PMFs of a feature from two different DataFrames for comparison.

    Parameters:
        values1 (DataFrame): The first DataFrame.
        values2 (DataFrame): The second DataFrame.
        name1 (str): Name label for the first PMF.
        name2 (str): Name label for the second PMF.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    """
        
    pmf1 = Pmf.from_seq(values1, name=label1)
    pmf2 = Pmf.from_seq(values2, name=label2)

    plt.figure(figsize=figsize)
    plt.title("Probability Mass Function")
    two_bar_plots(pmf1, pmf2)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_diff_pmfs(values1, values2, label1="Name1", label2="Name2", xlabel="Feature", ylabel="Difference (%)", figsize=(12,8)):
    """
    Plots the percentage difference between the PMFs of a feature from two DataFrames.

    Parameters:
        values1 (DataFrame): The first DataFrame.
        values2 (DataFrame): The second DataFrame.
        name1 (str): Name label for the first PMF.
        name2 (str): Name label for the second PMF.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """

    pmf1 = Pmf.from_seq(values1, name=label1)
    pmf2 = Pmf.from_seq(values2, name=label2)

    diff = (pmf1-pmf2)*100
    plt.figure(figsize=figsize)
    plt.title("Probability Mass Function - Percentage difference")
    diff.bar()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def pmf_skewness(pmf):
    """
    Calculates the skewness of a PMF.

    Parameters:
        pmf (Pmf): The PMF for which to calculate skewness.

    Returns:
        float: The skewness value.
    """

    qs = np.array(pmf.qs)
    ps = np.array(pmf.ps)

    mean = np.sum(qs * ps)
    std = np.sqrt(np.sum(((qs - mean) ** 2) * ps))
    return np.sum(((qs - mean) ** 3) * ps) / (std ** 3)


######################################
# Cumulative Distribution Function
######################################

def create_cdf(values):
    """
    Create a CDF (Cumulative Distribution Function) from a sequence of values.

    Parameters:
    values (sequence): The input data.

    Returns:
    Cdf: The cumulative distribution function of the input data.
    """
    return Cdf.from_seq(values)

def print_percentile_rank(ref, values, label="Reference"):
    """
    Print the percentile rank of a reference value compared to a set of values.

    Parameters:
    ref (float): The reference value.
    values (sequence): The dataset to compare against.
    label (str, optional): A label for the reference value, used in the print statement.
    """
    print(f"{label} - Percentile rank: {percentile_rank(ref, values):.2f}")

def plot_cdf(ref, values, figsize=(12,8), label="Reference", xlabel="Feature", ylabel="CDF"):
    """
    Plot the CDF of a dataset with a vertical line for a reference value.

    Parameters:
    ref (float): The reference value to mark on the plot.
    values (sequence): The dataset to plot.
    figsize (tuple, optional): Size of the plot.
    label (str, optional): Label for the reference value.
    xlabel (str, optional): Label for the x-axis.
    ylabel (str, optional): Label for the y-axis.
    """
    cdf = create_cdf(values)

    plt.figure(figsize=figsize)
    cdf.step()
    plt.axvline(ref, ls=":", color="red", label=f"{label}: {ref:.2f}")
    plt.legend()
    plt.title("Cumulative Distribution Function")
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_two_cdfs(values1, values2, label1="Name1", label2="Name2", xlabel="Feature", ylabel="CDF", figsize=(12,8)):
    """
    Plot two CDFs on the same figure for comparison.

    Parameters:
    values1 (sequence): First dataset.
    values2 (sequence): Second dataset.
    name1 (str, optional): Label for the first dataset.
    name2 (str, optional): Label for the second dataset.
    xlabel (str, optional): Label for the x-axis.
    ylabel (str, optional): Label for the y-axis.
    figsize (tuple, optional): Size of the plot.
    """

    pmf1 = Pmf.from_seq(values1, name=label1)
    pmf2 = Pmf.from_seq(values2, name=label2)
    cdf1 = pmf1.make_cdf()
    cdf2 = pmf2.make_cdf()
    plt.figure(figsize=figsize)
    cdf1.plot()
    cdf2.plot(alpha=0.5)
    plt.title("Cumulative Distribution Function")
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def print_percentile_stats(cdf):
    """
    Print summary statistics based on percentiles:
    - Median (50th percentile)
    - Interquartile Range (difference between 75th and 25th percentile)
    - Quartile Skewness (asymmetry based on quartiles)

    Parameters:
    cdf (Cdf): A cumulative distribution function.
    """

    print(f"Median: {median(cdf):.2f}")
    print(f"Interquartile range: {iqr(cdf):.2f}")
    print(f"Quartile skewness: {quartile_skewness(cdf):.2f}")

######################################
# Probability Density Function
######################################


def plot_gaussian_kde(
    data,
    figsize=(12, 8),
    xlabel="Feature",
    ylabel="Density",
    title="Gaussian KDE"
):
    """
    Plot the Gaussian KDE (Kernel Density Estimate) of a given dataset
    along with a normal distribution fitted to the data.

    Parameters:
        data (array-like): Input data to estimate the density from.
        figsize (tuple): Size of the matplotlib figure.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot and name for the Pdf object.
    """

    # KDE estimation
    kde = gaussian_kde(data)

    # Plot KDE
    plt.figure(figsize=figsize)

    # Domain of the PDF (min and max of data)
    domain = (np.min(data), np.max(data))
    kde_pdf = Pdf(kde, domain, name=title)

    # Fit normal distribution
    mean = np.mean(data)
    std_dev = np.std(data)
    normal_pdf = NormalPdf(mean, std_dev, name="Normal Model")

    normal_pdf.plot(ls=":", color="gray")
    kde_pdf.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def create_pmf_from_kde(sample_data, lower_bound, upper_bound, num_points=200, weights=None):
    """
    Estimate a Probability Mass Function (PMF) by approximating a Kernel Density Estimate (KDE).

    Parameters:
    ----------
    sample_data : array-like
        The input data sample to estimate the distribution from.
    lower_bound : float
        The minimum value in the range over which to evaluate the KDE.
    upper_bound : float
        The maximum value in the range over which to evaluate the KDE.
    num_points : int, optional
        Number of equally spaced points between lower_bound and upper_bound (default is 201).

    Returns:
    -------
    Pmf
        A probability mass function approximated from the KDE over the specified range.
    """
    kde_estimator = gaussian_kde(sample_data, weights=weights)
    x_values = np.linspace(lower_bound, upper_bound, num_points)
    density_values = kde_estimator(x_values)
    return Pmf(density_values, x_values)

def create_pdf_from_pmf(data, name="Estimated PDF"):
    """
    Create a Probability Density Function (PDF) by applying a Kernel Density Estimate (KDE)
    to a given Probability Mass Function (PMF).

    Parameters:
    ----------
    pmf : Pmf
        A probability mass function with discrete values (pmf.qs) and their probabilities (pmf.ps).
    name : str, optional
        Name to assign to the resulting PDF (default is "Estimated PDF").

    Returns:
    -------
    Pdf
        A continuous probability density function estimated via KDE from the input PMF.
    """
    pmf = create_pmf(data)
    kde = gaussian_kde(pmf.qs, weights=pmf.ps)
    domain = (np.min(pmf.qs), np.max(pmf.qs))
    return Pdf(kde, domain=domain, name=name)
