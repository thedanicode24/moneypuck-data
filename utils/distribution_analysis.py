import numpy as np
from empiricaldist import FreqTab, Pmf, Cdf
from utils.thinkstats import cohen_effect_size, two_bar_plots, bias, percentile_rank, median, iqr, quartile_skewness, Pdf, NormalPdf
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

###############################
# Frequency Table
###############################

def plot_ftab(df, feature, xlabel="Feature", figsize=(12,8)):
    """
    Plot a frequency bar chart of a specific feature in a DataFrame.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name of the feature to analyze.
    xlabel (str, optional): Label for the x-axis. Default is "Feature".
    figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    """
    
    print_stats(df, feature)
    ftab = FreqTab.from_seq(df[feature], name=feature)
    plt.figure(figsize=figsize)
    ftab.bar()
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title("Frequency Table")
    plt.grid(True)

def plot_two_ftabs(df1, df2, feature, name1="Name1", name2="Name2", figsize=(12,8), xlabel="Feature"):
    """
    Plot two frequency bar charts for the same feature from two different DataFrames
    and print the Cohen's effect size between the distributions.

    Parameters:
    df1 (DataFrame): The first pandas DataFrame.
    df2 (DataFrame): The second pandas DataFrame.
    feature (str): The column name of the feature to analyze.
    name1 (str, optional): Label for the first dataset in the plot. Default is "Name1".
    name2 (str, optional): Label for the second dataset in the plot. Default is "Name2".
    figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    xlabel (str): Label for the x-axis. Default is "Feature".
    """

    print(f"Cohen's effect size: {cohen_effect_size(df1[feature], df2[feature]):.2f}")
    ftab1 = FreqTab.from_seq(df1[feature], name=name1)
    ftab2 = FreqTab.from_seq(df2[feature], name=name2)
    plt.figure(figsize=figsize)
    two_bar_plots(ftab1, ftab2)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title("Frequency Table")

def print_stats(df, feature):
    """
    Print basic statistics (mean, variance, standard deviation, and mode) for a given feature.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name of the feature to analyze.
    """

    print(f"Mean: {df[feature].mean():.3f}")
    print(f"Variance: {df[feature].var():.3f}")
    print(f"Standard deviation: {df[feature].std(ddof=0):.3f}")
    mode_values = df[feature].mode()
    if len(mode_values) == 1:
        print(f"Mode: {mode_values.iloc[0]}")
    else:
        print(f"Mode: {mode_values.values}")


###################################
# Probability Mass Function
####################################

def create_pmf(df, feature, name="pmf"):
    """
    Creates a Probability Mass Function (PMF) from a given DataFrame column.

    Parameters:
        df (DataFrame): The input DataFrame.
        feature (str): The column name to create the PMF from.
        name (str): The name to assign to the PMF.

    Returns:
        Pmf: The resulting PMF.
    """
        
    return Pmf.from_seq(df[feature], name=name)

def plot_pmf(df, feature, width=2, xlabel="Feature", ylabel="PMF", figsize=(12,8)):
    """
    Plots the actual and observed (biased) PMFs of a given feature from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame.
        feature (str): The column name to analyze.
        width (int): Bar width for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    """

    actual_pmf = create_pmf(df, feature, name="Actual")
    observed_pmf = bias(actual_pmf, name="Observed")

    plt.figure(figsize=figsize)
    two_bar_plots(actual_pmf, observed_pmf, width=width)
    plt.grid(True)
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

def plot_two_pmfs(df1, df2, feature, name1="Name1", name2="Name2", xlabel="Feature", ylabel="Probability", figsize=(12,8)):
    """
    Plots the PMFs of a feature from two different DataFrames for comparison.

    Parameters:
        df1 (DataFrame): The first DataFrame.
        df2 (DataFrame): The second DataFrame.
        feature (str): The feature/column name to analyze.
        name1 (str): Name label for the first PMF.
        name2 (str): Name label for the second PMF.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (tuple, optional): Dimension of the plot. Default is (12,8).
    """
        
    pmf1 = Pmf.from_seq(df1[feature], name=name1)
    pmf2 = Pmf.from_seq(df2[feature], name=name2)

    plt.figure(figsize=figsize)
    two_bar_plots(pmf1, pmf2)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_diff_pmfs(df1, df2, feature, name1="Name1", name2="Name2", xlabel="Feature", ylabel="Difference (%)", figsize=(12,8)):
    """
    Plots the percentage difference between the PMFs of a feature from two DataFrames.

    Parameters:
        df1 (DataFrame): The first DataFrame.
        df2 (DataFrame): The second DataFrame.
        feature (str): The feature/column name to analyze.
        name1 (str): Name label for the first PMF.
        name2 (str): Name label for the second PMF.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """

    pmf1 = Pmf.from_seq(df1[feature], name=name1)
    pmf2 = Pmf.from_seq(df2[feature], name=name2)

    diff = (pmf1-pmf2)*100
    plt.figure(figsize=figsize)
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

    print_percentile_stats(cdf)
    plt.figure(figsize=figsize)
    cdf.step()
    plt.axvline(ref, ls=":", color="red", label=f"{label}: {ref:.2f}")
    plt.legend()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_two_cdfs(values1, values2, name1="Name1", name2="Name2", xlabel="Feature", ylabel="CDF", figsize=(12,8)):
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

    pmf1 = Pmf.from_seq(values1, name=name1)
    pmf2 = Pmf.from_seq(values2, name=name2)
    cdf1 = pmf1.make_cdf()
    cdf2 = pmf2.make_cdf()
    plt.figure(figsize=figsize)
    cdf1.plot()
    cdf2.plot(alpha=0.5)
    plt.legend()
    plt.grid(True)
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
    num_points=1000,
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
        num_points (int): Number of points to evaluate the KDE and normal PDF over.
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
    plt.grid(True)
    plt.show()

def plot_kde(
    df, feature, 
    dataset_name="data", 
    figsize=(12, 8), 
    xlabel="Interval", 
    ylabel="Density",
    title="Kernel Density Estimate",
    num_points=1000
):
    """
    Computes and plots a Kernel Density Estimate (KDE) based on the PMF of a given feature in a DataFrame.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data.
    feature_column : str
        The name of the column to analyze.
    dataset_name : str, optional (default="data")
        A label used when generating the PMF.
    figsize : tuple, optional (default=(12, 8))
        The size of the figure to be plotted.
    xlabel : str, optional (default="Interval (minutes)")
        Label for the x-axis.
    ylabel : str, optional (default="Density")
        Label for the y-axis.

    Returns:
    -------
    None
    """

    pmf = create_pmf(df, feature, name=dataset_name)
    
    # Estimate KDE using the PMF's values and probabilities
    kde_estimator = gaussian_kde(pmf.qs, weights=pmf.ps)
    domain = (np.min(pmf.qs), np.max(pmf.qs))
    
    # Generate KDE plot
    plt.figure(figsize=figsize)
    plt.title(title)

    estimated_pdf = Pdf(kde_estimator, domain=domain, name="Estimated Density")
    estimated_pdf.plot(ls=":", color="gray")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

