from utils import thinkstats
from empiricaldist import Cdf, Pmf
import matplotlib.pyplot as plt
from utils import thinkstats

def create_cdf(values):
    """
    Create a CDF (Cumulative Distribution Function) from a sequence of values.

    Parameters:
    values (sequence): The input data.

    Returns:
    Cdf: The cumulative distribution function of the input data.
    """
    return Cdf.from_seq(values)

def percentile_rank(ref, values, label="Reference"):
    """
    Print the percentile rank of a reference value compared to a set of values.

    Parameters:
    ref (float): The reference value.
    values (sequence): The dataset to compare against.
    label (str, optional): A label for the reference value, used in the print statement.
    """
    print(f"{label} - Percentile rank: {thinkstats.percentile_rank(ref, values):.2f}")

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
    plt.axvline(ref, ls=":", color="red", label=f"{label}: {int(ref)}")
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
    cdf1.plot(ls="--")
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

    print(f"Median: {thinkstats.median(cdf):.2f}")
    print(f"Interquartile range: {thinkstats.iqr(cdf):.2f}")
    print(f"Quartile skewness: {thinkstats.quartile_skewness(cdf):.2f}")
