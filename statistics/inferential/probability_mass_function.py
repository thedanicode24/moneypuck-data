from empiricaldist import Pmf
from thinkstats import decorate, two_bar_plots
import matplotlib as plt

"""
Glossary:
    - Probability Mass Function (PMF): A function that represents a distribution by mapping each quantity to its probability.
"""

def get_pmf(df):
    """
    Generate a PMF from a sequence.

    Args:
        df (array-like): A sequence of discrete values (e.g., list, Series).

    Returns:
        Pmf: The PMF of the input sequence.
    """
    return Pmf.from_seq(df)

def pmf_bar(df, xlabel="Value", ylabel="PMF"):
    """
    Plot the PMF of a sequence as a bar chart.

    Args:
        df (array-like): A sequence of discrete values (e.g., list, Series).
        xlabel (str): Label for the x-axis (default: "Value").
        ylabel (str): Label for the y-axis (default: "PMF").

    Returns:
        None
    """
    pmf = get_pmf(df)
    pmf.bar()
    decorate(xlabel=xlabel, ylabel=ylabel)

def pmf_line(df, xlabel="Value", ylabel="Probability"):
    """
    Plot the PMF of a sequence as a line chart.

    Args:
        df (array-like): A sequence of discrete values (e.g., list, Series).
        xlabel (str): Label for the x-axis. Default is "Value".
        ylabel (str): Label for the y-axis. Default is "PMF".

    Returns:
        None
    """
    pmf = get_pmf(df)
    pmf.plot()
    decorate(xlabel=xlabel, ylabel=ylabel)

def pmf_var(df):
    """
    Compute the variance of the PMF of a sequence.

    Args:
        df (array-like): A sequence of discrete values.

    Returns:
        float: Variance of the PMF.
    """
    pmf = get_pmf(df)
    return pmf.var()

def pmf_std(df):
    """
    Compute the standard deviation of the PMF of a sequence.

    Args:
        df (array-like): A sequence of discrete values.

    Returns:
        float: Standard deviation of the PMF.
    """
    pmf = get_pmf(df)
    return pmf.std()

def pmf_mode(df):
    """
    Compute the mode (most likely value) of the PMF.

    Args:
        df (array-like): A sequence of discrete values.

    Returns:
        int or float: Mode of the PMF.
    """
    pmf = get_pmf(df)
    return pmf.mode()

def biased_pmf(pmf, name="Observed"):
    """
    Applies a size bias to a probability mass function (PMF).

    This function modifies the input PMF by weighting each probability 
    by the corresponding value (qs), effectively biasing the distribution 
    toward larger values. The resulting PMF is then normalized.

    Parameters:
    - pmf: a Pmf object representing the original distribution.
    - name: a string to name the new biased PMF (default: "Observed").

    Returns:
    - A new normalized Pmf object with size-biased probabilities.
    """
    ps = pmf.ps * pmf.qs
    new_pmf = Pmf(ps, pmf.qs, name=name)
    new_pmf.normalize()
    return new_pmf

def plot_biased_pmf(actual_pmf, observed_pmf, width=2, xlabel="Value", ylabel="Probability"):
    """
    Plots a comparison between the actual and the size-biased PMF.

    This function creates a side-by-side bar plot showing the difference 
    between the original (actual) distribution and the biased (observed) 
    distribution. It's useful for visualizing how size bias affects the 
    perceived distribution.

    Parameters:
    - actual_pmf: the original unbiased Pmf object.
    - observed_pmf: the biased Pmf object (e.g. after applying bias()).
    - width: width of the bars in the plot (default: 2).
    - xlabel: label for the x-axis (default: "Value").
    - ylabel: label for the y-axis (default: "PMF").

    Returns:
    - None. Displays the plot.
    """
    two_bar_plots(actual_pmf, observed_pmf, width=width)
    decorate(xlabel=xlabel, ylabel=ylabel)

def debiased_pmf(pmf, name):
    """
    Reverses size bias from a probability mass function (PMF).

    This function adjusts a size-biased PMF by dividing each probability 
    by the corresponding value (qs), effectively removing the bias toward 
    larger values. The resulting PMF is then normalized to ensure it sums to 1.

    Parameters:
    - pmf: a size-biased Pmf object.
    - name: a string to name the new unbiased PMF.

    Returns:
    - A new normalized Pmf object with the size bias removed.
    """

    ps = pmf.ps / pmf.qs
    new_pmf = Pmf(ps, pmf.qs, name=name)
    new_pmf.normalize()
    return new_pmf

def plots_two_pmf(
        df1, df2, 
        xlim=[20,50], 
        name1="First", 
        name2="Others", 
        xlabel="Value", 
        ylabel="Probability"
        ):
    
    """
    Computes and plots the PMFs of two datasets for comparison.

    This function computes the probability mass functions (PMFs) of two input
    datasets and creates a side-by-side bar plot to visualize their differences. 
    Useful for comparing distributions (e.g. one biased, one not, or different groups).

    Parameters:
    - df1: first dataset (e.g. a pandas Series or DataFrame column).
    - df2: second dataset to compare against the first.
    - xlim: list specifying the x-axis limits for the plot (default: [20, 50]).
    - name1: label for the first dataset's PMF (default: "First").
    - name2: label for the second dataset's PMF (default: "Others").
    - xlabel: label for the x-axis (default: "Value").
    - ylabel: label for the y-axis (default: "Probability").

    Returns:
    - None. Displays the plot.
    """

    pmf1 = get_pmf(df1, name=name1)
    pmf2 = get_pmf(df2, name=name2)
    two_bar_plots(pmf1, pmf2)
    decorate(xlabel=xlabel, ylabel=ylabel, xlim=xlim)

def plot_pmf_diff(
        pmf1, pmf2, values, 
        xlabel="Value", 
        ylabel="Difference (%)", 
        title="PMF Difference"
        ):
    """
    Plots the percentage difference between two PMFs over a set of values.

    Parameters:
    - pmf1: the first Pmf object.
    - pmf2: the second Pmf object.
    - values: iterable of values to evaluate the PMFs on.
    - xlabel : label for the x-axis (default: "Value").
    - ylabel: label for the y-axis (default: "Difference (%)").
    - title: title of the plot (default: "PMF Difference").

    Returns:
    - None. Displays a bar plot of percentage differences.
    """
    diffs = pmf1(values) - pmf2(values)
    plt.bar(values, diffs * 100)
    plt.title(title)
    decorate(xlabel=xlabel, ylabel=ylabel)

def pmf_skewness(pmf):
    """
    Computes the skewness of a probability mass function (PMF).

    Skewness measures the asymmetry of a distribution:
    - Positive skew: long tail on the right.
    - Negative skew: long tail on the left.
    - Zero skew: symmetric distribution.

    Parameters:
    - pmf: a Pmf object.

    Returns:
    - A float representing the skewness of the distribution.
    """
    return pmf.skew()