from empiricaldist import Pmf
from thinkstats import decorate

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
        xlabel (str): Label for the x-axis. Default is "Value".
        ylabel (str): Label for the y-axis. Default is "PMF".

    Returns:
        None
    """
    pmf = get_pmf(df)
    pmf.bar()
    decorate(xlabel=xlabel, ylabel=ylabel)

def pmf_line(df, xlabel="Value", ylabel="PMF"):
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