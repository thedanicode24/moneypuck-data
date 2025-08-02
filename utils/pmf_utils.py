import numpy as np
from empiricaldist import Pmf
from utils import thinkstats
import matplotlib.pyplot as plt

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
    observed_pmf = thinkstats.bias(actual_pmf, name="Observed")

    plt.figure(figsize=figsize)
    thinkstats.two_bar_plots(actual_pmf, observed_pmf, width=width)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    print("Actual PMF: ")
    print_stats(actual_pmf)
    print()
    print(f"Observed PMF: ")
    print_stats(observed_pmf)

def print_stats(pmf):
    """
    Prints statistical measures for a given PMF.

    Parameters:
        pmf (Pmf): The PMF to analyze.
    """
        
    print(f"Mean: {pmf.mean():.2f}")
    print(f"Variance: {pmf.var():.2f}")
    print(f"Standard deviation: {pmf.std():.2f}")
    print(f"Mode: {pmf.mode()}")
    print(f"Skewness: {pmf_skewness(pmf):.2f}")

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
    thinkstats.two_bar_plots(pmf1, pmf2)
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


