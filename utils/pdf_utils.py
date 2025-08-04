import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from utils import thinkstats, pmf_utils

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
    kde_pdf = thinkstats.Pdf(kde, domain, name=title)

    # Fit normal distribution
    mean = np.mean(data)
    std_dev = np.std(data)
    normal_pdf = thinkstats.NormalPdf(mean, std_dev, name="Normal Model")

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
    xlabel="Interval (minutes)", 
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

    pmf = pmf_utils.create_pmf(df, feature, name=dataset_name)
    
    # Estimate KDE using the PMF's values and probabilities
    kde_estimator = gaussian_kde(pmf.qs, weights=pmf.ps)
    domain = (np.min(pmf.qs), np.max(pmf.qs))
    
    # Generate KDE plot
    plt.figure(figsize=figsize)
    plt.title(title)

    estimated_pdf = thinkstats.Pdf(kde_estimator, domain=domain, name="Estimated Density")
    estimated_pdf.plot(ls=":", color="gray")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()

