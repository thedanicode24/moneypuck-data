import os
import matplotlib.pyplot as plt
import seaborn as sns
from empiricaldist import Cdf

def save_histogram(
    df,
    column,
    output_dir,
    filename_prefix,
    title='Histogram',
    xlabel='Value',
    ylabel='Frequency',
    bins=30,
    kde=True, 
    color='skyblue'
):
    """
    Create and save a histogram from a specified column in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column : str
        Name of the column to plot.
    output_dir : str
        Path to the directory where the histogram image will be saved.
    filename_prefix : str
        Prefix for the saved file name.
    title : str, optional
        Title of the plot. Default is 'Histogram'.
    xlabel : str, optional
        Label for the x-axis. Default is 'Value'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Frequency'.
    bins : int, optional
        Number of histogram bins. Default is 30.
    kde : bool, optional
        Whether to include KDE curve. Default is True.
    color : str, optional
        Color for the histogram bars (e.g., 'blue', '#1f77b4', 'skyblue'). Default is 'skyblue'.
    """

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=bins, kde=kde, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    filename = f"{filename_prefix}_{column}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()


def save_boxplot(
    df,
    x_column,
    y_column,
    output_dir,
    filename_prefix,
    title='Boxplot',
    xlabel='X',
    ylabel='Y',
    order=None,
    palette=None
):
    """
    Create and save a boxplot from specified columns in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    x_column : str
        Column name for the x-axis categories.
    y_column : str
        Column name for the numeric y-axis values.
    output_dir : str
        Directory path to save the plot image.
    filename_prefix : str
        Prefix for the saved file name.
    title : str, optional
        Title of the plot. Default is 'Boxplot'.
    xlabel : str, optional
        Label for the x-axis. Default is 'X'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Y'.
    order : list, optional
        Order of categories on the x-axis.
    palette : dict or list, optional
        Colors for the boxes (used with hue).

    Returns:
    --------
    None
    """

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x=x_column,
        y=y_column,
        hue=x_column,
        order=order,
        palette=palette,
        dodge=False,
        legend=False
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    filename = f"{filename_prefix}_{y_column}_by_{x_column}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

def save_scatterplot(
    df,
    x_column,
    y_column,
    hue_column,
    output_dir,
    filename_prefix,
    title='Scatterplot',
    xlabel='X-axis',
    ylabel='Y-axis',
    palette=None,
    legend=True
):
    """
    Create and save a scatter plot using seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    x_column : str
        Column name for x-axis values.
    y_column : str
        Column name for y-axis values.
    hue_column : str
        Column name for color grouping (e.g. position).
    output_dir : str
        Path to the directory to save the plot.
    filename_prefix : str
        Prefix for the saved plot file.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for x-axis.
    ylabel : str, optional
        Label for y-axis.
    palette : dict or str, optional
        Color palette to use.
    legend : bool, optional
        Whether to show the legend. Default is True.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, palette=palette, legend=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    filename = f"{filename_prefix}_{y_column}_vs_{x_column}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()


def save_cdf(values, reference_value, output_dir, title="CDF", reference_label="Reference"):
    """
    Generates and saves a CDF (Cumulative Distribution Function) plot from a sequence of values,
    with a vertical line marking a reference value and a custom label.

    Parameters:
    -----------
    values : array-like
        A sequence of numeric values to compute the CDF from.
    reference_value : float
        A single numeric value to mark with a vertical line on the plot.
    output_dir : str
        Path where the plot will be saved.
    title : str, optional
        Title of the plot. Default is None.
    reference_label : str, optional
        Description of the reference value, shown in the legend. Default is 'Reference'.

    Returns:
    --------
    None
        The plot is saved to the specified output path.
    """
    cdf = Cdf.from_seq(values)

    plt.figure(figsize=(8,6))
    cdf.step()
    plt.axvline(reference_value, ls=":", color="gray", label=f"{reference_label}: {reference_value:.2f}")
    plt.xlabel("Value")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()

    filename = f"{output_dir}.png"

    plt.savefig(filename, dpi=300)
    plt.close()



def save_pmf(values, output_dir, title="PMF"):
    """
    Generate and save a PMF plot comparing the actual and observed distributions.

    Parameters
    ----------
    values : sequence
        A sequence of values from which to compute the PMF.
    output_dir : str
        The directory where the plot will be saved. It will be created if it doesn't exist.
    title : str
        A label for the plot title, typically describing the context or situation. Default: "PMF"
    filename : str, optional
        The name of the file to save the plot as. Default: "PMF"

    Returns
    -------
    str
        The full path to the saved plot file.
    """
    actual_pmf = Pmf.from_seq(values, name="Actual")
    observed_pmf = thinkstats.bias(actual_pmf, name="Observed")

    plt.figure(figsize=(8,6))
    thinkstats.two_bar_plots(actual_pmf, observed_pmf)

    plt.title(title)
    plt.tight_layout()
    filename = f"{output_dir}.png"
    plt.savefig(filename)
    plt.close()
