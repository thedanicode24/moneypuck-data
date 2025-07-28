from empiricaldist import FreqTab
from thinkstats import decorate

"""
Glossary:
    - Frequency: The number of times a value appears in a sample.
    - Frequency table: A mapping from values to frequencies.
"""

def get_ftab(df, name=None):
    """
    Create a frequency table from the given dataset.

    Parameters:
        df (sequence): A sequence of values (e.g., list, Series, array).

    Returns:
        FreqTab: The frequency table object.
    """
    return FreqTab.from_seq(df, name=name)

def plot_frequency_table(df, xlabel="Value", ylabel="Frequency"):
    """
    Plot a frequency table as a bar chart.

    Parameters:
        ftab (FreqTab): A frequency table object.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Returns:
        None
    """
    ftab = get_ftab(df)
    ftab.bar()
    decorate(xlabel=xlabel, ylabel=ylabel)

def get_frequency(df, val):
    """
    Get the frequency of a specific value in the dataset.

    Parameters:
        df (sequence): The data to analyze.
        val: The value whose frequency is to be returned.

    Returns:
        int: The frequency of the specified value.
    """
    ftab = get_ftab(df)
    return ftab(val)

def get_frequencies(df):
    """
    Get the list of frequencies from the dataset.

    Parameters:
        df (sequence): The input data.

    Returns:
        list: List of frequencies corresponding to each unique value.
    """
    ftab = get_ftab(df)
    return ftab.fs

def print_frequencies(df):
    """
    Print each unique value and its corresponding frequency.

    Parameters:
        df (sequence): The input data.

    Returns:
        None
    """
    ftab = get_ftab(df)
    for x, freq in ftab.items():
        print(x, freq)

    
def plots_two_ftab(
        df1, df2, 
        xlim=[20,50], 
        width=0.45, 
        alpha=0.5, 
        name1="first", 
        name2="others", 
        xlabel="Value", 
        ylabel="Frequency"):
    """
    Plot two frequency tables side-by-side for comparison.

    Parameters:
        df1 (sequence): First dataset.
        df2 (sequence): Second dataset.
        xlim (list): Limits for the x-axis.
        width (float): Width of the bars.
        alpha (float): Transparency of the second dataset bars.
        name1 (str): Label for the first dataset.
        name2 (str): Label for the second dataset.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Returns:
        None
    """
    ftab1 = get_ftab(df1, name=name1)
    ftab2 = get_ftab(df2, name=name2)
    ftab1.bar(align="edge", width=-width)
    ftab2.bar(align="edge", width=width, alpha=alpha)
    decorate(xlabel=xlabel, ylabel=ylabel, xlim=xlim)
