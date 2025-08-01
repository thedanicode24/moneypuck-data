from empiricaldist import FreqTab
from utils import thinkstats

def plot_ftab(df, feature, xlabel="Feature", ylabel="Frequency"):
    """
    Plot a frequency bar chart of a specific feature in a DataFrame.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name of the feature to analyze.
    xlabel (str, optional): Label for the x-axis. Default is "Feature".
    ylabel (str, optional): Label for the y-axis. Default is "Frequency".
    """
    
    print_stats(df, feature)
    ftab = FreqTab.from_seq(df[feature], name=feature)
    ftab.bar()
    thinkstats.decorate(xlabel=xlabel, ylabel=ylabel, legend=False)

def plot_two_ftabs(df1, df2, feature, name1="Name1", name2="Name2"):
    """
    Plot two frequency bar charts for the same feature from two different DataFrames
    and print the Cohen's effect size between the distributions.

    Parameters:
    df1 (DataFrame): The first pandas DataFrame.
    df2 (DataFrame): The second pandas DataFrame.
    feature (str): The column name of the feature to analyze.
    name1 (str, optional): Label for the first dataset in the plot. Default is "Name1".
    name2 (str, optional): Label for the second dataset in the plot. Default is "Name2".
    """

    print(f"Cohen's effect size: {thinkstats.cohen_effect_size(df1[feature], df2[feature]):.2f}")
    ftab1 = FreqTab.from_seq(df1[feature], name=name1)
    ftab2 = FreqTab.from_seq(df2[feature], name=name2)
    thinkstats.two_bar_plots(ftab1, ftab2)

def print_stats(df, feature):
    """
    Print basic statistics (mean, variance, standard deviation, and mode) for a given feature.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.
    feature (str): The column name of the feature to analyze.
    """

    print(f"Mean: {df[feature].mean():.2f}")
    print(f"Variance: {df[feature].var():.2f}")
    print(f"Standard deviation: {df[feature].std(ddof=0):.2f}")
    mode_values = df[feature].mode()
    if len(mode_values) == 1:
        print(f"Mode: {mode_values.iloc[0]}")
    else:
        print(f"Mode: {mode_values.values}")

