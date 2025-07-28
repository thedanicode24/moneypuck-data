from empiricaldist import FreqTab
from thinkstats import decorate

"""
Glossary:
    - Frequency: The number of times a value appears in a sample.
    - Frequency table: A mapping from values to frequencies.
"""

def frequency_table(df, xlabel="Value", ylabel="Frequency"):
    """
    Plot the frequency table as a bar chart
    """
    ftab = FreqTab.from_seq(df)
    ftab.bar()
    decorate(xlabel=xlabel, ylabel=ylabel)

def get_frequency(df, val):
    ftab = FreqTab.from_seq(df)
    return ftab(val)

def get_frequencies(df):
    ftab = FreqTab.from_seq(df)
    return ftab.fs

def print_frequencies(df):
    ftab = FreqTab.from_seq(df)
    for x, freq in ftab.items():
        print(x, freq)

    
def plots_two_bar(df1, df2, width=0.45, name1="first", name2="others"):
    """
    The following function plots two frequency tables side-by-side.
    """
    ftab1 = FreqTab.from_seq(df1, name=name1)
    ftab2 = FreqTab.from_seq(df2, name=name2)
    ftab1.bar(align="edge", width=-width)
    ftab2.bar(align="edge", width=width, alpha=0.5)
    decorate(xlabel="Weeks", ylabel="Frequency", xlim=[20, 50])
