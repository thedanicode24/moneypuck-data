from empiricaldist import FreqTab

def mode(df):
    """
    Mode: The most frequent quantity in a sample, or one of the most frequent quantities.
    """
    ftab = FreqTab.from_seq(df)
    return ftab.mode()

def mean(df):
    return df.mean()