from empiricaldist import FreqTab

def get_ouliers(df, n=10):
    """
    Outlier: An extreme quantity in a distribution.
    """
    ftab = FreqTab.from_seq(df)
    smallest = ftab[:n]
    largest = ftab[-n:]
    return smallest, largest
