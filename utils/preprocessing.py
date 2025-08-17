import pandas as pd

def one_hot_encoding(df, columns):
    df = pd.get_dummies(df, columns=columns)
    return df