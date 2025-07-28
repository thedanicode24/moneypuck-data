import pandas as pd

def print_df_size(df):
    print("Number of samples: ", df.shape[0])
    print("Number of features: ", df.shape[1])