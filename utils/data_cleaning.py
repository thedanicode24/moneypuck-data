import pandas as pd

def print_df_size(df: pd.DataFrame) -> None:
    """
    Prints the number of rows (samples) and columns (features) in a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame whose size will be printed.

    Returns:
    --------
    None
        This function prints the size of the DataFrame and does not return any value.
    """
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")

def report_nan(df: pd.DataFrame) -> None:
    """
    Checks for missing (NaN) values in a pandas DataFrame and prints a summary.

    If there are missing values, it prints the number of NaNs per column (only for columns with at least one NaN).
    If there are no missing values, it prints a message indicating the absence of NaNs.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to check for missing values.

    Returns:
    --------
    None
        This function only prints output and does not return any value.
    """
    if df.isna().any().any():
        nan_counts = df.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        print("Missing values detected:\n")
        print(nan_counts)
    else:
        print("No missing values found.")

def save_column_names(df: pd.DataFrame, filename: str = "names_columns.txt") -> None:
    """
    Saves the column names of a pandas DataFrame to a text file, one per line.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame whose column names will be saved.
    filename : str, optional
        The name of the text file to save the column names. Default is "names_columns.txt".

    Returns:
    --------
    None
        This function writes to a file and does not return any value.
    """
    cols = df.columns.tolist()
    with open(filename, "w") as f:
        for col in cols:
            f.write(col + '\n')
    print(f"Saved: {filename}")

def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all duplicate columns, keeping only the first occurrence of each set of identical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with duplicate columns dropped (keeping only the first of each duplicate group).
    """

    duplicate_cols = df.columns[df.T.duplicated()]
    
    if len(duplicate_cols) > 0:
        print("Removed duplicate columns:", list(duplicate_cols))
    else:
        print("No duplicate columns found.")
    
    df = df.loc[:, ~df.T.duplicated()]
    print_df_size(df)
    
    return df