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

import pandas as pd

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

def drop_duplicate_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Checks for duplicate columns (by values) among the given list of columns.
    Drops all duplicate columns, keeping only the first occurrence of each set of identical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame.
    columns : list of str
        List of column names to check for duplicates.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame with duplicate columns dropped (keeping only the first of each duplicate group).
    """
    to_drop = []

    for i, col_i in enumerate(columns):
        if col_i in to_drop:
            continue
        for col_j in columns[i+1:]:
            if col_j in to_drop:
                continue
            if (df[col_i] == df[col_j]).all():
                to_drop.append(col_j)

    df = df.drop(columns=to_drop)
    print(f"Dropped columns: {to_drop}")
    print(f"Number of features: {df.shape[1]}")
    return df
