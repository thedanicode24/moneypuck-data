import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_interactive_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    filter_col: str,
    tooltip_cols: list,
    title: str = '',
    default_filter_value: str = None,
    point_size: int = 60,
    width: int = 600,
    height: int = 400
):
    """
    Create an interactive scatter plot using Altair with a dropdown filter.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to plot.
        x (str): Column name to use for the x-axis.
        y (str): Column name to use for the y-axis.
        category (str): Column name to use for color encoding (categorical variable).
        filter_col (str): Column name to use for dropdown filtering.
        tooltip_cols (list): List of column names to show in tooltips on hover.
        title (str, optional): Title of the chart. Defaults to ''.
        default_filter_value (str, optional): Default selected value in the dropdown. 
            If None, the first unique value of the filter_col is used.
        point_size (int, optional): Size of the scatter plot points. Defaults: 60.
        width (int, optional): Width of the chart in pixels. Defaults: 600.
        height (int, optional): Height of the chart in pixels. Defaults: 400.

    Returns:
        alt.Chart: An Altair Chart object representing the interactive scatter plot.
    """

    options = df[filter_col].unique().tolist()
    
    if default_filter_value is None:
        default_filter_value = options[0]

    dropdown = alt.binding_select(options=options, name=f'{filter_col}: ')
    param = alt.param(name=f'{filter_col}_param', bind=dropdown, value=default_filter_value)

    chart = alt.Chart(df).mark_circle(size=point_size).encode(
        x=x,
        y=y,
        color=category,
        tooltip=tooltip_cols
    ).add_params(
        param
    ).transform_filter(
        f'datum.{filter_col} == {filter_col}_param'
    ).properties(
        title=title,
        width=width,
        height=height
    )

    return chart


def create_interactive_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    filter_col: str,
    title: str = '',
    default_filter_value: str = None,
    width: int = 600,
    height: int = 400
):
    """
    Create an interactive boxplot with a dropdown filter using Altair.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to visualize.
        x (str): Column name to use for the x-axis (typically a categorical variable).
        y (str): Column name to use for the y-axis (typically numeric).
        filter_col (str): Column used to filter the data via dropdown menu.
        title (str, optional): Title of the chart. Defaults to ''.
        default_filter_value (str, optional): Default selected value in the dropdown.
            If None, the first unique value of filter_col is used.
        width (int, optional): Width of the chart in pixels. Defaults to 600.
        height (int, optional): Height of the chart in pixels. Defaults to 400.

    Returns:
        alt.Chart: An Altair Chart object representing the interactive boxplot.
    """

    options = df[filter_col].unique().tolist()
    if default_filter_value is None:
        default_filter_value = options[0]

    dropdown = alt.binding_select(options=options, name=f'{filter_col}: ')
    param = alt.param(name=f'{filter_col}_param', bind=dropdown, value=default_filter_value)

    chart = alt.Chart(df).mark_boxplot().encode(
        x=x,
        y=y
    ).add_params(
        param
    ).transform_filter(
        f'datum.{filter_col} == {filter_col}_param'
    ).properties(
        title=title,
        width=width,
        height=height
    )

    return chart

def plot_histograms_by_group(
    df: pd.DataFrame,
    group_col: str,
    hist_col: str,
    derived_cols_funcs: dict = None,
    bins: int = 30,
    figsize_per_plot: tuple = (6,5),
    kde: bool = True
):
    """
    Plot histograms of a specified column grouped by unique values in another column.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        group_col (str): The column name to group by.
        hist_col (str): The numeric column to plot histogram of.
        derived_cols_funcs (dict, optional): 
            Dictionary where keys are new column names and values are functions
            that take df and return a Series to be added to df. Default is None.
        bins (int, optional): Number of bins for the histogram. Default is 30.
        figsize_per_plot (tuple, optional):
            Width and height per subplot in inches. Default is (6, 5).
        kde (bool, optional): Whether to show KDE curve. Default is True.

    Returns: 
        None: Displays the plot.
    """
    if derived_cols_funcs:
        for new_col, func in derived_cols_funcs.items():
            df[new_col] = func(df)

    groups = df[group_col].unique().tolist()
    n = len(groups)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize_per_plot[0], figsize_per_plot[1]*n))
    plt.subplots_adjust(hspace=0.4)

    if n==1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        df_g = df[df[group_col] == group]
        sns.histplot(df_g[hist_col], bins=bins, kde=kde, ax=ax)
        ax.set_title(f"Distribution {hist_col} - {group_col}: {group}")
        ax.set_xlabel(hist_col)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()