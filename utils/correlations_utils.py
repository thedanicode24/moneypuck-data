import matplotlib.pyplot as plt
import pandas as pd
from utils.thinkstats import jitter

def plot_scatter(val1, val2, figsize=(12,8), xlabel="Feature 1", ylabel="Feature 2"):
    plt.figure(figsize=figsize)
    plt.scatter(val1, val2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def plot_scatter_with_jitter(
        val1, val2, 
        figsize=(12,8), 
        std_dev=3, 
        size_points=20, 
        xlabel="Feature 1", 
        ylabel="Feature 2",
        alpha=1):
    val1_jittered = jitter(val1, std_dev)
    val2_jittered = jitter(val2, std_dev)
    plt.figure(figsize=figsize)
    plt.scatter(val1_jittered, val2, s=size_points, alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def plot_decile(df, feature1, feature2, figsize=(12,8), alpha=0.2, xlabel="Feature 1", ylabel="Feature 2", color="C0"):
    deciles = pd.qcut(df[feature1], 10, labels=False) + 1
    df_groupby = df.groupby(deciles)
    series_groupby = df_groupby[feature2]
    
    low = series_groupby.quantile(0.1)
    median = series_groupby.quantile(0.5)
    high = series_groupby.quantile(0.9)
    
    xs = df_groupby[feature1].median()

    x_vals = xs.sort_index().values
    y_low = low.sort_index().values
    y_med = median.sort_index().values
    y_high = high.sort_index().values
    
    plt.figure(figsize=figsize)
    plt.fill_between(x_vals, y_low, y_high, alpha=alpha)
    plt.plot(x_vals, y_med, color=color, label="Median")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()