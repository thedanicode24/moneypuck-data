import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from utils.thinkstats import jitter, standardize


def plot_scatter(val1, val2, figsize=(12,8), xlabel="Feature 1", ylabel="Feature 2"):
    plt.figure(figsize=figsize)
    plt.title("Scatter plot")
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
    plt.title("Scatter plot with jitter")
    plt.scatter(val1_jittered, val2_jittered, s=size_points, alpha=alpha)
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
    plt.title("Decile plot")
    plt.fill_between(x_vals, y_low, y_high, alpha=alpha)
    plt.plot(x_vals, y_med, color=color, label="Median")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()

def plot_zscore_and_corr(vals1, vals2, 
                         lw=1, 
                         alpha=0.5, 
                         figsize=(12,8),
                         color_line="gray", 
                         label1="Feature 1", 
                         label2="Feature 2", 
                         start_sample=0,
                         end_sample=None):
    """
    Pearson correlation coefficient: A statistic that measures the strength and sign (positive or negative) of the linear relationship between two variables.
    """
    # zscores
    feature1_standard = standardize(vals1)
    feature2_standard = standardize(vals2)

    if end_sample is None:
        end_sample = min(len(feature1_standard), len(feature2_standard))
    
    plt.figure(figsize=figsize)

    plt.subplot(2, 1, 1)
    plt.axhline(0, color=color_line, lw=lw, alpha=alpha)
    plt.plot(feature1_standard.values[start_sample:end_sample], label=label1)
    plt.legend()
    plt.ylabel("Z-score")
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.axhline(0, color=color_line, lw=lw, alpha=alpha)
    plt.plot(feature2_standard.values[start_sample:end_sample], label=label2, color="C1")
    plt.legend()
    plt.ylabel("Z-score")
    plt.grid()
    
    print(f"Pearson correlation coefficient: {np.corrcoef(vals1, vals2)[0, 1]:.3f}")

def plot_rank_correlation(vals1, vals2, 
                          figsize=(12,8), 
                          size_points=5, 
                          alpha=0.5, 
                          xlabel="Feature 1", 
                          ylabel="Feature 2"):
    vals1_rank = vals1.rank(method="first")
    vals2_rank = vals2.rank(method="first")

    plt.figure(figsize=figsize)
    plt.scatter(vals1_rank, vals2_rank, s=size_points, alpha=alpha)
    plt.title("Rank correlation")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

    print(f"Spearman's rank correlation coefficient: {spearmanr(vals1, vals2).statistic:.3f}")
    
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_top_correlations(data, target_feature, top_n=10, cmap=plt.cm.coolwarm, figsize=(9,6)):
    """
    Plot top correlated features with the target feature from raw data.

    Parameters:
    - data: pandas DataFrame with all features (including target)
    - target_feature: string, name of target column
    - top_n: int, number of top correlated features to show (default 10)
    - cmap: matplotlib colormap
    - figsize: tuple for figure size
    """

    correlations = data.corr()[target_feature].drop(target_feature)

    top_features = correlations.abs().sort_values(ascending=False).head(top_n).index
    top_correlations = correlations.loc[top_features]
    
    df = pd.DataFrame({
        'feature': top_correlations.index,
        'correlation': top_correlations.values
    })
    
    norm_values = np.abs(df['correlation']) / np.max(np.abs(df['correlation']))
    colors = cmap(norm_values)
    color_list = [colors[i] for i in range(len(colors))]
    
    plt.figure(figsize=figsize)
    
    sns.barplot(data=df, x='correlation', y='feature', palette=color_list, hue='feature', dodge=False, legend=False)
    
    plt.xlabel(f"Correlation with {target_feature}")
    plt.title(f"Top {top_n} features correlated with {target_feature}")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    xmin = df['correlation'].min() * 1.1 if df['correlation'].min() < 0 else 0
    xmax = df['correlation'].max() * 1.1 if df['correlation'].max() > 0 else 0
    plt.xlim(xmin, xmax)
    
    for i, v in enumerate(df['correlation']):
        if v < 0:
            plt.text(v - 0.02, i, f"{v:.2f}", color='black', va='center', ha='right')
        else:
            plt.text(v + 0.02, i, f"{v:.2f}", color='black', va='center', ha='left')
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()
