import numpy as np

def perc_effect_size(df, group1, group2):
    first_mean = group1.mean()
    other_mean = group2.mean()
    diff = first_mean - other_mean
    return diff / df.mean() * 100

def std_effect_size(df, group1, group2):
    first_mean = group1.mean()
    other_mean = group2.mean()
    diff = first_mean - other_mean
    return diff / df.std()

def cohen_effect_size(group1, group2):
    """
    Cohen’s effect size: A standardized statistic that quantifies the difference in the means of two groups.
    """
    diff = group1.mean() - group2.mean()
    v1, v2 = group1.var(), group2.var()
    n1, n2 = group1.count(), group2.count()
    pooled_var = (n1 * v1 + n2 * v2) / (n1 + n2)
    return diff / np.sqrt(pooled_var)