import numpy as np

def std_pooled_deviation(group1, group2):
    """
    Pooled standard deviation: A statistic that combines data from two or more groups to compute a common standard deviation.
    """
    v1, v2 = group1.var(), group2.var()
    n1, n2 = group1.count(), group2.count()
    pooled_var = (n1 * v1 + n2 * v2) / (n1 + n2)
    np.sqrt(pooled_var)
