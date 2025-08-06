import matplotlib.pyplot as plt
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