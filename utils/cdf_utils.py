from utils import thinkstats
from empiricaldist import Cdf, Pmf
import matplotlib.pyplot as plt

def create_cdf(values):
    return Cdf.from_seq(values)

def percentile_rank(ref, values, label="Reference"):
    print(f"{label} - Percentile rank: {thinkstats.percentile_rank(ref, values):.2f}")

def plot_cdf(ref, values, figsize=(12,8), label="Reference", xlabel="Feature", ylabel="CDF"):
    cdf = create_cdf(values)

    plt.figure(figsize=figsize)
    cdf.step()
    plt.axvline(ref, ls=":", color="red", label=f"{label}: {int(ref)}")
    plt.legend()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_two_cdfs(values1, values2, name1="Name1", name2="Name2", xlabel="Feature", ylabel="CDF", figsize=(12,8)):
    pmf1 = Pmf.from_seq(values1, name=name1)
    pmf2 = Pmf.from_seq(values2, name=name2)
    cdf1 = pmf1.make_cdf()
    cdf2 = pmf2.make_cdf()
    plt.figure(figsize=figsize)
    cdf1.plot(ls="--")
    cdf2.plot(alpha=0.5)
    plt.legend()
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)